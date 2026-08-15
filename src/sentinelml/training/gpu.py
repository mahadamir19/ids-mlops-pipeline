"""GPU data helpers for XGBoost-backed SentinelML models."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinelml.data.config import PROJECT_ROOT

CPU_TREE_METHOD = "hist"
CUDA_TREE_METHOD = "hist"
GPU_ONLY_XGBOOST_PARAMS = ("gpu_id", "predictor")
GPU_TREE_METHODS = {"gpu_hist"}
GPU_SAMPLING_METHODS = {"gradient_based"}


def configure_gpu_runtime_environment() -> None:
    """Keep GPU library scratch files inside the project when possible."""

    tmp_dir = PROJECT_ROOT / ".tmp"
    cupy_cache_dir = tmp_dir / "cupy-cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cupy_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cupy_cache_dir))
    if not _tempdir_is_writable(Path(tempfile.gettempdir())):
        os.environ["TEMP"] = str(tmp_dir)
        os.environ["TMP"] = str(tmp_dir)
        tempfile.tempdir = str(tmp_dir)


def estimator_requests_cuda(estimator: Any) -> bool:
    """Return whether an estimator is configured to run on CUDA."""

    get_params = getattr(estimator, "get_params", None)
    params = get_params() if callable(get_params) else {}
    device = str(params.get("device", "")).lower()
    tree_method = str(params.get("tree_method", "")).lower()
    predictor = str(params.get("predictor", "")).lower()
    return (
        device.startswith("cuda")
        or device.startswith("gpu")
        or tree_method in GPU_TREE_METHODS
        or predictor == "gpu_predictor"
    )


def estimator_effective_device(estimator: Any) -> str | None:
    """Read the fitted XGBoost booster device from its saved configuration."""

    booster = _booster(estimator)
    if booster is None:
        return None
    try:
        config = json.loads(booster.save_config())
    except Exception:
        return None
    return config.get("learner", {}).get("generic_param", {}).get("device")


def force_estimator_cpu(estimator: Any) -> Any:
    """Best-effort CPU coercion for loaded estimators in non-GPU runtimes."""

    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list):
        for _, step in steps:
            force_estimator_cpu(step)

    set_params = getattr(estimator, "set_params", None)
    if callable(set_params):
        try:
            params = _cpu_set_params_for_estimator(estimator)
            set_params(**params)
        except (TypeError, ValueError):
            pass
    elif hasattr(estimator, "device"):
        try:
            estimator.device = "cpu"
        except Exception:
            pass

    booster = _booster(estimator)
    if booster is not None:
        try:
            booster.set_param({"device": "cpu", "tree_method": CPU_TREE_METHOD})
        except Exception:
            pass
    return estimator


def normalize_xgboost_execution_params(
    params: dict[str, Any],
    *,
    requested_device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize XGBoost runtime-only parameters for the active environment."""

    requested = str(requested_device or "cpu").strip().lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("XGBoost execution device must be cpu or cuda")

    normalized = dict(params)
    removed: dict[str, Any] = {}
    changed: dict[str, dict[str, Any]] = {}

    def set_changed(key: str, value: Any) -> None:
        previous = normalized.get(key)
        if previous != value:
            changed[key] = {"from": previous, "to": value}
        normalized[key] = value

    if requested == "cpu":
        for key in GPU_ONLY_XGBOOST_PARAMS:
            if key in normalized:
                removed[key] = normalized.pop(key)
        set_changed("device", "cpu")
        tree_method = str(normalized.get("tree_method", "")).lower()
        if tree_method in GPU_TREE_METHODS | {"auto", ""}:
            set_changed("tree_method", CPU_TREE_METHOD)
        else:
            normalized.setdefault("tree_method", CPU_TREE_METHOD)
        if str(normalized.get("sampling_method", "")).lower() in GPU_SAMPLING_METHODS:
            set_changed("sampling_method", "uniform")
    else:
        set_changed("device", "cuda")
        normalized.setdefault("tree_method", CUDA_TREE_METHOD)

    runtime_config = {
        "requested_device": requested,
        "effective_device": normalized.get("device"),
        "effective_tree_method": normalized.get("tree_method"),
        "removed_gpu_params": sorted(removed),
        "removed_gpu_param_values": removed,
        "changed_runtime_params": changed,
        "xgboost_version": _xgboost_version(),
    }
    return normalized, runtime_config


def validate_xgboost_runtime_available(requested_device: str) -> None:
    """Fail fast for explicitly requested CUDA in environments without a GPU."""

    requested = str(requested_device or "cpu").strip().lower()
    if requested != "cuda":
        return
    try:
        cp = _cupy()
    except Exception as exc:
        raise RuntimeError(
            "Configured XGBoost device 'cuda' is unavailable in the retrainer "
            "environment. Set retraining training.device to 'cpu' for local V1 "
            "execution or run on a CUDA-capable host with compatible drivers."
        ) from exc
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "Configured XGBoost device 'cuda' is unavailable in the retrainer "
            "environment because CUDA device discovery failed."
        ) from exc
    if count < 1:
        raise RuntimeError(
            "Configured XGBoost device 'cuda' is unavailable in the retrainer "
            "environment because no CUDA devices were found."
        )


def fit_estimator_on_configured_device(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    **fit_kwargs: Any,
) -> Any:
    """Fit with CuPy arrays when CUDA is requested, otherwise use pandas."""

    if not estimator_requests_cuda(estimator):
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        return estimator.fit(features, target, **fit_kwargs)

    gpu_features = to_gpu_features(features)
    gpu_target = to_gpu_target(target)
    if eval_set is not None:
        fit_kwargs["eval_set"] = [
            (to_gpu_features(eval_features), to_gpu_target(eval_target))
            for eval_features, eval_target in eval_set
        ]
    return estimator.fit(gpu_features, gpu_target, **fit_kwargs)


def _cpu_set_params_for_estimator(estimator: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"device": "cpu"}
    get_params = getattr(estimator, "get_params", None)
    existing = get_params() if callable(get_params) else {}
    if "tree_method" in existing:
        params["tree_method"] = CPU_TREE_METHOD
    if existing.get("predictor") == "gpu_predictor":
        params["predictor"] = "auto"
    if "gpu_id" in existing:
        params["gpu_id"] = None
    if str(existing.get("sampling_method", "")).lower() in GPU_SAMPLING_METHODS:
        params["sampling_method"] = "uniform"
    return params


def _xgboost_version() -> str | None:
    try:
        import xgboost
    except Exception:
        return None
    return str(getattr(xgboost, "__version__", "unknown"))


def xgboost_positive_scores(
    estimator: Any,
    features: pd.DataFrame,
) -> np.ndarray | None:
    """Use XGBoost GPU in-place prediction for CUDA estimators."""

    if not estimator_requests_cuda(estimator):
        return None
    gpu_features = to_gpu_features(features)
    booster = _booster(estimator)
    if booster is not None:
        try:
            predictions = booster.inplace_predict(gpu_features)
        except Exception:
            predictions = _predict_proba(estimator, gpu_features)
    else:
        predictions = _predict_proba(estimator, gpu_features)
    scores = to_host_array(predictions)
    if scores.ndim == 2:
        return np.asarray(scores)[:, 1]
    return np.asarray(scores).reshape(-1)


def to_gpu_features(features: pd.DataFrame) -> Any:
    configure_gpu_runtime_environment()
    cp = _cupy()
    values = np.ascontiguousarray(features.to_numpy(dtype=np.float32, copy=False))
    return cp.asarray(values)


def to_gpu_target(target: pd.Series) -> Any:
    configure_gpu_runtime_environment()
    cp = _cupy()
    values = np.ascontiguousarray(target.to_numpy(dtype=np.int32, copy=False))
    return cp.asarray(values)


def to_host_array(values: Any) -> np.ndarray:
    cp = _optional_cupy()
    if cp is not None and isinstance(values, cp.ndarray):
        return cp.asnumpy(values)
    return np.asarray(values)


def _cupy() -> Any:
    configure_gpu_runtime_environment()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy as cp
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested for this model, but CuPy could not create GPU "
            "arrays. Install cupy-cuda12x plus the NVIDIA CUDA runtime/NVRTC/NVCC "
            "wheels, or set the estimator device to cpu."
        ) from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA was requested, but CuPy found no CUDA devices.")
    return cp


def _optional_cupy() -> Any | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy as cp
    except Exception:
        return None
    return cp


def _booster(estimator: Any) -> Any | None:
    get_booster = getattr(estimator, "get_booster", None)
    if not callable(get_booster):
        return None
    try:
        return get_booster()
    except Exception:
        return None


def _predict_proba(estimator: Any, features: Any) -> Any:
    predict_proba = getattr(estimator, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError(
            f"{type(estimator).__name__} exposes neither GPU in-place prediction "
            "nor predict_proba"
        )
    return predict_proba(features)


def _tempdir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sentinelml_gpu_tmp_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
