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
    return device.startswith("cuda") or device.startswith("gpu")


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


def xgboost_positive_scores(estimator: Any, features: pd.DataFrame) -> np.ndarray | None:
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
