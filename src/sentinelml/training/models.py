"""Baseline model factories for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelml.data.config import PROJECT_ROOT

DEFAULT_TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
MODEL_FAMILIES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "hist_gradient_boosting",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    parameters: dict[str, Any]


def _parse_scalar(value: str) -> int | float | str:
    value = value.strip()
    if value == "":
        raise ValueError("empty scalar values are not supported in training config")
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG_PATH) -> dict[str, Any]:
    """Load the simple Phase 2 YAML training configuration."""

    config: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, config)]
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indent = len(content) - len(content.lstrip(" "))
        if indent % 2:
            raise ValueError(f"invalid indentation on line {line_number}")
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"expected key/value pair on line {line_number}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))

    _validate_training_config(config)
    return config


def _validate_training_config(config: dict[str, Any]) -> None:
    required_models = set(MODEL_FAMILIES)
    if "random_seed" not in config:
        raise ValueError("training config is missing random_seed")
    baseline = config.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("training config is missing baseline model settings")
    missing_models = sorted(required_models - set(baseline))
    if missing_models:
        raise ValueError(f"training config is missing models: {missing_models}")


def _class_balance_ratio(target: pd.Series) -> float:
    counts = target.value_counts().to_dict()
    positives = max(int(counts.get(1, 0)), 1)
    negatives = max(int(counts.get(0, 0)), 1)
    return negatives / positives


def build_baseline_model_specs(
    target: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> list[ModelSpec]:
    """Create the four required baseline model families with modest defaults."""

    config = config or load_training_config()
    seed = int(config["random_seed"])
    baseline_config = config["baseline"]
    scale_pos_weight = _class_balance_ratio(target)
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "XGBoost is required for Phase 2. Install the project ML extras or xgboost."
        ) from exc

    logistic_params = dict(baseline_config["logistic_regression"])
    logistic_params.setdefault("booster", "gblinear")
    logistic_params.setdefault("eval_metric", "logloss")
    logistic_params.setdefault("device", "cuda")
    logistic = XGBClassifier(
        **logistic_params,
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    random_forest_params = dict(baseline_config["random_forest"])
    random_forest_params.setdefault("n_estimators", 1)
    random_forest_params.setdefault("learning_rate", 1.0)
    random_forest_params.setdefault("eval_metric", "logloss")
    random_forest_params.setdefault("device", "cuda")
    random_forest = XGBClassifier(
        **random_forest_params,
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    xgboost_params = dict(baseline_config["xgboost"])
    xgboost_params.setdefault("eval_metric", "logloss")
    xgboost_params.setdefault("device", "cuda")
    xgboost = XGBClassifier(
        **xgboost_params,
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    hist_gradient_boosting_params = dict(baseline_config["hist_gradient_boosting"])
    hist_gradient_boosting_params.setdefault("eval_metric", "logloss")
    hist_gradient_boosting_params.setdefault("device", "cuda")
    hist_gradient_boosting = XGBClassifier(
        **hist_gradient_boosting_params,
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=logistic,
            parameters=logistic.get_params(),
        ),
        ModelSpec(
            name="random_forest",
            estimator=random_forest,
            parameters=random_forest.get_params(),
        ),
        ModelSpec(
            name="xgboost",
            estimator=xgboost,
            parameters=xgboost.get_params(),
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=hist_gradient_boosting,
            parameters=hist_gradient_boosting.get_params(),
        ),
    ]


def build_baseline_model_spec(
    model_family: str,
    target: pd.Series,
    *,
    config: dict[str, Any] | None = None,
) -> ModelSpec:
    """Create one Phase 2 model family using the existing baseline factories."""

    specs = {
        spec.name: spec
        for spec in build_baseline_model_specs(
            target,
            config=config,
        )
    }
    if model_family not in specs:
        raise ValueError(f"unsupported model family: {model_family}")
    return specs[model_family]
