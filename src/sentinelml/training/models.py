"""Baseline model factories for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinelml.data.config import PROJECT_ROOT

DEFAULT_TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"


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
    required_models = {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "hist_gradient_boosting",
    }
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

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    **baseline_config["logistic_regression"],
                    random_state=seed,
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        **baseline_config["random_forest"],
        random_state=seed,
    )
    xgboost = XGBClassifier(
        **baseline_config["xgboost"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    hist_gradient_boosting = HistGradientBoostingClassifier(
        **baseline_config["hist_gradient_boosting"],
        random_state=seed,
    )
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=logistic,
            parameters={
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "LogisticRegression": logistic.named_steps["classifier"].get_params(),
            },
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
