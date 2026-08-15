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

from sentinelml.config import load_yaml_mapping
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


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG_PATH) -> dict[str, Any]:
    """Load the Phase 2 YAML training configuration."""

    config = load_yaml_mapping(path, label="training config")
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


def _pipeline_parameters(pipeline: Pipeline) -> dict[str, Any]:
    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    return {
        "pipeline": ["StandardScaler", type(classifier).__name__],
        "scaler": scaler.get_params(),
        "classifier": classifier.get_params(),
        "flat": pipeline.get_params(),
    }


def _filter_params(estimator_cls: type[Any], params: dict[str, Any]) -> dict[str, Any]:
    valid = estimator_cls().get_params().keys()
    return {key: value for key, value in params.items() if key in valid}


def _hist_gradient_boosting(params: dict[str, Any], seed: int) -> Any:
    normalized = dict(params)
    if "n_estimators" in normalized and "max_iter" not in normalized:
        normalized["max_iter"] = normalized.pop("n_estimators")
    if "max_leaves" in normalized and "max_leaf_nodes" not in normalized:
        normalized["max_leaf_nodes"] = normalized.pop("max_leaves")
    normalized = _filter_params(HistGradientBoostingClassifier, normalized)
    normalized.setdefault("max_iter", 200)
    normalized.setdefault("learning_rate", 0.1)
    normalized.setdefault("max_leaf_nodes", 31)
    normalized.setdefault("class_weight", "balanced")
    normalized.setdefault("random_state", seed)
    try:
        return HistGradientBoostingClassifier(**normalized)
    except TypeError:
        normalized.pop("class_weight", None)
        return HistGradientBoostingClassifier(**normalized)


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

    logistic_params = _filter_params(
        LogisticRegression,
        dict(baseline_config["logistic_regression"]),
    )
    logistic_params.setdefault("class_weight", "balanced")
    logistic_params.setdefault("max_iter", 500)
    logistic_params.setdefault("random_state", seed)
    logistic_params.setdefault("solver", "lbfgs")
    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**logistic_params)),
        ]
    )
    random_forest_params = _filter_params(
        RandomForestClassifier,
        dict(baseline_config["random_forest"]),
    )
    random_forest_params.setdefault("n_estimators", 80)
    random_forest_params.setdefault("max_depth", 18)
    random_forest_params.setdefault("min_samples_leaf", 2)
    random_forest_params.setdefault("class_weight", "balanced_subsample")
    random_forest_params.setdefault("n_jobs", 1)
    random_forest_params.setdefault("random_state", seed)
    random_forest = RandomForestClassifier(**random_forest_params)
    xgboost_params = dict(baseline_config["xgboost"])
    xgboost_params.setdefault("eval_metric", "logloss")
    xgboost_params.setdefault("tree_method", "hist")
    xgboost_params.setdefault("device", "cpu")
    xgboost_params.setdefault("n_jobs", 1)
    xgboost = XGBClassifier(
        **xgboost_params,
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    hist_gradient_boosting_params = dict(baseline_config["hist_gradient_boosting"])
    hist_gradient_boosting = _hist_gradient_boosting(
        hist_gradient_boosting_params,
        seed,
    )
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=logistic,
            parameters=_pipeline_parameters(logistic),
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
