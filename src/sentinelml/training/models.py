"""Baseline model factories for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    parameters: dict[str, Any]


def _class_balance_ratio(target: pd.Series) -> float:
    counts = target.value_counts().to_dict()
    positives = max(int(counts.get(1, 0)), 1)
    negatives = max(int(counts.get(0, 0)), 1)
    return negatives / positives


def build_baseline_model_specs(target: pd.Series, *, seed: int = 42) -> list[ModelSpec]:
    """Create the four required baseline model families with modest defaults."""

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
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    xgboost = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    hist_gradient_boosting = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        class_weight="balanced",
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
