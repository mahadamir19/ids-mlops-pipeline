"""Evaluation and feature-importance helpers for Phase 2 baselines."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sentinelml.training.gpu import xgboost_positive_scores


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return probability-like positive-class scores for binary metrics."""

    gpu_scores = xgboost_positive_scores(model, features)
    if gpu_scores is not None:
        return gpu_scores
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(features))
        return 1.0 / (1.0 + np.exp(-raw_scores))
    raise TypeError(
        f"{type(model).__name__} exposes neither predict_proba nor decision_function"
    )


def evaluate_binary_classifier(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Calculate the common Phase 2 validation/test metric suite."""

    start = perf_counter()
    scores = positive_class_scores(model, features)
    inference_seconds = perf_counter() - start
    predictions = (scores >= 0.5).astype("int8")
    labels = target.to_numpy(dtype="int8")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    roc_auc: float | None
    if len(np.unique(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = None

    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "attack_class_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "inference_time_seconds": float(inference_seconds),
        "inference_latency_ms_per_row": float(
            (inference_seconds / len(labels)) * 1000 if len(labels) else 0.0
        ),
    }


def unwrap_estimator(model: Any) -> Any:
    """Return the final estimator from an sklearn Pipeline if needed."""

    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def feature_importance_table(
    *,
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Build global feature-importance rows where a model supports them."""

    estimator = unwrap_estimator(model)
    values: np.ndarray | None = None
    signed_values: np.ndarray | None = None
    method: str | None = None

    if model_name == "logistic_regression" and hasattr(estimator, "coef_"):
        signed_values = np.asarray(estimator.coef_).reshape(-1)
        values = np.abs(signed_values)
        method = "absolute_scaled_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
        method = "native_feature_importances"
    else:
        return [], (
            "No native global feature-importance mechanism was used for this model."
        )

    rows = []
    for feature, importance, signed in zip(
        feature_columns,
        values,
        signed_values if signed_values is not None else [None] * len(feature_columns),
        strict=True,
    ):
        row = {
            "feature": feature,
            "importance": float(importance),
            "method": method,
        }
        if signed is not None:
            row["signed_coefficient"] = float(signed)
        rows.append(row)

    rows.sort(key=lambda item: item["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows, None


def destination_port_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract destination_port's rank from a feature-importance table."""

    for row in rows:
        if row["feature"] == "destination_port":
            return {
                "rank": row["rank"],
                "importance": row["importance"],
                "method": row["method"],
            }
    return None
