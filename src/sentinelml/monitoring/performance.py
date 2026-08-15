"""Prediction distribution and labelled performance metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sentinelml.monitoring.config import MonitoringConfig


def class_distribution(values: list[str | int]) -> dict[str, Any]:
    counts = Counter(_class_name(value) for value in values)
    total = sum(counts.values())
    return {
        "counts": {
            "BENIGN": counts.get("BENIGN", 0),
            "ATTACK": counts.get("ATTACK", 0),
        },
        "shares": {
            "BENIGN": _share(counts.get("BENIGN", 0), total),
            "ATTACK": _share(counts.get("ATTACK", 0), total),
        },
        "total": total,
    }


def prediction_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return class_distribution([str(row["prediction"]) for row in rows])


def true_label_distribution(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    labels = [
        row["ground_truth"] for row in rows if row.get("ground_truth") is not None
    ]
    if not labels:
        return None
    return class_distribution([int(label) for label in labels])


def model_version_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["model_version"]) for row in rows)
    return dict(sorted(counts.items()))


def calculate_performance(
    rows: list[dict[str, Any]],
    config: MonitoringConfig,
) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("ground_truth") is not None]
    labelled_count = len(labelled)
    true_attack = sum(1 for row in labelled if int(row["ground_truth"]) == 1)
    true_benign = sum(1 for row in labelled if int(row["ground_truth"]) == 0)
    support = {
        "labelled_rows": labelled_count,
        "true_attack": true_attack,
        "true_benign": true_benign,
    }
    reasons: list[str] = []
    if labelled_count < config.minimum_labelled_rows:
        reasons.append("minimum_labelled_rows")
    if true_attack < config.min_attack_support:
        reasons.append("min_attack_support")
    if true_benign < config.min_benign_support:
        reasons.append("min_benign_support")
    if reasons:
        return {
            "status": "insufficient_labels",
            "availability": "unavailable",
            "support": support,
            "insufficient_reasons": reasons,
            "confusion_matrix": None,
            "attack_recall": None,
            "precision": None,
            "f1": None,
            "false_positive_rate": None,
            "accuracy": None,
        }

    tp = sum(
        1
        for row in labelled
        if row["prediction"] == "ATTACK" and int(row["ground_truth"]) == 1
    )
    fn = sum(
        1
        for row in labelled
        if row["prediction"] == "BENIGN" and int(row["ground_truth"]) == 1
    )
    fp = sum(
        1
        for row in labelled
        if row["prediction"] == "ATTACK" and int(row["ground_truth"]) == 0
    )
    tn = sum(
        1
        for row in labelled
        if row["prediction"] == "BENIGN" and int(row["ground_truth"]) == 0
    )
    attack_recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    false_positive_rate = _safe_div(fp, fp + tn)
    if precision is None or attack_recall is None:
        f1 = None
    elif precision + attack_recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * attack_recall / (precision + attack_recall)
    accuracy = _safe_div(tp + tn, labelled_count)
    return {
        "status": "available",
        "availability": "available",
        "support": support,
        "insufficient_reasons": [],
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "attack_recall": attack_recall,
        "precision": precision,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
    }


def _class_name(value: str | int) -> str:
    if value in {1, "1", "ATTACK"}:
        return "ATTACK"
    return "BENIGN"


def _share(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
