"""Baseline comparison and reporting for Phase 2."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPARISON_COLUMNS = [
    "model",
    "selection_score",
    "precision",
    "recall",
    "attack_class_recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
    "false_negative_rate",
    "training_time_seconds",
    "inference_latency_ms_per_row",
]


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def selection_score(metrics: dict[str, Any]) -> float:
    """Composite validation score for baseline ranking, not a promotion threshold."""

    roc_auc = metrics["roc_auc"] if metrics["roc_auc"] is not None else 0.0
    return float(
        0.35 * metrics["attack_class_recall"]
        + 0.25 * metrics["f1"]
        + 0.25 * metrics["pr_auc"]
        + 0.10 * roc_auc
        + 0.05 * (1.0 - metrics["false_positive_rate"])
    )


def comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_name, result in results.items():
        metrics = result["validation_metrics"]
        row = {
            "model": model_name,
            "selection_score": selection_score(metrics),
            "training_time_seconds": result["training_time_seconds"],
        }
        for key in COMPARISON_COLUMNS:
            if key in {"model", "selection_score", "training_time_seconds"}:
                continue
            row[key] = metrics[key]
        rows.append(row)
    return sorted(rows, key=lambda row: row["selection_score"], reverse=True)


def select_best_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select a baseline from an empty comparison table")
    selected = rows[0]
    return {
        "recommended_baseline": selected["model"],
        "selection_metric": "validation composite score",
        "selection_score": selected["selection_score"],
        "selection_rationale": (
            "Selected from validation behavior using attack recall, F1, PR-AUC, "
            "ROC-AUC, and false-positive rate. Accuracy is reported but not used "
            "as the ranking criterion."
        ),
    }


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_report(
    *,
    path: Path,
    mode: str,
    sample_metadata: dict[str, Any],
    comparison: list[dict[str, Any]],
    selected: dict[str, Any],
    validation_results: dict[str, dict[str, Any]],
    selected_test_metrics: dict[str, Any],
    feature_importance: dict[str, Any],
) -> None:
    lines = [
        "# Phase 2 Baseline ML Report",
        "",
        f"Generated at `{datetime.now(timezone.utc).isoformat()}`.",
        f"Run mode: `{mode}`.",
        "",
        "## Data Usage",
        *[
            f"- {partition}: {meta['rows_used']:,} of {meta['original_rows']:,} rows; "
            f"target distribution={meta['target_distribution']}"
            for partition, meta in sample_metadata.items()
        ],
        "",
        "## Validation Comparison",
        "| Model | Score | Precision | Attack Recall | F1 | PR-AUC | "
        "ROC-AUC | FPR | FNR | Train s | Latency ms/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            "| {model} | {selection_score} | {precision} | {attack_class_recall} | "
            "{f1} | {pr_auc} | {roc_auc} | {false_positive_rate} | "
            "{false_negative_rate} | {training_time_seconds} | "
            "{inference_latency_ms_per_row} |".format(
                **{key: _format_metric(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Baseline",
            f"- Model: `{selected['recommended_baseline']}`",
            f"- Selection score: `{selected['selection_score']:.6f}`",
            f"- Rationale: {selected['selection_rationale']}",
            "",
            "## Selected Baseline Test Assessment",
            *[
                f"- {key}: `{_format_metric(value)}`"
                for key, value in selected_test_metrics.items()
                if key != "confusion_matrix"
            ],
            f"- confusion_matrix: `{selected_test_metrics['confusion_matrix']}`",
            "",
            "## Feature Importance",
        ]
    )
    for model_name, details in feature_importance.items():
        lines.append(f"### {model_name}")
        if details.get("unsupported_reason"):
            lines.append(f"- {details['unsupported_reason']}")
            continue
        top_rows = details["rows"][:10]
        for row in top_rows:
            lines.append(
                f"- #{row['rank']} `{row['feature']}`: "
                f"{_format_metric(row['importance'])}"
            )
        destination_port = details.get("destination_port")
        if destination_port:
            lines.append(
                "- destination_port rank: "
                f"#{destination_port['rank']} "
                f"({_format_metric(destination_port['importance'])})"
            )
    lines.extend(
        [
            "",
            "## Scope Notes",
            "- No promotion thresholds were defined.",
            "- No DVC stages, MLflow, Optuna, model registry, serving, "
            "monitoring, or CI infrastructure were added.",
            "- `destination_port` was retained and inspected, not removed.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
