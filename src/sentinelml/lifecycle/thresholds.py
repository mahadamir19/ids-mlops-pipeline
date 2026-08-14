"""Baseline-derived promotion thresholds and composite scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinelml.lifecycle.audit import utc_now, write_json

METRIC_ALIASES = {
    "attack_recall": "attack_class_recall",
    "latency": "inference_latency_ms_per_row",
    "max_latency_ms": "inference_latency_ms_per_row",
}


def metric_value(metrics: dict[str, Any], metric: str) -> float:
    key = METRIC_ALIASES.get(metric, metric)
    if key not in metrics:
        raise ValueError(f"metrics are missing required value: {metric}")
    value = metrics[key]
    if value is None:
        raise ValueError(f"metric {metric} is null")
    return float(value)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def derive_thresholds(
    *,
    config: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_metrics_path: Path,
    baseline_manifest_path: Path | None = None,
    selected_baseline: dict[str, Any] | None = None,
    selected_baseline_path: Path | None = None,
    canonical_dataset: dict[str, Any] | None = None,
    baseline_evaluation: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create absolute gate thresholds from compatible baseline evidence."""

    policy = config["threshold_policy"]
    thresholds: dict[str, float] = {}
    formulas: dict[str, str] = {}
    evidence: dict[str, Any] = {
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_manifest_path": str(baseline_manifest_path)
        if baseline_manifest_path is not None
        else None,
        "selected_baseline_path": str(selected_baseline_path)
        if selected_baseline_path is not None
        else None,
        "baseline_mode": policy.get("baseline_mode"),
        "evaluation_split": policy.get("evaluation_split"),
    }
    if selected_baseline is not None:
        evidence["selected_baseline"] = selected_baseline
    if canonical_dataset is not None:
        evidence["canonical_promotion_dataset"] = canonical_dataset
    if baseline_evaluation is not None:
        evidence["baseline_evaluation"] = {
            "report_path": str(baseline_evaluation.get("report_path")),
            "model_family": baseline_evaluation.get("model_family"),
            "baseline_source": baseline_evaluation.get("baseline_source"),
            "evaluation_type": baseline_evaluation.get("evaluation_type"),
        }
    if baseline_manifest_path is not None and baseline_manifest_path.exists():
        baseline_manifest = load_json(baseline_manifest_path)
        evidence["baseline_report_mode"] = baseline_manifest.get("mode")
        evidence["baseline_validation_sample_metadata"] = (
            baseline_manifest.get("sample_metadata", {}).get("validation")
        )

    for gate_name, gate_policy in policy["metrics"].items():
        source_metric = str(gate_policy["source_metric"])
        reference = metric_value(baseline_metrics, source_metric)
        if gate_policy["direction"] == "min":
            fraction = float(gate_policy["fraction_of_baseline"])
            thresholds[gate_name] = reference * fraction
            formulas[gate_name] = f"{source_metric} * {fraction}"
        elif gate_policy["direction"] == "max":
            multiplier = float(gate_policy["multiplier_of_baseline"])
            floor = float(gate_policy.get("minimum_ceiling", 0.0))
            thresholds[gate_name] = max(reference * multiplier, floor)
            formulas[gate_name] = f"max({source_metric} * {multiplier}, {floor})"
        else:
            raise ValueError(f"unsupported threshold direction for {gate_name}")

    report = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "evidence": evidence,
        "baseline_metrics": baseline_metrics,
        "thresholds": thresholds,
        "formulas": formulas,
        "policy": policy,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def composite_score(
    metrics: dict[str, Any],
    *,
    thresholds: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    weights = config["composite_score"]["weights"]
    components: dict[str, float] = {}
    for metric, weight in weights.items():
        raw = metric_value(metrics, metric)
        if metric in {"false_positive_rate", "inference_latency_ms_per_row"}:
            ceiling = max(float(thresholds[metric]), 1e-12)
            normalized = 1.0 - min(max(raw / ceiling, 0.0), 1.0)
        else:
            normalized = min(max(raw, 0.0), 1.0)
        components[metric] = float(weight) * normalized
    score = sum(components.values())
    return {
        "score": float(score),
        "components": components,
        "weights": weights,
        "formula": (
            "weighted sum of PR-AUC, attack recall, and F1 as [0,1] metrics; "
            "FPR and latency contribute 1 - min(value / configured_ceiling, 1)"
        ),
    }


def evaluate_absolute_gates(
    *,
    candidate_metrics: dict[str, Any],
    thresholds: dict[str, float],
    threshold_policy: dict[str, Any],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for gate_name, gate_policy in threshold_policy["metrics"].items():
        candidate = metric_value(candidate_metrics, str(gate_policy["source_metric"]))
        threshold = float(thresholds[gate_name])
        if gate_policy["direction"] == "min":
            passed = candidate >= threshold
            operator = ">="
        else:
            passed = candidate <= threshold
            operator = "<="
        gates[gate_name] = {
            "passed": bool(passed),
            "candidate": candidate,
            "threshold": threshold,
            "operator": operator,
            "source_metric": gate_policy["source_metric"],
        }
    return {
        "passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "failed_gates": [
            name for name, gate in gates.items() if not bool(gate["passed"])
        ],
    }
