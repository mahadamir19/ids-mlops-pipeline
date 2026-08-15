"""Phase 8 trigger policy over persisted Phase 7 monitoring reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinelml.retraining.config import RetrainingConfig


@dataclass(frozen=True)
class PerformanceThresholds:
    min_attack_recall: float | None
    max_false_positive_rate: float | None
    min_f1: float | None
    source: str
    baseline_f1: float | None = None


def thresholds_from_lifecycle_report(
    report: dict[str, Any] | None,
    config: RetrainingConfig,
) -> PerformanceThresholds:
    """Resolve production-performance trigger thresholds from Phase 4 evidence."""

    thresholds = (report or {}).get("thresholds", {})
    baseline_metrics = (report or {}).get("baseline_metrics", {})
    min_attack_recall = config.min_attack_recall
    if min_attack_recall is None and thresholds.get("attack_recall") is not None:
        min_attack_recall = float(thresholds["attack_recall"])
    max_false_positive_rate = config.max_false_positive_rate
    if (
        max_false_positive_rate is None
        and thresholds.get("false_positive_rate") is not None
    ):
        max_false_positive_rate = float(thresholds["false_positive_rate"])

    baseline_f1 = None
    if baseline_metrics.get("f1") is not None:
        baseline_f1 = float(baseline_metrics["f1"])
    min_f1 = None
    if baseline_f1 is not None:
        min_f1 = baseline_f1 - float(config.max_f1_drop)

    return PerformanceThresholds(
        min_attack_recall=min_attack_recall,
        max_false_positive_rate=max_false_positive_rate,
        min_f1=min_f1,
        baseline_f1=baseline_f1,
        source="phase4_threshold_report" if report else "retraining_config",
    )


def evaluate_trigger(
    monitoring_report: dict[str, Any],
    config: RetrainingConfig,
    *,
    performance_thresholds: PerformanceThresholds,
    already_processed: bool = False,
) -> dict[str, Any]:
    """Evaluate whether a completed Phase 7 report should launch retraining."""

    monitoring_run_id = str(monitoring_report.get("monitoring_run_id", ""))
    health = str(monitoring_report.get("monitoring_health") or "unhealthy")
    status = str(monitoring_report.get("status") or health)
    decision: dict[str, Any] = {
        "monitoring_run_id": monitoring_run_id,
        "monitoring_timestamp": monitoring_report.get("finished_at"),
        "monitoring_health": health,
        "monitoring_status": status,
        "already_processed": already_processed,
        "trigger_data_drift": False,
        "trigger_performance_degradation": False,
        "trigger_reasons": [],
        "performance_unavailable_reasons": [],
        "should_retrain": False,
        "decision": "not_triggered",
        "drift_share": monitoring_report.get("drift_share"),
        "drift_threshold_used": None,
        "performance_metrics": _performance_summary(monitoring_report),
        "performance_thresholds": performance_thresholds.__dict__,
    }
    if already_processed:
        decision.update(
            {
                "decision": "already_processed",
                "reason": "monitoring_run_already_processed",
            }
        )
        return decision
    if health != "healthy":
        blocked = (
            "blocked_monitoring_warming_up"
            if health == "warming_up" or status == "warming_up"
            else "blocked_monitoring_unhealthy"
        )
        decision.update({"decision": blocked, "reason": blocked})
        return decision

    if config.drift_enabled:
        drift_signal = bool(monitoring_report.get("data_drift_detected") is True)
        decision["trigger_data_drift"] = drift_signal
        decision["drift_threshold_used"] = "phase7.configured_data_drift_detected"
        if drift_signal:
            decision["trigger_reasons"].append("data_drift")

    if config.performance_enabled:
        performance_decision = evaluate_performance_trigger(
            monitoring_report,
            config,
            performance_thresholds=performance_thresholds,
        )
        decision.update(performance_decision)
        if performance_decision["trigger_performance_degradation"]:
            decision["trigger_reasons"].extend(performance_decision["performance_reasons"])

    decision["trigger_reasons"] = sorted(set(decision["trigger_reasons"]))
    decision["should_retrain"] = bool(decision["trigger_reasons"])
    decision["decision"] = (
        "triggered" if decision["should_retrain"] else "not_triggered"
    )
    return decision


def evaluate_performance_trigger(
    monitoring_report: dict[str, Any],
    config: RetrainingConfig,
    *,
    performance_thresholds: PerformanceThresholds,
) -> dict[str, Any]:
    performance = monitoring_report.get("performance") or {}
    support = performance.get("support") or {}
    unavailable: list[str] = []
    if (
        int(support.get("labelled_rows") or 0)
        < config.performance_minimum_labelled_rows
    ):
        unavailable.append("insufficient_labelled_rows")
    if int(support.get("true_attack") or 0) < config.performance_minimum_attack_support:
        unavailable.append("insufficient_attack_support")
    if int(support.get("true_benign") or 0) < config.performance_minimum_benign_support:
        unavailable.append("insufficient_benign_support")

    reasons: list[str] = []
    if unavailable:
        return {
            "trigger_performance_degradation": False,
            "performance_reasons": [],
            "performance_unavailable_reasons": unavailable,
        }

    attack_recall = _optional_float(performance.get("attack_recall"))
    f1 = _optional_float(performance.get("f1"))
    false_positive_rate = _optional_float(performance.get("false_positive_rate"))
    if attack_recall is None:
        unavailable.append("attack_recall_unavailable")
    elif (
        performance_thresholds.min_attack_recall is not None
        and attack_recall < performance_thresholds.min_attack_recall
    ):
        reasons.append("attack_recall_below_threshold")
    if false_positive_rate is None:
        unavailable.append("false_positive_rate_unavailable")
    elif (
        performance_thresholds.max_false_positive_rate is not None
        and false_positive_rate > performance_thresholds.max_false_positive_rate
    ):
        reasons.append("false_positive_rate_above_threshold")
    if f1 is None:
        unavailable.append("f1_unavailable")
    elif (
        performance_thresholds.min_f1 is not None
        and f1 < performance_thresholds.min_f1
    ):
        reasons.append("f1_drop_above_threshold")

    return {
        "trigger_performance_degradation": bool(reasons),
        "performance_reasons": reasons,
        "performance_unavailable_reasons": unavailable,
    }


def _performance_summary(report: dict[str, Any]) -> dict[str, Any]:
    performance = report.get("performance") or {}
    return {
        "attack_recall": performance.get("attack_recall"),
        "f1": performance.get("f1"),
        "false_positive_rate": performance.get("false_positive_rate"),
        "support": performance.get("support", {}),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
