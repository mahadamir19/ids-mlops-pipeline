"""Prometheus metrics for the Phase 7 monitoring service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sentinelml.metrics import PrometheusTextRegistry

monitoring_metrics = PrometheusTextRegistry()

for name, metric_type, help_text in [
    (
        "sentinelml_monitoring_health",
        "gauge",
        "Monitoring health: healthy=1, warming=0.5, unhealthy=0.",
    ),
    (
        "sentinelml_monitoring_last_success_timestamp",
        "gauge",
        "Unix timestamp of the last successful healthy monitoring run.",
    ),
    (
        "sentinelml_monitoring_last_check_timestamp",
        "gauge",
        "Unix timestamp of the last successful monitor poll/check.",
    ),
    (
        "sentinelml_monitoring_unchanged_skips_total",
        "counter",
        "Total monitor polls skipped because input fingerprint was unchanged.",
    ),
    (
        "sentinelml_monitoring_window_size",
        "gauge",
        "Rows in the latest monitoring window.",
    ),
    (
        "sentinelml_drift_detected",
        "gauge",
        "Whether data drift is currently detected.",
    ),
    (
        "sentinelml_drifting_feature_count",
        "gauge",
        "Number of monitored features marked drifting.",
    ),
    ("sentinelml_monitored_feature_count", "gauge", "Number of monitored features."),
    (
        "sentinelml_drift_feature_share",
        "gauge",
        "Share of monitored features marked drifting.",
    ),
    (
        "sentinelml_attack_recall",
        "gauge",
        "Attack recall for labelled rows when available.",
    ),
    ("sentinelml_f1", "gauge", "F1 for labelled rows when available."),
    (
        "sentinelml_false_positive_rate",
        "gauge",
        "False positive rate for labelled rows when available.",
    ),
    (
        "sentinelml_labelled_window_rows",
        "gauge",
        "Labelled rows in the current monitoring window.",
    ),
    (
        "sentinelml_retraining_state",
        "gauge",
        "Current Phase 8 retraining state.",
    ),
    (
        "sentinelml_retraining_cooldown_active",
        "gauge",
        "Whether Phase 8 retraining cooldown is active.",
    ),
    (
        "sentinelml_retraining_last_result",
        "gauge",
        "Last terminal Phase 8 retraining result.",
    ),
    (
        "sentinelml_lifecycle_state",
        "gauge",
        "Current model lifecycle state exposed for dashboards.",
    ),
    ("sentinelml_probation_active", "gauge", "Whether probation is active."),
    (
        "sentinelml_probation_promoted_model_version",
        "gauge",
        "Numeric promoted model version under probation.",
    ),
    (
        "sentinelml_probation_labelled_rows",
        "gauge",
        "Labelled rows observed during active probation.",
    ),
    (
        "sentinelml_probation_severe_violation",
        "gauge",
        "Whether probation has severe guardrail violations.",
    ),
    (
        "sentinelml_promotion_pending",
        "gauge",
        "Whether a pending promotion retry exists.",
    ),
    (
        "sentinelml_monitoring_stale",
        "gauge",
        "Whether the latest monitor heartbeat is stale.",
    ),
]:
    monitoring_metrics.describe(name, metric_type, help_text)


def update_monitoring_metrics(report: dict[str, Any]) -> None:
    health = str(report.get("monitoring_health", "unhealthy"))
    health_value = {"healthy": 1.0, "warming_up": 0.5, "unhealthy": 0.0}.get(
        health,
        0.0,
    )
    monitoring_metrics.set_gauge("sentinelml_monitoring_health", health_value)
    monitoring_metrics.set_gauge(
        "sentinelml_monitoring_last_check_timestamp",
        _unix_timestamp(
            str(report.get("last_check_timestamp") or report.get("finished_at"))
        ),
    )
    if report.get("poll_status") == "unchanged":
        monitoring_metrics.inc_counter("sentinelml_monitoring_unchanged_skips_total")
    if health == "healthy" and report.get("poll_status") != "unchanged":
        monitoring_metrics.set_gauge(
            "sentinelml_monitoring_last_success_timestamp",
            _unix_timestamp(str(report.get("finished_at"))),
        )
    window = report.get("window", {})
    monitoring_metrics.set_gauge(
        "sentinelml_monitoring_window_size",
        window.get("actual_window_size"),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_drift_detected",
        1 if report.get("data_drift_detected") else 0,
    )
    monitoring_metrics.set_gauge(
        "sentinelml_drifting_feature_count",
        report.get("drifting_feature_count"),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_monitored_feature_count",
        report.get("monitored_feature_count"),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_drift_feature_share",
        report.get("drift_share"),
    )
    performance = report.get("performance", {})
    support = performance.get("support", {})
    monitoring_metrics.set_gauge(
        "sentinelml_labelled_window_rows",
        support.get("labelled_rows", report.get("labelled_row_count", 0)),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_attack_recall",
        performance.get("attack_recall"),
    )
    monitoring_metrics.set_gauge("sentinelml_f1", performance.get("f1"))
    monitoring_metrics.set_gauge(
        "sentinelml_false_positive_rate",
        performance.get("false_positive_rate"),
    )
    retraining_state = report.get("retraining_state", "idle")
    if isinstance(retraining_state, dict):
        cooldown = retraining_state.get("cooldown", {})
        latest = retraining_state.get("latest_run", {})
        state_name = str(retraining_state.get("state", "idle"))
        last_result = str(latest.get("status", "none"))
        cooldown_active = 1 if cooldown.get("active") else 0
    else:
        state_name = str(retraining_state or "idle")
        last_result = "none"
        cooldown_active = 0
    monitoring_metrics.clear_gauge_family("sentinelml_retraining_state")
    monitoring_metrics.set_gauge(
        "sentinelml_retraining_state",
        1,
        labels={"state": state_name},
    )
    monitoring_metrics.set_gauge(
        "sentinelml_retraining_cooldown_active",
        cooldown_active,
    )
    monitoring_metrics.clear_gauge_family("sentinelml_retraining_last_result")
    monitoring_metrics.set_gauge(
        "sentinelml_retraining_last_result",
        1,
        labels={"result": last_result},
    )
    monitoring_metrics.clear_gauge_family("sentinelml_lifecycle_state")
    monitoring_metrics.set_gauge(
        "sentinelml_lifecycle_state",
        1,
        labels={"state": "champion_active"},
    )
    resilience = _resilience_summary(report.get("resilience_state", {}))
    monitoring_metrics.set_gauge(
        "sentinelml_probation_active",
        1 if resilience.get("active_probation") else 0,
    )
    monitoring_metrics.set_gauge(
        "sentinelml_probation_promoted_model_version",
        _numeric(resilience.get("promoted_version")),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_probation_labelled_rows",
        resilience.get("labelled_rows", 0),
    )
    monitoring_metrics.set_gauge(
        "sentinelml_probation_severe_violation",
        1 if resilience.get("severe_violation") else 0,
    )
    monitoring_metrics.set_gauge(
        "sentinelml_promotion_pending",
        1 if resilience.get("promotion_pending") else 0,
    )
    monitoring_metrics.set_gauge("sentinelml_monitoring_stale", 0)


def _unix_timestamp(value: str) -> float | None:
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(UTC).timestamp()
    except Exception:
        return None


def _resilience_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    latest = value.get("latest")
    if not isinstance(latest, dict):
        return {}
    payload = latest.get("payload", latest)
    if not isinstance(payload, dict):
        return {}
    active = payload.get("active_probation")
    if isinstance(active, dict):
        return {
            "active_probation": True,
            "promoted_version": active.get("promoted_version"),
            "labelled_rows": active.get("labelled_rows", 0),
            "severe_violation": bool(active.get("severe_violation_reasons")),
            "promotion_pending": False,
        }
    category = latest.get("category")
    retry_payload = payload.get("outcome", {})
    return {
        "active_probation": False,
        "promoted_version": None,
        "labelled_rows": 0,
        "severe_violation": False,
        "promotion_pending": category == "promotion_retry"
        and isinstance(retry_payload, dict)
        and retry_payload.get("event") == "promotion_pending",
    }


def _numeric(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
