"""Prometheus metrics for Phase 9 resilience state."""

from __future__ import annotations

from typing import Any

from sentinelml.metrics import PrometheusTextRegistry

resilience_metrics = PrometheusTextRegistry()

for name, metric_type, help_text in [
    ("sentinelml_probation_active", "gauge", "Whether a probation is active."),
    (
        "sentinelml_probation_promoted_model_version",
        "gauge",
        "Numeric promoted model version currently under probation.",
    ),
    (
        "sentinelml_probation_labelled_rows",
        "gauge",
        "Labelled production rows in the active probation window.",
    ),
    (
        "sentinelml_probation_severe_violation",
        "gauge",
        "Whether active probation has a severe guardrail violation.",
    ),
    ("sentinelml_rollback_total", "counter", "Rollback attempts by source/status."),
    (
        "sentinelml_last_rollback_timestamp",
        "gauge",
        "Unix timestamp of the last completed rollback event.",
    ),
    (
        "sentinelml_promotion_pending",
        "gauge",
        "Whether a candidate is waiting for promotion retry.",
    ),
    (
        "sentinelml_promotion_retry_total",
        "counter",
        "Promotion retry attempts by outcome status.",
    ),
    (
        "sentinelml_registry_connectivity",
        "gauge",
        "Registry connectivity: available=1, unavailable=0.",
    ),
    (
        "sentinelml_monitoring_stale",
        "gauge",
        "Whether monitoring heartbeat is stale for retraining.",
    ),
    (
        "sentinelml_db_logging_health",
        "gauge",
        "Prediction DB logging health: healthy=1, unhealthy=0, unconfigured=-1.",
    ),
]:
    resilience_metrics.describe(name, metric_type, help_text)


def record_probation_status(status: dict[str, Any]) -> None:
    active = status.get("active_probation")
    resilience_metrics.set_gauge("sentinelml_probation_active", 1 if active else 0)
    if not isinstance(active, dict):
        resilience_metrics.set_gauge("sentinelml_probation_promoted_model_version", 0)
        resilience_metrics.set_gauge("sentinelml_probation_labelled_rows", 0)
        resilience_metrics.set_gauge("sentinelml_probation_severe_violation", 0)
        return
    resilience_metrics.set_gauge(
        "sentinelml_probation_promoted_model_version",
        _numeric(active.get("promoted_version")),
    )
    resilience_metrics.set_gauge(
        "sentinelml_probation_labelled_rows",
        active.get("labelled_rows", 0),
    )
    resilience_metrics.set_gauge(
        "sentinelml_probation_severe_violation",
        1 if active.get("severe_violation_reasons") else 0,
    )


def record_rollback(*, source: str, status: str, completed_timestamp: float | None):
    resilience_metrics.inc_counter(
        "sentinelml_rollback_total",
        labels={"source": source, "status": status},
    )
    resilience_metrics.set_gauge(
        "sentinelml_last_rollback_timestamp",
        completed_timestamp,
    )


def record_promotion_pending(count: int) -> None:
    resilience_metrics.set_gauge("sentinelml_promotion_pending", 1 if count else 0)


def record_promotion_retry(*, status: str) -> None:
    resilience_metrics.inc_counter(
        "sentinelml_promotion_retry_total",
        labels={"status": status},
    )


def _numeric(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
