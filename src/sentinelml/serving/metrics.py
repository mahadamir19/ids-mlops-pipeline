"""Operational Prometheus metrics for the SentinelML inference API."""

from __future__ import annotations

from sentinelml.metrics import PrometheusTextRegistry

api_metrics = PrometheusTextRegistry()

api_metrics.describe(
    "sentinelml_api_requests_total",
    "counter",
    "Total API requests labelled by bounded route, method, and status code.",
)
api_metrics.describe(
    "sentinelml_api_request_duration_seconds",
    "summary",
    "API request latency in seconds.",
)
api_metrics.describe(
    "sentinelml_api_errors_total",
    "counter",
    "Total API requests that returned a 5xx status code.",
)
api_metrics.describe(
    "sentinelml_api_rejected_validation_requests_total",
    "counter",
    "Prediction requests rejected by strict schema validation.",
)
api_metrics.describe(
    "sentinelml_predictions_total",
    "counter",
    "Total production predictions served.",
)
api_metrics.describe(
    "sentinelml_predictions_by_class_total",
    "counter",
    "Total production predictions by predicted class.",
)
api_metrics.describe(
    "sentinelml_prediction_attack_share",
    "gauge",
    "Cumulative ATTACK share among predictions served by this process.",
)
api_metrics.describe(
    "sentinelml_active_model_version",
    "gauge",
    "Numeric active model version when parseable.",
)
api_metrics.describe(
    "sentinelml_active_model_version_info",
    "gauge",
    "Active model metadata exposed as a one-hot info metric.",
)
api_metrics.describe(
    "sentinelml_model_reload_failures_total",
    "counter",
    "Champion reload failures observed by the API.",
)
api_metrics.describe(
    "sentinelml_db_prediction_logging_failures_total",
    "counter",
    "Prediction records queued because database logging failed.",
)
api_metrics.describe(
    "sentinelml_registry_connectivity",
    "gauge",
    "Registry connectivity: available=1, unavailable=0.",
)
api_metrics.describe(
    "sentinelml_db_logging_health",
    "gauge",
    "Prediction DB logging health: healthy=1, unhealthy=0, unconfigured=-1.",
)
api_metrics.describe(
    "sentinelml_loaded_registry_divergence",
    "gauge",
    "Whether loaded model version differs from registry champion.",
)
api_metrics.describe(
    "sentinelml_retraining_state",
    "gauge",
    "Current retraining state; Phase 7 exports idle only.",
)

_prediction_counts = {"ATTACK": 0, "BENIGN": 0}


def record_request(
    *,
    method: str,
    route: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    labels = {"method": method, "route": route, "status_code": status_code}
    api_metrics.inc_counter("sentinelml_api_requests_total", labels=labels)
    api_metrics.observe_summary(
        "sentinelml_api_request_duration_seconds",
        duration_seconds,
        labels={"method": method, "route": route},
    )
    if status_code >= 500:
        api_metrics.inc_counter(
            "sentinelml_api_errors_total",
            labels={"method": method, "route": route, "status_code": status_code},
        )


def record_validation_rejection(*, endpoint: str, category: str) -> None:
    api_metrics.inc_counter(
        "sentinelml_api_rejected_validation_requests_total",
        labels={"endpoint": endpoint, "category": category},
    )


def record_prediction(*, prediction: str, count: int = 1) -> None:
    normalized = prediction if prediction in _prediction_counts else "UNKNOWN"
    api_metrics.inc_counter("sentinelml_predictions_total", amount=count)
    api_metrics.inc_counter(
        "sentinelml_predictions_by_class_total",
        amount=count,
        labels={"prediction": normalized},
    )
    if normalized in _prediction_counts:
        _prediction_counts[normalized] += count
    total = _prediction_counts["ATTACK"] + _prediction_counts["BENIGN"]
    if total:
        api_metrics.set_gauge(
            "sentinelml_prediction_attack_share",
            _prediction_counts["ATTACK"] / total,
        )


def record_model_loaded(
    *,
    model_name: str,
    model_version: str,
    lifecycle_state: str = "champion",
) -> None:
    api_metrics.clear_gauge_family("sentinelml_active_model_version_info")
    api_metrics.set_gauge(
        "sentinelml_active_model_version_info",
        1,
        labels={
            "model_name": model_name,
            "model_version": model_version,
            "lifecycle_state": lifecycle_state,
        },
    )
    try:
        version_value: float | None = float(model_version)
    except ValueError:
        version_value = None
    api_metrics.set_gauge("sentinelml_active_model_version", version_value)
    api_metrics.clear_gauge_family("sentinelml_retraining_state")
    api_metrics.set_gauge(
        "sentinelml_retraining_state",
        1,
        labels={"state": "idle"},
    )


def record_model_reload_failure() -> None:
    api_metrics.inc_counter("sentinelml_model_reload_failures_total")


def record_prediction_logging_failure() -> None:
    api_metrics.inc_counter("sentinelml_db_prediction_logging_failures_total")


def record_registry_connectivity(status: str, *, divergent: bool = False) -> None:
    api_metrics.set_gauge(
        "sentinelml_registry_connectivity",
        1 if status == "available" else 0,
    )
    api_metrics.set_gauge(
        "sentinelml_loaded_registry_divergence",
        1 if divergent else 0,
    )


def record_db_logging_health(status: str) -> None:
    value = {"healthy": 1, "unhealthy": 0, "unconfigured": -1}.get(status, 0)
    api_metrics.set_gauge("sentinelml_db_logging_health", value)
