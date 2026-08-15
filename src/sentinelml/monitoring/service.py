"""Phase 7 monitoring orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.monitoring.config import (
    DEFAULT_MONITORING_CONFIG_PATH,
    MonitoringConfig,
    load_monitoring_config,
)
from sentinelml.monitoring.data import (
    CurrentWindow,
    ReferenceDataset,
    load_current_window,
    load_reference_dataset,
)
from sentinelml.monitoring.drift import EvidentlyDriftRunner, calculate_drift
from sentinelml.monitoring.fingerprint import monitoring_input_fingerprint
from sentinelml.monitoring.health import health_from_status
from sentinelml.monitoring.metrics import update_monitoring_metrics
from sentinelml.monitoring.performance import (
    calculate_performance,
    model_version_distribution,
    prediction_distribution,
    true_label_distribution,
)
from sentinelml.monitoring.reports import (
    latest_monitoring_report,
    write_monitoring_report,
    write_monitoring_status,
)
from sentinelml.retraining.config import load_retraining_config
from sentinelml.retraining.repository import latest_retraining_state
from sentinelml.serving.validation import load_serving_feature_schema


def run_monitoring_once(
    *,
    config_path: Path = DEFAULT_MONITORING_CONFIG_PATH,
    window_size: int | None = None,
    evidently_runner: EvidentlyDriftRunner | None = None,
    write_report: bool = True,
    force_recompute: bool = False,
) -> dict[str, Any]:
    config = load_monitoring_config(config_path, window_size=window_size)
    return run_monitoring_cycle(
        config,
        evidently_runner=evidently_runner,
        write_report=write_report,
        force_recompute=force_recompute,
    )


def run_monitoring_cycle(
    config: MonitoringConfig,
    *,
    evidently_runner: EvidentlyDriftRunner | None = None,
    write_report: bool = True,
    force_recompute: bool = False,
) -> dict[str, Any]:
    checked_at = datetime.now(UTC).isoformat()
    errors: list[dict[str, Any]] = []
    try:
        schema = load_serving_feature_schema(config.feature_schema_path)
        reference = load_reference_dataset(config, schema)
        current = load_current_window(config, schema)
        fingerprint = monitoring_input_fingerprint(
            config=config,
            schema=schema,
            reference=reference,
            current=current,
        )
        previous = latest_monitoring_report(config.reports_dir)
        if not force_recompute and _unchanged(previous, fingerprint):
            report = _unchanged_report(
                checked_at=checked_at,
                previous=previous,
                fingerprint=fingerprint,
            )
            if write_report:
                status_path = write_monitoring_status(report, config.reports_dir)
                report["status_path"] = str(status_path)
            update_monitoring_metrics(report)
            return report
        started_at = checked_at
        run_id = str(uuid4())
        report = _build_report(
            config=config,
            run_id=run_id,
            started_at=started_at,
            reference=reference,
            current=current,
            fingerprint=fingerprint,
            force_recompute=force_recompute,
            errors=errors,
            evidently_runner=evidently_runner,
        )
    except Exception as exc:
        errors.append({"stage": "monitoring_cycle", "error": str(exc)})
        report = _failure_report(
            config=config,
            run_id=str(uuid4()),
            started_at=checked_at,
            errors=errors,
        )
    if write_report:
        report_path = write_monitoring_report(report, config.reports_dir)
        report["report_path"] = str(report_path)
    update_monitoring_metrics(report)
    return report


def _build_report(
    *,
    config: MonitoringConfig,
    run_id: str,
    started_at: str,
    reference: ReferenceDataset,
    current: CurrentWindow,
    fingerprint: dict[str, Any],
    force_recompute: bool,
    errors: list[dict[str, Any]],
    evidently_runner: EvidentlyDriftRunner | None,
) -> dict[str, Any]:
    finished_at = datetime.now(UTC).isoformat()
    prediction_dist = prediction_distribution(current.rows)
    true_dist = true_label_distribution(current.rows)
    performance = calculate_performance(current.rows, config)
    base = {
        "monitoring_run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "reference": reference.metadata,
        "window": current.metadata,
        "monitoring_input_fingerprint": fingerprint["value"],
        "monitoring_input_fingerprint_metadata": fingerprint,
        "force_recompute": bool(force_recompute),
        "poll_status": "computed",
        "last_check_timestamp": finished_at,
        "last_full_monitoring_report_timestamp": finished_at,
        "window_size": current.metadata["actual_window_size"],
        "model_version_distribution": model_version_distribution(current.rows),
        "prediction_distribution": prediction_dist,
        "true_label_distribution": true_dist,
        "labelled_row_count": performance["support"]["labelled_rows"],
        "performance": performance,
        "errors": errors,
        "retraining_state": _phase8_retraining_state(),
        "resilience_state": _phase9_resilience_state(),
        "lifecycle_state": "champion_active",
    }
    if current.metadata["actual_window_size"] < config.minimum_window_size:
        return {
            **base,
            "status": "warming_up",
            "monitoring_health": "warming_up",
            "drifting_features": [],
            "drifting_feature_count": None,
            "monitored_feature_count": len(config.monitored_features),
            "drift_share": None,
            "data_drift_detected": None,
            "zero_variance_reference_features": [],
            "zero_variance_current_features": [],
            "evidently_result_summary": None,
        }
    try:
        drift = calculate_drift(
            reference.frame,
            current.frame,
            config,
            evidently_runner=evidently_runner,
        )
    except Exception as exc:
        errors.append({"stage": "evidently_drift", "error": str(exc)})
        return {
            **base,
            "status": "unhealthy",
            "monitoring_health": "unhealthy",
            "drifting_features": [],
            "drifting_feature_count": None,
            "monitored_feature_count": len(config.monitored_features),
            "drift_share": None,
            "data_drift_detected": None,
            "zero_variance_reference_features": [],
            "zero_variance_current_features": [],
            "evidently_result_summary": None,
        }
    return {
        **base,
        "status": "healthy",
        "monitoring_health": "healthy",
        "drifting_features": drift.drifting_features,
        "drift_feature_scores": drift.feature_scores,
        "drifting_feature_count": drift.drifting_feature_count,
        "monitored_feature_count": drift.monitored_feature_count,
        "drift_share": drift.drift_share,
        "data_drift_detected": drift.data_drift_detected,
        "zero_variance_reference_features": drift.zero_variance_reference_features,
        "zero_variance_current_features": drift.zero_variance_current_features,
        "evidently_result_summary": drift.evidently_summary,
        "evidently_compact_payload": drift.evidently_payload,
    }


def _failure_report(
    *,
    config: MonitoringConfig,
    run_id: str,
    started_at: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    status = "unhealthy"
    finished_at = datetime.now(UTC).isoformat()
    return {
        "monitoring_run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "last_check_timestamp": finished_at,
        "last_full_monitoring_report_timestamp": None,
        "poll_status": "failed",
        "status": status,
        "monitoring_health": health_from_status(status),
        "reference": {"path": str(config.reference_path)},
        "window": {
            "requested_window_size": config.window_size,
            "minimum_window_size": config.minimum_window_size,
            "actual_window_size": 0,
        },
        "window_size": 0,
        "model_version_distribution": {},
        "prediction_distribution": None,
        "true_label_distribution": None,
        "labelled_row_count": 0,
        "drifting_features": [],
        "drifting_feature_count": None,
        "monitored_feature_count": len(config.monitored_features),
        "drift_share": None,
        "data_drift_detected": None,
        "zero_variance_reference_features": [],
        "zero_variance_current_features": [],
        "performance": {
            "status": "unavailable",
            "availability": "unavailable",
            "support": {"labelled_rows": 0, "true_attack": 0, "true_benign": 0},
            "attack_recall": None,
            "f1": None,
            "false_positive_rate": None,
        },
        "evidently_result_summary": None,
        "errors": errors,
        "retraining_state": _phase8_retraining_state(),
        "resilience_state": _phase9_resilience_state(),
        "lifecycle_state": "champion_active",
    }


def _unchanged(previous: dict[str, Any] | None, fingerprint: dict[str, Any]) -> bool:
    if previous is None:
        return False
    if previous.get("monitoring_health") != "healthy":
        return False
    return previous.get("monitoring_input_fingerprint") == fingerprint["value"]


def _unchanged_report(
    *,
    checked_at: str,
    previous: dict[str, Any],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "unchanged",
        "monitoring_health": "healthy",
        "action": "skipped_unchanged_input",
        "poll_status": "unchanged",
        "last_check_timestamp": checked_at,
        "last_full_monitoring_report_timestamp": previous.get("finished_at"),
        "last_monitoring_run_id": previous.get("monitoring_run_id"),
        "monitoring_run_id": previous.get("monitoring_run_id"),
        "monitoring_input_fingerprint": fingerprint["value"],
        "monitoring_input_fingerprint_metadata": fingerprint,
        "finished_at": previous.get("finished_at"),
        "errors": [],
        "window": previous.get("window", {}),
        "window_size": previous.get("window_size"),
        "data_drift_detected": previous.get("data_drift_detected"),
        "drifting_feature_count": previous.get("drifting_feature_count"),
        "monitored_feature_count": previous.get("monitored_feature_count"),
        "drift_share": previous.get("drift_share"),
        "performance": previous.get("performance", {}),
        "retraining_state": _phase8_retraining_state(),
        "resilience_state": _phase9_resilience_state(),
        "lifecycle_state": previous.get("lifecycle_state", "champion_active"),
    }


def _phase8_retraining_state() -> dict[str, Any] | str:
    try:
        config = load_retraining_config(validate_data_paths=False)
        return latest_retraining_state(config)
    except Exception:
        return "idle"


def _phase9_resilience_state() -> dict[str, Any]:
    path = PROJECT_ROOT / "reports" / "resilience" / "latest.json"
    if not path.exists():
        return {"state": "idle", "source": "reports/resilience/latest.json"}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "state": "unavailable",
            "source": "reports/resilience/latest.json",
            "error": str(exc),
        }
    return {
        "state": "available",
        "source": "reports/resilience/latest.json",
        "latest": payload,
    }
