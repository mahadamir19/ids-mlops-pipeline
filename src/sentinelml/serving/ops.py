"""Read-only operations dashboard data facade."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from sentinelml.retraining.config import load_retraining_config
from sentinelml.retraining.repository import RetrainingRepository

STATUS_HEALTHY = {"healthy", "available", "unchanged"}


@dataclass(frozen=True)
class OpsConfig:
    monitor_status_url: str | None
    resilience_status_url: str | None
    timeout_seconds: float


def load_ops_config() -> OpsConfig:
    return OpsConfig(
        monitor_status_url=_optional_env("SENTINELML_MONITOR_STATUS_URL"),
        resilience_status_url=_optional_env("SENTINELML_RESILIENCE_STATUS_URL"),
        timeout_seconds=float(os.environ.get("SENTINELML_OPS_TIMEOUT_SECONDS", "2")),
    )


def ops_overview(app_state: Any, config: OpsConfig | None = None) -> dict[str, Any]:
    config = config or load_ops_config()
    model = _safe(_active_model, app_state)
    registry = _safe(_registry_status, app_state)
    database = _safe(_database_status, app_state)
    monitoring = ops_monitoring(config)
    retraining = ops_retraining()
    resilience = ops_resilience(config)
    components = {
        "inference": _status("available" if model.get("available") else "unavailable"),
        "registry": registry,
        "database_logging": database,
        "monitoring": _component_status(monitoring),
        "retraining": _component_status(retraining),
        "resilience": _component_status(resilience),
    }
    return {
        "status": _aggregate_status(components.values()),
        "champion": model,
        "health": components,
        "metrics": _current_metrics(monitoring),
        "monitoring": monitoring,
        "retraining": retraining,
        "resilience": resilience,
        "latest_event": _latest_event(retraining, resilience),
    }


def ops_models(app_state: Any) -> dict[str, Any]:
    manager = getattr(app_state, "model_manager", None)
    if manager is None:
        return {
            "status": "unavailable",
            "models": [],
            "error": "model manager unavailable",
        }
    try:
        versions = manager.client.search_model_versions(
            f"name='{manager.config.model_name}'"
        )
    except Exception as exc:
        return {"status": "unavailable", "models": [], "error": str(exc)}
    rows = []
    for version in versions:
        tags = getattr(version, "tags", {}) or {}
        aliases = list(getattr(version, "aliases", []) or [])
        rows.append(
            {
                "version": str(getattr(version, "version", "")),
                "model_family": tags.get("model_family"),
                "lifecycle_state": tags.get("lifecycle_state")
                or ("champion" if "champion" in aliases else None),
                "run_id": getattr(version, "run_id", None),
                "execution_mode": tags.get("execution_mode"),
                "rejection_reason": tags.get("rejection_reason")
                or tags.get("failed_reason"),
                "created_timestamp": getattr(version, "creation_timestamp", None),
                "promoted_timestamp": tags.get("promoted_at_utc"),
                "source_model_uri": tags.get("source_model_uri")
                or getattr(version, "source", None),
                "aliases": aliases,
            }
        )
    return {"status": "available", "models": rows}


def ops_monitoring(config: OpsConfig | None = None) -> dict[str, Any]:
    config = config or load_ops_config()
    if not config.monitor_status_url:
        return {
            "status": "no_data",
            "source": "monitor_service",
            "message": "monitor status URL is not configured",
        }
    response = _get_json(config.monitor_status_url, timeout=config.timeout_seconds)
    if not response["ok"]:
        return {
            "status": "unavailable",
            "source": "monitor_service",
            "status_url": config.monitor_status_url,
            "error": response["error"],
        }
    payload = response["payload"]
    status = _normalize_monitoring_status(payload)
    return {
        "status": status,
        "source": "monitor_service",
        "status_url": config.monitor_status_url,
        "latest": payload,
        "heartbeat": _monitoring_heartbeat(payload),
    }


def ops_retraining() -> dict[str, Any]:
    try:
        config = load_retraining_config(validate_data_paths=False)
        repository = RetrainingRepository(config)
        try:
            configured = repository.configured
            schema_available = repository.schema_available()
            if schema_available:
                state = repository.current_state()
                latest_run = repository.latest_run()
                latest_processed = repository.latest_processed_monitoring()
                cooldown = repository.cooldown_status()
            else:
                state = {"state": "idle", "configured": configured}
                latest_run = None
                latest_processed = None
                cooldown = {
                    "active": False,
                    "last_cycle_finished_at": None,
                    "cooldown_until": None,
                }
        finally:
            if repository.engine is not None:
                repository.engine.dispose()
        latest_status = "no_data" if latest_run is None else str(latest_run["status"])
        status = "inactive" if not config.enabled else _retraining_status(state)
        return {
            "status": status,
            "source": "retraining_repository",
            "enabled": config.enabled,
            "execution_mode": config.execution_mode,
            "configured": configured,
            "schema_available": schema_available,
            "state": state,
            "cooldown": cooldown,
            "latest": latest_run,
            "latest_status": latest_status,
            "latest_processed_monitoring": latest_processed,
            "latest_decision": latest_processed,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "retraining_repository",
            "error": str(exc),
        }


def ops_resilience(config: OpsConfig | None = None) -> dict[str, Any]:
    config = config or load_ops_config()
    if not config.resilience_status_url:
        return {
            "status": "no_data",
            "source": "resilience_service",
            "message": "resilience status URL is not configured",
        }
    response = _get_json(config.resilience_status_url, timeout=config.timeout_seconds)
    if not response["ok"]:
        return {
            "status": "unavailable",
            "source": "resilience_service",
            "status_url": config.resilience_status_url,
            "error": response["error"],
        }
    payload = response["payload"]
    return {
        "status": _normalize_resilience_status(payload),
        "source": "resilience_service",
        "status_url": config.resilience_status_url,
        "latest": payload,
        "heartbeat": _resilience_heartbeat(payload),
    }


def _safe(fn: Any, *args: Any) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def _status(value: str) -> dict[str, Any]:
    return {"status": value}


def _active_model(app_state: Any) -> dict[str, Any]:
    manager = app_state.model_manager
    active = manager.current()
    return {
        "status": "available",
        "available": True,
        "model_name": active.model_name,
        "version": active.model_version,
        "family": active.model_family,
        "execution_mode": active.execution_mode,
        "demo_model": active.demo_model,
        "loaded_at": active.loaded_at,
        "source_run_id": active.source_run_id,
    }


def _registry_status(app_state: Any) -> dict[str, Any]:
    raw = app_state.model_manager.registry_status()
    connectivity = str(raw.get("connectivity") or "unavailable")
    return {
        **raw,
        "status": "available" if connectivity == "available" else "unavailable",
    }


def _database_status(app_state: Any) -> dict[str, Any]:
    raw = app_state.repository.database_status()
    return {"status": raw}


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSONDecodeError: {exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "status endpoint returned non-object JSON"}
    return {"ok": True, "payload": payload}


def _component_status(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(payload.get("status", "unavailable")),
        "error": payload.get("error"),
        "message": payload.get("message"),
    }


def _aggregate_status(components: Any) -> str:
    statuses = [str(component.get("status", "unavailable")) for component in components]
    if any(status == "unavailable" for status in statuses):
        return "degraded"
    if any(status in {"stale", "degraded", "unhealthy"} for status in statuses):
        return "degraded"
    if all(status in STATUS_HEALTHY | {"inactive", "no_data"} for status in statuses):
        return "healthy"
    return "degraded"


def _normalize_monitoring_status(payload: dict[str, Any]) -> str:
    health = str(payload.get("monitoring_health") or payload.get("status") or "")
    status = str(payload.get("status") or health)
    if health == "healthy" or status in {"healthy", "unchanged"}:
        return "healthy"
    if health == "warming_up" or status == "warming_up":
        return "warming_up"
    if health == "stale" or status == "stale":
        return "stale"
    if health == "unhealthy" or status == "unhealthy":
        return "degraded"
    return status or "no_data"


def _normalize_resilience_status(payload: dict[str, Any]) -> str:
    raw = str(payload.get("resilience_health") or payload.get("status") or "")
    if raw in {"healthy", "available"}:
        active_count = int(payload.get("active_probation_count") or 0)
        return "healthy" if active_count else "inactive"
    if raw == "warming_up":
        return "warming_up"
    if raw in {"unhealthy", "degraded"}:
        return "degraded"
    return raw or "no_data"


def _retraining_status(state: dict[str, Any]) -> str:
    if not bool(state.get("configured", False)):
        return "no_data"
    raw = str(state.get("state") or "idle")
    if raw == "idle" and state.get("latest_run") is None:
        return "no_data"
    if raw in {"idle", "cooldown"}:
        return raw
    return "active"


def _monitoring_heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _normalize_monitoring_status(payload),
        "last_check_timestamp": payload.get("last_check_timestamp"),
        "last_full_monitoring_report_timestamp": payload.get(
            "last_full_monitoring_report_timestamp"
        ),
        "monitoring_run_id": payload.get("monitoring_run_id")
        or payload.get("last_monitoring_run_id"),
        "poll_status": payload.get("poll_status"),
    }


def _resilience_heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _normalize_resilience_status(payload),
        "database_health": payload.get("database_health"),
        "registry_connectivity": payload.get("registry_connectivity"),
        "active_probation_count": payload.get("active_probation_count"),
    }


def _current_metrics(monitoring: dict[str, Any]) -> dict[str, Any]:
    latest = monitoring.get("latest") or {}
    performance = latest.get("performance") or {}
    support = performance.get("support") or {}
    return {
        "drift_share": latest.get("drift_share"),
        "attack_recall": performance.get("attack_recall"),
        "f1": performance.get("f1"),
        "false_positive_rate": performance.get("false_positive_rate"),
        "labelled_rows": latest.get("labelled_row_count")
        or support.get("labelled_rows"),
        "prediction_distribution": latest.get("prediction_distribution"),
    }


def _latest_event(
    retraining: dict[str, Any],
    resilience: dict[str, Any],
) -> dict[str, Any] | None:
    return (
        retraining.get("latest")
        or retraining.get("latest_processed_monitoring")
        or resilience.get("latest")
    )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None
