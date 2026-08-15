"""FastAPI exporter and controller for Phase 9 resilience."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response

from sentinelml.resilience.config import load_resilience_config
from sentinelml.resilience.metrics import resilience_metrics
from sentinelml.resilience.service import ResilienceService


def create_app() -> FastAPI:
    holder: dict[str, Any] = {"latest": None, "service": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = ResilienceService(config=load_resilience_config())
        holder["service"] = service
        app.state.service = service
        stop_event = asyncio.Event()
        task = asyncio.create_task(_watch(holder, stop_event))
        try:
            yield
        finally:
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="SentinelML Phase 9 Resilience",
        version="0.9.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest = holder["latest"]
        if latest is None:
            return {"status": "warming_up", "resilience_health": "warming_up"}
        return {
            "status": latest.get("status"),
            "resilience_health": latest.get("resilience_health"),
            "database_health": latest.get("database_health"),
            "registry_connectivity": latest.get("registry_connectivity"),
            "active_probation_count": latest.get("active_probation_count"),
            "error": latest.get("error"),
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        latest = _run_status(holder["service"])
        holder["latest"] = latest
        return latest

    @app.post("/evaluate-probation")
    def evaluate_probation() -> dict[str, Any]:
        service = _service(holder["service"])
        return {"status": "processed", "results": service.evaluate_probation()}

    @app.post("/retry-pending")
    def retry_pending() -> dict[str, Any]:
        service = _service(holder["service"])
        return {"status": "processed", "results": service.retry_pending()}

    @app.get("/metrics")
    def metrics() -> Response:
        if holder["latest"] is None:
            holder["latest"] = _run_status(holder["service"])
        return Response(
            resilience_metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


async def _watch(holder: dict[str, Any], stop_event: asyncio.Event) -> None:
    service = _service(holder["service"])
    while not stop_event.is_set():
        holder["latest"] = await asyncio.to_thread(_run_cycle, service)
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=service.config.poll_interval_seconds,
            )
        except TimeoutError:
            continue


def _run_cycle(service: ResilienceService) -> dict[str, Any]:
    try:
        evaluations = service.evaluate_probation()
        retries = service.retry_pending()
        status = service.status()
        status.update(
            {
                "resilience_health": "healthy",
                "status": "healthy",
                "database_health": (
                    "healthy" if status.get("configured") else "unconfigured"
                ),
                "registry_connectivity": _registry_connectivity(service),
                "last_evaluation_count": len(evaluations),
                "last_retry_count": len(retries),
            }
        )
        resilience_metrics.set_gauge(
            "sentinelml_db_logging_health",
            1 if status.get("configured") else -1,
        )
        return status
    except Exception as exc:
        resilience_metrics.set_gauge("sentinelml_db_logging_health", 0)
        return {
            "resilience_health": "unhealthy",
            "status": "unhealthy",
            "database_health": "unhealthy",
            "registry_connectivity": "unknown",
            "error": str(exc),
        }


def _run_status(service_value: Any) -> dict[str, Any]:
    service = _service(service_value)
    try:
        status = service.status()
        registry_connectivity = _registry_connectivity(service)
        db_health = "healthy" if status.get("configured") else "unconfigured"
        resilience_metrics.set_gauge(
            "sentinelml_db_logging_health",
            1 if db_health == "healthy" else -1,
        )
        status.update(
            {
                "resilience_health": "healthy",
                "status": "healthy",
                "database_health": db_health,
                "registry_connectivity": registry_connectivity,
            }
        )
        return status
    except Exception as exc:
        resilience_metrics.set_gauge("sentinelml_db_logging_health", 0)
        return {
            "resilience_health": "unhealthy",
            "status": "unhealthy",
            "database_health": "unhealthy",
            "registry_connectivity": "unknown",
            "error": str(exc),
        }


def _registry_connectivity(service: ResilienceService) -> str:
    try:
        service.lifecycle_service.get_champion()
    except Exception:
        resilience_metrics.set_gauge("sentinelml_registry_connectivity", 0)
        return "unavailable"
    resilience_metrics.set_gauge("sentinelml_registry_connectivity", 1)
    return "available"


def _service(value: Any) -> ResilienceService:
    if not isinstance(value, ResilienceService):
        raise RuntimeError("resilience service is not initialized")
    return value


app = create_app()
