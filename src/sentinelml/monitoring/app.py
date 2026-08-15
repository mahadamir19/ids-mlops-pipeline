"""FastAPI exporter for the dedicated Phase 7 monitoring service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response

from sentinelml.monitoring.config import load_monitoring_config
from sentinelml.monitoring.metrics import monitoring_metrics
from sentinelml.monitoring.service import run_monitoring_cycle


def create_app() -> FastAPI:
    holder: dict[str, Any] = {"latest": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = load_monitoring_config()
        app.state.config = config
        stop_event = asyncio.Event()
        task = asyncio.create_task(_watch(app, holder, stop_event))
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
        title="SentinelML Phase 7 Monitoring",
        version="0.7.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest = holder["latest"]
        if latest is None:
            return {"status": "warming_up", "monitoring_health": "warming_up"}
        return {
            "status": latest.get("status"),
            "monitoring_health": latest.get("monitoring_health"),
            "finished_at": latest.get("finished_at"),
            "last_check_timestamp": latest.get("last_check_timestamp"),
            "last_monitoring_run_id": latest.get("last_monitoring_run_id")
            or latest.get("monitoring_run_id"),
            "poll_status": latest.get("poll_status"),
            "errors": latest.get("errors", [])[:3],
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        latest = holder["latest"]
        if latest is None:
            return {"status": "warming_up", "monitoring_health": "warming_up"}
        return latest

    @app.post("/run")
    def run_once(force_recompute: bool = False) -> dict[str, Any]:
        latest = run_monitoring_cycle(
            app.state.config,
            force_recompute=force_recompute,
        )
        holder["latest"] = latest
        return latest

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            monitoring_metrics.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


async def _watch(
    app: FastAPI,
    holder: dict[str, Any],
    stop_event: asyncio.Event,
) -> None:
    config = app.state.config
    while not stop_event.is_set():
        holder["latest"] = run_monitoring_cycle(config)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=config.interval_seconds)
        except TimeoutError:
            continue


app = create_app()
