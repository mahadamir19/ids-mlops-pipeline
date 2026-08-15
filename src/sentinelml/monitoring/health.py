"""Monitoring health state helpers."""

from __future__ import annotations

from typing import Literal

MonitoringHealth = Literal["healthy", "warming_up", "unhealthy"]


def health_from_status(status: str) -> MonitoringHealth:
    if status == "healthy":
        return "healthy"
    if status in {"warming_up", "insufficient_data"}:
        return "warming_up"
    return "unhealthy"
