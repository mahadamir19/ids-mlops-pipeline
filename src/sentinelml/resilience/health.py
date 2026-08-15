"""Operational health helpers for Phase 9 resilience gates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def monitoring_heartbeat_status(
    report: dict[str, Any],
    *,
    maximum_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify a Phase 7 report as fresh, stale, or unavailable."""

    now = now or datetime.now(UTC)
    timestamp = report.get("last_check_timestamp") or report.get("finished_at")
    if not timestamp:
        return {
            "monitoring_available": False,
            "monitoring_stale": True,
            "reason": "monitoring_heartbeat_missing",
            "last_check_timestamp": None,
            "age_seconds": None,
            "maximum_age_seconds": maximum_age_seconds,
        }
    checked_at = _parse_datetime(str(timestamp))
    age = (now - checked_at).total_seconds()
    stale = age > maximum_age_seconds
    return {
        "monitoring_available": not stale,
        "monitoring_stale": stale,
        "reason": "monitoring_stale" if stale else "monitoring_fresh",
        "last_check_timestamp": checked_at.isoformat(),
        "age_seconds": max(0.0, float(age)),
        "maximum_age_seconds": maximum_age_seconds,
    }


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
