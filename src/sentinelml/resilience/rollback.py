"""Manual and automatic rollback orchestration for Phase 9."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sentinelml.lifecycle.service import LifecycleError, LifecycleService
from sentinelml.resilience.config import ResilienceConfig
from sentinelml.resilience.metrics import record_rollback
from sentinelml.resilience.reports import write_resilience_report
from sentinelml.resilience.repository import ResilienceRepository, utc_now


class RollbackService:
    def __init__(
        self,
        *,
        config: ResilienceConfig,
        repository: ResilienceRepository,
        lifecycle_service: LifecycleService,
    ) -> None:
        self.config = config
        self.repository = repository
        self.lifecycle_service = lifecycle_service

    def rollback(
        self,
        *,
        to_version: str,
        source: str,
        reason: str,
        probation_id: str | None = None,
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.repository.initialize()
        if probation_id:
            existing = self.repository.rollback_for_probation(probation_id)
            if existing is not None:
                return {
                    "event": "rollback",
                    "status": "already_rolled_back",
                    "rollback": existing,
                }
        rollback_id = str(uuid4())
        champion = self.lifecycle_service.get_champion()
        started_at = utc_now()
        initial = {
            "rollback_id": rollback_id,
            "source": source,
            "probation_id": probation_id,
            "from_version": champion.version if champion else None,
            "to_version": str(to_version),
            "trigger_metrics": trigger_metrics or {},
            "reason": reason,
            "started_at": started_at,
            "completed_at": None,
            "status": "started",
            "registry_result": {},
            "reload_result": {},
            "error": None,
        }
        self.repository.insert_rollback_event(initial)
        try:
            registry_result = self.lifecycle_service.rollback(
                version=to_version,
                reason=reason,
                source=source,
                probation_id=probation_id,
                trigger_metrics=trigger_metrics,
            )
        except Exception as exc:
            self.repository.update_rollback_event(
                rollback_id,
                completed_at=utc_now(),
                status="failed",
                error=str(exc),
            )
            raise
        status = str(registry_result.get("status", "succeeded"))
        completed_at = utc_now()
        self.repository.update_rollback_event(
            rollback_id,
            completed_at=completed_at,
            status=status,
            registry_result=registry_result,
            reload_result=registry_result.get("serving_reload_notification", {}),
        )
        payload = {
            **initial,
            "completed_at": completed_at,
            "status": status,
            "registry_result": registry_result,
            "reload_result": registry_result.get("serving_reload_notification", {}),
        }
        write_resilience_report(
            payload,
            self.config.resilience_reports_dir,
            category="rollback",
            name=f"{rollback_id}.json",
        )
        record_rollback(
            source=source,
            status=status,
            completed_timestamp=_unix_timestamp(completed_at),
        )
        return {"event": "rollback", "status": status, "rollback": payload}

    def rollback_previous_approved(
        self,
        *,
        source: str,
        reason: str,
        probation_id: str | None = None,
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        champion = self.lifecycle_service.get_champion()
        if champion is None:
            raise LifecycleError("no current champion is available")
        previous = champion.tags.get("lifecycle.previous_champion_version")
        if not previous:
            raise LifecycleError("current champion has no previous champion recorded")
        return self.rollback(
            to_version=previous,
            source=source,
            reason=reason,
            probation_id=probation_id,
            trigger_metrics=trigger_metrics,
        )


def _unix_timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        UTC
    ).timestamp()
