"""Promotion-pending retry coordination for Phase 9."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sentinelml.lifecycle.service import LifecycleService
from sentinelml.resilience.config import ResilienceConfig
from sentinelml.resilience.metrics import (
    record_promotion_pending,
    record_promotion_retry,
)
from sentinelml.resilience.reports import write_resilience_report
from sentinelml.resilience.repository import ResilienceRepository, utc_now


class PromotionRetryService:
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

    def retry_pending(self) -> list[dict[str, Any]]:
        self.repository.initialize()
        if not self.config.promotion_retry_enabled:
            return []
        results = self.lifecycle_service.retry_pending()
        record_promotion_pending(self._pending_count_after_retry(results))
        for result in results:
            outcome = result.get("outcome", {})
            status = _retry_status(outcome)
            payload = {
                "retry_id": str(uuid4()),
                "candidate_version": str(
                    result.get("pending", {}).get("candidate_version", "")
                ),
                "started_at": result.get("retried_at", utc_now()),
                "completed_at": utc_now(),
                "status": status,
                "current_champion_version": _current_champion_version(
                    self.lifecycle_service
                ),
                "gate_evaluation": outcome.get("evaluation")
                or outcome.get("original_gate_evaluation")
                or {},
                "outcome": outcome,
                "error": outcome.get("registry_error"),
            }
            self.repository.insert_retry_attempt(payload)
            write_resilience_report(
                payload,
                self.config.resilience_reports_dir,
                category="promotion_retry",
                name=f"{payload['retry_id']}.json",
            )
            record_promotion_retry(status=status)
        return results

    def _pending_count_after_retry(self, results: list[dict[str, Any]]) -> int:
        return sum(
            1
            for result in results
            if _retry_status(result.get("outcome", {})) == "promotion_pending"
        )


def _retry_status(outcome: dict[str, Any]) -> str:
    if outcome.get("event") == "promotion_pending":
        return "promotion_pending"
    if outcome.get("operation_state") == "promotion_pending":
        return "promotion_pending"
    if outcome.get("event") == "promoted":
        return str(outcome.get("status", "promoted"))
    return str(outcome.get("event", "unknown"))


def _current_champion_version(lifecycle_service: LifecycleService) -> str | None:
    champion = lifecycle_service.get_champion()
    return champion.version if champion else None
