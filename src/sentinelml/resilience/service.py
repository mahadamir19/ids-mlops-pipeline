"""Phase 9 resilience controller."""

from __future__ import annotations

import time
from typing import Any

from sentinelml.lifecycle.config import load_lifecycle_config
from sentinelml.lifecycle.service import LifecycleService
from sentinelml.resilience.config import ResilienceConfig, load_resilience_config
from sentinelml.resilience.metrics import (
    record_probation_status,
    record_promotion_pending,
)
from sentinelml.resilience.probation import ProbationService
from sentinelml.resilience.promotion_retry import PromotionRetryService
from sentinelml.resilience.reports import write_resilience_report
from sentinelml.resilience.repository import ResilienceRepository
from sentinelml.resilience.rollback import RollbackService


class ResilienceService:
    def __init__(
        self,
        *,
        config: ResilienceConfig | None = None,
        repository: ResilienceRepository | None = None,
        lifecycle_service: LifecycleService | None = None,
    ) -> None:
        self.config = config or load_resilience_config()
        self.repository = repository or ResilienceRepository(self.config)
        self.lifecycle_service = lifecycle_service or LifecycleService(
            config=load_lifecycle_config()
        )
        self.probation_service = ProbationService(
            config=self.config,
            repository=self.repository,
        )
        self.rollback_service = RollbackService(
            config=self.config,
            repository=self.repository,
            lifecycle_service=self.lifecycle_service,
        )
        self.retry_service = PromotionRetryService(
            config=self.config,
            repository=self.repository,
            lifecycle_service=self.lifecycle_service,
        )

    def status(self) -> dict[str, Any]:
        self.repository.initialize()
        status = self.repository.summary()
        record_probation_status(status)
        record_promotion_pending(self._pending_promotion_count())
        write_resilience_report(
            status,
            self.config.resilience_reports_dir,
            category="status",
            name="status.json",
        )
        return status

    def start_probation_after_promotion(
        self,
        *,
        promoted_version: str,
        previous_champion_version: str | None,
        promotion_timestamp: str | None = None,
    ) -> dict[str, Any]:
        result = self.probation_service.start_probation(
            promoted_version=promoted_version,
            previous_champion_version=previous_champion_version,
            promotion_timestamp=promotion_timestamp,
        )
        write_resilience_report(
            result,
            self.config.resilience_reports_dir,
            category="probation",
            name=(
                f"{result['probation']['probation_id']}.json"
                if result.get("started")
                else "not_started.json"
            ),
        )
        return result

    def evaluate_probation(self) -> list[dict[str, Any]]:
        self.repository.initialize()
        results: list[dict[str, Any]] = []
        for probation in self.repository.active_probations():
            evaluation = self.probation_service.evaluate(probation)
            payload = {
                "event": "probation_evaluated",
                "probation_id": evaluation.probation_id,
                "promoted_version": evaluation.promoted_version,
                "status": evaluation.status,
                "support": evaluation.support,
                "performance": evaluation.performance,
                "severe_violation_reasons": evaluation.severe_violation_reasons,
                "threshold_source": evaluation.threshold_source,
                "rollback_required": evaluation.rollback_required,
            }
            if evaluation.rollback_required:
                rollback = self.rollback_service.rollback(
                    to_version=str(probation["previous_champion_version"]),
                    source="automatic",
                    reason=";".join(evaluation.severe_violation_reasons),
                    probation_id=evaluation.probation_id,
                    trigger_metrics=evaluation.performance,
                )
                payload["rollback"] = rollback
                self.repository.update_probation(
                    evaluation.probation_id,
                    status=(
                        "rolled_back"
                        if rollback.get("status")
                        in {"succeeded", "already_current", "already_rolled_back"}
                        else "rollback_failed"
                    ),
                    rollback_attempt_id=rollback.get("rollback", {}).get(
                        "rollback_id"
                    ),
                )
            write_resilience_report(
                payload,
                self.config.resilience_reports_dir,
                category="probation",
                name=f"{evaluation.probation_id}.json",
            )
            results.append(payload)
        record_probation_status(self.repository.summary())
        return results

    def rollback(
        self,
        *,
        to_version: str | None,
        reason: str,
    ) -> dict[str, Any]:
        if to_version is None:
            return self.rollback_service.rollback_previous_approved(
                source="manual",
                reason=reason,
            )
        return self.rollback_service.rollback(
            to_version=to_version,
            source="manual",
            reason=reason,
        )

    def retry_pending(self) -> list[dict[str, Any]]:
        return self.retry_service.retry_pending()

    def watch(self) -> None:
        while True:
            self.evaluate_probation()
            self.retry_pending()
            time.sleep(self.config.poll_interval_seconds)

    def _pending_promotion_count(self) -> int:
        if not self.lifecycle_service.pending_dir.exists():
            return 0
        return sum(1 for _ in self.lifecycle_service.pending_dir.glob("*.json"))
