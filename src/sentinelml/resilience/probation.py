"""Post-promotion probation evaluation for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sentinelml.monitoring.performance import calculate_performance
from sentinelml.resilience.config import ResilienceConfig
from sentinelml.resilience.repository import ResilienceRepository, utc_now


@dataclass(frozen=True)
class ProbationEvaluation:
    probation_id: str
    promoted_version: str
    status: str
    support: dict[str, int]
    performance: dict[str, Any]
    severe_violation_reasons: list[str]
    threshold_source: str
    rollback_required: bool


class ProbationService:
    def __init__(
        self,
        *,
        config: ResilienceConfig,
        repository: ResilienceRepository,
    ) -> None:
        self.config = config
        self.repository = repository

    def start_probation(
        self,
        *,
        promoted_version: str,
        previous_champion_version: str | None,
        promotion_timestamp: str | None = None,
    ) -> dict[str, Any]:
        self.repository.initialize()
        if not self.config.probation_enabled:
            return {"started": False, "reason": "probation_disabled"}
        if not previous_champion_version:
            return {"started": False, "reason": "no_previous_champion"}
        started_at = promotion_timestamp or utc_now()
        started = _parse_datetime(started_at)
        ends_at = (
            started + timedelta(seconds=self.config.probation_duration_seconds)
        ).isoformat()
        probation_id = str(uuid4())
        payload = {
            "probation_id": probation_id,
            "promoted_version": str(promoted_version),
            "previous_champion_version": str(previous_champion_version),
            "promotion_timestamp": started_at,
            "started_at": started_at,
            "ends_at": ends_at,
            "status": "active",
            "minimum_labelled_rows": self.config.probation_minimum_labelled_rows,
            "minimum_attack_support": self.config.probation_minimum_attack_support,
            "minimum_benign_support": self.config.probation_minimum_benign_support,
            "latest_evaluation_at": None,
            "labelled_rows": 0,
            "attack_support": 0,
            "benign_support": 0,
            "performance_metrics": {},
            "severe_violation_reasons": [],
            "rollback_attempt_id": None,
            "completed_at": None,
        }
        self.repository.create_probation(payload)
        return {"started": True, "probation": payload}

    def evaluate(self, probation: dict[str, Any]) -> ProbationEvaluation:
        now = datetime.now(UTC)
        rows = self.repository.probation_observations(
            model_version=str(probation["promoted_version"]),
            started_at=str(probation["started_at"]),
            ends_at=str(probation["ends_at"]),
        )
        performance = calculate_performance(
            rows,
            _performance_config(
                minimum_labelled_rows=int(probation["minimum_labelled_rows"]),
                min_attack_support=int(probation["minimum_attack_support"]),
                min_benign_support=int(probation["minimum_benign_support"]),
            ),
        )
        support = {
            "labelled_rows": int(performance["support"]["labelled_rows"]),
            "true_attack": int(performance["support"]["true_attack"]),
            "true_benign": int(performance["support"]["true_benign"]),
        }
        sufficient = performance.get("status") == "available"
        ended = now >= _parse_datetime(str(probation["ends_at"]))
        severe_reasons = (
            self._severe_violation_reasons(performance) if sufficient else []
        )
        if severe_reasons and self.config.automatic_rollback_enabled:
            status = "rollback_triggered"
        elif not sufficient:
            status = "inconclusive" if ended else "awaiting_evidence"
        elif ended:
            status = "passed"
        else:
            status = "active"
        evaluation = ProbationEvaluation(
            probation_id=str(probation["probation_id"]),
            promoted_version=str(probation["promoted_version"]),
            status=status,
            support=support,
            performance=performance,
            severe_violation_reasons=severe_reasons,
            threshold_source=self.config.severe_guardrails.threshold_source,
            rollback_required=status == "rollback_triggered",
        )
        completed_at = utc_now() if status in {"passed", "inconclusive"} else None
        self.repository.update_probation(
            evaluation.probation_id,
            status=status,
            latest_evaluation_at=utc_now(),
            labelled_rows=support["labelled_rows"],
            attack_support=support["true_attack"],
            benign_support=support["true_benign"],
            performance_metrics=performance,
            severe_violation_reasons=severe_reasons,
            completed_at=completed_at,
        )
        return evaluation

    def _severe_violation_reasons(
        self,
        performance: dict[str, Any],
    ) -> list[str]:
        guardrails = self.config.severe_guardrails
        reasons: list[str] = []
        attack_recall = _optional_float(performance.get("attack_recall"))
        f1 = _optional_float(performance.get("f1"))
        fpr = _optional_float(performance.get("false_positive_rate"))
        if (
            guardrails.min_attack_recall is not None
            and attack_recall is not None
            and attack_recall < guardrails.min_attack_recall
        ):
            reasons.append("severe_attack_recall_below_threshold")
        if guardrails.min_f1 is not None and f1 is not None and f1 < guardrails.min_f1:
            reasons.append("severe_f1_below_threshold")
        if (
            guardrails.max_false_positive_rate is not None
            and fpr is not None
            and fpr > guardrails.max_false_positive_rate
        ):
            reasons.append("severe_false_positive_rate_above_threshold")
        return reasons


@dataclass(frozen=True)
class _PerformanceConfig:
    minimum_labelled_rows: int
    min_attack_support: int
    min_benign_support: int


def _performance_config(
    *,
    minimum_labelled_rows: int,
    min_attack_support: int,
    min_benign_support: int,
) -> _PerformanceConfig:
    return _PerformanceConfig(
        minimum_labelled_rows=minimum_labelled_rows,
        min_attack_support=min_attack_support,
        min_benign_support=min_benign_support,
    )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
