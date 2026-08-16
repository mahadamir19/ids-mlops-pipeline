"""Promotion worker for the LifecycleService facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sentinelml.lifecycle.audit import audit_path, utc_now, write_json
from sentinelml.lifecycle.errors import LifecycleError
from sentinelml.lifecycle.registry import ModelVersionInfo, set_version_tags, tag_json
from sentinelml.lifecycle.thresholds import load_json


class PromotionManager:
    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        alias: str,
        report_root: Path,
        pending_dir: Path,
        get_version: Callable[[str | int], ModelVersionInfo],
        evaluate_candidate: Callable[..., dict[str, Any]],
        get_champion: Callable[[], ModelVersionInfo | None],
        notify_serving_reload: Callable[[], dict[str, Any]],
        start_probation_after_promotion: Callable[..., dict[str, Any]],
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.alias = alias
        self.report_root = report_root
        self.pending_dir = pending_dir
        self.get_version = get_version
        self.evaluate_candidate = evaluate_candidate
        self.get_champion = get_champion
        self.notify_serving_reload = notify_serving_reload
        self.start_probation_after_promotion = start_probation_after_promotion

    def promote_or_reject(self, *, version: str | int) -> dict[str, Any]:
        candidate = self.get_version(version)
        evaluation = self.evaluate_candidate(version=version)
        champion = self.get_champion()
        if champion is not None and champion.version == candidate.version:
            return {
                "event": "promoted",
                "status": "already_champion",
                "model_name": self.model_name,
                "model_version": candidate.version,
                "evaluation": evaluation,
            }
        if not evaluation["passed"]:
            return self.reject_candidate(candidate, evaluation)
        return self.promote_candidate(candidate, evaluation, champion)

    def retry_pending(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.pending_dir.glob("*.json")):
            pending = load_json(path)
            version = pending["candidate_version"]
            champion_before = self.get_champion()
            outcome = self.promote_or_reject(version=version)
            archive_payload = {
                "event": "promotion_retry",
                "retried_at": utc_now(),
                "pending": pending,
                "current_champion_version": champion_before.version
                if champion_before
                else None,
                "outcome": outcome,
            }
            still_pending = (
                outcome.get("event") == "promotion_pending"
                or outcome.get("operation_state") == "promotion_pending"
            )
            if still_pending:
                retry_history = pending.get("retry_history", [])
                pending["retry_history"] = [*retry_history, archive_payload]
                pending["last_retry_at"] = archive_payload["retried_at"]
                pending["last_retry_outcome"] = outcome
                write_json(path, pending)
            else:
                archive = self.pending_dir.parent / "resolved" / path.name
                write_json(archive, archive_payload)
                path.unlink()
            write_json(
                audit_path(self.report_root, "promotion_retry", version),
                archive_payload,
            )
            results.append(archive_payload)
        return results

    def reject_candidate(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        failed = evaluation["absolute_gates"]["failed_gates"]
        if evaluation["comparison"].get("mode_mismatch"):
            failed = [*failed, "execution_mode"]
        elif not evaluation["comparison"].get("relative_passed", True):
            failed = [*failed, "composite_improvement"]
        rejected_at = utc_now()
        tags = {
            "lifecycle_state": "rejected",
            "lifecycle.rejected_at": rejected_at,
            "lifecycle.failed_gates": ",".join(failed),
            "lifecycle.gate_result_json": tag_json(
                {"passed": False, "failed_gates": failed}
            ),
        }
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags=tags,
        )
        payload = {
            "event": "rejected",
            "model_name": self.model_name,
            "model_version": candidate.version,
            "rejected_at": rejected_at,
            "failed_gates": failed,
            "evaluation": evaluation,
        }
        write_json(audit_path(self.report_root, "rejected", candidate.version), payload)
        return payload

    def mark_candidate_failed(
        self,
        *,
        version: str | int,
        error: str,
        retraining_run_id: str | None = None,
        monitoring_run_id: str | None = None,
    ) -> dict[str, Any]:
        candidate = self.get_version(version)
        failed_at = utc_now()
        payload = {
            "event": "failed",
            "model_name": self.model_name,
            "model_version": candidate.version,
            "failed_at": failed_at,
            "error": error,
            "retraining_run_id": retraining_run_id,
            "monitoring_run_id": monitoring_run_id,
        }
        tags = {
            "lifecycle_state": "failed",
            "lifecycle.failed_at": failed_at,
            "lifecycle.failure_reason": error[:1000],
        }
        if retraining_run_id:
            tags["retraining_run_id"] = retraining_run_id
        if monitoring_run_id:
            tags["trigger_monitoring_run_id"] = monitoring_run_id
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags=tags,
        )
        write_json(audit_path(self.report_root, "failed", candidate.version), payload)
        return payload

    def promote_candidate(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
        previous_champion: ModelVersionInfo | None,
    ) -> dict[str, Any]:
        promoted_at = utc_now()
        previous_version = previous_champion.version if previous_champion else None
        try:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "promotion_pending",
                    "lifecycle.promotion_attempted_at": promoted_at,
                    "lifecycle.previous_champion_version": previous_version or "",
                },
            )
            self.client.set_registered_model_alias(
                self.model_name,
                self.alias,
                candidate.version,
            )
            verified = self.get_champion()
            if verified is None or verified.version != candidate.version:
                raise LifecycleError("champion alias verification failed")
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "champion",
                    "lifecycle.promoted_at": promoted_at,
                    "lifecycle.previous_champion_version": previous_version or "",
                    "lifecycle.evaluated_against_version": previous_version or "",
                },
            )
            if previous_champion is not None:
                set_version_tags(
                    self.client,
                    model_name=self.model_name,
                    version=previous_champion.version,
                    tags={
                        "lifecycle_state": "superseded",
                        "lifecycle.superseded_at": promoted_at,
                        "lifecycle.superseded_by_version": candidate.version,
                    },
                )
            payload = {
                "event": "promoted",
                "model_name": self.model_name,
                "model_version": candidate.version,
                "previous_champion_version": previous_version,
                "promoted_at": promoted_at,
                "evaluation": evaluation,
            }
            payload["serving_reload_notification"] = self.notify_serving_reload()
            payload["probation"] = self.start_probation_after_promotion(
                promoted_version=candidate.version,
                previous_champion_version=previous_version,
                promotion_timestamp=promoted_at,
            )
            write_json(
                audit_path(self.report_root, "promoted", candidate.version),
                payload,
            )
            return payload
        except Exception as exc:
            return self.write_pending_promotion(
                candidate,
                evaluation,
                previous_champion,
                str(exc),
            )

    def write_pending_promotion(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
        previous_champion: ModelVersionInfo | None,
        error: str,
    ) -> dict[str, Any]:
        created_at = utc_now()
        try:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "promotion_pending",
                    "lifecycle.promotion_pending_at": created_at,
                },
            )
        except Exception:
            pass
        payload = {
            "event": "promotion_pending",
            "operation_state": "promotion_pending",
            "model_name": self.model_name,
            "candidate_version": candidate.version,
            "candidate_source_run_id": candidate.tags.get("source_run_id"),
            "candidate_source_model_uri": candidate.tags.get("source_model_uri"),
            "previous_champion_version": previous_champion.version
            if previous_champion
            else None,
            "created_at": created_at,
            "registry_error": error,
            "original_gate_evaluation": evaluation,
        }
        path = self.pending_dir / f"{self.model_name}_version_{candidate.version}.json"
        write_json(path, payload)
        write_json(
            audit_path(self.report_root, "promotion_pending", candidate.version),
            payload,
        )
        return payload
