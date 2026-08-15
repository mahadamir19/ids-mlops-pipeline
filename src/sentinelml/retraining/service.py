"""Phase 8 continuous-retraining controller."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from sentinelml.lifecycle.config import load_lifecycle_config
from sentinelml.lifecycle.service import LifecycleService
from sentinelml.lifecycle.thresholds import load_json
from sentinelml.production_data.repository import ProductionDataRepository
from sentinelml.production_data.service import ProductionDataService
from sentinelml.retraining.config import RetrainingConfig, load_retraining_config
from sentinelml.retraining.dataset import build_retraining_dataset
from sentinelml.retraining.lock import RetrainingLock
from sentinelml.retraining.reports import write_retraining_report
from sentinelml.retraining.repository import RetrainingRepository
from sentinelml.retraining.trainer import RetrainingTrainer
from sentinelml.retraining.triggers import (
    evaluate_trigger,
    thresholds_from_lifecycle_report,
)
from sentinelml.serving.database import PredictionDatabase


class RetrainingService:
    def __init__(
        self,
        *,
        config: RetrainingConfig | None = None,
        repository: RetrainingRepository | None = None,
        lifecycle_service: LifecycleService | None = None,
        production_service: ProductionDataService | None = None,
        trainer: RetrainingTrainer | None = None,
    ) -> None:
        self.config = config or load_retraining_config()
        self.repository = repository or RetrainingRepository(self.config)
        self.lifecycle_service = lifecycle_service or LifecycleService(
            config=load_lifecycle_config(self.config.lifecycle_config_path)
        )
        self.production_service = production_service or self._production_service()
        self.trainer = trainer or RetrainingTrainer(
            config=self.config,
            lifecycle_service=self.lifecycle_service,
        )

    def _production_service(self) -> ProductionDataService:
        database = PredictionDatabase(
            self.config.database_url,
            connect_timeout_seconds=self.config.database_connect_timeout_seconds,
            statement_timeout_ms=self.config.database_statement_timeout_ms,
        )
        return ProductionDataService(ProductionDataRepository(database))

    def status(self) -> dict[str, Any]:
        self.repository.initialize()
        return {
            "enabled": self.config.enabled,
            "execution_mode": self.config.execution_mode,
            "cooldown": self.repository.cooldown_status(),
            "state": self.repository.current_state(),
            "latest_monitoring_report": str(self.latest_monitoring_report_path()),
        }

    def latest_monitoring_report_path(self) -> Path:
        return self.config.monitoring_reports_dir / "latest.json"

    def load_latest_monitoring_report(self) -> dict[str, Any]:
        path = self.latest_monitoring_report_path()
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def evaluate_latest_trigger(self, *, force_recheck: bool = False) -> dict[str, Any]:
        self.repository.initialize()
        report = self.load_latest_monitoring_report()
        return self.evaluate_trigger_report(report, force_recheck=force_recheck)

    def evaluate_trigger_report(
        self,
        report: dict[str, Any],
        *,
        force_recheck: bool = False,
        consider_execution_state: bool = False,
    ) -> dict[str, Any]:
        monitoring_run_id = str(report.get("monitoring_run_id", ""))
        threshold_report = self._load_threshold_report()
        execution_already_processed = self.repository.already_processed(
            monitoring_run_id
        )
        decision = evaluate_trigger(
            report,
            self.config,
            performance_thresholds=thresholds_from_lifecycle_report(
                threshold_report,
                self.config,
            ),
            already_processed=(
                execution_already_processed
                if consider_execution_state and not force_recheck
                else False
            ),
        )
        decision["execution_already_processed"] = execution_already_processed
        decision["action_taken"] = "evaluate_only"
        return self._write_trigger_decision(decision)

    def process_once(self) -> dict[str, Any]:
        self.repository.initialize()
        report = self.load_latest_monitoring_report()
        monitoring_run_id = str(report.get("monitoring_run_id", ""))
        if self.repository.already_processed(monitoring_run_id):
            decision = self.evaluate_trigger_report(
                report,
                consider_execution_state=True,
            )
            decision["action_taken"] = "skipped"
            return self._write_trigger_decision(decision)

        decision = self.evaluate_trigger_report(report)
        if decision["decision"] in {
            "blocked_monitoring_unhealthy",
            "blocked_monitoring_warming_up",
            "not_triggered",
        }:
            self.repository.mark_processed(
                monitoring_run_id=monitoring_run_id,
                decision=decision["decision"],
                action_taken="skipped",
                report_path=report.get("report_path"),
            )
            decision["action_taken"] = "skipped"
            return self._write_trigger_decision(decision)

        if not self.config.enabled:
            decision.update(
                {
                    "action_taken": "disabled",
                    "reason": "retraining_disabled",
                    "should_retrain": False,
                }
            )
            return self._write_trigger_decision(decision)

        cooldown = self.repository.cooldown_status()
        if cooldown["active"]:
            decision.update(
                {
                    "action_taken": "cooldown",
                    "reason": "cooldown_active",
                    "cooldown": cooldown,
                }
            )
            return self._write_trigger_decision(decision)

        approved = self.production_service.get_approved_production_observations()
        consumed = self.repository.consumed_prediction_ids()
        new_count = len(
            [
                row
                for row in approved
                if str(row.get("prediction_id")) not in consumed
                and row.get("validation_status") == "approved"
            ]
        )
        decision["eligible_new_approved_production_rows"] = new_count
        if new_count < self.config.minimum_approved_production_rows:
            decision.update(
                {
                    "action_taken": "deferred",
                    "reason": "insufficient_new_approved_production_data",
                }
            )
            return self._write_trigger_decision(decision)

        if not self.repository.claim_monitoring_run(
            monitoring_run_id=monitoring_run_id,
            decision=decision["decision"],
            report_path=report.get("report_path"),
        ):
            decision.update(
                {
                    "action_taken": "skipped",
                    "decision": "already_processed",
                    "reason": "monitoring_run_already_processed",
                    "should_retrain": False,
                    "execution_already_processed": True,
                }
            )
            return self._write_trigger_decision(decision)

        with RetrainingLock(self.repository) as lock:
            if not lock.acquired:
                decision.update(
                    {
                        "action_taken": "skipped",
                        "retraining_skipped": True,
                        "reason": "active_retraining_lock",
                    }
                )
                self.repository.update_processed_action(
                    monitoring_run_id=monitoring_run_id,
                    decision=decision["decision"],
                    action_taken="active_retraining_lock",
                    report_path=report.get("report_path"),
                )
                return self._write_trigger_decision(decision)

            return self._run_cycle(
                report=report,
                decision=decision,
                approved_observations=approved,
                consumed_prediction_ids=consumed,
            )

    def watch(self) -> None:
        while True:
            try:
                self.process_once()
            except Exception as exc:
                payload = {
                    "action_taken": "watch_error",
                    "error": str(exc),
                    "retraining_state": self.repository.current_state(),
                }
                write_retraining_report(
                    payload,
                    self.config.retraining_reports_dir,
                    filename="watch_error_latest.json",
                )
            time.sleep(self.config.poll_interval_seconds)

    def _run_cycle(
        self,
        *,
        report: dict[str, Any],
        decision: dict[str, Any],
        approved_observations: list[dict[str, Any]],
        consumed_prediction_ids: set[str],
    ) -> dict[str, Any]:
        retraining_run_id = str(uuid4())
        monitoring_run_id = str(report["monitoring_run_id"])
        run_dir = self.config.retraining_reports_dir / retraining_run_id
        self.repository.create_run(
            retraining_run_id=retraining_run_id,
            monitoring_run_id=monitoring_run_id,
            trigger_reasons=list(decision.get("trigger_reasons", [])),
            drift_share=decision.get("drift_share"),
            performance_metrics=decision.get("performance_metrics", {}),
        )
        used_prediction_ids: list[str] = []
        try:
            self.repository.update_run(retraining_run_id, status="training")
            dataset = build_retraining_dataset(
                retraining_run_id=retraining_run_id,
                config=self.config,
                approved_observations=approved_observations,
                consumed_prediction_ids=consumed_prediction_ids,
                output_dir=run_dir,
            )
            used_prediction_ids = dataset.consumed_prediction_ids
            self.repository.update_run(
                retraining_run_id,
                historical_rows=dataset.manifest["historical"][
                    "row_count_after_sampling"
                ],
                production_rows=dataset.manifest["production"][
                    "row_count_after_sampling"
                ],
                deduplicated_rows=dataset.manifest["deduplication"][
                    "cross_source_duplicates_removed"
                ],
                dataset_fingerprint=dataset.manifest["dataset_fingerprint"],
            )
            train_result = self.trainer.train_candidate(
                retraining_run_id=retraining_run_id,
                monitoring_run_id=monitoring_run_id,
                trigger_decision=decision,
                dataset=dataset,
                output_dir=run_dir,
            )
            self.repository.update_run(
                retraining_run_id,
                status="candidate_created",
                mlflow_run_id=train_result["mlflow_run_id"],
            )
            lifecycle_result = self._register_evaluate_promote(
                train_result["manifest_path"],
                retraining_run_id=retraining_run_id,
                monitoring_run_id=monitoring_run_id,
            )
            final_status = _status_from_lifecycle(lifecycle_result)
            cooldown_until = self.repository.finish_run(
                retraining_run_id,
                status=final_status,
            )
            self.repository.mark_observations_consumed(
                retraining_run_id=retraining_run_id,
                prediction_ids=used_prediction_ids,
            )
            self.repository.update_processed_action(
                monitoring_run_id=monitoring_run_id,
                decision=decision["decision"],
                action_taken=final_status,
                report_path=report.get("report_path"),
            )
            payload = {
                "retraining_run_id": retraining_run_id,
                "monitoring_run_id": monitoring_run_id,
                "trigger_decision": decision,
                "dataset_manifest": str(dataset.manifest_path),
                "candidate_manifest": str(train_result["manifest_path"]),
                "lifecycle": lifecycle_result,
                "status": final_status,
                "cooldown_until": cooldown_until,
            }
            write_retraining_report(
                payload,
                run_dir,
                filename="retraining_manifest.json",
            )
            write_retraining_report(
                payload,
                self.config.retraining_reports_dir,
                filename=f"{retraining_run_id}.json",
                latest_name="latest.json",
            )
            return payload
        except Exception as exc:
            cooldown_until = self.repository.finish_run(
                retraining_run_id,
                status="failed",
                error=str(exc),
            )
            self.repository.update_processed_action(
                monitoring_run_id=monitoring_run_id,
                decision=decision["decision"],
                action_taken="failed",
                report_path=report.get("report_path"),
            )
            payload = {
                "retraining_run_id": retraining_run_id,
                "monitoring_run_id": monitoring_run_id,
                "status": "failed",
                "error": str(exc),
                "cooldown_until": cooldown_until,
            }
            write_retraining_report(
                payload,
                self.config.retraining_reports_dir,
                filename=f"{retraining_run_id}.json",
                latest_name="latest.json",
            )
            return payload

    def _register_evaluate_promote(
        self,
        manifest_path: Path,
        *,
        retraining_run_id: str,
        monitoring_run_id: str,
    ) -> dict[str, Any]:
        if not self.config.auto_register:
            return {"event": "not_registered", "reason": "auto_register_disabled"}
        self.repository.update_run(retraining_run_id, status="candidate_created")
        registration = self.lifecycle_service.register_candidate(
            mode=self.config.execution_mode,
            force_new_version=True,
            manifest_path=manifest_path,
            extra_tags={
                "source": "continuous_retraining",
                "retraining_run_id": retraining_run_id,
                "trigger_monitoring_run_id": monitoring_run_id,
            },
        )
        version = registration["model_version"]
        self.repository.update_run(
            retraining_run_id,
            registered_model_version=version,
            status="evaluating",
        )
        try:
            if not self.config.auto_evaluate:
                return {"registration": registration, "event": "registered_only"}
            if not self.config.auto_promote:
                evaluation = self.lifecycle_service.evaluate_candidate(version=version)
                self.repository.update_run(
                    retraining_run_id,
                    candidate_evaluation=evaluation,
                    status="evaluated",
                )
                return {"registration": registration, "evaluation": evaluation}
            self.repository.update_run(retraining_run_id, status="promoting")
            outcome = self.lifecycle_service.promote_or_reject(version=version)
            self.repository.update_run(
                retraining_run_id,
                candidate_evaluation=outcome.get("evaluation")
                or outcome.get("original_gate_evaluation"),
                promotion_result=outcome,
            )
            return {"registration": registration, "outcome": outcome}
        except Exception as exc:
            mark_failed = getattr(self.lifecycle_service, "mark_candidate_failed", None)
            if callable(mark_failed):
                mark_failed(
                    version=version,
                    error=str(exc),
                    retraining_run_id=retraining_run_id,
                    monitoring_run_id=monitoring_run_id,
                )
            raise

    def _write_trigger_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        run_id = decision.get("monitoring_run_id") or "unknown"
        write_retraining_report(
            decision,
            self.config.retraining_reports_dir / "decisions",
            filename=f"{run_id}.json",
        )
        write_retraining_report(
            decision,
            self.config.retraining_reports_dir,
            filename="latest_decision.json",
        )
        return decision

    def _load_threshold_report(self) -> dict[str, Any] | None:
        path = self.lifecycle_service.config["paths"].get("threshold_report")
        if not path:
            return None
        resolved = Path(path)
        if not resolved.is_absolute():
            from sentinelml.data.config import resolve_project_path

            resolved = resolve_project_path(path)
        if not resolved.exists():
            return None
        return load_json(resolved)


def _status_from_lifecycle(result: dict[str, Any]) -> str:
    outcome = result.get("outcome") or result
    event = outcome.get("event")
    if event == "promoted":
        return "promoted"
    if event == "rejected":
        return "rejected"
    if (
        event == "promotion_pending"
        or outcome.get("operation_state") == "promotion_pending"
    ):
        return "promotion_pending"
    return "candidate_created"
