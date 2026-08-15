from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.lifecycle.service import LifecycleError
from sentinelml.resilience.config import ResilienceConfig, SevereGuardrails
from sentinelml.resilience.health import monitoring_heartbeat_status
from sentinelml.resilience.probation import ProbationService
from sentinelml.resilience.repository import ResilienceRepository
from sentinelml.resilience.rollback import RollbackService
from sentinelml.resilience.service import ResilienceService
from sentinelml.retraining.triggers import PerformanceThresholds, evaluate_trigger
from sentinelml.serving.app import create_app
from sentinelml.serving.config import ServingConfig
from sentinelml.serving.database import PredictionDatabase, PredictionRecord
from sentinelml.serving.model_manager import ModelManager
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.repository import PredictionRepository
from sentinelml.serving.validation import load_serving_feature_schema

FEATURES = ["a", "b", "c"]


class Phase9ResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_path = self.root / "feature_schema.json"
        write_schema(self.schema_path)
        self.config = resilience_config(self.root)
        self.engine = create_engine(f"sqlite:///{self.root / 'state.db'}")
        self.repository = ResilienceRepository(self.config, engine=self.engine)
        self.repository.initialize()
        self.prediction_db = PredictionDatabase(None, engine=self.engine)
        self.prediction_db.initialize()

    def tearDown(self) -> None:
        self.engine.dispose()
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_promotion_starts_probation_with_previous_champion(self) -> None:
        service = ProbationService(config=self.config, repository=self.repository)

        result = service.start_probation(
            promoted_version="9",
            previous_champion_version="2",
            promotion_timestamp="2026-08-15T00:00:00+00:00",
        )

        self.assertTrue(result["started"])
        probation = self.repository.latest_probation()
        self.assertEqual(probation["promoted_version"], "9")
        self.assertEqual(probation["previous_champion_version"], "2")
        self.assertEqual(probation["status"], "active")

    def test_probation_uses_only_new_model_window_approved_labels(self) -> None:
        service = ProbationService(config=self.config, repository=self.repository)
        started = datetime.now(UTC) - timedelta(seconds=20)
        result = service.start_probation(
            promoted_version="9",
            previous_champion_version="2",
            promotion_timestamp=started.isoformat(),
        )
        probation = result["probation"]
        self._record_prediction("old", "9", started - timedelta(seconds=1), 1)
        self._record_prediction("wrong-model", "2", started + timedelta(seconds=1), 1)
        self._record_prediction("unlabelled", "9", started + timedelta(seconds=2), None)
        self._record_prediction("new-attack", "9", started + timedelta(seconds=3), 1)
        self._record_prediction(
            "new-benign",
            "9",
            started + timedelta(seconds=4),
            0,
            "BENIGN",
        )

        evaluation = service.evaluate(probation)

        self.assertEqual(evaluation.support["labelled_rows"], 2)
        self.assertEqual(evaluation.support["true_attack"], 1)
        self.assertEqual(evaluation.support["true_benign"], 1)
        self.assertEqual(evaluation.status, "active")

    def test_insufficient_labels_wait_then_expire_inconclusive(self) -> None:
        service = ProbationService(config=self.config, repository=self.repository)
        recent = service.start_probation(
            promoted_version="9",
            previous_champion_version="2",
        )["probation"]

        self.assertEqual(service.evaluate(recent).status, "awaiting_evidence")

        expired = service.start_probation(
            promoted_version="10",
            previous_champion_version="9",
            promotion_timestamp=(
                datetime.now(UTC) - timedelta(seconds=999)
            ).isoformat(),
        )["probation"]

        self.assertEqual(service.evaluate(expired).status, "inconclusive")

    def test_severe_guardrails_trigger_automatic_rollback_once(self) -> None:
        lifecycle = FakeLifecycleService(current="9", previous="2")
        service = ResilienceService(
            config=self.config,
            repository=self.repository,
            lifecycle_service=lifecycle,
        )
        started = datetime.now(UTC) - timedelta(seconds=20)
        probation = service.start_probation_after_promotion(
            promoted_version="9",
            previous_champion_version="2",
            promotion_timestamp=started.isoformat(),
        )["probation"]
        # Low recall, low F1, and high false-positive rate with enough support.
        self._record_prediction("fn1", "9", started + timedelta(seconds=1), 1, "BENIGN")
        self._record_prediction("fn2", "9", started + timedelta(seconds=2), 1, "BENIGN")
        self._record_prediction("fp1", "9", started + timedelta(seconds=3), 0, "ATTACK")
        self._record_prediction("fp2", "9", started + timedelta(seconds=4), 0, "ATTACK")

        first = service.evaluate_probation()
        second = service.evaluate_probation()

        self.assertEqual(first[0]["status"], "rollback_triggered")
        self.assertIn("rollback", first[0])
        self.assertEqual(lifecycle.rollback_calls, ["2"])
        self.assertEqual(second, [])
        updated = self.repository.latest_probation()
        self.assertEqual(updated["probation_id"], probation["probation_id"])
        self.assertEqual(updated["status"], "rolled_back")

    def test_manual_rollback_audits_lifecycle_result(self) -> None:
        lifecycle = FakeLifecycleService(current="9", previous="2")
        service = RollbackService(
            config=self.config,
            repository=self.repository,
            lifecycle_service=lifecycle,
        )

        result = service.rollback(
            to_version="2",
            source="manual",
            reason="operator requested",
        )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(lifecycle.rollback_calls, ["2"])
        self.assertEqual(self.repository.latest_rollback()["source"], "manual")

    def test_rejected_failed_and_missing_targets_are_invalid(self) -> None:
        for state in ["rejected", "failed", "candidate"]:
            with self.subTest(state=state):
                lifecycle = FakeLifecycleService(
                    current="9",
                    previous="2",
                    target_state=state,
                )
                service = RollbackService(
                    config=self.config,
                    repository=self.repository,
                    lifecycle_service=lifecycle,
                )
                with self.assertRaisesRegex(LifecycleError, "approved"):
                    service.rollback(to_version="2", source="manual", reason="bad")

        lifecycle = FakeLifecycleService(current="9", previous="2", missing=True)
        service = RollbackService(
            config=self.config,
            repository=self.repository,
            lifecycle_service=lifecycle,
        )
        with self.assertRaisesRegex(LifecycleError, "not found"):
            service.rollback(to_version="2", source="manual", reason="missing")

    def test_monitoring_heartbeat_classifies_stale_and_fresh(self) -> None:
        now = datetime(2026, 8, 15, tzinfo=UTC)
        fresh = monitoring_heartbeat_status(
            {"last_check_timestamp": (now - timedelta(seconds=10)).isoformat()},
            maximum_age_seconds=60,
            now=now,
        )
        stale = monitoring_heartbeat_status(
            {"last_check_timestamp": (now - timedelta(seconds=90)).isoformat()},
            maximum_age_seconds=60,
            now=now,
        )

        self.assertFalse(fresh["monitoring_stale"])
        self.assertTrue(stale["monitoring_stale"])

    def test_retraining_trigger_blocks_stale_monitoring(self) -> None:
        retraining = retraining_config(self.root, heartbeat_age=1)
        report = monitoring_report(
            last_check_timestamp=(
                datetime.now(UTC) - timedelta(seconds=60)
            ).isoformat()
        )

        decision = evaluate_trigger(
            report,
            retraining,
            performance_thresholds=PerformanceThresholds(
                min_attack_recall=0.8,
                max_false_positive_rate=0.1,
                min_f1=0.8,
                source="test",
            ),
        )

        self.assertEqual(decision["decision"], "blocked_monitoring_stale")
        self.assertFalse(decision["should_retrain"])

    def test_db_outage_queues_prediction_and_flush_is_idempotent(self) -> None:
        database = ToggleDatabase()
        queue = DurableJsonlQueue(self.root / "queue.jsonl")
        repository = PredictionRepository(database, queue)
        record = prediction_record("p1")

        self.assertFalse(repository.persist_prediction(record))
        self.assertEqual(queue.depth(), 1)

        database.fail = False
        first = repository.flush_queue()
        second = repository.flush_queue()

        self.assertEqual(first.persisted, 1)
        self.assertEqual(second.persisted, 0)
        self.assertEqual(database.inserted_ids, ["p1"])
        self.assertEqual(queue.depth(), 0)

    def test_ground_truth_db_outage_returns_retryable_503(self) -> None:
        app = create_app(
            serving_config(self.root, self.schema_path),
            model_manager_factory=lambda _config: FakeReadyModelManager(
                self.schema_path
            ),
            repository_factory=lambda _config: FailingGroundTruthRepository(),
        )

        with TestClient(app) as client:
            response = client.post(
                "/ground-truth",
                json={"prediction_id": "p1", "ground_truth": 1},
            )

        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json()["retryable"])

    def test_mlflow_inference_outage_does_not_break_loaded_prediction(self) -> None:
        client = FakeRegistryClient(FakeVersion("2"))
        manager = ModelManager(
            serving_config(self.root, self.schema_path),
            mlflow_module=FakeMLflow(FakeModel()),
            client=client,
        )
        manager.load_startup()
        database = PredictionDatabase(f"sqlite:///{self.root / 'predictions.db'}")
        repository = PredictionRepository(
            database,
            DurableJsonlQueue(self.root / "queue.jsonl"),
        )
        repository.initialize()
        app = create_app(
            serving_config(self.root, self.schema_path),
            model_manager_factory=lambda _config: manager,
            repository_factory=lambda _config: repository,
        )

        with TestClient(app) as test_client:
            client.fail_alias = True
            response = test_client.post(
                "/predict",
                json={"a": 1, "b": 2, "c": 3},
            )
            health = test_client.get("/health").json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_version"], "2")
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["registry_connectivity"], "unavailable")
        database.engine.dispose()

    def test_reload_failure_keeps_previous_loaded_model(self) -> None:
        client = FakeRegistryClient(FakeVersion("2"))
        manager = ModelManager(
            serving_config(self.root, self.schema_path),
            mlflow_module=FakeMLflow(FakeModel()),
            client=client,
        )
        manager.load_startup()
        client.champion = FakeVersion("3", source="runs:/bad/model")
        manager.model_loader = lambda _uri: (_ for _ in ()).throw(
            RuntimeError("load failed")
        )

        result = manager.reload_champion()

        self.assertFalse(result["success"])
        self.assertEqual(result["active_model_version"], "2")
        self.assertEqual(result["registry_champion_version"], "3")
        self.assertEqual(manager.current().model_version, "2")

    def _record_prediction(
        self,
        prediction_id: str,
        model_version: str,
        timestamp: datetime,
        ground_truth: int | None,
        prediction: str = "ATTACK",
    ) -> None:
        self.prediction_db.insert_prediction(
            prediction_record(
                prediction_id,
                model_version=model_version,
                timestamp=timestamp.isoformat(),
                prediction=prediction,
            )
        )
        if ground_truth is not None:
            schema = load_serving_feature_schema(self.schema_path)
            self.prediction_db.record_ground_truth(
                prediction_id=prediction_id,
                ground_truth=ground_truth,
                schema_fingerprint=schema.fingerprint,
                validation_status="approved",
                validation_errors=[],
            )


def resilience_config(root: Path) -> ResilienceConfig:
    return ResilienceConfig(
        poll_interval_seconds=1,
        probation_enabled=True,
        probation_duration_seconds=120,
        probation_minimum_labelled_rows=2,
        probation_minimum_attack_support=1,
        probation_minimum_benign_support=1,
        automatic_rollback_enabled=True,
        severe_guardrails=SevereGuardrails(
            min_attack_recall=0.5,
            min_f1=0.5,
            max_false_positive_rate=0.5,
            threshold_source="test_config",
        ),
        monitoring_maximum_heartbeat_age_seconds=60,
        promotion_retry_enabled=True,
        promotion_retry_interval_seconds=1,
        promotion_retry_max_attempts=2,
        resilience_reports_dir=root / "reports" / "resilience",
        database_url_env="SENTINELML_TEST_DATABASE_URL",
        database_connect_timeout_seconds=1,
        database_statement_timeout_ms=1000,
        config_path=root / "resilience.yaml",
    )


def retraining_config(root: Path, *, heartbeat_age: int | None) -> Any:
    from tests.unit.test_phase8_retraining import config

    cfg = config(root)
    object.__setattr__(cfg, "monitoring_maximum_heartbeat_age_seconds", heartbeat_age)
    return cfg


def monitoring_report(**overrides: Any) -> dict[str, Any]:
    payload = {
        "monitoring_run_id": "mon-1",
        "finished_at": datetime.now(UTC).isoformat(),
        "last_check_timestamp": datetime.now(UTC).isoformat(),
        "status": "healthy",
        "monitoring_health": "healthy",
        "data_drift_detected": True,
        "drift_share": 1.0,
        "performance": {
            "attack_recall": 0.9,
            "f1": 0.9,
            "false_positive_rate": 0.0,
            "support": {"labelled_rows": 10, "true_attack": 5, "true_benign": 5},
        },
    }
    payload.update(overrides)
    return payload


def write_schema(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "feature_columns": FEATURES,
                "target_column": "target",
                "excluded_from_model_features": ["target"],
            }
        ),
        encoding="utf-8",
    )


def serving_config(root: Path, schema_path: Path) -> ServingConfig:
    return ServingConfig(
        model_name="sentinelml-ids",
        champion_alias="champion",
        host="127.0.0.1",
        port=8000,
        max_batch_size=4,
        schema_path=schema_path,
        queue_path=root / "queue.jsonl",
        queue_flush_interval_seconds=0,
        rejection_log_path=root / "rejections.jsonl",
        database_url_env="SENTINELML_TEST_DATABASE_URL",
        database_connect_timeout_seconds=1,
        database_statement_timeout_ms=1000,
        reload_endpoint_enabled=True,
        reload_notification_timeout_seconds=1,
        mlflow_tracking_uri=None,
        config_path=root / "serving.yaml",
    )


def prediction_record(
    prediction_id: str,
    *,
    model_version: str = "9",
    timestamp: str = "2026-08-15T00:00:00+00:00",
    prediction: str = "ATTACK",
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        timestamp=timestamp,
        model_name="sentinelml-ids",
        model_version=model_version,
        model_family="xgboost",
        execution_mode="smoke",
        demo_model=True,
        features={"a": 1.0, "b": 2.0, "c": 3.0},
        prediction=prediction,
        probability=0.8,
        latency_ms=1.0,
    )


class FakeVersion:
    def __init__(
        self,
        version: str,
        *,
        source: str = "runs:/fake/model",
        state: str = "champion",
    ) -> None:
        self.name = "sentinelml-ids"
        self.version = version
        self.source = source
        self.run_id = f"run-{version}"
        self.tags = {
            "lifecycle_state": state,
            "model_family": "xgboost",
            "execution_mode": "smoke",
            "demo_model": "true",
            "source_model_uri": source,
            "source_run_id": f"run-{version}",
            "lifecycle.previous_champion_version": "2",
        }
        self.current_stage = None


class FakeLifecycleService:
    def __init__(
        self,
        *,
        current: str,
        previous: str,
        target_state: str = "superseded",
        missing: bool = False,
    ) -> None:
        self.current = current
        self.previous = previous
        self.target_state = target_state
        self.missing = missing
        self.rollback_calls: list[str] = []

    def get_champion(self) -> Any:
        champion = FakeVersion(self.current, state="champion")
        champion.tags["lifecycle.previous_champion_version"] = self.previous
        return champion

    def rollback(self, *, version: str, **kwargs: Any) -> dict[str, Any]:
        if self.missing:
            raise LifecycleError("model version not found")
        if self.target_state not in {"champion", "superseded"}:
            raise LifecycleError(
                "rollback target must be a current or superseded approved champion; "
                "rejected, failed, and candidate versions are not valid targets"
            )
        self.rollback_calls.append(str(version))
        self.current = str(version)
        return {
            "event": "rollback",
            "status": "succeeded",
            "target_version": str(version),
            "previous_champion_version": "9",
            "source": kwargs.get("source", "manual"),
            "serving_reload_notification": {"attempted": True, "success": True},
        }


class ToggleDatabase:
    configured = True

    def __init__(self) -> None:
        self.fail = True
        self.inserted_ids: list[str] = []

    def initialize(self) -> None:
        return None

    def check(self) -> bool:
        if self.fail:
            raise RuntimeError("db down")
        return True

    def insert_prediction(self, record: Any) -> None:
        if self.fail:
            raise RuntimeError("db down")
        if hasattr(record, "prediction_id"):
            prediction_id = record.prediction_id
        else:
            prediction_id = record["prediction_id"]
        if prediction_id not in self.inserted_ids:
            self.inserted_ids.append(prediction_id)


class FailingGroundTruthRepository:
    queue = types.SimpleNamespace(depth=lambda: 0, malformed_count=lambda: 0)

    def initialize(self) -> None:
        return None

    def database_status(self) -> str:
        return "unhealthy"

    def flush_queue(self) -> Any:
        return types.SimpleNamespace(
            attempted=0,
            persisted=0,
            remaining=0,
            malformed=0,
            database_logging="unhealthy",
        )

    def record_ground_truth(self, **_kwargs: Any) -> Any:
        raise RuntimeError("db down")


class FakeReadyModelManager:
    def __init__(self, schema_path: Path) -> None:
        self.loaded = types.SimpleNamespace(
            model_name="sentinelml-ids",
            model_version="2",
            model_family="xgboost",
            execution_mode="smoke",
            demo_model=True,
            source_run_id="run-2",
            source_model_uri="runs:/run-2/model",
            loaded_at="2026-08-15T00:00:00+00:00",
            feature_schema=load_serving_feature_schema(schema_path),
        )

    def load_startup(self) -> Any:
        return self.loaded

    def current(self) -> Any:
        return self.loaded

    def is_ready(self) -> bool:
        return True


class FakeRegistryClient:
    def __init__(self, champion: FakeVersion) -> None:
        self.champion = champion
        self.fail_alias = False

    def get_model_version_by_alias(self, _name: str, _alias: str) -> FakeVersion:
        if self.fail_alias:
            raise RuntimeError("MLflow unavailable")
        return self.champion


class FakeMLflow:
    def __init__(self, model: Any) -> None:
        self.xgboost = types.SimpleNamespace(load_model=lambda _uri: model)

    def set_tracking_uri(self, _uri: str) -> None:
        return None

    def MlflowClient(self) -> Any:
        raise AssertionError("client should be injected")


class FakeModel:
    n_features_in_ = 3

    def predict_proba(self, _features: Any) -> Any:
        return [[0.2, 0.8]]


if __name__ == "__main__":
    unittest.main()
