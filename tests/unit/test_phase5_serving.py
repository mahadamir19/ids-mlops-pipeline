from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.serving.config import ServingConfig
from sentinelml.serving.database import PredictionDatabase, PredictionRecord
from sentinelml.serving.model_manager import ModelLoadError, ModelManager
from sentinelml.serving.prediction_service import (
    PredictionService,
    attack_probabilities,
)
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.rejection_logging import StructuredRejectionLogger
from sentinelml.serving.repository import PredictionRepository
from sentinelml.serving.validation import (
    FeatureValidationError,
    load_serving_feature_schema,
    validate_feature_batch,
    validate_feature_record,
)


def write_schema(path: Path, features: list[str] | None = None) -> list[str]:
    feature_columns = features or ["a", "b", "c"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-14T00:00:00+00:00",
                "feature_columns": feature_columns,
                "target_column": "target",
                "excluded_from_model_features": ["target"],
            }
        ),
        encoding="utf-8",
    )
    return feature_columns


def serving_config(
    root: Path,
    schema_path: Path,
    max_batch_size: int = 2,
) -> ServingConfig:
    return ServingConfig(
        model_name="sentinelml-ids",
        champion_alias="champion",
        host="127.0.0.1",
        port=8000,
        max_batch_size=max_batch_size,
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
        config_path=root / "serving_config.yaml",
    )


class FakeProbabilityModel:
    n_features_in_ = 3

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = probabilities or [0.8]
        self.calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        values = self.probabilities
        if len(values) != len(features):
            values = [values[0]] * len(features)
        return np.asarray([[1.0 - value, value] for value in values], dtype=float)


class FakeVersion:
    def __init__(
        self,
        version: str,
        *,
        source: str = "runs:/fake/model",
        tags: dict[str, str] | None = None,
    ) -> None:
        self.name = "sentinelml-ids"
        self.version = version
        self.source = source
        self.run_id = f"run-{version}"
        self.tags = tags or {
            "lifecycle_state": "champion",
            "model_family": "xgboost",
            "execution_mode": "smoke",
            "demo_model": "true",
            "source_model_uri": source,
            "source_run_id": f"run-{version}",
        }
        self.current_stage = None


class FakeClient:
    def __init__(self, champion: FakeVersion | None) -> None:
        self.champion = champion
        self.alias_calls = 0

    def get_model_version_by_alias(self, _name: str, _alias: str) -> FakeVersion:
        self.alias_calls += 1
        if self.champion is None:
            raise Exception("alias champion not found")
        return self.champion


class FakeMLflow:
    def __init__(self, model: object) -> None:
        self.model = model
        self.loaded_uris: list[str] = []
        self.xgboost = types.SimpleNamespace(load_model=self.load_model)

    def set_tracking_uri(self, _uri: str) -> None:
        return None

    def MlflowClient(self) -> FakeClient:
        raise AssertionError("client should be injected in tests")

    def load_model(self, uri: str) -> object:
        self.loaded_uris.append(uri)
        return self.model


class FakeFailingDatabase:
    configured = True

    def __init__(self) -> None:
        self.records: list[object] = []
        self.fail = True

    def initialize(self) -> None:
        return None

    def check(self) -> bool:
        if self.fail:
            raise RuntimeError("db down")
        return True

    def insert_prediction(self, record: object) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.records.append(record)


class FakeHealthDatabase:
    configured = True

    def __init__(self) -> None:
        self.fail = False

    def initialize(self) -> None:
        return None

    def check(self) -> bool:
        if self.fail:
            raise RuntimeError("db unavailable")
        return True

    def insert_prediction(self, record: object) -> None:
        return None


class Phase5ServingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.schema_path = self.root / "feature_schema.json"
        self.features = write_schema(self.schema_path)
        self.config = serving_config(self.root, self.schema_path)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_feature_schema_loading_and_ordering(self) -> None:
        schema = load_serving_feature_schema(self.schema_path)
        ordered = validate_feature_record({"c": 3, "a": 1, "b": 2}, schema)

        self.assertEqual(list(ordered), ["a", "b", "c"])
        self.assertEqual(schema.feature_count, 3)
        self.assertEqual(len(schema.fingerprint), 64)

    def test_valid_request_and_strict_rejections(self) -> None:
        schema = load_serving_feature_schema(self.schema_path)
        valid = validate_feature_record({"a": 1, "b": 2.0, "c": 3}, schema)
        self.assertEqual(valid["b"], 2.0)

        cases = [
            ({"a": 1, "b": 2}, "schema_mismatch"),
            ({"a": 1, "b": 2, "c": 3, "d": 4}, "schema_mismatch"),
            ({"a": 1, "b": "2", "c": 3}, "invalid_value"),
            ({"a": 1, "b": True, "c": 3}, "invalid_value"),
            ({"a": 1, "b": float("nan"), "c": 3}, "invalid_value"),
            ({"a": 1, "b": float("inf"), "c": 3}, "invalid_value"),
        ]
        for payload, category in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(FeatureValidationError) as context:
                    validate_feature_record(payload, schema)
                self.assertEqual(context.exception.category, category)

    def test_batch_validation_atomic_and_size_limit(self) -> None:
        schema = load_serving_feature_schema(self.schema_path)
        records = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        valid_batch = validate_feature_batch(records, schema, max_batch_size=2)
        self.assertEqual(len(valid_batch), 2)

        with self.assertRaisesRegex(FeatureValidationError, "maximum batch size"):
            validate_feature_batch([*records, records[0]], schema, max_batch_size=2)
        with self.assertRaises(FeatureValidationError) as context:
            validate_feature_batch([records[0], {"a": 1}], schema, max_batch_size=2)
        self.assertEqual(context.exception.category, "batch_validation_failed")
        self.assertEqual(context.exception.details["invalid_rows"][0]["index"], 1)

    def test_probability_extraction_and_prediction_response_schema(self) -> None:
        model = FakeProbabilityModel([0.2, 0.8])
        frame = pd.DataFrame([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])

        probabilities = attack_probabilities(model, frame)

        self.assertEqual(probabilities.tolist(), [0.2, 0.8])

    def test_model_manager_champion_resolution_and_no_champion_behavior(self) -> None:
        manager = ModelManager(
            self.config,
            mlflow_module=FakeMLflow(FakeProbabilityModel()),
            client=FakeClient(FakeVersion("2")),
        )

        loaded = manager.load_startup()

        self.assertEqual(loaded.model_version, "2")
        self.assertEqual(loaded.execution_mode, "smoke")
        self.assertTrue(loaded.demo_model)

        no_champion = ModelManager(
            self.config,
            mlflow_module=FakeMLflow(FakeProbabilityModel()),
            client=FakeClient(None),
        )
        with self.assertRaises(ModelLoadError):
            no_champion.load_startup()

    def test_prediction_path_does_not_query_mlflow_per_request(self) -> None:
        client = FakeClient(FakeVersion("2"))
        manager = ModelManager(
            self.config,
            mlflow_module=FakeMLflow(FakeProbabilityModel()),
            client=client,
        )
        manager.load_startup()
        database = PredictionDatabase(f"sqlite:///{self.root / 'predictions.db'}")
        repository = PredictionRepository(
            database,
            DurableJsonlQueue(self.root / "queue.jsonl"),
        )
        repository.initialize()
        service = PredictionService(manager, repository)

        first = service.predict_one({"a": 1, "b": 2, "c": 3})
        second = service.predict_one({"a": 1, "b": 2, "c": 3})

        self.assertEqual(first["prediction"], "ATTACK")
        self.assertEqual(second["model_version"], "2")
        self.assertEqual(client.alias_calls, 1)
        database.engine.dispose()

    def test_successful_reload_and_failed_reload_retains_previous_model(self) -> None:
        client = FakeClient(FakeVersion("2"))
        manager = ModelManager(
            self.config,
            mlflow_module=FakeMLflow(FakeProbabilityModel()),
            client=client,
        )
        manager.load_startup()
        client.champion = FakeVersion("3", source="runs:/new/model")

        result = manager.reload_champion()

        self.assertTrue(result["reloaded"])
        self.assertEqual(manager.current().model_version, "3")

        def failing_loader(_uri: str) -> object:
            raise RuntimeError("load failed")

        manager.model_loader = failing_loader
        client.champion = FakeVersion("4", source="runs:/bad/model")
        with self.assertRaises(RuntimeError):
            manager.reload_champion()
        self.assertEqual(manager.current().model_version, "3")

    def test_prediction_logging_success_and_idempotency(self) -> None:
        database = PredictionDatabase(f"sqlite:///{self.root / 'predictions.db'}")
        database.initialize()
        record = prediction_record("p1")

        database.insert_prediction(record)
        database.insert_prediction(record)

        with database.engine.connect() as connection:
            count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM predictions"
            ).scalar()
        self.assertEqual(count, 1)
        database.engine.dispose()

    def test_db_failure_queues_and_flushes_later(self) -> None:
        failing_db = FakeFailingDatabase()
        queue = DurableJsonlQueue(self.root / "queue.jsonl")
        repository = PredictionRepository(failing_db, queue)

        persisted = repository.persist_prediction(prediction_record("p1"))

        self.assertFalse(persisted)
        self.assertEqual(queue.depth(), 1)

        failing_db.fail = False
        result = repository.flush_queue()

        self.assertEqual(result.persisted, 1)
        self.assertEqual(queue.depth(), 0)
        self.assertEqual(len(failing_db.records), 1)

    def test_database_status_reports_healthy_and_unhealthy(self) -> None:
        database = FakeHealthDatabase()
        repository = PredictionRepository(
            database,
            DurableJsonlQueue(self.root / "queue.jsonl"),
        )

        self.assertEqual(repository.database_status(), "healthy")

        database.fail = True
        self.assertEqual(repository.database_status(), "unhealthy")

    def test_docker_api_uses_compose_service_hostnames(self) -> None:
        compose_path = Path.cwd() / "docker" / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        api_env = compose["services"]["api"]["environment"]
        mlflow_command = compose["services"]["mlflow-server"]["command"]

        self.assertEqual(api_env["MLFLOW_TRACKING_URI"], "http://mlflow-server:5000")
        self.assertIn("@app-postgres:5432/", api_env["SENTINELML_DATABASE_URL"])
        self.assertNotIn("127.0.0.1:5434", api_env["SENTINELML_DATABASE_URL"])
        self.assertIn("--allowed-hosts", mlflow_command)
        self.assertIn("healthcheck", compose["services"]["app-postgres"])
        self.assertEqual(
            compose["services"]["api"]["depends_on"]["app-postgres"]["condition"],
            "service_healthy",
        )

    def test_durable_queue_duplicate_and_malformed_handling(self) -> None:
        queue = DurableJsonlQueue(self.root / "queue.jsonl")
        record = prediction_record("p1").to_jsonable()

        self.assertTrue(queue.append(record))
        self.assertFalse(queue.append(record))
        with queue.path.open("a", encoding="utf-8") as handle:
            handle.write("{not-json\n")

        snapshot = queue.snapshot()
        archived = queue.archive_malformed(snapshot.malformed_lines)
        queue.replace_records(snapshot.records)

        self.assertEqual(len(snapshot.records), 1)
        self.assertEqual(len(snapshot.malformed_lines), 1)
        self.assertIsNotNone(archived)
        self.assertTrue(archived.exists())

    def test_structured_rejection_logging_omits_payload(self) -> None:
        logger = StructuredRejectionLogger(self.root / "rejections.jsonl")
        logger.log(
            endpoint="/predict",
            request_id="request-1",
            schema_fingerprint="abc",
            error={
                "category": "schema_mismatch",
                "missing_fields": ["a"],
                "unexpected_fields": ["secret"],
            },
        )

        event = json.loads((self.root / "rejections.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(event["endpoint"], "/predict")
        self.assertEqual(event["missing_fields"], ["a"])
        self.assertNotIn("payload", event)


def prediction_record(prediction_id: str) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        timestamp="2026-08-14T00:00:00+00:00",
        model_name="sentinelml-ids",
        model_version="2",
        model_family="xgboost",
        execution_mode="smoke",
        demo_model=True,
        features={"a": 1.0, "b": 2.0, "c": 3.0},
        prediction="ATTACK",
        probability=0.8,
        latency_ms=1.0,
    )


if __name__ == "__main__":
    unittest.main()
