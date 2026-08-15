from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.production_data.repository import ProductionDataRepository
from sentinelml.serving.app import create_app
from sentinelml.serving.config import ServingConfig
from sentinelml.serving.database import PredictionDatabase, PredictionRecord
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.repository import PredictionRepository
from sentinelml.serving.validation import (
    FeatureSchema,
    load_serving_feature_schema,
)
from sentinelml.simulation.client import ApiError
from sentinelml.simulation.config import SimulationConfig, reject_test_source
from sentinelml.simulation.data import deterministic_sample
from sentinelml.simulation.labels import BatchLabeler, RandomizedDelayLabeler
from sentinelml.simulation.reports import write_simulation_report
from sentinelml.simulation.scenarios import (
    apply_scenario,
    build_scenario,
    predefined_scenario_names,
    scenario_progress,
)
from sentinelml.simulation.simulator import TrafficSimulator

FEATURES = [
    "flow_duration",
    "packet_length_mean",
    "flow_iat_mean",
    "flow_iat_max",
    "average_packet_size",
    "max_packet_length",
    "total_fwd_packets",
]


class Phase6SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_path = self.root / "feature_schema.json"
        write_schema(self.schema_path, FEATURES)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_deterministic_row_selection_and_fingerprints(self) -> None:
        frame = simulation_frame()

        first = deterministic_sample(
            frame,
            feature_columns=FEATURES,
            target_column="target",
            request_count=4,
            seed=7,
        )
        second = deterministic_sample(
            frame,
            feature_columns=FEATURES,
            target_column="target",
            request_count=4,
            seed=7,
        )
        different = deterministic_sample(
            frame,
            feature_columns=FEATURES,
            target_column="target",
            request_count=4,
            seed=8,
        )

        self.assertEqual(
            [record.row_fingerprint for record in first],
            [record.row_fingerprint for record in second],
        )
        self.assertNotEqual(
            [record.row_fingerprint for record in first],
            [record.row_fingerprint for record in different],
        )

    def test_normal_gradual_and_sudden_scenarios(self) -> None:
        normal = build_scenario(
            "normal",
            feature_columns=FEATURES,
            configured={"normal": {}},
        )
        unchanged, normal_meta = apply_scenario(
            base_features(),
            definition=normal,
            index=1,
            total=3,
        )
        self.assertEqual(unchanged, base_features())
        self.assertFalse(normal_meta["applied"])

        gradual = build_scenario(
            "gradual_drift",
            feature_columns=FEATURES,
            configured={
                "gradual_drift": {
                    "affected_features": ["flow_duration"],
                    "additive_shift": 10.0,
                    "multiplicative_scale": 2.0,
                }
            },
        )
        middle, _ = apply_scenario(
            base_features(),
            definition=gradual,
            index=1,
            total=3,
        )
        self.assertEqual(middle["flow_duration"], 155.0)

        sudden = build_scenario(
            "sudden_drift",
            feature_columns=FEATURES,
            configured={
                "sudden_drift": {
                    "affected_features": ["flow_duration"],
                    "additive_shift": 10.0,
                    "change_point_fraction": 0.5,
                }
            },
        )
        self.assertEqual(
            scenario_progress(
                schedule="sudden",
                index=0,
                total=4,
                change_point_fraction=0.5,
            ),
            0.0,
        )
        shifted, _ = apply_scenario(
            base_features(),
            definition=sudden,
            index=2,
            total=4,
        )
        self.assertEqual(shifted["flow_duration"], 110.0)

    def test_attack_rate_spike_changes_prevalence_not_features(self) -> None:
        scenario = build_scenario(
            "attack_rate_spike",
            feature_columns=FEATURES,
            configured={"attack_rate_spike": {"target_attack_fraction": 0.75}},
        )
        sampled = deterministic_sample(
            simulation_frame(),
            feature_columns=FEATURES,
            target_column="target",
            request_count=4,
            seed=3,
            attack_fraction=scenario.target_attack_fraction,
        )
        transformed, _ = apply_scenario(
            sampled[0].features,
            definition=scenario,
            index=0,
            total=4,
        )

        self.assertEqual(sum(record.true_label for record in sampled), 3)
        self.assertEqual(transformed, sampled[0].features)
        self.assertEqual(scenario.category, "class_prevalence_change")

    def test_predefined_and_custom_scenario_validation(self) -> None:
        self.assertIn("packet_size_shift", predefined_scenario_names())
        scenario = build_scenario(
            "packet_size_shift",
            feature_columns=FEATURES,
            configured={"packet_size_shift": {}},
        )
        self.assertEqual(scenario.category, "feature_drift")
        with self.assertRaisesRegex(ValueError, "unknown scenario features"):
            build_scenario(
                "custom",
                feature_columns=FEATURES,
                configured={},
                custom={"affected_features": ["missing"], "additive_shift": 1},
            )
        with self.assertRaisesRegex(ValueError, "requires a transform"):
            build_scenario(
                "custom",
                feature_columns=FEATURES,
                configured={},
                custom={},
            )

    def test_test_source_exclusion_and_report_serialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "test.parquet"):
            reject_test_source(Path("data/processed/test.parquet"))

        report_path = write_simulation_report(
            {"simulation_run_id": "run-1", "requests_succeeded": 1},
            self.root / "reports",
        )

        self.assertTrue(report_path.exists())
        self.assertTrue((self.root / "reports" / "latest.json").exists())

    def test_seeded_randomized_delay_and_retryable_failure(self) -> None:
        first_client = FakeGroundTruthClient(failures=1, retryable=True)
        first = RandomizedDelayLabeler(
            first_client,
            seed=99,
            min_delay_seconds=1,
            max_delay_seconds=2,
            max_attempts=2,
            retry_delay_seconds=3,
        )
        second = RandomizedDelayLabeler(
            FakeGroundTruthClient(),
            seed=99,
            min_delay_seconds=1,
            max_delay_seconds=2,
            max_attempts=2,
            retry_delay_seconds=3,
        )

        first.schedule(prediction_id="p1", ground_truth=1, now=10)
        second.schedule(prediction_id="p1", ground_truth=1, now=10)
        self.assertEqual(first.pending[0].due_at, second.pending[0].due_at)

        first.deliver_due(now=20)
        self.assertEqual(len(first.pending), 1)
        self.assertEqual(first.pending[0].attempts, 1)
        first.deliver_due(now=30)
        self.assertEqual(len(first.pending), 0)
        self.assertEqual(len(first.delivered), 1)

    def test_batch_label_delivery(self) -> None:
        client = FakeGroundTruthClient()
        labeler = BatchLabeler(
            client,
            batch_size=2,
            max_attempts=2,
            retry_delay_seconds=1,
        )

        labeler.schedule(prediction_id="p1", ground_truth=0, now=1)
        self.assertEqual(client.batch_calls, 0)
        labeler.schedule(prediction_id="p2", ground_truth=1, now=1)

        self.assertEqual(client.batch_calls, 1)
        self.assertEqual(len(labeler.pending), 0)

    def test_simulator_uses_http_client_and_keeps_label_association(self) -> None:
        source_path = self.root / "reference.parquet"
        simulation_frame().to_parquet(source_path, index=False)
        config = SimulationConfig(
            api_base_url="http://127.0.0.1:8000",
            default_seed=42,
            default_request_count=2,
            request_interval_seconds=0,
            timeout_seconds=1,
            source_data_path=source_path,
            feature_schema_path=self.schema_path,
            reports_dir=self.root / "reports",
            randomized_delay={"min_delay_seconds": 0, "max_delay_seconds": 0},
            batch_delivery={"batch_size": 2},
            retry={"max_attempts": 1, "retry_delay_seconds": 0},
            scenarios={"normal": {}},
            custom={},
            config_path=self.root / "simulation_config.yaml",
        )
        client = FakeSimulationClient()

        report = TrafficSimulator(config, client=client).run(
            scenario_name="normal",
            request_count=2,
            seed=5,
            label_mode="randomized",
            write_report=False,
            wait_for_labels=True,
        )

        self.assertEqual(client.predict_calls, 2)
        self.assertEqual(len(client.labels), 2)
        self.assertEqual(report["labels_delivered"], 2)
        self.assertEqual(report["requests_succeeded"], 2)


class Phase6GroundTruthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema_path = self.root / "feature_schema.json"
        write_schema(self.schema_path, ["a", "b"])
        self.schema = load_serving_feature_schema(self.schema_path)
        self.config = serving_config(self.root, self.schema_path)
        self.database = PredictionDatabase(f"sqlite:///{self.root / 'predictions.db'}")
        self.repository = PredictionRepository(
            self.database,
            DurableJsonlQueue(self.root / "queue.jsonl"),
        )
        self.app = create_app(
            self.config,
            model_manager_factory=lambda _config: FakeManager(self.schema),
            repository_factory=lambda _config: self.repository,
        )

    def tearDown(self) -> None:
        if self.database.engine is not None:
            self.database.engine.dispose()
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_single_ground_truth_endpoint_idempotency_conflict_and_unknown(
        self,
    ) -> None:
        with TestClient(self.app) as client:
            self.database.insert_prediction(prediction_record("p1"))

            first = client.post(
                "/ground-truth",
                json={"prediction_id": "p1", "ground_truth": 1},
            )
            duplicate = client.post(
                "/ground-truth",
                json={"prediction_id": "p1", "ground_truth": 1},
            )
            conflict = client.post(
                "/ground-truth",
                json={"prediction_id": "p1", "ground_truth": 0},
            )
            unknown = client.post(
                "/ground-truth",
                json={"prediction_id": "missing", "ground_truth": 1},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "recorded")
        self.assertEqual(duplicate.json()["status"], "idempotent")
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(unknown.status_code, 404)
        self.assertTrue(unknown.json()["retryable"])

    def test_batch_ground_truth_endpoint_and_production_approval_boundary(self) -> None:
        with TestClient(self.app) as client:
            self.database.insert_prediction(prediction_record("p1"))
            self.database.insert_prediction(prediction_record("p2"))
            response = client.post(
                "/ground-truth/batch",
                json=[
                    {"prediction_id": "p1", "ground_truth": 1},
                    {"prediction_id": "p2", "ground_truth": 0},
                ],
            )

        approved = ProductionDataRepository(
            self.database
        ).get_approved_production_observations()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["atomic"])
        self.assertEqual(len(approved), 2)
        json.dumps({"observations": approved})

    def test_rejected_malformed_production_observation_is_not_eligible(self) -> None:
        self.repository.initialize()
        self.database.insert_prediction(
            prediction_record("bad", features={"a": 1.0})
        )

        result = self.repository.record_ground_truth(
            prediction_id="bad",
            ground_truth=1,
            feature_schema=self.schema,
        )
        approved = ProductionDataRepository(
            self.database
        ).get_approved_production_observations()

        self.assertEqual(result.production_observation_status, "rejected")
        self.assertEqual(approved, [])


def write_schema(path: Path, features: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-15T00:00:00+00:00",
                "feature_columns": features,
                "target_column": "target",
                "excluded_from_model_features": ["target"],
            }
        ),
        encoding="utf-8",
    )


def simulation_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(8):
        row = {feature: float((index + 1) * 10) for feature in FEATURES}
        row["target"] = 0 if index < 4 else 1
        row["row_digest"] = f"row-{index}"
        rows.append(row)
    return pd.DataFrame(rows)


def base_features() -> dict[str, float]:
    return {
        "flow_duration": 100.0,
        "packet_length_mean": 20.0,
        "flow_iat_mean": 5.0,
        "flow_iat_max": 10.0,
        "average_packet_size": 25.0,
        "max_packet_length": 50.0,
        "total_fwd_packets": 2.0,
    }


def serving_config(root: Path, schema_path: Path) -> ServingConfig:
    return ServingConfig(
        model_name="sentinelml-ids",
        champion_alias="champion",
        host="127.0.0.1",
        port=8000,
        max_batch_size=10,
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


def prediction_record(
    prediction_id: str,
    *,
    features: dict[str, float] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        timestamp="2026-08-15T00:00:00+00:00",
        model_name="sentinelml-ids",
        model_version="2",
        model_family="xgboost",
        execution_mode="smoke",
        demo_model=True,
        features=features or {"a": 1.0, "b": 2.0},
        prediction="ATTACK",
        probability=0.9,
        latency_ms=1.0,
    )


class FakeManager:
    def __init__(self, schema: FeatureSchema) -> None:
        self.loaded = types.SimpleNamespace(
            model_name="sentinelml-ids",
            model_version="2",
            model_family="xgboost",
            execution_mode="smoke",
            demo_model=True,
            source_run_id="run-2",
            source_model_uri="runs:/fake/model",
            loaded_at="2026-08-15T00:00:00+00:00",
            feature_schema=schema,
        )

    def load_startup(self) -> object:
        return self.loaded

    def is_ready(self) -> bool:
        return True

    def current(self) -> object:
        return self.loaded


class FakeGroundTruthClient:
    def __init__(self, *, failures: int = 0, retryable: bool = True) -> None:
        self.failures = failures
        self.retryable = retryable
        self.batch_calls = 0

    def submit_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
    ) -> dict[str, Any]:
        if self.failures:
            self.failures -= 1
            raise ApiError(None, "temporary", retryable=self.retryable)
        return {"prediction_id": prediction_id, "ground_truth": ground_truth}

    def submit_ground_truth_batch(
        self,
        labels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.batch_calls += 1
        return {"labels": labels, "count": len(labels)}


class FakeSimulationClient(FakeGroundTruthClient):
    def __init__(self) -> None:
        super().__init__()
        self.predict_calls = 0
        self.labels: list[dict[str, Any]] = []

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        del features
        self.predict_calls += 1
        return {
            "prediction_id": f"p{self.predict_calls}",
            "prediction": "ATTACK",
            "probability": 0.9,
            "model_version": "2",
            "latency_ms": 1.0,
        }

    def submit_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
    ) -> dict[str, Any]:
        label = {"prediction_id": prediction_id, "ground_truth": ground_truth}
        self.labels.append(label)
        return label


if __name__ == "__main__":
    unittest.main()
