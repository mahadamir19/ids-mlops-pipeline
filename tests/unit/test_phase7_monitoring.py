from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.monitoring.config import MonitoringConfig
from sentinelml.monitoring.data import (
    MonitoringDataError,
    load_current_window,
    load_reference_dataset,
    reject_test_reference,
    validate_feature_payload,
)
from sentinelml.monitoring.drift import (
    _run_with_expected_numpy_warning_filter,
    calculate_drift,
    population_stability_index,
    zero_variance_features,
)
from sentinelml.monitoring.fingerprint import monitoring_input_fingerprint
from sentinelml.monitoring.metrics import monitoring_metrics, update_monitoring_metrics
from sentinelml.monitoring.performance import (
    calculate_performance,
    prediction_distribution,
    true_label_distribution,
)
from sentinelml.monitoring.reports import write_monitoring_report
from sentinelml.monitoring.service import run_monitoring_cycle
from sentinelml.serving.database import PredictionDatabase, PredictionRecord
from sentinelml.serving.validation import load_serving_feature_schema


class FakeEvidentlyRunner:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        config: MonitoringConfig,
    ) -> tuple[dict[str, Any], str | None]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("evidently failed")
        return {
            "reference_rows": len(reference),
            "current_rows": len(current),
            "features": list(config.monitored_features),
        }, None


class Phase7MonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        self.databases: list[PredictionDatabase] = []
        self.schema_path = self.root / "feature_schema.json"
        self.features = ["a", "b", "c"]
        self.schema_path.write_text(
            json.dumps(
                {
                    "feature_columns": self.features,
                    "target_column": "target",
                    "excluded_from_model_features": ["target"],
                    "generated_at_utc": "2026-08-15T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        self.reference_path = self.root / "reference.parquet"
        pd.DataFrame(
            {
                "a": range(100),
                "b": [float(value * 2) for value in range(100)],
                "c": [1.0] * 100,
                "target": [0, 1] * 50,
            }
        ).to_parquet(self.reference_path)
        self.database_url = f"sqlite:///{self.root / 'predictions.db'}"
        self.database_env = f"SENTINELML_TEST_DB_{self._testMethodName}".upper()
        os.environ[self.database_env] = self.database_url
        self.config = self._config()

    def tearDown(self) -> None:
        for database in self.databases:
            if database.engine is not None:
                database.engine.dispose()
        os.environ.pop(self.database_env, None)
        if self.root.exists():
            shutil.rmtree(self.root)

    def _config(
        self,
        *,
        window_size: int = 5,
        minimum_window_size: int = 3,
    ) -> MonitoringConfig:
        return MonitoringConfig(
            window_size=window_size,
            minimum_window_size=minimum_window_size,
            interval_seconds=60,
            reference_path=self.reference_path,
            reference_sample_size=10,
            reference_sample_seed=7,
            feature_schema_path=self.schema_path,
            monitored_features=self.features,
            drift_share_threshold=0.34,
            psi_threshold=0.10,
            evidently_method="psi",
            evidently_include_html=False,
            minimum_labelled_rows=2,
            min_attack_support=1,
            min_benign_support=1,
            database_url_env=self.database_env,
            database_connect_timeout_seconds=1,
            database_statement_timeout_ms=1000,
            reports_dir=self.root / "monitoring",
            host="127.0.0.1",
            port=9101,
            config_path=self.root / "monitoring_config.yaml",
        )

    def _seed_predictions(self) -> PredictionDatabase:
        database = PredictionDatabase(self.database_url)
        self.databases.append(database)
        database.initialize()
        for index in range(5):
            database.insert_prediction(
                PredictionRecord(
                    prediction_id=f"p{index}",
                    timestamp=f"2026-08-15T00:00:0{index}+00:00",
                    model_name="sentinelml-ids",
                    model_version="2",
                    model_family="xgboost",
                    execution_mode="smoke",
                    demo_model=True,
                    features={"a": float(index), "b": float(index * 2), "c": 1.0},
                    prediction="ATTACK" if index % 2 else "BENIGN",
                    probability=0.8,
                    latency_ms=1.0,
                    ground_truth=1 if index in {1, 3} else 0,
                    ground_truth_received_at=f"2026-08-15T00:01:0{index}+00:00",
                )
            )
        return database

    def test_reference_path_excludes_test_partition(self) -> None:
        with self.assertRaises(MonitoringDataError):
            reject_test_reference(Path("data/processed/test.parquet"))

    def test_deterministic_reference_sampling(self) -> None:
        schema = load_serving_feature_schema(self.schema_path)
        first = load_reference_dataset(self.config, schema)
        second = load_reference_dataset(self.config, schema)

        self.assertEqual(first.metadata["row_count"], 100)
        self.assertEqual(first.metadata["selected_row_count"], 10)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_current_window_retrieval_uses_latest_timestamp_order(self) -> None:
        database = self._seed_predictions()
        window = load_current_window(
            self.config,
            load_serving_feature_schema(self.schema_path),
        )

        self.assertEqual(
            [row["prediction_id"] for row in window.rows],
            ["p0", "p1", "p2", "p3", "p4"],
        )
        self.assertEqual(window.metadata["actual_window_size"], 5)
        database.engine.dispose()

    def test_insufficient_window_state_does_not_claim_no_drift(self) -> None:
        database = self._seed_predictions()
        config = self._config(window_size=2, minimum_window_size=3)

        report = run_monitoring_cycle(config, evidently_runner=FakeEvidentlyRunner())

        self.assertEqual(report["monitoring_health"], "warming_up")
        self.assertIsNone(report["data_drift_detected"])
        database.engine.dispose()

    def test_schema_mismatch_handling_for_corrupted_features(self) -> None:
        with self.assertRaises(MonitoringDataError):
            validate_feature_payload({"a": 1.0, "b": 2.0}, self.features)

    def test_drift_share_and_boolean_from_config_threshold(self) -> None:
        reference = pd.DataFrame({"a": range(50), "b": range(50), "c": [1.0] * 50})
        current = pd.DataFrame({"a": range(100, 150), "b": range(50), "c": [1.0] * 50})

        result = calculate_drift(
            reference,
            current,
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )

        self.assertIn("a", result.drifting_features)
        self.assertEqual(result.monitored_feature_count, 3)
        self.assertAlmostEqual(result.drift_share, result.drifting_feature_count / 3)
        self.assertEqual(result.data_drift_detected, result.drift_share >= 0.34)

    def test_zero_variance_features_stay_in_drift_monitoring(self) -> None:
        reference = pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0, 1.0],
                "b": [0.0, 1.0, 2.0, 3.0],
                "c": [5.0, 5.0, 5.0, 5.0],
            }
        )
        current = pd.DataFrame(
            {
                "a": [2.0, 2.0, 2.0, 2.0],
                "b": [10.0, 11.0, 12.0, 13.0],
                "c": [5.0, 5.0, 5.0, 5.0],
            }
        )

        result = calculate_drift(
            reference,
            current,
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )

        self.assertEqual(result.monitored_feature_count, 3)
        self.assertEqual(set(result.zero_variance_reference_features), {"a", "c"})
        self.assertEqual(set(result.zero_variance_current_features), {"a", "c"})
        self.assertIn("a", result.feature_scores)
        self.assertIn("c", result.feature_scores)
        self.assertIsInstance(result.drift_share, float)

    def test_zero_variance_detection_reference_current_and_both(self) -> None:
        frame = pd.DataFrame(
            {
                "reference_constant": [1.0, 1.0, 1.0],
                "current_constant": [2.0, 2.0, 2.0],
                "variable": [1.0, 2.0, 3.0],
            }
        )

        self.assertEqual(
            zero_variance_features(
                frame,
                ["reference_constant", "current_constant", "variable"],
            ),
            ["reference_constant", "current_constant"],
        )

    def test_targeted_numpy_warning_suppression_is_scoped(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _run_with_expected_numpy_warning_filter(
                lambda: np.corrcoef(np.ones(3), np.ones(3))
            )

        self.assertFalse(
            [
                item
                for item in captured
                if "invalid value encountered in divide" in str(item.message)
            ]
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warnings.warn("different runtime warning", RuntimeWarning, stacklevel=1)

        self.assertTrue(
            [
                item
                for item in captured
                if "different runtime warning" in str(item.message)
            ]
        )

    def test_normal_fixture_has_lower_drift_than_shifted_fixture(self) -> None:
        reference = pd.Series(range(100), dtype=float)
        normal = pd.Series(range(100), dtype=float)
        shifted = pd.Series(range(1000, 1100), dtype=float)

        self.assertLess(
            population_stability_index(reference, normal),
            population_stability_index(reference, shifted),
        )

    def test_prediction_and_true_label_distribution_metrics(self) -> None:
        rows = [
            {"prediction": "ATTACK", "ground_truth": 1},
            {"prediction": "BENIGN", "ground_truth": 0},
            {"prediction": "ATTACK", "ground_truth": None},
        ]

        self.assertEqual(prediction_distribution(rows)["counts"]["ATTACK"], 2)
        self.assertEqual(true_label_distribution(rows)["counts"]["BENIGN"], 1)

    def test_labelled_performance_calculations(self) -> None:
        rows = [
            {"prediction": "ATTACK", "ground_truth": 1},
            {"prediction": "BENIGN", "ground_truth": 1},
            {"prediction": "ATTACK", "ground_truth": 0},
            {"prediction": "BENIGN", "ground_truth": 0},
        ]

        performance = calculate_performance(rows, self.config)

        self.assertEqual(
            performance["confusion_matrix"],
            {"tp": 1, "fn": 1, "fp": 1, "tn": 1},
        )
        self.assertEqual(performance["attack_recall"], 0.5)
        self.assertEqual(performance["false_positive_rate"], 0.5)
        self.assertEqual(performance["f1"], 0.5)

    def test_insufficient_label_status_and_undefined_support_metrics(self) -> None:
        rows = [{"prediction": "BENIGN", "ground_truth": 0}]

        performance = calculate_performance(rows, self.config)

        self.assertEqual(performance["status"], "insufficient_labels")
        self.assertIsNone(performance["attack_recall"])
        self.assertIsNone(performance["false_positive_rate"])

    def test_monitoring_report_serialization(self) -> None:
        report = {
            "monitoring_run_id": "run-1",
            "finished_at": "2026-08-15T00:00:00+00:00",
            "status": "healthy",
        }

        path = write_monitoring_report(report, self.root / "reports")

        self.assertTrue(path.exists())
        self.assertTrue((self.root / "reports" / "latest.json").exists())

    def test_monitoring_failure_isolation_returns_unhealthy_report(self) -> None:
        config = self._config()
        os.environ.pop(self.database_env)

        report = run_monitoring_cycle(config, evidently_runner=FakeEvidentlyRunner())

        self.assertEqual(report["monitoring_health"], "unhealthy")
        self.assertTrue(report["errors"])

    def test_unchanged_input_skips_full_report_and_evidently(self) -> None:
        database = self._seed_predictions()
        runner = FakeEvidentlyRunner()

        first = run_monitoring_cycle(self.config, evidently_runner=runner)
        latest_before = (self.root / "monitoring" / "latest.json").read_text(
            encoding="utf-8"
        )
        second = run_monitoring_cycle(self.config, evidently_runner=runner)
        latest_after = (self.root / "monitoring" / "latest.json").read_text(
            encoding="utf-8"
        )

        self.assertEqual(first["status"], "healthy")
        self.assertEqual(second["status"], "unchanged")
        self.assertEqual(second["action"], "skipped_unchanged_input")
        self.assertEqual(second["monitoring_run_id"], first["monitoring_run_id"])
        self.assertEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        self.assertEqual(latest_before, latest_after)
        self.assertEqual(runner.calls, 1)
        database.engine.dispose()

    def test_new_prediction_changes_fingerprint_and_generates_report(self) -> None:
        database = self._seed_predictions()
        runner = FakeEvidentlyRunner()
        first = run_monitoring_cycle(self.config, evidently_runner=runner)
        database.insert_prediction(
            PredictionRecord(
                prediction_id="p5",
                timestamp="2026-08-15T00:00:09+00:00",
                model_name="sentinelml-ids",
                model_version="2",
                model_family="xgboost",
                execution_mode="smoke",
                demo_model=True,
                features={"a": 50.0, "b": 100.0, "c": 1.0},
                prediction="ATTACK",
                probability=0.9,
                latency_ms=1.0,
                ground_truth=1,
                ground_truth_received_at="2026-08-15T00:01:09+00:00",
            )
        )

        second = run_monitoring_cycle(self.config, evidently_runner=runner)

        self.assertEqual(second["status"], "healthy")
        self.assertNotEqual(second["monitoring_run_id"], first["monitoring_run_id"])
        self.assertNotEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        self.assertEqual(runner.calls, 2)
        database.engine.dispose()

    def test_sliding_latest_window_change_generates_report(self) -> None:
        database = self._seed_predictions()
        config = self._config(window_size=3, minimum_window_size=3)
        runner = FakeEvidentlyRunner()
        first = run_monitoring_cycle(config, evidently_runner=runner)
        database.insert_prediction(
            PredictionRecord(
                prediction_id="p5",
                timestamp="2026-08-15T00:00:09+00:00",
                model_name="sentinelml-ids",
                model_version="2",
                model_family="xgboost",
                execution_mode="smoke",
                demo_model=True,
                features={"a": 99.0, "b": 198.0, "c": 1.0},
                prediction="BENIGN",
                probability=0.1,
                latency_ms=1.0,
                ground_truth=0,
                ground_truth_received_at="2026-08-15T00:01:09+00:00",
            )
        )

        second = run_monitoring_cycle(config, evidently_runner=runner)

        self.assertNotEqual(second["monitoring_run_id"], first["monitoring_run_id"])
        self.assertEqual(runner.calls, 2)
        database.engine.dispose()

    def test_delayed_label_update_changes_fingerprint_and_performance(self) -> None:
        database = self._seed_predictions()
        runner = FakeEvidentlyRunner()
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE predictions
                    SET ground_truth = NULL, ground_truth_received_at = NULL
                    WHERE prediction_id = 'p4'
                    """
                )
            )
        first = run_monitoring_cycle(self.config, evidently_runner=runner)
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE predictions
                    SET ground_truth = 1,
                        ground_truth_received_at = '2026-08-15T00:02:00+00:00'
                    WHERE prediction_id = 'p4'
                    """
                )
            )

        second = run_monitoring_cycle(self.config, evidently_runner=runner)

        self.assertNotEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        self.assertGreater(
            second["performance"]["support"]["labelled_rows"],
            first["performance"]["support"]["labelled_rows"],
        )
        self.assertEqual(runner.calls, 2)
        database.engine.dispose()

    def test_label_value_change_where_allowed_changes_fingerprint(self) -> None:
        database = self._seed_predictions()
        first = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE predictions
                    SET ground_truth = 0,
                        ground_truth_received_at = '2026-08-15T00:03:00+00:00'
                    WHERE prediction_id = 'p3'
                    """
                )
            )

        second = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )

        self.assertNotEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        database.engine.dispose()

    def test_reference_and_config_changes_generate_new_report(self) -> None:
        database = self._seed_predictions()
        first = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )
        pd.DataFrame(
            {
                "a": range(100, 200),
                "b": [float(value * 2) for value in range(100, 200)],
                "c": [1.0] * 100,
                "target": [0, 1] * 50,
            }
        ).to_parquet(self.reference_path)
        second = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )
        changed_config = replace(self.config, drift_share_threshold=0.50)
        third = run_monitoring_cycle(
            changed_config,
            evidently_runner=FakeEvidentlyRunner(),
        )

        self.assertNotEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        self.assertNotEqual(
            third["monitoring_input_fingerprint"],
            second["monitoring_input_fingerprint"],
        )
        database.engine.dispose()

    def test_reference_absolute_path_does_not_change_fingerprint(self) -> None:
        database = self._seed_predictions()
        schema = load_serving_feature_schema(self.schema_path)
        reference = load_reference_dataset(self.config, schema)
        current = load_current_window(self.config, schema)
        host_identity = monitoring_input_fingerprint(
            config=self.config,
            schema=schema,
            reference=reference,
            current=current,
        )
        container_reference = replace(
            reference,
            metadata={**reference.metadata, "path": "/app/data/reference.parquet"},
        )
        container_identity = monitoring_input_fingerprint(
            config=self.config,
            schema=schema,
            reference=container_reference,
            current=current,
        )

        self.assertEqual(host_identity["value"], container_identity["value"])
        database.engine.dispose()

    def test_manual_window_size_fingerprint_and_restart_recovery(self) -> None:
        database = self._seed_predictions()
        default = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )
        window_four = self._config(window_size=4, minimum_window_size=3)
        changed = run_monitoring_cycle(
            window_four,
            evidently_runner=FakeEvidentlyRunner(),
        )
        repeated = run_monitoring_cycle(
            window_four,
            evidently_runner=FakeEvidentlyRunner(),
        )

        self.assertNotEqual(
            changed["monitoring_input_fingerprint"],
            default["monitoring_input_fingerprint"],
        )
        self.assertEqual(repeated["status"], "unchanged")
        self.assertEqual(repeated["monitoring_run_id"], changed["monitoring_run_id"])
        database.engine.dispose()

    def test_force_recompute_generates_report_despite_same_input(self) -> None:
        database = self._seed_predictions()
        runner = FakeEvidentlyRunner()
        first = run_monitoring_cycle(self.config, evidently_runner=runner)
        second = run_monitoring_cycle(
            self.config,
            evidently_runner=runner,
            force_recompute=True,
        )

        self.assertEqual(second["status"], "healthy")
        self.assertNotEqual(second["monitoring_run_id"], first["monitoring_run_id"])
        self.assertEqual(
            second["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        self.assertEqual(runner.calls, 2)
        database.engine.dispose()

    def test_fingerprint_failure_is_unhealthy_not_unchanged(self) -> None:
        database = self._seed_predictions()
        run_monitoring_cycle(self.config, evidently_runner=FakeEvidentlyRunner())

        with patch(
            "sentinelml.monitoring.service.monitoring_input_fingerprint",
            side_effect=RuntimeError("fingerprint failed"),
        ):
            failed = run_monitoring_cycle(
                self.config,
                evidently_runner=FakeEvidentlyRunner(),
            )

        self.assertEqual(failed["monitoring_health"], "unhealthy")
        self.assertEqual(failed["poll_status"], "failed")
        self.assertNotEqual(failed["status"], "unchanged")
        database.engine.dispose()

    def test_monitoring_poll_is_read_only_for_predictions_table(self) -> None:
        database = self._seed_predictions()
        before = prediction_count(database)

        run_monitoring_cycle(self.config, evidently_runner=FakeEvidentlyRunner())
        run_monitoring_cycle(self.config, evidently_runner=FakeEvidentlyRunner())

        self.assertEqual(prediction_count(database), before)
        database.engine.dispose()

    def test_phase8_latest_pointer_stays_stable_on_unchanged_poll(self) -> None:
        database = self._seed_predictions()
        first = run_monitoring_cycle(
            self.config,
            evidently_runner=FakeEvidentlyRunner(),
        )
        run_monitoring_cycle(self.config, evidently_runner=FakeEvidentlyRunner())
        latest = json.loads(
            (self.root / "monitoring" / "latest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(latest["monitoring_run_id"], first["monitoring_run_id"])
        self.assertEqual(
            latest["monitoring_input_fingerprint"],
            first["monitoring_input_fingerprint"],
        )
        database.engine.dispose()

    def test_evidently_failure_marks_monitor_unhealthy(self) -> None:
        database = self._seed_predictions()

        report = run_monitoring_cycle(
            config=self.config,
            evidently_runner=FakeEvidentlyRunner(fail=True),
        )

        self.assertEqual(report["monitoring_health"], "unhealthy")
        self.assertIn("evidently_drift", report["errors"][0]["stage"])
        database.engine.dispose()

    def test_prometheus_monitoring_metric_registration(self) -> None:
        update_monitoring_metrics(
            {
                "monitoring_health": "healthy",
                "finished_at": "2026-08-15T00:00:00+00:00",
                "window": {"actual_window_size": 5},
                "data_drift_detected": True,
                "drifting_feature_count": 1,
                "monitored_feature_count": 3,
                "drift_share": 1 / 3,
                "performance": {
                    "attack_recall": 0.5,
                    "f1": 0.5,
                    "false_positive_rate": 0.25,
                    "support": {"labelled_rows": 4},
                },
            }
        )

        rendered = monitoring_metrics.render()

        self.assertIn("sentinelml_monitoring_health", rendered)
        self.assertIn("sentinelml_drift_feature_share", rendered)

    def test_prometheus_and_grafana_config_validity(self) -> None:
        compose = yaml.safe_load(
            (Path.cwd() / "docker" / "docker-compose.yml").read_text(
                encoding="utf-8"
            )
        )
        prometheus = yaml.safe_load(
            (Path.cwd() / "docker" / "prometheus" / "prometheus.yml").read_text(
                encoding="utf-8"
            )
        )
        dashboard = json.loads(
            (
                Path.cwd()
                / "docker"
                / "grafana"
                / "dashboards"
                / "sentinelml.json"
            ).read_text(encoding="utf-8")
        )

        self.assertIn("monitor", compose["services"])
        self.assertIn("prometheus", compose["services"])
        self.assertIn("grafana", compose["services"])
        targets = [
            target
            for job in prometheus["scrape_configs"]
            for group in job["static_configs"]
            for target in group["targets"]
        ]
        self.assertIn("api:8000", targets)
        self.assertIn("monitor:9101", targets)
        self.assertEqual(dashboard["uid"], "sentinelml-phase7")
        panel_queries = json.dumps(dashboard["panels"])
        self.assertIn("sentinelml_attack_recall", panel_queries)
        self.assertIn("sentinelml_retraining_state", panel_queries)

def prediction_count(database: PredictionDatabase) -> int:
    with database.engine.connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
    return int(count)


if __name__ == "__main__":
    unittest.main()
