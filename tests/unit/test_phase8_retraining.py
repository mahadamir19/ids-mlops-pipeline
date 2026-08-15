from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.retraining.config import RetrainingConfig
from sentinelml.retraining.dataset import (
    RetrainingDatasetError,
    build_retraining_dataset,
)
from sentinelml.retraining.lock import RetrainingLock
from sentinelml.retraining.repository import RetrainingRepository
from sentinelml.retraining.service import RetrainingService
from sentinelml.retraining.trainer import (
    RetrainingTrainer,
    _apply_xgboost_execution_policy,
    _force_cpu_training_config,
)
from sentinelml.training.gpu import force_estimator_cpu
from sentinelml.retraining.triggers import (
    PerformanceThresholds,
    evaluate_trigger,
    thresholds_from_lifecycle_report,
)
from sentinelml.tracking.mlflow import sha256_file

FEATURES = ["a", "b", "c"]


class Phase8TriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = config(Path.cwd() / "tmp_tests" / self._testMethodName)
        self.thresholds = PerformanceThresholds(
            min_attack_recall=0.8,
            max_false_positive_rate=0.05,
            min_f1=0.75,
            baseline_f1=0.85,
            source="test",
        )

    def test_healthy_no_drift_no_degradation_no_trigger(self) -> None:
        decision = evaluate_trigger(
            report(),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertFalse(decision["should_retrain"])

    def test_drift_only_triggers(self) -> None:
        decision = evaluate_trigger(
            report(data_drift_detected=True),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertTrue(decision["should_retrain"])
        self.assertIn("data_drift", decision["trigger_reasons"])

    def test_performance_degradation_only_triggers(self) -> None:
        decision = evaluate_trigger(
            report(attack_recall=0.2, data_drift_detected=False),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertTrue(decision["trigger_performance_degradation"])
        self.assertIn("attack_recall_below_threshold", decision["trigger_reasons"])

    def test_both_triggers_record_both_reasons(self) -> None:
        decision = evaluate_trigger(
            report(data_drift_detected=True, false_positive_rate=0.2),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertIn("data_drift", decision["trigger_reasons"])
        self.assertIn(
            "false_positive_rate_above_threshold",
            decision["trigger_reasons"],
        )

    def test_insufficient_labels_make_performance_unavailable(self) -> None:
        decision = evaluate_trigger(
            report(labelled_rows=1),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertFalse(decision["trigger_performance_degradation"])
        self.assertIn(
            "insufficient_labelled_rows",
            decision["performance_unavailable_reasons"],
        )

    def test_unhealthy_and_warming_up_are_blocked(self) -> None:
        unhealthy = evaluate_trigger(
            report(monitoring_health="unhealthy"),
            self.config,
            performance_thresholds=self.thresholds,
        )
        warming = evaluate_trigger(
            report(monitoring_health="warming_up", status="warming_up"),
            self.config,
            performance_thresholds=self.thresholds,
        )

        self.assertEqual(unhealthy["decision"], "blocked_monitoring_unhealthy")
        self.assertEqual(warming["decision"], "blocked_monitoring_warming_up")

    def test_same_monitoring_run_is_idempotent(self) -> None:
        decision = evaluate_trigger(
            report(),
            self.config,
            performance_thresholds=self.thresholds,
            already_processed=True,
        )

        self.assertEqual(decision["decision"], "already_processed")

    def test_thresholds_derive_from_phase4_report(self) -> None:
        thresholds = thresholds_from_lifecycle_report(
            {
                "thresholds": {"attack_recall": 0.36, "false_positive_rate": 0.005},
                "baseline_metrics": {"f1": 0.50},
            },
            self.config,
        )

        self.assertEqual(thresholds.min_attack_recall, 0.36)
        self.assertEqual(thresholds.max_false_positive_rate, 0.005)
        self.assertEqual(thresholds.min_f1, 0.40)


class Phase8DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        write_schema(self.root / "feature_schema.json")
        train = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [10.0, 20.0, 30.0, 40.0],
                "c": [100.0, 200.0, 300.0, 400.0],
                "target": [0, 1, 0, 1],
            }
        )
        train.to_parquet(self.root / "train.parquet")
        self.config = config(self.root)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_dataset_uses_train_and_new_approved_production_only(self) -> None:
        observations = [
            approved("p1", {"a": 5.0, "b": 50.0, "c": 500.0}, 1, self.config),
            approved("p2", {"a": 1.0, "b": 10.0, "c": 100.0}, 0, self.config),
            {
                **approved("p3", {"a": 6.0, "b": 60.0, "c": 600.0}, 0, self.config),
                "validation_status": "rejected",
            },
            approved("p4", {"a": 7.0, "b": 70.0, "c": 700.0}, 0, self.config),
        ]

        dataset = build_retraining_dataset(
            retraining_run_id="run-1",
            config=self.config,
            approved_observations=observations,
            consumed_prediction_ids={"p4"},
            output_dir=self.root / "reports",
        )

        self.assertEqual(dataset.manifest["historical"]["row_count_after_sampling"], 4)
        self.assertEqual(
            dataset.manifest["production"]["row_count_after_validation"],
            2,
        )
        self.assertEqual(
            dataset.manifest["deduplication"]["cross_source_duplicates_removed"],
            1,
        )
        self.assertEqual(dataset.consumed_prediction_ids, ["p1", "p2"])
        self.assertEqual(list(dataset.bundle.features.columns), FEATURES)
        self.assertIn(
            "data/processed/test.parquet",
            dataset.manifest["excluded_partitions"],
        )
        self.assertTrue(dataset.manifest_path.exists())

    def test_dataset_fingerprint_is_deterministic(self) -> None:
        observations = [
            approved("p1", {"a": 5.0, "b": 50.0, "c": 500.0}, 1, self.config)
        ]

        first = build_retraining_dataset(
            retraining_run_id="run-1",
            config=self.config,
            approved_observations=observations,
            consumed_prediction_ids=set(),
            output_dir=self.root / "first",
        )
        second = build_retraining_dataset(
            retraining_run_id="run-2",
            config=self.config,
            approved_observations=observations,
            consumed_prediction_ids=set(),
            output_dir=self.root / "second",
        )

        self.assertEqual(
            first.manifest["dataset_fingerprint"],
            second.manifest["dataset_fingerprint"],
        )

    def test_schema_mismatch_and_test_source_are_rejected(self) -> None:
        bad = approved("bad", {"a": 1.0, "b": 2.0}, 1, self.config)
        with self.assertRaises(RetrainingDatasetError):
            build_retraining_dataset(
                retraining_run_id="run-1",
                config=self.config,
                approved_observations=[bad],
                consumed_prediction_ids=set(),
                output_dir=self.root / "bad",
            )

        test_config = config(self.root, historical_source=self.root / "test.parquet")
        pd.DataFrame(
            {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0], "target": [0, 1]}
        ).to_parquet(test_config.historical_source)
        object.__setattr__(
            test_config,
            "historical_source",
            Path("data/processed/test.parquet"),
        )
        with self.assertRaises(RetrainingDatasetError):
            build_retraining_dataset(
                retraining_run_id="run-2",
                config=test_config,
                approved_observations=[],
                consumed_prediction_ids=set(),
                output_dir=self.root / "test",
            )


class Phase8RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        write_schema(self.root / "feature_schema.json")
        pd.DataFrame(
            {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0], "target": [0, 1]}
        ).to_parquet(self.root / "train.parquet")
        self.config = config(self.root, cooldown_seconds=60)
        self.engine = create_engine(f"sqlite:///{self.root / 'state.db'}")
        self.repository = RetrainingRepository(self.config, engine=self.engine)
        self.repository.initialize()

    def tearDown(self) -> None:
        self.engine.dispose()
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_lock_denies_concurrent_cycle(self) -> None:
        first = RetrainingLock(self.repository)
        second = RetrainingLock(self.repository)
        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
        finally:
            first.release()
            second.release()

    def test_cooldown_blocks_then_expires(self) -> None:
        self.repository.create_run(
            retraining_run_id="run-1",
            monitoring_run_id="mon-1",
            trigger_reasons=["data_drift"],
            drift_share=1.0,
            performance_metrics={},
        )
        self.repository.finish_run("run-1", status="rejected")

        self.assertTrue(self.repository.cooldown_status()["active"])

        self.repository.update_run(
            "run-1",
            cooldown_until="2000-01-01T00:00:00+00:00",
        )
        self.assertFalse(self.repository.cooldown_status()["active"])

    def test_failed_monitoring_run_can_be_claimed_again(self) -> None:
        self.repository.mark_processed(
            monitoring_run_id="mon-failed",
            decision="triggered",
            action_taken="failed",
            report_path=None,
        )

        self.assertFalse(self.repository.already_processed("mon-failed"))
        claimed = self.repository.claim_monitoring_run(
            monitoring_run_id="mon-failed",
            decision="triggered",
            report_path=None,
        )

        self.assertTrue(claimed)
        self.assertTrue(self.repository.already_processed("mon-failed"))

    def test_consumed_ids_ignore_failed_retraining_runs(self) -> None:
        self.repository.create_run(
            retraining_run_id="run-failed",
            monitoring_run_id="mon-failed",
            trigger_reasons=["data_drift"],
            drift_share=1.0,
            performance_metrics={},
        )
        self.repository.mark_observations_consumed(
            retraining_run_id="run-failed",
            prediction_ids=["p-failed"],
        )
        self.repository.finish_run("run-failed", status="failed")
        self.repository.create_run(
            retraining_run_id="run-rejected",
            monitoring_run_id="mon-rejected",
            trigger_reasons=["data_drift"],
            drift_share=1.0,
            performance_metrics={},
        )
        self.repository.mark_observations_consumed(
            retraining_run_id="run-rejected",
            prediction_ids=["p-rejected"],
        )
        self.repository.finish_run("run-rejected", status="rejected")

        self.assertEqual(self.repository.consumed_prediction_ids(), {"p-rejected"})


class Phase8CandidateLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
        write_schema(self.root / "feature_schema.json")
        pd.DataFrame(
            {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0], "target": [0, 1]}
        ).to_parquet(self.root / "train.parquet")
        self.engine = create_engine(f"sqlite:///{self.root / 'state.db'}")
        self.repository = RetrainingRepository(config(self.root), engine=self.engine)
        self.repository.initialize()
        self.lifecycle = FakeLifecycle()
        self.service = RetrainingService(
            config=config(self.root),
            repository=self.repository,
            lifecycle_service=self.lifecycle,
            production_service=types.SimpleNamespace(
                get_approved_production_observations=lambda: []
            ),
            trainer=types.SimpleNamespace(),
        )

    def tearDown(self) -> None:
        self.engine.dispose()
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_phase4_registration_evaluation_and_promotion_are_called(self) -> None:
        manifest = self.root / "candidate_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        self.repository.create_run(
            retraining_run_id="run-1",
            monitoring_run_id="mon-1",
            trigger_reasons=["data_drift"],
            drift_share=1.0,
            performance_metrics={},
        )

        result = self.service._register_evaluate_promote(
            manifest,
            retraining_run_id="run-1",
            monitoring_run_id="mon-1",
        )

        self.assertEqual(result["outcome"]["event"], "rejected")
        self.assertEqual(self.lifecycle.registered_manifest_path, manifest)
        self.assertEqual(self.lifecycle.promoted_version, "7")

    def test_registered_candidate_is_marked_failed_when_lifecycle_raises(self) -> None:
        manifest = self.root / "candidate_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        self.repository.create_run(
            retraining_run_id="run-1",
            monitoring_run_id="mon-1",
            trigger_reasons=["data_drift"],
            drift_share=1.0,
            performance_metrics={},
        )
        self.lifecycle.raise_on_promote = RuntimeError("cuda failure")

        with self.assertRaisesRegex(RuntimeError, "cuda failure"):
            self.service._register_evaluate_promote(
                manifest,
                retraining_run_id="run-1",
                monitoring_run_id="mon-1",
            )

        self.assertEqual(self.lifecycle.failed_version, "7")
        self.assertEqual(self.lifecycle.failed_retraining_run_id, "run-1")
        self.assertEqual(self.lifecycle.failed_monitoring_run_id, "mon-1")

    def test_phase8_retraining_overrides_cuda_to_cpu(self) -> None:
        training_config = {
            "baseline": {
                "xgboost": {
                    "device": "cuda",
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "gpu_id": 0,
                    "n_estimators": 2,
                }
            }
        }

        updated = _force_cpu_training_config(training_config, "xgboost")

        self.assertEqual(updated["baseline"]["xgboost"]["device"], "cpu")
        self.assertEqual(updated["baseline"]["xgboost"]["tree_method"], "hist")
        self.assertNotIn("predictor", updated["baseline"]["xgboost"])
        self.assertNotIn("gpu_id", updated["baseline"]["xgboost"])
        self.assertEqual(training_config["baseline"]["xgboost"]["device"], "cuda")

    def test_phase8_policy_preserves_predictive_hyperparameters(self) -> None:
        training_config = {
            "baseline": {
                "xgboost": {
                    "device": "cuda",
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "gpu_id": 0,
                    "sampling_method": "gradient_based",
                    "max_depth": 7,
                    "learning_rate": 0.03,
                    "n_estimators": 11,
                    "subsample": 0.7,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 2.5,
                    "reg_alpha": 0.1,
                    "reg_lambda": 3.0,
                }
            }
        }

        updated, runtime = _apply_xgboost_execution_policy(
            training_config,
            "xgboost",
            requested_device="cpu",
        )
        params = updated["baseline"]["xgboost"]

        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["tree_method"], "hist")
        self.assertEqual(params["sampling_method"], "uniform")
        self.assertEqual(runtime["requested_device"], "cpu")
        self.assertEqual(runtime["effective_device"], "cpu")
        self.assertEqual(runtime["effective_tree_method"], "hist")
        self.assertEqual(runtime["removed_gpu_params"], ["gpu_id", "predictor"])
        for key in [
            "max_depth",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ]:
            self.assertEqual(params[key], training_config["baseline"]["xgboost"][key])

    def test_reconstructed_champion_gpu_params_are_normalized_without_registry_write(
        self,
    ) -> None:
        champion = types.SimpleNamespace(
            version="2",
            run_id="champion-run",
            tags={"source_run_id": "champion-run"},
        )
        fake_run = types.SimpleNamespace(
            data=types.SimpleNamespace(
                params={
                    "winning.device": "cuda",
                    "winning.tree_method": "gpu_hist",
                    "winning.predictor": "gpu_predictor",
                    "winning.gpu_id": "0",
                    "winning.max_depth": "9",
                }
            )
        )
        client = types.SimpleNamespace(
            get_run=lambda run_id: fake_run,
            set_model_version_tag=Mock(),
        )
        lifecycle = types.SimpleNamespace(client=client)
        trainer = RetrainingTrainer(
            config=config(self.root),
            lifecycle_service=lifecycle,
        )
        training_path = self.root / "training.yaml"
        training_path.write_text(
            "\n".join(
                [
                    "random_seed: 42",
                    "baseline:",
                    "  xgboost:",
                    "    n_estimators: 2",
                    "    tree_method: hist",
                    "    device: cuda",
                    "  logistic_regression: {}",
                    "  random_forest: {}",
                    "  hist_gradient_boosting: {}",
                ]
            ),
            encoding="utf-8",
        )

        updated, runtime = trainer._candidate_training_config(champion, "xgboost")

        params = updated["baseline"]["xgboost"]
        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["tree_method"], "hist")
        self.assertNotIn("predictor", params)
        self.assertNotIn("gpu_id", params)
        self.assertEqual(params["max_depth"], 9)
        self.assertEqual(runtime["removed_gpu_params"], ["gpu_id", "predictor"])
        client.set_model_version_tag.assert_not_called()

    def test_cpu_xgboost_candidate_fits_serializes_reloads_and_predicts(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            self.skipTest("xgboost is not installed")
        features = pd.DataFrame(
            {
                "a": np.linspace(0, 1, 12, dtype=np.float32),
                "b": np.linspace(1, 2, 12, dtype=np.float32),
                "c": np.linspace(2, 3, 12, dtype=np.float32),
            }
        )
        target = pd.Series([0, 1] * 6, dtype="int8")
        model = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            device="cpu",
            eval_metric="logloss",
        )
        model.fit(features, target)
        model_path = self.root / "candidate.json"
        model.save_model(model_path)

        loaded = XGBClassifier()
        loaded.load_model(model_path)
        force_estimator_cpu(loaded)
        probabilities = loaded.predict_proba(features.head(3))
        predictions = loaded.predict(features.head(3))

        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(predictions.shape, (3,))
        self.assertEqual(loaded.get_params()["device"], "cpu")

    def test_manual_trigger_evaluation_does_not_consume_execution_state(self) -> None:
        write_monitoring_latest(
            self.root,
            report(monitoring_run_id="mon-trigger", data_drift_detected=True),
        )

        first = self.service.evaluate_latest_trigger()
        second = self.service.evaluate_latest_trigger()
        forced = self.service.evaluate_latest_trigger(force_recheck=True)

        self.assertTrue(first["should_retrain"])
        self.assertTrue(second["should_retrain"])
        self.assertTrue(forced["should_retrain"])
        self.assertFalse(self.repository.already_processed("mon-trigger"))
        self.assertEqual(processed_count(self.repository), 0)

    def test_no_trigger_manual_evaluation_does_not_consume_execution_state(
        self,
    ) -> None:
        write_monitoring_latest(self.root, report(monitoring_run_id="mon-clean"))

        result = self.service.evaluate_latest_trigger()

        self.assertFalse(result["should_retrain"])
        self.assertFalse(self.repository.already_processed("mon-clean"))

    def test_disabled_retraining_once_does_not_consume_trigger(self) -> None:
        disabled_service = RetrainingService(
            config=config(self.root, enabled=False),
            repository=self.repository,
            lifecycle_service=self.lifecycle,
            production_service=types.SimpleNamespace(
                get_approved_production_observations=lambda: []
            ),
            trainer=types.SimpleNamespace(),
        )
        write_monitoring_latest(
            self.root,
            report(monitoring_run_id="mon-disabled", data_drift_detected=True),
        )

        result = disabled_service.process_once()

        self.assertEqual(result["action_taken"], "disabled")
        self.assertFalse(self.repository.already_processed("mon-disabled"))

    def test_once_after_evaluation_claims_and_executes_exactly_once(self) -> None:
        write_monitoring_latest(
            self.root,
            report(monitoring_run_id="mon-execute", data_drift_detected=True),
        )
        approved_rows = [
            approved("p1", {"a": 9.0, "b": 9.0, "c": 9.0}, 1, self.service.config)
        ]
        self.service.production_service = types.SimpleNamespace(
            get_approved_production_observations=lambda: approved_rows
        )
        trainer = Mock()
        trainer.train_candidate.return_value = {
            "manifest_path": self.root / "candidate_manifest.json",
            "mlflow_run_id": "run-1",
        }
        self.service.trainer = trainer
        dataset = fake_retraining_dataset(self.root)

        self.service.evaluate_latest_trigger(force_recheck=True)
        with patch(
            "sentinelml.retraining.service.build_retraining_dataset",
            return_value=dataset,
        ):
            first = self.service.process_once()
            second = self.service.process_once()

        self.assertIn(first["status"], {"rejected", "promoted", "promotion_pending"})
        self.assertEqual(second["decision"], "already_processed")
        self.assertEqual(trainer.train_candidate.call_count, 1)
        self.assertTrue(self.repository.already_processed("mon-execute"))

    def test_atomic_execution_claim_allows_only_one_worker(self) -> None:
        first = self.repository.claim_monitoring_run(
            monitoring_run_id="mon-concurrent",
            decision="triggered",
            report_path=None,
        )
        second = self.repository.claim_monitoring_run(
            monitoring_run_id="mon-concurrent",
            decision="triggered",
            report_path=None,
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(processed_count(self.repository), 1)

    def test_failed_cycle_does_not_consume_observations(self) -> None:
        write_monitoring_latest(
            self.root,
            report(monitoring_run_id="mon-failure", data_drift_detected=True),
        )
        approved_rows = [
            approved("p1", {"a": 9.0, "b": 9.0, "c": 9.0}, 1, self.service.config)
        ]
        self.service.production_service = types.SimpleNamespace(
            get_approved_production_observations=lambda: approved_rows
        )
        trainer = Mock()
        trainer.train_candidate.side_effect = RuntimeError("infrastructure failure")
        self.service.trainer = trainer
        dataset = fake_retraining_dataset(self.root)

        with patch(
            "sentinelml.retraining.service.build_retraining_dataset",
            return_value=dataset,
        ):
            first = self.service.process_once()
            self.repository.update_run(
                first["retraining_run_id"],
                cooldown_until="2000-01-01T00:00:00+00:00",
            )
            second = self.service.process_once()

        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["status"], "failed")
        self.assertEqual(trainer.train_candidate.call_count, 2)
        self.assertEqual(self.repository.consumed_prediction_ids(), set())


def config(
    root: Path,
    *,
    historical_source: Path | None = None,
    cooldown_seconds: int = 300,
    enabled: bool = True,
) -> RetrainingConfig:
    return RetrainingConfig(
        enabled=enabled,
        poll_interval_seconds=60,
        cooldown_seconds=cooldown_seconds,
        execution_mode="smoke",
        minimum_approved_production_rows=1,
        drift_enabled=True,
        performance_enabled=True,
        performance_minimum_labelled_rows=2,
        performance_minimum_attack_support=1,
        performance_minimum_benign_support=1,
        min_attack_recall=None,
        max_false_positive_rate=None,
        max_f1_drop=0.10,
        historical_source=historical_source or root / "train.parquet",
        max_historical_rows_smoke=10,
        max_production_rows_smoke=10,
        sampling_seed=42,
        deduplication="canonical_feature_target_sha256",
        production_weighting_ratio=1.0,
        candidate_strategy="fresh_champion_family_retrain",
        training_device="cpu",
        force_cpu=True,
        auto_register=True,
        auto_evaluate=True,
        auto_promote=True,
        monitoring_reports_dir=root / "monitoring",
        retraining_reports_dir=root / "retraining",
        feature_schema_path=root / "feature_schema.json",
        lifecycle_config_path=root / "lifecycle.yaml",
        training_config_path=root / "training.yaml",
        database_url_env="SENTINELML_TEST_DATABASE_URL",
        database_connect_timeout_seconds=1,
        database_statement_timeout_ms=1000,
        config_path=root / "retraining.yaml",
    )


def report(**overrides: Any) -> dict[str, Any]:
    payload = {
        "monitoring_run_id": "mon-1",
        "finished_at": "2026-08-15T00:00:00+00:00",
        "status": "healthy",
        "monitoring_health": "healthy",
        "data_drift_detected": False,
        "drift_share": 0.0,
        "performance": {
            "attack_recall": 0.9,
            "f1": 0.9,
            "false_positive_rate": 0.0,
            "support": {"labelled_rows": 10, "true_attack": 5, "true_benign": 5},
        },
    }
    performance_keys = {
        "attack_recall",
        "f1",
        "false_positive_rate",
        "labelled_rows",
        "true_attack",
        "true_benign",
    }
    for key, value in overrides.items():
        if key in {"labelled_rows", "true_attack", "true_benign"}:
            payload["performance"]["support"][key] = value
        elif key in performance_keys:
            payload["performance"][key] = value
        else:
            payload[key] = value
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


def write_monitoring_latest(root: Path, payload: dict[str, Any]) -> None:
    reports_dir = root / "monitoring"
    reports_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    (reports_dir / "latest.json").write_text(encoded + "\n", encoding="utf-8")


def fake_retraining_dataset(root: Path) -> Any:
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return types.SimpleNamespace(
        manifest_path=manifest_path,
        consumed_prediction_ids=["p1"],
        manifest={
            "historical": {"row_count_after_sampling": 2},
            "production": {"row_count_after_sampling": 1},
            "deduplication": {"cross_source_duplicates_removed": 0},
            "dataset_fingerprint": "dataset-fp",
        },
    )


def processed_count(repository: RetrainingRepository) -> int:
    with repository.engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM retraining_monitoring_processed")
        ).scalar()
    return int(count)


def approved(
    prediction_id: str,
    features: dict[str, float],
    ground_truth: int,
    cfg: RetrainingConfig,
) -> dict[str, Any]:
    return {
        "prediction_id": prediction_id,
        "features": features,
        "ground_truth": ground_truth,
        "validation_status": "approved",
        "schema_fingerprint": sha256_file(cfg.feature_schema_path),
    }


class FakeLifecycle:
    def __init__(self) -> None:
        self.model_name = "sentinelml-ids"
        self.config = {"paths": {}}
        self.registered_manifest_path: Path | None = None
        self.promoted_version: str | None = None
        self.raise_on_promote: Exception | None = None
        self.failed_version: str | None = None
        self.failed_retraining_run_id: str | None = None
        self.failed_monitoring_run_id: str | None = None

    def register_candidate(self, **kwargs: Any) -> dict[str, Any]:
        self.registered_manifest_path = kwargs["manifest_path"]
        self.extra_tags = kwargs["extra_tags"]
        return {"model_version": "7", "registered": True}

    def promote_or_reject(self, *, version: str) -> dict[str, Any]:
        if self.raise_on_promote is not None:
            raise self.raise_on_promote
        self.promoted_version = version
        return {"event": "rejected", "evaluation": {"passed": False}}

    def mark_candidate_failed(
        self,
        *,
        version: str,
        error: str,
        retraining_run_id: str | None = None,
        monitoring_run_id: str | None = None,
    ) -> dict[str, Any]:
        self.failed_version = version
        self.failed_error = error
        self.failed_retraining_run_id = retraining_run_id
        self.failed_monitoring_run_id = monitoring_run_id
        return {"event": "failed", "model_version": version}


if __name__ == "__main__":
    unittest.main()
