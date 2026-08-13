from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.final_candidate.dependencies import (
    dependency_lock_metadata,
    ensure_dependency_lock,
)
from sentinelml.final_candidate.selection import select_optimization_winner
from sentinelml.final_candidate.training import (
    MLFLOW_EXPERIMENT_NAME,
    log_model_artifact,
    run_phase3_final_candidate,
)
from sentinelml.training.data import DatasetBundle


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeEstimator:
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.fitted = True

    def predict(self, features: pd.DataFrame) -> list[int]:
        return [0 for _ in range(len(features))]


class FakeRunInfo:
    run_id = "final-run-id"


class FakeRun:
    info = FakeRunInfo()

    def __enter__(self) -> FakeRun:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeFlavor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_model(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            model_uri="runs:/final-run-id/model",
            artifact_path="model",
            model_uuid="model-uuid",
        )


class FakeMLflow:
    def __init__(self) -> None:
        self.sklearn = FakeFlavor()
        self.xgboost = FakeFlavor()
        self.tags: dict[str, str] = {}
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[tuple[str, str | None]] = []
        self.experiment_name: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def get_tracking_uri(self) -> str:
        return self.tracking_uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, run_name: str) -> FakeRun:
        self.run_name = run_name
        return FakeRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))


class Phase3FinalCandidateTests(unittest.TestCase):
    def test_selects_validation_winner_and_loads_best_params(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase3e_select"
        try:
            write_json(
                root / "optimization_summary.json",
                {
                    "execution_mode": "smoke",
                    "direction": "maximize",
                    "validation_only_comparison": [
                        {
                            "model_family": "random_forest",
                            "best_objective_value": 0.3,
                            "best_trial_number": 1,
                            "best_mlflow_run_id": "rf-run",
                        },
                        {
                            "model_family": "xgboost",
                            "best_objective_value": 0.2,
                            "best_trial_number": 0,
                            "best_mlflow_run_id": "xgb-run",
                        },
                    ],
                },
            )
            write_json(
                root / "random_forest" / "study_summary.json",
                {
                    "study_name": "optuna-study-random_forest",
                    "model_family": "random_forest",
                    "execution_mode": "smoke",
                    "objective_metric": "pr_auc",
                    "direction": "maximize",
                    "best_trial_number": 1,
                    "best_objective_value": 0.3,
                    "best_mlflow_run_id": "rf-run",
                    "best_params": {"n_estimators": 80},
                },
            )
            write_json(
                root / "random_forest" / "best_params.json",
                {"n_estimators": 80},
            )

            selected = select_optimization_winner(
                mode="smoke",
                optimization_report_dir=root,
            )

            self.assertEqual(selected["model_family"], "random_forest")
            self.assertEqual(selected["best_params"], {"n_estimators": 80})
            self.assertEqual(selected["best_mlflow_run_id"], "rf-run")
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_rejects_mode_mismatch(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase3e_mode"
        try:
            write_json(
                root / "optimization_summary.json",
                {
                    "execution_mode": "smoke",
                    "direction": "maximize",
                    "validation_only_comparison": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                select_optimization_winner(mode="full", optimization_report_dir=root)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_dependency_lock_hashing_reuses_existing_lock(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase3e_lock"
        try:
            root.mkdir(parents=True, exist_ok=True)
            lock = root / "pylock.toml"
            lock.write_text("[lock]\n", encoding="utf-8")

            selected = ensure_dependency_lock(
                python_executable=sys.executable,
                repo_root=root,
            )
            metadata = dependency_lock_metadata(selected)

            self.assertEqual(selected, lock)
            self.assertEqual(metadata["dependency_lock_name"], "pylock.toml")
            self.assertEqual(
                metadata["dependency_lock_size_bytes"],
                lock.stat().st_size,
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_model_flavor_dispatch_does_not_use_registry(self) -> None:
        fake_mlflow = FakeMLflow()

        rf_info = log_model_artifact(
            fake_mlflow,
            model_family="random_forest",
            model=object(),
            signature="signature",
            input_example="example",
        )
        xgboost_info = log_model_artifact(
            fake_mlflow,
            model_family="xgboost",
            model=object(),
            signature="signature",
            input_example="example",
        )

        self.assertEqual(rf_info["flavor"], "xgboost")
        self.assertEqual(xgboost_info["flavor"], "xgboost")
        self.assertEqual(len(fake_mlflow.sklearn.calls), 0)
        self.assertEqual(fake_mlflow.xgboost.calls[0]["name"], "model")
        self.assertEqual(fake_mlflow.xgboost.calls[1]["name"], "model")
        self.assertNotIn("registered_model_name", fake_mlflow.xgboost.calls[0])
        self.assertNotIn("registered_model_name", fake_mlflow.xgboost.calls[1])

    def test_final_candidate_manifest_and_mlflow_traceability(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase3e_run"
        events: list[str] = []
        fake_mlflow = FakeMLflow()

        selected = {
            "study_name": "optuna-study-hist_gradient_boosting",
            "model_family": "hist_gradient_boosting",
            "execution_mode": "smoke",
            "best_trial_number": 1,
            "best_objective_value": 0.3,
            "best_mlflow_run_id": "trial-run-id",
            "objective_metric": "pr_auc",
            "direction": "maximize",
            "best_params": {"n_estimators": 120},
        }
        dataset = DatasetBundle(
            features=pd.DataFrame({"feature": [0.0, 1.0, 2.0]}),
            target=pd.Series([0, 1, 0]),
            metadata={"rows_used": 3},
        )
        lineage = {
            "git_commit": "abc123",
            "git_branch": "main",
            "git_dirty": False,
            "dvc_lock_sha256": "dvc-sha",
            "dvc_status_clean": True,
            "training_config_sha256": "train-sha",
            "optimization_config_sha256": "opt-sha",
            "feature_schema_sha256": "schema-sha",
            "dependency_lock_sha256": "lock-sha",
            "python_version": "3.x",
            "platform": "test-platform",
        }

        def fake_select(**kwargs: object) -> dict[str, object]:
            events.append("select")
            return selected

        def fake_load_partition(path: Path, **kwargs: object) -> DatasetBundle:
            events.append(path.name)
            return dataset

        captured_config: dict[str, object] = {}

        def fake_build(
            model_family: str,
            target: pd.Series,
            **kwargs: object,
        ) -> object:
            captured_config.update(kwargs["config"]["baseline"][model_family])
            return types.SimpleNamespace(
                estimator=FakeEstimator(),
                parameters={"n_estimators": captured_config["n_estimators"]},
            )

        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / "pylock.toml").write_text("[lock]\n", encoding="utf-8")
            with (
                patch.dict(sys.modules, {"mlflow": fake_mlflow}),
                patch(
                    "sentinelml.final_candidate.training.select_optimization_winner",
                    side_effect=fake_select,
                ),
                patch(
                    "sentinelml.final_candidate.training.ensure_dependency_lock",
                    return_value=root / "pylock.toml",
                ),
                patch(
                    "sentinelml.final_candidate.training.write_pip_freeze_snapshot",
                    return_value=root / "pip_freeze.txt",
                ),
                patch(
                    "sentinelml.final_candidate.training.load_optimization_config",
                    return_value={
                        "modes": {
                            "smoke": {
                                "train_sample_size": 2,
                                "validation_sample_size": 2,
                            },
                            "full": {
                                "train_sample_size": None,
                                "validation_sample_size": None,
                            },
                        }
                    },
                ),
                patch(
                    "sentinelml.final_candidate.training.load_training_config",
                    return_value={
                        "random_seed": 42,
                        "baseline": {
                            "hist_gradient_boosting": {
                                "n_estimators": 200,
                                "learning_rate": 0.1,
                                "grow_policy": "lossguide",
                                "max_leaves": 31,
                                "max_depth": 0,
                                "min_child_weight": 1.0,
                                "reg_lambda": 1.0,
                                "tree_method": "hist",
                                "device": "cuda",
                                "n_jobs": 1,
                            }
                        },
                    },
                ),
                patch(
                    "sentinelml.final_candidate.training.load_feature_schema",
                    return_value={
                        "feature_columns": ["feature"],
                        "target_column": "target",
                    },
                ),
                patch(
                    "sentinelml.final_candidate.training.load_partition",
                    side_effect=fake_load_partition,
                ),
                patch(
                    "sentinelml.final_candidate.training.build_baseline_model_spec",
                    side_effect=fake_build,
                ),
                patch(
                    "sentinelml.final_candidate.training.evaluate_binary_classifier",
                    return_value={
                        "pr_auc": 0.5,
                        "precision": 0.4,
                        "inference_latency_ms_per_row": 0.01,
                    },
                ),
                patch(
                    "sentinelml.final_candidate.training.feature_importance_table",
                    return_value=([], "unsupported"),
                ),
                patch(
                    "sentinelml.final_candidate.training.collect_reproducibility_lineage",
                    return_value=lineage,
                ),
                patch(
                    "sentinelml.final_candidate.training.configure_mlflow_runtime_environment",
                    return_value="http://127.0.0.1:5000",
                ),
            ):
                manifest = run_phase3_final_candidate(
                    mode="smoke",
                    enable_mlflow=True,
                    output_dir=root / "reports",
                    python_executable=sys.executable,
                )

            self.assertEqual(events, [
                "select",
                "train.parquet",
                "validation.parquet",
                "test.parquet",
            ])
            self.assertEqual(captured_config["n_estimators"], 120)
            self.assertEqual(manifest["model_family"], "hist_gradient_boosting")
            self.assertEqual(
                manifest["source_optimization"]["best_trial_mlflow_run_id"],
                "trial-run-id",
            )
            self.assertEqual(manifest["registry_state"], {
                "registered": False,
                "model_version": None,
            })
            self.assertEqual(
                manifest["mlflow"]["experiment_name"],
                MLFLOW_EXPERIMENT_NAME,
            )
            self.assertEqual(
                fake_mlflow.tags["source_optuna_mlflow_run_id"],
                "trial-run-id",
            )
            self.assertEqual(fake_mlflow.tags["registry_status"], "unregistered")
            self.assertIn("val.pr_auc", fake_mlflow.metrics)
            self.assertIn("test.pr_auc", fake_mlflow.metrics)
            self.assertTrue((root / "reports" / "model_manifest.json").exists())
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
