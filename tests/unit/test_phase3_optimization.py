from __future__ import annotations

import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.optimization.config import (
    enabled_model_families,
    load_optimization_config,
    mode_sample_sizes,
    overlay_model_params,
    trial_count,
)
from sentinelml.optimization.objective import (
    OptimizationObjective,
    objective_metric_value,
)
from sentinelml.optimization.search_spaces import suggest_hyperparameters
from sentinelml.optimization.study import (
    create_pruner,
    create_sampler,
    run_phase3_optimization,
    study_summary,
)
from sentinelml.training.data import DatasetBundle


class FakeTrial:
    def __init__(self) -> None:
        self.number = 0
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int:
        self.params[name] = low
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        **kwargs: object,
    ) -> float:
        self.params[name] = low
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.params[name] = choices[0]
        return choices[0]

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


class Phase3OptimizationTests(unittest.TestCase):
    def test_loads_optimization_config_defaults(self) -> None:
        config = load_optimization_config()

        self.assertEqual(config["objective_metric"], "pr_auc")
        self.assertEqual(config["direction"], "maximize")
        self.assertEqual(config["sampler"]["seed"], 42)
        self.assertEqual(trial_count(config, "xgboost", "smoke"), 2)
        self.assertIn("xgboost", enabled_model_families(config))
        self.assertEqual(
            mode_sample_sizes(config, "smoke"),
            {
                "train": config["modes"]["smoke"]["train_sample_size"],
                "validation": config["modes"]["smoke"]["validation_sample_size"],
            },
        )
        self.assertTrue(config["models"]["xgboost"]["pruning_enabled"])
        self.assertFalse(config["models"]["random_forest"]["pruning_enabled"])

    def test_baseline_values_are_inside_search_spaces(self) -> None:
        config = load_optimization_config()
        spaces = {
            model: config["models"][model]["search_space"]
            for model in enabled_model_families(config)
        }

        self.assertLessEqual(spaces["logistic_regression"]["C"]["low"], 1.0)
        self.assertGreaterEqual(
            spaces["logistic_regression"]["C"]["high"],
            1.0,
        )
        self.assertLessEqual(spaces["random_forest"]["n_estimators"]["low"], 80)
        self.assertGreaterEqual(
            spaces["random_forest"]["n_estimators"]["high"],
            80,
        )
        self.assertLessEqual(spaces["xgboost"]["max_depth"]["low"], 6)
        self.assertGreaterEqual(spaces["xgboost"]["max_depth"]["high"], 6)
        self.assertLessEqual(
            spaces["hist_gradient_boosting"]["max_iter"]["low"],
            200,
        )
        self.assertGreaterEqual(
            spaces["hist_gradient_boosting"]["max_iter"]["high"],
            200,
        )

    def test_search_space_generation_uses_configured_bounds(self) -> None:
        trial = FakeTrial()
        params = suggest_hyperparameters(
            trial,
            "random_forest",
            {
                "n_estimators": {
                    "type": "int",
                    "low": 40,
                    "high": 80,
                    "step": 20,
                },
                "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]},
            },
        )

        self.assertEqual(params, {"n_estimators": 40, "max_features": "sqrt"})
        self.assertEqual(trial.params["n_estimators"], 40)

    def test_objective_metric_value_uses_project_metrics(self) -> None:
        metrics = {"pr_auc": 0.42, "confusion_matrix": {"tp": 3}}

        self.assertEqual(objective_metric_value(metrics, "pr_auc"), 0.42)
        self.assertEqual(objective_metric_value(metrics, "confusion_matrix.tp"), 3.0)
        with self.assertRaises(KeyError):
            objective_metric_value(metrics, "accuracy")

    def test_xgboost_trial_config_preserves_derived_parameters(self) -> None:
        training_config = {
            "random_seed": 42,
            "baseline": {
                "xgboost": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                }
            },
        }

        trial_config = overlay_model_params(
            training_config,
            "xgboost",
            {"max_depth": 4},
        )

        self.assertEqual(trial_config["baseline"]["xgboost"]["max_depth"], 4)
        self.assertEqual(trial_config["baseline"]["xgboost"]["eval_metric"], "aucpr")
        self.assertNotIn("scale_pos_weight", trial_config["baseline"]["xgboost"])

    def test_sampler_and_pruner_setup_are_seeded_and_configured(self) -> None:
        class FakeTPESampler:
            def __init__(self, seed: int) -> None:
                self.seed = seed

        class FakeMedianPruner:
            def __init__(
                self,
                n_startup_trials: int,
                n_warmup_steps: int,
                interval_steps: int,
            ) -> None:
                self.n_startup_trials = n_startup_trials
                self.n_warmup_steps = n_warmup_steps
                self.interval_steps = interval_steps

        class FakeNopPruner:
            pass

        fake_optuna = types.SimpleNamespace(
            samplers=types.SimpleNamespace(TPESampler=FakeTPESampler),
            pruners=types.SimpleNamespace(
                MedianPruner=FakeMedianPruner,
                NopPruner=FakeNopPruner,
            ),
        )
        config = load_optimization_config()

        with patch.dict(sys.modules, {"optuna": fake_optuna}):
            sampler = create_sampler(config)
            xgb_pruner = create_pruner(config, "xgboost")
            rf_pruner = create_pruner(config, "random_forest")

        self.assertEqual(sampler.seed, 42)
        self.assertEqual(xgb_pruner.n_startup_trials, 1)
        self.assertIsInstance(rf_pruner, FakeNopPruner)

    def test_objective_logs_nested_mlflow_trial_without_test_metrics(self) -> None:
        class FakeRunInfo:
            run_id = "child-run-id"

        class FakeRun:
            info = FakeRunInfo()

            def __enter__(self) -> FakeRun:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeMLflow:
            def __init__(self) -> None:
                self.start_calls: list[dict[str, object]] = []
                self.tags: dict[str, str] = {}
                self.metrics: dict[str, float] = {}
                self.params: dict[str, object] = {}

            def start_run(self, **kwargs: object) -> FakeRun:
                self.start_calls.append(kwargs)
                return FakeRun()

            def set_tags(self, tags: dict[str, str]) -> None:
                self.tags.update(tags)

            def set_tag(self, key: str, value: str) -> None:
                self.tags[key] = value

            def log_params(self, params: dict[str, object]) -> None:
                self.params.update(params)

            def log_metric(self, key: str, value: float) -> None:
                self.metrics[key] = value

            def log_metrics(self, metrics: dict[str, float]) -> None:
                self.metrics.update(metrics)

        class FakeEstimator:
            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                return None

        fake_spec = types.SimpleNamespace(
            estimator=FakeEstimator(),
            parameters={"C": 1.0, "random_state": 42},
        )
        dataset = DatasetBundle(
            features=pd.DataFrame({"feature": [0.0, 1.0]}),
            target=pd.Series([0, 1]),
            metadata={},
        )
        config = load_optimization_config()
        objective = OptimizationObjective(
            model_family="logistic_regression",
            mode="smoke",
            training_config={
                "random_seed": 42,
                "baseline": {"logistic_regression": {"solver": "lbfgs"}},
            },
            optimization_config=config,
            train=dataset,
            validation=dataset,
            enable_mlflow=True,
            mlflow=FakeMLflow(),
        )

        with (
            patch(
                "sentinelml.optimization.objective.suggest_hyperparameters",
                return_value={"C": 1.0},
            ),
            patch(
                "sentinelml.optimization.objective.build_baseline_model_spec",
                return_value=fake_spec,
            ),
            patch(
                "sentinelml.optimization.objective.evaluate_binary_classifier",
                return_value={
                    "pr_auc": 0.5,
                    "precision": 0.4,
                    "inference_latency_ms_per_row": 0.01,
                },
            ),
        ):
            value = objective(FakeTrial())

        self.assertEqual(value, 0.5)
        self.assertEqual(objective.mlflow.start_calls[0]["nested"], True)
        self.assertEqual(objective.mlflow.start_calls[0]["run_name"], "trial-000")
        self.assertEqual(objective.mlflow.tags["run_type"], "optuna_trial")
        self.assertEqual(objective.mlflow.tags["trial_state"], "complete")
        self.assertIn("objective_value", objective.mlflow.metrics)
        self.assertNotIn("test.pr_auc", objective.mlflow.metrics)

    def test_study_summary_serializes_best_trial_and_state_counts(self) -> None:
        state_complete = types.SimpleNamespace(name="COMPLETE")
        state_pruned = types.SimpleNamespace(name="PRUNED")
        best_trial = types.SimpleNamespace(
            number=2,
            state=state_complete,
            value=0.77,
            params={"C": 1.0},
            user_attrs={"mlflow_run_id": "run-2"},
        )
        study = types.SimpleNamespace(
            study_name="optuna-study-logistic_regression",
            trials=[
                types.SimpleNamespace(
                    number=1,
                    state=state_pruned,
                    value=None,
                    params={},
                    user_attrs={},
                ),
                best_trial,
            ],
            best_trial=best_trial,
        )

        summary = study_summary(
            study,
            model_family="logistic_regression",
            mode="smoke",
            requested_n_trials=2,
            config=load_optimization_config(),
        )

        self.assertEqual(summary["trial_state_counts"]["COMPLETE"], 1)
        self.assertEqual(summary["trial_state_counts"]["PRUNED"], 1)
        self.assertEqual(summary["best_mlflow_run_id"], "run-2")

    def test_run_phase3_optimization_does_not_load_test_partition(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase3_no_test"
        loaded_paths: list[Path] = []

        class FakeStudy:
            study_name = "optuna-study-logistic_regression"
            trials: list[object] = []

            def optimize(self, objective: object, **kwargs: object) -> None:
                return None

        fake_optuna = types.SimpleNamespace(
            create_study=lambda **kwargs: FakeStudy(),
            samplers=types.SimpleNamespace(TPESampler=lambda seed: object()),
            pruners=types.SimpleNamespace(
                MedianPruner=lambda **kwargs: object(),
                NopPruner=lambda: object(),
            ),
        )

        def fake_load_partition(path: Path, **kwargs: object) -> DatasetBundle:
            loaded_paths.append(path)
            return DatasetBundle(
                features=pd.DataFrame({"feature": [0.0, 1.0]}),
                target=pd.Series([0, 1]),
                metadata={"path": str(path)},
            )

        config = load_optimization_config()
        config["models"]["logistic_regression"]["smoke_trials"] = 0
        try:
            with (
                patch.dict(sys.modules, {"optuna": fake_optuna}),
                patch(
                    "sentinelml.optimization.study.load_optimization_config",
                    return_value=config,
                ),
                patch(
                    "sentinelml.optimization.study.load_training_config",
                    return_value={
                        "random_seed": 42,
                        "baseline": {"logistic_regression": {"solver": "lbfgs"}},
                    },
                ),
                patch(
                    "sentinelml.optimization.study.load_feature_schema",
                    return_value={
                        "feature_columns": ["feature"],
                        "target_column": "target",
                    },
                ),
                patch(
                    "sentinelml.optimization.study.load_partition",
                    side_effect=fake_load_partition,
                ),
            ):
                run_phase3_optimization(
                    mode="smoke",
                    model="logistic_regression",
                    output_dir=root,
                )

            self.assertEqual([path.name for path in loaded_paths], [
                "train.parquet",
                "validation.parquet",
            ])
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
