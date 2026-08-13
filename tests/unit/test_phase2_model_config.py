from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.training.models import (
    build_baseline_model_specs,
    load_training_config,
)


class Phase2ModelConfigTests(unittest.TestCase):
    def test_loads_training_config_values(self) -> None:
        config = load_training_config()

        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(
            config["baseline"]["logistic_regression"],
            {
                "booster": "gblinear",
                "n_estimators": 120,
                "learning_rate": 0.1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "device": "cuda",
                "n_jobs": 1,
            },
        )
        self.assertEqual(config["baseline"]["random_forest"]["num_parallel_tree"], 80)
        self.assertEqual(config["baseline"]["xgboost"]["tree_method"], "hist")
        self.assertEqual(config["baseline"]["xgboost"]["device"], "cuda")
        self.assertEqual(
            config["baseline"]["hist_gradient_boosting"]["grow_policy"],
            "lossguide",
        )

    def test_builds_all_four_models_from_config_without_behavior_change(self) -> None:
        target = pd.Series([0] * 13 + [1] * 2)
        specs = {spec.name: spec for spec in build_baseline_model_specs(target)}

        self.assertEqual(
            set(specs),
            {
                "logistic_regression",
                "random_forest",
                "xgboost",
                "hist_gradient_boosting",
            },
        )

        logistic = specs["logistic_regression"].estimator
        self.assertIsInstance(logistic, XGBClassifier)
        self.assertEqual(logistic.booster, "gblinear")
        self.assertEqual(logistic.n_estimators, 120)
        self.assertEqual(logistic.learning_rate, 0.1)
        self.assertEqual(logistic.reg_alpha, 0.0)
        self.assertEqual(logistic.reg_lambda, 1.0)
        self.assertEqual(logistic.device, "cuda")
        self.assertEqual(logistic.random_state, 42)
        self.assertEqual(logistic.scale_pos_weight, 6.5)

        random_forest = specs["random_forest"].estimator
        self.assertIsInstance(random_forest, XGBClassifier)
        self.assertEqual(random_forest.n_estimators, 1)
        self.assertEqual(random_forest.num_parallel_tree, 80)
        self.assertEqual(random_forest.max_depth, 18)
        self.assertEqual(random_forest.learning_rate, 1.0)
        self.assertEqual(random_forest.subsample, 0.8)
        self.assertEqual(random_forest.colsample_bynode, 0.8)
        self.assertEqual(random_forest.min_child_weight, 2.0)
        self.assertEqual(random_forest.tree_method, "hist")
        self.assertEqual(random_forest.device, "cuda")
        self.assertEqual(random_forest.n_jobs, 1)
        self.assertEqual(random_forest.random_state, 42)
        self.assertEqual(random_forest.scale_pos_weight, 6.5)

        xgboost = specs["xgboost"].estimator
        self.assertIsInstance(xgboost, XGBClassifier)
        self.assertEqual(xgboost.n_estimators, 200)
        self.assertEqual(xgboost.max_depth, 6)
        self.assertEqual(xgboost.learning_rate, 0.1)
        self.assertEqual(xgboost.subsample, 0.8)
        self.assertEqual(xgboost.colsample_bytree, 0.8)
        self.assertEqual(xgboost.objective, "binary:logistic")
        self.assertEqual(xgboost.eval_metric, "logloss")
        self.assertEqual(xgboost.tree_method, "hist")
        self.assertEqual(xgboost.device, "cuda")
        self.assertEqual(xgboost.n_jobs, 1)
        self.assertEqual(xgboost.random_state, 42)
        self.assertEqual(xgboost.scale_pos_weight, 6.5)
        self.assertEqual(xgboost.verbosity, 0)

        hist = specs["hist_gradient_boosting"].estimator
        self.assertIsInstance(hist, XGBClassifier)
        self.assertEqual(hist.n_estimators, 200)
        self.assertEqual(hist.learning_rate, 0.1)
        self.assertEqual(hist.grow_policy, "lossguide")
        self.assertEqual(hist.max_leaves, 31)
        self.assertEqual(hist.max_depth, 0)
        self.assertEqual(hist.min_child_weight, 1.0)
        self.assertEqual(hist.reg_lambda, 1.0)
        self.assertEqual(hist.device, "cuda")
        self.assertEqual(hist.random_state, 42)

    def test_custom_config_controls_common_seed_for_all_models(self) -> None:
        root = Path.cwd() / "tmp_tests" / "phase2_model_config"
        path = root / "training_config.yaml"
        try:
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        "random_seed: 7",
                        "",
                        "baseline:",
                        "  logistic_regression:",
                        "    booster: gblinear",
                        "    n_estimators: 120",
                        "    learning_rate: 0.1",
                        "    reg_alpha: 0.0",
                        "    reg_lambda: 1.0",
                        "    tree_method: hist",
                        "    device: cuda",
                        "    n_jobs: 1",
                        "  random_forest:",
                        "    n_estimators: 1",
                        "    num_parallel_tree: 80",
                        "    max_depth: 18",
                        "    learning_rate: 1.0",
                        "    subsample: 0.8",
                        "    colsample_bynode: 0.8",
                        "    min_child_weight: 2.0",
                        "    tree_method: hist",
                        "    device: cuda",
                        "    n_jobs: 1",
                        "  xgboost:",
                        "    n_estimators: 200",
                        "    max_depth: 6",
                        "    learning_rate: 0.1",
                        "    subsample: 0.8",
                        "    colsample_bytree: 0.8",
                        "    tree_method: hist",
                        "    device: cuda",
                        "    n_jobs: 1",
                        "  hist_gradient_boosting:",
                        "    n_estimators: 200",
                        "    learning_rate: 0.1",
                        "    grow_policy: lossguide",
                        "    max_leaves: 31",
                        "    max_depth: 0",
                        "    min_child_weight: 1.0",
                        "    reg_lambda: 1.0",
                        "    tree_method: hist",
                        "    device: cuda",
                        "    n_jobs: 1",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_training_config(path)
            specs = {
                spec.name: spec
                for spec in build_baseline_model_specs(
                    pd.Series([0, 0, 1]),
                    config=config,
                )
            }
        finally:
            if path.exists():
                path.unlink()
            if root.exists():
                root.rmdir()

        self.assertEqual(
            specs["logistic_regression"].estimator.random_state,
            7,
        )
        self.assertEqual(specs["random_forest"].estimator.random_state, 7)
        self.assertEqual(specs["xgboost"].estimator.random_state, 7)
        self.assertEqual(specs["xgboost"].estimator.device, "cuda")
        self.assertEqual(specs["hist_gradient_boosting"].estimator.random_state, 7)


if __name__ == "__main__":
    unittest.main()
