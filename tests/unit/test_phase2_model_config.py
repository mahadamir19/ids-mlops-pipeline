from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
            },
        )
        self.assertEqual(config["baseline"]["random_forest"]["n_estimators"], 80)
        self.assertEqual(config["baseline"]["xgboost"]["tree_method"], "hist")
        self.assertEqual(config["baseline"]["xgboost"]["device"], "cpu")
        self.assertEqual(
            config["baseline"]["hist_gradient_boosting"]["max_leaf_nodes"],
            31,
        )

    def test_builds_all_four_true_model_families_from_config(self) -> None:
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
        self.assertIsInstance(logistic, Pipeline)
        self.assertIsInstance(logistic.named_steps["classifier"], LogisticRegression)
        self.assertEqual(logistic.named_steps["classifier"].class_weight, "balanced")
        self.assertEqual(logistic.named_steps["classifier"].random_state, 42)
        self.assertEqual(specs["logistic_regression"].parameters["pipeline"], ["StandardScaler", "LogisticRegression"])

        random_forest = specs["random_forest"].estimator
        self.assertIsInstance(random_forest, RandomForestClassifier)
        self.assertEqual(random_forest.n_estimators, 80)
        self.assertEqual(random_forest.max_depth, 18)
        self.assertEqual(random_forest.class_weight, "balanced_subsample")
        self.assertEqual(random_forest.n_jobs, 1)
        self.assertEqual(random_forest.random_state, 42)

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
        self.assertEqual(xgboost.device, "cpu")
        self.assertEqual(xgboost.n_jobs, 1)
        self.assertEqual(xgboost.random_state, 42)
        self.assertEqual(xgboost.scale_pos_weight, 6.5)
        self.assertEqual(xgboost.verbosity, 0)

        hist = specs["hist_gradient_boosting"].estimator
        self.assertIsInstance(hist, HistGradientBoostingClassifier)
        self.assertEqual(hist.max_iter, 200)
        self.assertEqual(hist.learning_rate, 0.1)
        self.assertEqual(hist.max_leaf_nodes, 31)
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
                        "    C: 0.5",
                        "    penalty: l2",
                        "    solver: lbfgs",
                        "    class_weight: balanced",
                        "    max_iter: 300",
                        "  random_forest:",
                        "    n_estimators: 80",
                        "    max_depth: 18",
                        "    class_weight: balanced_subsample",
                        "    n_jobs: 1",
                        "  xgboost:",
                        "    n_estimators: 200",
                        "    max_depth: 6",
                        "    learning_rate: 0.1",
                        "    subsample: 0.8",
                        "    colsample_bytree: 0.8",
                        "    tree_method: hist",
                        "    device: cpu",
                        "    n_jobs: 1",
                        "  hist_gradient_boosting:",
                        "    max_iter: 200",
                        "    learning_rate: 0.1",
                        "    max_leaf_nodes: 31",
                        "    class_weight: balanced",
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
            specs["logistic_regression"].estimator.named_steps[
                "classifier"
            ].random_state,
            7,
        )
        self.assertEqual(specs["random_forest"].estimator.random_state, 7)
        self.assertEqual(specs["xgboost"].estimator.random_state, 7)
        self.assertEqual(specs["xgboost"].estimator.device, "cpu")
        self.assertEqual(specs["hist_gradient_boosting"].estimator.random_state, 7)


if __name__ == "__main__":
    unittest.main()
