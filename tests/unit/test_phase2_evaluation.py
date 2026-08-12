from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.training.compare import comparison_rows, select_best_baseline
from sentinelml.training.evaluate import evaluate_binary_classifier


class TinyProbabilityModel:
    def __init__(self, positive_scores: list[float]) -> None:
        self.positive_scores = np.asarray(positive_scores)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = self.positive_scores[: len(features)]
        return np.column_stack([1.0 - scores, scores])


class Phase2EvaluationTests(unittest.TestCase):
    def test_binary_metrics_include_confusion_rates_and_auc(self) -> None:
        features = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        target = pd.Series([0, 0, 1, 1])
        model = TinyProbabilityModel([0.1, 0.8, 0.7, 0.2])

        metrics = evaluate_binary_classifier(model, features, target)

        self.assertEqual(
            metrics["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertEqual(metrics["attack_class_recall"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.5)
        self.assertIn("pr_auc", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["inference_latency_ms_per_row"], 0.0)

    def test_comparison_selects_validation_behavior_not_accuracy(self) -> None:
        results = {
            "low_recall_high_accuracy": {
                "training_time_seconds": 1.0,
                "validation_metrics": {
                    "precision": 1.0,
                    "recall": 0.0,
                    "attack_class_recall": 0.0,
                    "f1": 0.0,
                    "pr_auc": 0.2,
                    "roc_auc": 0.5,
                    "false_positive_rate": 0.0,
                    "false_negative_rate": 1.0,
                    "inference_latency_ms_per_row": 0.01,
                },
            },
            "balanced_detector": {
                "training_time_seconds": 2.0,
                "validation_metrics": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "attack_class_recall": 0.75,
                    "f1": 0.77,
                    "pr_auc": 0.82,
                    "roc_auc": 0.9,
                    "false_positive_rate": 0.05,
                    "false_negative_rate": 0.25,
                    "inference_latency_ms_per_row": 0.02,
                },
            },
        }

        rows = comparison_rows(results)
        selected = select_best_baseline(rows)

        self.assertEqual(selected["recommended_baseline"], "balanced_detector")


if __name__ == "__main__":
    unittest.main()
