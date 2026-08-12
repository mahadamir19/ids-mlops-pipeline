from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.training.data import deterministic_stratified_sample


class Phase2SamplingTests(unittest.TestCase):
    def test_smoke_sample_is_deterministic_and_keeps_both_classes(self) -> None:
        frame = pd.DataFrame(
            {
                "feature": range(100),
                "target": [0] * 97 + [1] * 3,
            }
        )

        first = deterministic_stratified_sample(
            frame,
            target_column="target",
            sample_size=10,
            seed=42,
        )
        second = deterministic_stratified_sample(
            frame,
            target_column="target",
            sample_size=10,
            seed=42,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(set(first["target"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
