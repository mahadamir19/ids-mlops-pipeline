"""Run Phase 2 baseline ML training and evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.training.baselines import run_phase2_baselines
from sentinelml.training.models import DEFAULT_TRAINING_CONFIG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Phase 2 baseline ML models."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help=(
            "Use smoke for quick deterministic subsets; full trains on Phase 1 "
            "partitions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report output directory.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Path to the Phase 2 training configuration YAML.",
    )
    parser.add_argument("--train-sample-size", type=int, default=20_000)
    parser.add_argument("--validation-sample-size", type=int, default=10_000)
    parser.add_argument("--test-sample-size", type=int, default=10_000)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help=(
            "Register the run in MLflow under the sentinelml-baselines experiment "
            "with one parent run and one nested child run per baseline model."
        ),
    )
    args = parser.parse_args()

    report = run_phase2_baselines(
        mode=args.mode,
        output_dir=args.output_dir,
        train_sample_size=args.train_sample_size,
        validation_sample_size=args.validation_sample_size,
        test_sample_size=args.test_sample_size,
        training_config_path=args.training_config,
        enable_mlflow=args.mlflow,
    )
    selected = report["selected"]["recommended_baseline"]
    print(f"Phase 2 {args.mode} baseline workflow complete")
    print(f"Recommended baseline: {selected}")
    for row in report["comparison"]:
        print(
            f"{row['model']}: attack_recall={row['attack_class_recall']:.4f}, "
            f"f1={row['f1']:.4f}, pr_auc={row['pr_auc']:.4f}, "
            f"train_s={row['training_time_seconds']:.2f}, "
            f"latency_ms_per_row={row['inference_latency_ms_per_row']:.6f}"
        )


if __name__ == "__main__":
    main()
