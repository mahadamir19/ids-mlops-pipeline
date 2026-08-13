"""Run Phase 3D Optuna hyperparameter optimization."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.optimization.config import DEFAULT_OPTIMIZATION_CONFIG_PATH
from sentinelml.optimization.study import (
    DEFAULT_OPTIMIZATION_OUTPUT_DIR,
    run_phase3_optimization,
)
from sentinelml.training.models import DEFAULT_TRAINING_CONFIG_PATH, MODEL_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3D Optuna optimization studies."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help=(
            "Use smoke for cheap deterministic studies; full uses configured "
            "larger samples."
        ),
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_FAMILIES, "all"],
        default="all",
        help="Model family to optimize, or all enabled families.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OPTIMIZATION_OUTPUT_DIR,
        help="Directory for optimization reports.",
    )
    parser.add_argument(
        "--optimization-config",
        type=Path,
        default=DEFAULT_OPTIMIZATION_CONFIG_PATH,
        help="Path to the Phase 3D optimization configuration YAML.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Path to the Phase 2 training configuration YAML.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help=(
            "Track studies and nested trials in the sentinelml-optimization "
            "experiment."
        ),
    )
    args = parser.parse_args()

    summary = run_phase3_optimization(
        mode=args.mode,
        model=args.model,
        enable_mlflow=args.mlflow,
        output_dir=args.output_dir,
        optimization_config_path=args.optimization_config,
        training_config_path=args.training_config,
    )
    print(f"Phase 3D {args.mode} optimization workflow complete")
    for row in summary["validation_only_comparison"]:
        print(
            f"{row['model_family']}: best_{summary['objective_metric']}="
            f"{row['best_objective_value']:.6f}, "
            f"trial={row['best_trial_number']}, "
            f"mlflow_run_id={row['best_mlflow_run_id']}"
        )


if __name__ == "__main__":
    main()
