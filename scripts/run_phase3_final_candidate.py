"""Run Phase 3E final-candidate training."""

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

from sentinelml.final_candidate.selection import DEFAULT_OPTIMIZATION_REPORT_DIR
from sentinelml.final_candidate.training import (
    DEFAULT_FINAL_CANDIDATE_OUTPUT_DIR,
    run_phase3_final_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and track the Phase 3E final candidate."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Mode must match the consumed Phase 3D optimization outputs.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Track the final candidate in the sentinelml-final-training experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FINAL_CANDIDATE_OUTPUT_DIR,
        help="Directory for final-candidate reports.",
    )
    parser.add_argument(
        "--optimization-report-dir",
        type=Path,
        default=DEFAULT_OPTIMIZATION_REPORT_DIR,
        help="Directory containing Phase 3D optimization reports.",
    )
    args = parser.parse_args()

    manifest = run_phase3_final_candidate(
        mode=args.mode,
        enable_mlflow=args.mlflow,
        output_dir=args.output_dir,
        optimization_report_dir=args.optimization_report_dir,
    )
    source = manifest["source_optimization"]
    print(f"Phase 3E {args.mode} final-candidate workflow complete")
    print(f"Selected model family: {manifest['model_family']}")
    print(
        f"Source trial: {source['study_name']} trial "
        f"{source['best_trial_number']} ({source['best_trial_mlflow_run_id']})"
    )
    print(f"Logged model URI: {manifest['mlflow']['logged_model_uri']}")


if __name__ == "__main__":
    main()
