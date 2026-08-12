"""Generate clean Phase 1 model-ready datasets from immutable raw CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.data.preprocess import generate_phase1_datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean Phase 1 model-ready datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a Phase 1 data config JSON file.",
    )
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=None,
        help="Optional path to the binary label mapping JSON file.",
    )
    args = parser.parse_args()
    manifest = generate_phase1_datasets(
        config_path=args.config,
        label_mapping_path=args.label_mapping,
    )
    print("Phase 1 data foundation complete")
    for partition, stats in manifest["partitions"].items():
        counts = stats["target_distribution"]
        print(
            f"{partition}: {stats['row_count']:,} rows "
            f"(target 0={counts.get('0', 0):,}, target 1={counts.get('1', 0):,})"
        )


if __name__ == "__main__":
    main()
