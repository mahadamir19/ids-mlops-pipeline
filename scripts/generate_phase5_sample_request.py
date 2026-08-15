"""Generate a deterministic valid Phase 5 prediction request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.training.data import (
    DEFAULT_VALIDATION_PATH,
    load_feature_schema,
)


def build_sample(schema_path: Path, validation_path: Path | None) -> dict[str, float]:
    schema = load_feature_schema(schema_path)
    feature_columns = schema["feature_columns"]
    if validation_path is not None and validation_path.exists():
        import pandas as pd

        frame = pd.read_parquet(validation_path, columns=feature_columns)
        if not frame.empty:
            row = frame.iloc[0][feature_columns].astype(float)
            return {feature: float(row[feature]) for feature in feature_columns}
    return {feature: 0.0 for feature in feature_columns}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a valid SentinelML Phase 5 JSON feature record."
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data" / "feature_schema.json",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Optional processed partition used to emit a realistic row.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sample = build_sample(args.schema, args.validation)
    encoded = json.dumps(sample, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
