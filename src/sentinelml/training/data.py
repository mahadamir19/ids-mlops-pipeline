"""Data loading helpers for Phase 2 baseline training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelml.data.config import PROJECT_ROOT

DEFAULT_FEATURE_SCHEMA_PATH = PROJECT_ROOT / "reports" / "data" / "feature_schema.json"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference.parquet"


@dataclass(frozen=True)
class DatasetBundle:
    """Feature matrix, target vector, and lightweight provenance for one partition."""

    features: pd.DataFrame
    target: pd.Series
    metadata: dict[str, Any]


def load_feature_schema(path: Path = DEFAULT_FEATURE_SCHEMA_PATH) -> dict[str, Any]:
    """Load the Phase 1 feature schema instead of inferring model columns."""

    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    required = {"feature_columns", "target_column", "excluded_from_model_features"}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"feature schema is missing required fields: {missing}")
    forbidden = set(schema["excluded_from_model_features"])
    leaked = sorted(set(schema["feature_columns"]).intersection(forbidden))
    if leaked:
        raise ValueError(f"feature schema contains non-feature columns: {leaked}")
    return schema


def deterministic_stratified_sample(
    frame: pd.DataFrame,
    *,
    target_column: str,
    sample_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Sample a deterministic class-aware subset while preserving both classes."""

    if sample_size is None or len(frame) <= sample_size:
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2 for binary smoke sampling")

    counts = frame[target_column].value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError(
            f"expected both binary classes in {target_column}, "
            f"observed {counts.to_dict()}"
        )

    allocations: dict[int, int] = {}
    for label, count in counts.items():
        proportional = int(sample_size * (int(count) / len(frame)))
        allocations[int(label)] = max(1, min(int(count), proportional))

    while sum(allocations.values()) < sample_size:
        room = {
            int(label): int(count) - allocations[int(label)]
            for label, count in counts.items()
        }
        label = max(room, key=room.get)
        if room[label] <= 0:
            break
        allocations[label] += 1

    while sum(allocations.values()) > sample_size:
        label = max(allocations, key=allocations.get)
        if allocations[label] <= 1:
            break
        allocations[label] -= 1

    samples = []
    for label, n_rows in allocations.items():
        class_rows = frame.loc[frame[target_column] == label]
        samples.append(class_rows.sample(n=n_rows, random_state=seed + label))
    return (
        pd.concat(samples, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def load_partition(
    path: Path,
    *,
    feature_columns: list[str],
    target_column: str,
    sample_size: int | None = None,
    seed: int = 42,
) -> DatasetBundle:
    """Load one Phase 1 partition using only approved model feature columns."""

    columns = [target_column, *feature_columns]
    frame = pd.read_parquet(path, columns=columns)
    original_rows = len(frame)
    frame = deterministic_stratified_sample(
        frame,
        target_column=target_column,
        sample_size=sample_size,
        seed=seed,
    )
    target = frame[target_column].astype("int8")
    features = frame[feature_columns].astype("float32")
    metadata = {
        "path": str(path),
        "original_rows": int(original_rows),
        "rows_used": int(len(frame)),
        "sample_size_requested": sample_size,
        "target_distribution": {
            str(key): int(value)
            for key, value in target.value_counts().sort_index().items()
        },
    }
    return DatasetBundle(features=features, target=target, metadata=metadata)
