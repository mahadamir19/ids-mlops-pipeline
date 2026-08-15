"""Deterministic source data handling for Phase 6 simulation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinelml.simulation.config import reject_test_source
from sentinelml.tracking.mlflow import sha256_file


@dataclass(frozen=True)
class SimulationSource:
    path: Path
    fingerprint: str
    row_count: int
    target_distribution: dict[str, int]
    provenance: str


@dataclass(frozen=True)
class SimulationRecord:
    features: dict[str, float]
    true_label: int
    row_fingerprint: str


def load_simulation_frame(
    path: Path,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, SimulationSource]:
    reject_test_source(path)
    columns = [*feature_columns, target_column]
    optional = ["row_digest", "source_file", "source_day", "source_partition"]
    available_columns = pd.read_parquet(path, engine="pyarrow").columns
    for column in optional:
        if column in available_columns:
            columns.append(column)
    frame = pd.read_parquet(path, columns=columns)
    target = frame[target_column].astype(int)
    source = SimulationSource(
        path=path,
        fingerprint=sha256_file(path),
        row_count=int(len(frame)),
        target_distribution={
            str(key): int(value)
            for key, value in target.value_counts().sort_index().items()
        },
        provenance=(
            "Phase 1 reference partition derived from non-test train days "
            "(Monday, Tuesday, Wednesday)."
        ),
    )
    return frame, source


def deterministic_sample(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    request_count: int,
    seed: int,
    attack_fraction: float | None = None,
) -> list[SimulationRecord]:
    if request_count < 1:
        raise ValueError("request_count must be at least 1")
    if attack_fraction is None:
        sampled = frame.sample(
            n=request_count,
            replace=len(frame) < request_count,
            random_state=seed,
        )
    else:
        if not 0.0 <= attack_fraction <= 1.0:
            raise ValueError("attack_fraction must be between 0 and 1")
        attack_count = int(round(request_count * attack_fraction))
        benign_count = request_count - attack_count
        rng = np.random.default_rng(seed)
        sampled_parts = []
        for label, count in [(0, benign_count), (1, attack_count)]:
            if count == 0:
                continue
            label_frame = frame.loc[frame[target_column] == label]
            if label_frame.empty:
                raise ValueError(f"source data does not contain target label {label}")
            indices = rng.choice(
                label_frame.index.to_numpy(),
                size=count,
                replace=len(label_frame) < count,
            )
            sampled_parts.append(label_frame.loc[indices])
        sampled = (
            pd.concat(sampled_parts, axis=0)
            .sample(frac=1.0, random_state=seed + 17)
            .reset_index(drop=False)
        )

    records: list[SimulationRecord] = []
    for index, row in sampled.reset_index(drop=False).iterrows():
        features = {column: float(row[column]) for column in feature_columns}
        label = int(row[target_column])
        records.append(
            SimulationRecord(
                features=features,
                true_label=label,
                row_fingerprint=row_fingerprint(row, features, label, index=index),
            )
        )
    return records


def row_fingerprint(
    row: pd.Series,
    features: dict[str, float],
    true_label: int,
    *,
    index: int,
) -> str:
    if "row_digest" in row and pd.notna(row["row_digest"]):
        return str(row["row_digest"])
    payload: dict[str, Any] = {
        "index": int(index),
        "features": features,
        "target": int(true_label),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
