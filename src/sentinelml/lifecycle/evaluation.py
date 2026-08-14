"""Comparable lifecycle promotion evaluation on a canonical validation slice."""

from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any

import pandas as pd

from sentinelml.lifecycle.audit import utc_now
from sentinelml.tracking.mlflow import sha256_file
from sentinelml.training.data import DatasetBundle
from sentinelml.training.evaluate import evaluate_binary_classifier


def _target_counts(target: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in target.value_counts().sort_index().items()
    }


def _fingerprint_indices(indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{index}\n".encode())
    return digest.hexdigest()


def canonical_promotion_slice(
    frame: pd.DataFrame,
    *,
    target_column: str,
    sample_size: int,
    seed: int,
    min_positive_rows: int,
    max_positive_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select deterministic source-row positions for Phase 4 comparisons."""

    if sample_size < 2:
        raise ValueError("promotion validation sample_size must be at least 2")
    working = frame.copy()
    working["_sentinelml_source_row"] = range(len(working))
    source_counts = working[target_column].value_counts().sort_index()
    if set(source_counts.index) != {0, 1}:
        raise ValueError("promotion validation source must contain both classes")

    rows_to_select = min(sample_size, len(working))
    source_positive = int(source_counts.loc[1])
    natural_positive = round(rows_to_select * (source_positive / len(working)))
    positive_cap = max(1, int(rows_to_select * max_positive_fraction))
    positive_rows = min(
        source_positive,
        positive_cap,
        max(min_positive_rows, natural_positive),
    )
    positive_rows = min(positive_rows, rows_to_select - 1)
    negative_rows = rows_to_select - positive_rows

    positives = working.loc[working[target_column] == 1].sample(
        n=positive_rows,
        random_state=seed + 1,
    )
    negatives = working.loc[working[target_column] == 0].sample(
        n=negative_rows,
        random_state=seed,
    )
    selected = (
        pd.concat([negatives, positives], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    source_rows = [int(value) for value in selected["_sentinelml_source_row"].tolist()]
    metadata = {
        "source_row_count": int(len(working)),
        "selected_row_count": int(len(selected)),
        "benign_count": int(negative_rows),
        "attack_count": int(positive_rows),
        "target_distribution": _target_counts(selected[target_column]),
        "selected_row_fingerprint": _fingerprint_indices(source_rows),
        "selected_row_fingerprint_algorithm": (
            "sha256(newline-separated source row positions)"
        ),
        "sampling_strategy": "stratified_min_positive",
        "random_seed": int(seed),
        "min_positive_rows": int(min_positive_rows),
        "max_positive_fraction": float(max_positive_fraction),
        "created_at": utc_now(),
    }
    return selected.drop(columns=["_sentinelml_source_row"]), metadata


def load_promotion_validation_dataset(
    *,
    validation_path: Any,
    feature_schema: dict[str, Any],
    feature_schema_path: Any,
    config: dict[str, Any],
) -> DatasetBundle:
    policy = config["promotion_evaluation"]
    feature_columns = feature_schema["feature_columns"]
    target_column = feature_schema["target_column"]
    columns = [target_column, *feature_columns]
    frame = pd.read_parquet(validation_path, columns=columns)
    selected, metadata = canonical_promotion_slice(
        frame,
        target_column=target_column,
        sample_size=int(policy["validation_sample_size"]),
        seed=int(policy["random_seed"]) + int(policy.get("validation_seed_offset", 0)),
        min_positive_rows=int(policy["min_positive_rows"]),
        max_positive_fraction=float(policy["max_positive_fraction"]),
    )
    target = selected[target_column].astype("int8")
    features = selected[feature_columns].astype("float32")
    metadata.update(
        {
            "source_path": str(validation_path),
            "sample_size_requested": int(policy["validation_sample_size"]),
            "feature_schema_path": str(feature_schema_path),
            "feature_schema_sha256": sha256_file(feature_schema_path),
            "evaluation_type": "lifecycle_promotion_validation",
            "split": "validation",
        }
    )
    return DatasetBundle(features=features, target=target, metadata=metadata)


def evaluate_model_on_promotion_slice(
    *,
    model: Any,
    dataset: DatasetBundle,
) -> dict[str, Any]:
    start = perf_counter()
    metrics = evaluate_binary_classifier(model, dataset.features, dataset.target)
    metrics["promotion_evaluation_seconds"] = float(perf_counter() - start)
    return metrics
