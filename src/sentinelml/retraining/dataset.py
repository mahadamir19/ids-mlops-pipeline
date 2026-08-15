"""Build Phase 8 retraining datasets from TRAIN plus approved production data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelml.retraining.config import RetrainingConfig
from sentinelml.tracking.mlflow import sha256_file
from sentinelml.training.data import (
    DatasetBundle,
    deterministic_stratified_sample,
    load_feature_schema,
)


class RetrainingDatasetError(RuntimeError):
    """Raised when a retraining dataset cannot be safely constructed."""


@dataclass(frozen=True)
class RetrainingDataset:
    bundle: DatasetBundle
    manifest: dict[str, Any]
    consumed_prediction_ids: list[str]
    manifest_path: Path


def build_retraining_dataset(
    *,
    retraining_run_id: str,
    config: RetrainingConfig,
    approved_observations: list[dict[str, Any]],
    consumed_prediction_ids: set[str],
    output_dir: Path,
) -> RetrainingDataset:
    """Build a smoke-safe dataset using only Phase 1 TRAIN and new approved rows."""

    _reject_test_path(config.historical_source)
    schema = load_feature_schema(config.feature_schema_path)
    feature_columns = list(schema["feature_columns"])
    target_column = str(schema["target_column"])
    historical = _load_historical_train(
        config,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    production_frame, production_ids, production_meta = _production_frame(
        approved_observations,
        consumed_prediction_ids=consumed_prediction_ids,
        feature_columns=feature_columns,
        target_column=target_column,
        feature_schema_fingerprint=sha256_file(config.feature_schema_path),
        limit=(
            config.max_production_rows_smoke
            if config.execution_mode == "smoke"
            else None
        ),
    )

    historical_frame = pd.concat(
        [historical.features, historical.target.rename(target_column)],
        axis=1,
    )
    combined, duplicate_count = _deduplicate(
        historical_frame,
        production_frame,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    if len(combined) < 2 or set(combined[target_column].unique()) != {0, 1}:
        raise RetrainingDatasetError(
            "retraining dataset must contain at least one BENIGN and one ATTACK row"
        )

    dataset_fingerprint = dataframe_fingerprint(
        combined,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    target = combined[target_column].astype("int8").reset_index(drop=True)
    features = combined[feature_columns].astype("float32").reset_index(drop=True)
    class_distribution = {
        str(key): int(value)
        for key, value in target.value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": "1.0",
        "retraining_run_id": retraining_run_id,
        "execution_mode": config.execution_mode,
        "historical": {
            "source": str(config.historical_source),
            "source_fingerprint": sha256_file(config.historical_source),
            "row_count_before_sampling": historical.metadata["original_rows"],
            "row_count_after_sampling": historical.metadata["rows_used"],
            "provenance": "Phase 1 TRAIN partition only",
        },
        "production": production_meta,
        "deduplication": {
            "algorithm": config.deduplication,
            "cross_source_duplicates_removed": int(duplicate_count),
        },
        "final_row_count": int(len(combined)),
        "class_distribution": class_distribution,
        "feature_schema_path": str(config.feature_schema_path),
        "feature_schema_fingerprint": sha256_file(config.feature_schema_path),
        "feature_order": feature_columns,
        "target_column": target_column,
        "sampling_seed": config.sampling_seed,
        "dataset_fingerprint": dataset_fingerprint,
        "excluded_partitions": [
            "data/processed/validation.parquet",
            "data/processed/test.parquet",
            "data/reference/reference.parquet",
        ],
    }
    manifest_path = output_dir / "dataset_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bundle = DatasetBundle(
        features=features,
        target=target,
        metadata={
            "dataset_fingerprint": dataset_fingerprint,
            "rows_used": int(len(combined)),
            "target_distribution": class_distribution,
            "dataset_manifest_path": str(manifest_path),
        },
    )
    return RetrainingDataset(
        bundle=bundle,
        manifest=manifest,
        consumed_prediction_ids=production_ids,
        manifest_path=manifest_path,
    )


def dataframe_fingerprint(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> str:
    digest = hashlib.sha256()
    for _, row in frame.reset_index(drop=True).iterrows():
        digest.update(
            row_fingerprint(
                {feature: row[feature] for feature in feature_columns},
                int(row[target_column]),
                feature_columns=feature_columns,
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def row_fingerprint(
    features: dict[str, Any],
    target: int,
    *,
    feature_columns: list[str],
) -> str:
    payload = {
        "features": {feature: float(features[feature]) for feature in feature_columns},
        "target": int(target),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _load_historical_train(
    config: RetrainingConfig,
    *,
    feature_columns: list[str],
    target_column: str,
) -> DatasetBundle:
    columns = [target_column, *feature_columns]
    frame = pd.read_parquet(config.historical_source, columns=columns)
    original_rows = int(len(frame))
    sample_size = (
        config.max_historical_rows_smoke if config.execution_mode == "smoke" else None
    )
    sampled = deterministic_stratified_sample(
        frame,
        target_column=target_column,
        sample_size=sample_size,
        seed=config.sampling_seed,
    )
    return DatasetBundle(
        features=sampled[feature_columns].astype("float32").reset_index(drop=True),
        target=sampled[target_column].astype("int8").reset_index(drop=True),
        metadata={
            "path": str(config.historical_source),
            "original_rows": original_rows,
            "rows_used": int(len(sampled)),
            "sample_size_requested": sample_size,
        },
    )


def _production_frame(
    observations: list[dict[str, Any]],
    *,
    consumed_prediction_ids: set[str],
    feature_columns: list[str],
    target_column: str,
    feature_schema_fingerprint: str,
    limit: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_ids: list[str] = []
    skipped_consumed = 0
    for observation in observations:
        prediction_id = str(observation["prediction_id"])
        if prediction_id in consumed_prediction_ids:
            skipped_consumed += 1
            continue
        if observation.get("validation_status") != "approved":
            continue
        if observation.get("schema_fingerprint") != feature_schema_fingerprint:
            raise RetrainingDatasetError(
                f"approved production row {prediction_id} has schema mismatch"
            )
        features = _validate_features(observation.get("features"), feature_columns)
        target = int(observation["ground_truth"])
        if target not in {0, 1}:
            raise RetrainingDatasetError("production ground_truth must be 0 or 1")
        rows.append({**features, target_column: target})
        prediction_ids.append(prediction_id)
        if limit is not None and len(rows) >= limit:
            break
    frame = pd.DataFrame(rows, columns=[*feature_columns, target_column])
    meta = {
        "row_count_available": int(len(observations)),
        "row_count_skipped_already_consumed": int(skipped_consumed),
        "row_count_after_validation": int(len(frame)),
        "row_count_after_sampling": int(len(frame)),
        "source": "Phase 6 get_approved_production_observations boundary",
        "consumed_prediction_ids": prediction_ids,
    }
    return frame, prediction_ids, meta


def _deduplicate(
    historical: pd.DataFrame,
    production: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, int]:
    seen = {
        row_fingerprint(
            {feature: row[feature] for feature in feature_columns},
            int(row[target_column]),
            feature_columns=feature_columns,
        )
        for _, row in historical.iterrows()
    }
    retained_production: list[pd.Series] = []
    duplicates = 0
    for _, row in production.iterrows():
        fingerprint = row_fingerprint(
            {feature: row[feature] for feature in feature_columns},
            int(row[target_column]),
            feature_columns=feature_columns,
        )
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        retained_production.append(row)
    if retained_production:
        production_deduped = pd.DataFrame(retained_production)
        combined = pd.concat([historical, production_deduped], ignore_index=True)
    else:
        combined = historical.reset_index(drop=True)
    return combined.reset_index(drop=True), duplicates


def _validate_features(payload: Any, feature_columns: list[str]) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise RetrainingDatasetError("production features must be a JSON object")
    expected = set(feature_columns)
    actual = set(payload)
    if actual != expected:
        raise RetrainingDatasetError(
            "production feature schema mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    ordered = {}
    for feature in feature_columns:
        value = payload[feature]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RetrainingDatasetError(
                f"production feature is not numeric: {feature}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RetrainingDatasetError(f"production feature is not finite: {feature}")
        ordered[feature] = numeric
    return ordered


def _reject_test_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if normalized.endswith("data/processed/test.parquet"):
        raise RetrainingDatasetError("Phase 8 retraining must never consult TEST")
