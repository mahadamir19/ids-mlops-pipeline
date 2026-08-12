"""Raw and processed data validation for Phase 1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES_FILENAME = "feature_schema.json"
RAW_VALIDATION_FILENAME = "validation_raw.json"
PROCESSED_VALIDATION_FILENAME = "validation_processed.json"


@dataclass
class ValidationError(Exception):
    """Raised when data validation fails."""

    report: dict[str, Any]

    def __str__(self) -> str:
        failures = self.report.get("failures", [])
        return f"validation failed: {failures}"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def fail_if_needed(report: dict[str, Any], path: Path) -> None:
    write_report(path, report)
    if report.get("failures"):
        raise ValidationError(report)


def expected_labels(label_mapping: dict[str, Any]) -> set[str]:
    return set(label_mapping["known_original_labels"])


def validate_raw_report(
    *,
    raw_files: list[dict[str, Any]],
    expected_header: list[str],
    observed_headers: dict[str, list[str]],
    observed_labels: dict[str, int],
    label_mapping: dict[str, Any],
    row_length_mismatches: dict[str, int],
    duplicate_header_names: list[str],
    duplicate_trimmed_header_names: list[str],
    duplicate_column_equivalence: dict[str, Any],
    minimum_row_count: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    raw_filenames = [item["source_filename"] for item in raw_files]
    expected_filenames = {item["source_filename"] for item in raw_files}
    observed_filenames = set(observed_headers)

    missing_files = sorted(expected_filenames - observed_filenames)
    unexpected_files = sorted(observed_filenames - expected_filenames)
    if missing_files:
        failures.append(f"missing expected raw files: {missing_files}")
    if unexpected_files:
        failures.append(f"unexpected raw files: {unexpected_files}")
    for filename, header in observed_headers.items():
        if header != expected_header:
            failures.append(f"header mismatch in {filename}")
    unknown_labels = sorted(set(observed_labels) - expected_labels(label_mapping))
    if unknown_labels:
        failures.append(f"unexpected raw labels: {unknown_labels}")
    if any(count for count in row_length_mismatches.values()):
        failures.append(f"row length mismatches: {row_length_mismatches}")
    if not duplicate_header_names:
        failures.append("expected duplicate raw header name was not observed")
    else:
        warnings.append(f"raw duplicate header names allowed for preprocessing: {duplicate_header_names}")
    if duplicate_trimmed_header_names:
        warnings.append(
            "raw duplicate trimmed header names allowed only with explicit policy: "
            f"{duplicate_trimmed_header_names}"
        )
    if not duplicate_column_equivalence.get("equivalent", False):
        failures.append("configured duplicate Fwd Header Length columns are not equivalent")
    total_rows = sum(item.get("row_count_observed", 0) for item in raw_files)
    if total_rows < minimum_row_count:
        failures.append(f"raw row count below minimum: {total_rows} < {minimum_row_count}")

    return {
        "status": "failed" if failures else "passed",
        "validation_scope": "raw",
        "raw_files": raw_filenames,
        "observed_labels": observed_labels,
        "row_length_mismatches": row_length_mismatches,
        "duplicate_header_names": duplicate_header_names,
        "duplicate_trimmed_header_names": duplicate_trimmed_header_names,
        "duplicate_column_equivalence": duplicate_column_equivalence,
        "failures": failures,
        "warnings": warnings,
    }


def validate_processed_partitions(
    *,
    partition_paths: dict[str, Path],
    feature_columns: list[str],
    target_column: str,
    provenance_columns: list[str],
    minimum_row_count: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    partition_stats: dict[str, Any] = {}
    feature_schema: tuple[str, ...] | None = None
    partition_digests: dict[str, set[str]] = {}

    forbidden_features = {target_column, "original_label", *provenance_columns}
    leaked_features = sorted(forbidden_features.intersection(feature_columns))
    if leaked_features:
        failures.append(f"non-feature columns present in feature schema: {leaked_features}")

    for partition, path in partition_paths.items():
        if not path.exists():
            failures.append(f"missing processed partition: {path}")
            continue
        frame = pd.read_parquet(path)
        schema = tuple(feature_columns)
        if feature_schema is None:
            feature_schema = schema
        elif schema != feature_schema:
            failures.append(f"feature schema mismatch for {partition}")
        missing_columns = sorted(set(feature_columns + [target_column]) - set(frame.columns))
        if missing_columns:
            failures.append(f"{partition} missing columns: {missing_columns}")
            continue
        unexpected_targets = sorted(set(frame[target_column].dropna().unique()) - {0, 1})
        if unexpected_targets:
            failures.append(f"{partition} has unexpected target values: {unexpected_targets}")
        if len(frame) < minimum_row_count:
            failures.append(f"{partition} row count below minimum: {len(frame)}")
        feature_frame = frame[feature_columns]
        missing_count = int(feature_frame.isna().sum().sum())
        if missing_count:
            failures.append(f"{partition} has missing feature values: {missing_count}")
        numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = sorted(set(feature_columns) - set(numeric_columns))
        if non_numeric:
            failures.append(f"{partition} has non-numeric features: {non_numeric}")
        inf_count = int(np.isinf(feature_frame.to_numpy(dtype=float)).sum())
        if inf_count:
            failures.append(f"{partition} has infinite feature values: {inf_count}")
        feature_duplicate_count = int(
            frame.duplicated(subset=feature_columns + [target_column]).sum()
        )
        digest_duplicate_count = int(frame.duplicated(subset=["row_digest"]).sum())
        if digest_duplicate_count:
            failures.append(
                f"{partition} still has exact raw-observation duplicates: {digest_duplicate_count}"
            )
        if partition in {"train", "validation", "test"}:
            partition_digests[partition] = set(frame["row_digest"].astype(str))
        partition_stats[partition] = {
            "path": str(path),
            "row_count": int(len(frame)),
            "target_distribution": {
                str(key): int(value)
                for key, value in frame[target_column].value_counts().sort_index().items()
            },
            "source_files": sorted(frame["source_file"].astype(str).unique().tolist()),
            "missing_feature_values": missing_count,
            "infinite_feature_values": inf_count,
            "duplicate_raw_observation_rows": digest_duplicate_count,
            "duplicate_feature_target_rows": feature_duplicate_count,
        }

    partitions = ["train", "validation", "test"]
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            if left in partition_digests and right in partition_digests:
                overlap = partition_digests[left].intersection(partition_digests[right])
                if overlap:
                    failures.append(
                        f"exact raw-observation digests leak across {left}/{right}: {len(overlap)}"
                    )
    if "train" in partition_stats and "reference" in partition_stats:
        train_sources = set(partition_stats["train"]["source_files"])
        reference_sources = set(partition_stats["reference"]["source_files"])
        if not reference_sources.issubset(train_sources):
            failures.append("reference contains source files outside the training population")
    if not failures:
        warnings.append("Destination Port retained for baseline; review leakage risk later.")

    return {
        "status": "failed" if failures else "passed",
        "validation_scope": "processed",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "provenance_columns": provenance_columns,
        "partitions": partition_stats,
        "failures": failures,
        "warnings": warnings,
    }
