"""Production preprocessing for Phase 1 CICIDS2017 data foundation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sentinelml.data.config import (
    load_label_mapping,
    load_phase1_config,
    resolve_project_path,
)
from sentinelml.data.validate import (
    FEATURES_FILENAME,
    PROCESSED_VALIDATION_FILENAME,
    RAW_VALIDATION_FILENAME,
    fail_if_needed,
    validate_processed_partitions,
    validate_raw_report,
    write_report,
)

TARGET_COLUMN = "target"
ORIGINAL_LABEL_COLUMN = "original_label"
PROVENANCE_COLUMNS = [
    "source_file",
    "source_day",
    "source_partition",
    "source_row_number",
    "row_digest",
]
OUTPUT_COLUMNS = [*PROVENANCE_COLUMNS, ORIGINAL_LABEL_COLUMN, TARGET_COLUMN]
PARTITIONS = ("train", "validation", "test", "reference")


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot produce validated model-ready data."""


def canonicalize_column_name(name: str) -> str:
    cleaned = name.strip().replace("/", " per ")
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "unnamed"


def raw_file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_exact_header(path: Path) -> list[str]:
    with path.open("rb") as handle:
        return (
            handle.readline()
            .decode("utf-8-sig", errors="replace")
            .rstrip("\r\n")
            .split(",")
        )


def ordered_raw_paths(raw_dir: Path, eda_summary: dict[str, Any]) -> list[Path]:
    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
    files = eda_summary["files"]
    return [
        raw_dir / item["source_filename"]
        for item in sorted(
            files,
            key=lambda item: (
                day_order.get(item["day"], 99),
                item["source_filename"],
            ),
        )
    ]


def partition_for_day(day: str, config: dict[str, Any]) -> str:
    assignment = config["split_strategy"]["assignment"]
    for partition, days in assignment.items():
        if day in days:
            return partition
    raise PreprocessingError(f"no partition configured for day {day!r}")


def build_column_plan(
    *,
    original_columns: list[str],
    eda_summary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    constant_indices = {
        int(item["index"]) for item in eda_summary["overall"]["constant_features"]
    }
    duplicate_policy = config["duplicate_column_policy"]
    duplicate_drop_index = int(duplicate_policy["drop_original_index"])
    label_index = next(
        idx for idx, name in enumerate(original_columns) if name.strip().lower() == "label"
    )
    seen_names: Counter[str] = Counter()
    plan: list[dict[str, Any]] = []
    feature_columns: list[str] = []
    removed_features: list[str] = []

    for idx, original_name in enumerate(original_columns):
        trimmed = original_name.strip()
        base_name = canonicalize_column_name(trimmed)
        canonical = base_name
        if seen_names[base_name]:
            canonical = f"{base_name}_pos_{idx}"
        seen_names[base_name] += 1
        action = "keep"
        reason = "model feature"
        if idx == label_index:
            action = "target"
            reason = "original multiclass label mapped to binary target"
            canonical = ORIGINAL_LABEL_COLUMN
        elif idx == duplicate_drop_index:
            action = "drop"
            reason = "duplicate Fwd Header Length column dropped after value-equivalence check"
        elif idx in constant_indices:
            action = "drop"
            reason = "globally constant feature identified by Phase 1 EDA"
        if action == "keep":
            feature_columns.append(canonical)
        elif action == "drop":
            removed_features.append(canonical)
        plan.append(
            {
                "original_index": idx,
                "original_name": original_name,
                "trimmed_name": trimmed,
                "canonical_name": canonical,
                "action": action,
                "reason": reason,
            }
        )
    return plan, feature_columns, removed_features


def reference_selected(row_digest: str, config: dict[str, Any]) -> bool:
    modulus = int(config["reference_hash_modulus"])
    threshold = int(modulus * float(config["reference_fraction"]))
    value = int(hashlib.sha256(f"reference:{row_digest}".encode("ascii")).hexdigest()[:16], 16)
    return value % modulus < threshold


class PartitionWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self, empty_frame: pd.DataFrame) -> None:
        if self.writer is not None:
            self.writer.close()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(empty_frame, preserve_index=False)
        pq.write_table(table, self.path, compression="snappy")


def _normal_label_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def generate_phase1_datasets(
    *,
    config_path: Path | None = None,
    label_mapping_path: Path | None = None,
) -> dict[str, Any]:
    config = load_phase1_config(config_path) if config_path else load_phase1_config()
    label_mapping = (
        load_label_mapping(label_mapping_path) if label_mapping_path else load_label_mapping()
    )
    raw_dir = resolve_project_path(config["raw_data_dir"])
    eda_summary_path = resolve_project_path(config["eda_summary_path"])
    processed_dir = resolve_project_path(config["processed_dir"])
    reference_dir = resolve_project_path(config["reference_dir"])
    report_dir = resolve_project_path(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    eda_summary = json.loads(eda_summary_path.read_text(encoding="utf-8"))
    original_columns = [item["name"] for item in eda_summary["original_columns"]]
    raw_paths = ordered_raw_paths(raw_dir, eda_summary)
    missing_paths = [path for path in raw_paths if not path.exists()]
    if missing_paths:
        raise PreprocessingError(f"missing raw files: {missing_paths}")

    column_plan, feature_columns, removed_features = build_column_plan(
        original_columns=original_columns,
        eda_summary=eda_summary,
        config=config,
    )
    positional_names = [f"raw_{idx:03d}" for idx in range(len(original_columns))]
    label_index = next(
        idx for idx, name in enumerate(original_columns) if name.strip().lower() == "label"
    )
    label_positional = positional_names[label_index]
    keep_indices = [
        item["original_index"] for item in column_plan if item["action"] == "keep"
    ]
    keep_positional = [positional_names[idx] for idx in keep_indices]
    keep_canonical = [
        item["canonical_name"] for item in column_plan if item["action"] == "keep"
    ]
    duplicate_keep = int(config["duplicate_column_policy"]["keep_original_index"])
    duplicate_drop = int(config["duplicate_column_policy"]["drop_original_index"])
    duplicate_keep_pos = positional_names[duplicate_keep]
    duplicate_drop_pos = positional_names[duplicate_drop]
    known_labels = set(label_mapping["known_original_labels"])
    max_rows_per_file = config.get("max_rows_per_file")
    if max_rows_per_file is not None:
        max_rows_per_file = int(max_rows_per_file)

    output_paths = {
        "train": processed_dir / "train.parquet",
        "validation": processed_dir / "validation.parquet",
        "test": processed_dir / "test.parquet",
        "reference": reference_dir / "reference.parquet",
    }
    for output_path in output_paths.values():
        if output_path.exists():
            output_path.unlink()
    writers = {partition: PartitionWriter(path) for partition, path in output_paths.items()}
    empty_template = pd.DataFrame(
        {
            **{name: pd.Series(dtype="object") for name in PROVENANCE_COLUMNS},
            ORIGINAL_LABEL_COLUMN: pd.Series(dtype="object"),
            TARGET_COLUMN: pd.Series(dtype="int8"),
            **{name: pd.Series(dtype="float64") for name in feature_columns},
        }
    )[[*PROVENANCE_COLUMNS, ORIGINAL_LABEL_COLUMN, TARGET_COLUMN, *feature_columns]]

    raw_before = {path.name: raw_file_fingerprint(path) for path in raw_paths}
    observed_headers: dict[str, list[str]] = {}
    observed_labels: Counter[str] = Counter()
    row_length_mismatches: dict[str, int] = {}
    file_reports: dict[str, Any] = {}
    partition_counts: dict[str, Counter[str]] = {
        partition: Counter() for partition in PARTITIONS
    }
    row_removals: Counter[str] = Counter()
    row_removals_by_file: dict[str, Counter[str]] = defaultdict(Counter)
    seen_digests: set[str] = set()
    duplicate_equivalent = True
    duplicate_mismatch_examples: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        day, _ = raw_path.name.split("-", 1)
        day = day.capitalize()
        partition = partition_for_day(day, config)
        header = read_exact_header(raw_path)
        observed_headers[raw_path.name] = header
        row_length_mismatches[raw_path.name] = 0
        rows_seen = 0
        rows_written = 0
        with raw_path.open("rb") as raw_handle:
            raw_handle.readline()
            read_csv_kwargs: dict[str, Any] = {
                "header": 0,
                "names": positional_names,
                "dtype": str,
                "keep_default_na": False,
                "na_values": [],
                "chunksize": int(config["chunk_rows"]),
                "low_memory": False,
            }
            if max_rows_per_file is not None:
                read_csv_kwargs["nrows"] = max_rows_per_file
            chunk_iter = pd.read_csv(raw_path, **read_csv_kwargs)
            for chunk in chunk_iter:
                raw_lines = [
                    raw_handle.readline().rstrip(b"\r\n") for _ in range(len(chunk))
                ]
                rows_seen += len(chunk)
                source_row_numbers = np.arange(rows_seen - len(chunk) + 1, rows_seen + 1)
                labels = _normal_label_values(chunk[label_positional])
                observed_labels.update(labels.tolist())

                equivalent_mask = chunk[duplicate_keep_pos].eq(chunk[duplicate_drop_pos])
                if not bool(equivalent_mask.all()):
                    duplicate_equivalent = False
                    examples = chunk.loc[
                        ~equivalent_mask, [duplicate_keep_pos, duplicate_drop_pos]
                    ].head(5)
                    for offset, values in examples.iterrows():
                        duplicate_mismatch_examples.append(
                            {
                                "source_file": raw_path.name,
                                "source_row_number": int(
                                    source_row_numbers[chunk.index.get_loc(offset)]
                                ),
                                "keep_value": str(values[duplicate_keep_pos]),
                                "drop_value": str(values[duplicate_drop_pos]),
                            }
                        )

                raw_digest = pd.Series(
                    [hashlib.sha256(line).hexdigest() for line in raw_lines],
                    index=chunk.index,
                )
                duplicate_mask = raw_digest.isin(seen_digests) | raw_digest.duplicated()
                seen_digests.update(raw_digest.loc[~duplicate_mask].tolist())

                unknown_label_mask = ~labels.isin(known_labels)
                features = chunk[keep_positional].copy()
                features.columns = keep_canonical
                features = features.apply(pd.to_numeric, errors="coerce")
                values = features.to_numpy(dtype="float64", copy=False)
                finite_mask = np.isfinite(values).all(axis=1)
                invalid_numeric_mask = pd.Series(~finite_mask, index=chunk.index)
                keep_mask = ~(duplicate_mask | unknown_label_mask | invalid_numeric_mask)

                row_removals["duplicate_raw_observation"] += int(duplicate_mask.sum())
                row_removals["unexpected_label"] += int(unknown_label_mask.sum())
                row_removals["unresolved_missing_or_infinite_feature"] += int(
                    invalid_numeric_mask.sum()
                )
                row_removals_by_file[raw_path.name]["duplicate_raw_observation"] += int(
                    duplicate_mask.sum()
                )
                row_removals_by_file[raw_path.name]["unexpected_label"] += int(
                    unknown_label_mask.sum()
                )
                row_removals_by_file[raw_path.name][
                    "unresolved_missing_or_infinite_feature"
                ] += int(invalid_numeric_mask.sum())

                if not keep_mask.any():
                    continue
                kept = features.loc[keep_mask].reset_index(drop=True)
                kept_labels = labels.loc[keep_mask].reset_index(drop=True)
                kept_digests = raw_digest.loc[keep_mask].astype(str).reset_index(drop=True)
                kept_source_rows = source_row_numbers[keep_mask.to_numpy()]
                output = pd.DataFrame(
                    {
                        "source_file": raw_path.name,
                        "source_day": day,
                        "source_partition": partition,
                        "source_row_number": kept_source_rows.astype("int64"),
                        "row_digest": kept_digests,
                        ORIGINAL_LABEL_COLUMN: kept_labels,
                        TARGET_COLUMN: (kept_labels != "BENIGN").astype("int8"),
                    }
                )
                output = pd.concat([output, kept.astype("float64")], axis=1)
                output = output[
                    [
                        *PROVENANCE_COLUMNS,
                        ORIGINAL_LABEL_COLUMN,
                        TARGET_COLUMN,
                        *feature_columns,
                    ]
                ]
                writers[partition].write(output)
                rows_written += len(output)
                partition_counts[partition].update(
                    output[TARGET_COLUMN].astype(str).tolist()
                )

                if partition == "train":
                    reference_mask = kept_digests.map(
                        lambda value: reference_selected(value, config)
                    )
                    reference = output.loc[reference_mask.to_numpy()].copy()
                    if not reference.empty:
                        reference["source_partition"] = "reference"
                        writers["reference"].write(reference)
                        partition_counts["reference"].update(
                            reference[TARGET_COLUMN].astype(str).tolist()
                        )

        file_reports[raw_path.name] = {
            "source_day": day,
            "partition": partition,
            "rows_seen": int(rows_seen),
            "rows_written": int(rows_written),
            "rows_removed": {
                key: int(value) for key, value in row_removals_by_file[raw_path.name].items()
            },
        }
        print(
            f"{raw_path.name}: wrote {rows_written:,} clean rows to {partition}",
            flush=True,
        )

    for writer in writers.values():
        writer.close(empty_template)

    raw_after = {path.name: raw_file_fingerprint(path) for path in raw_paths}
    raw_unchanged = raw_before == raw_after
    if not raw_unchanged:
        raise PreprocessingError("raw file fingerprints changed during preprocessing")

    original_names = [item["name"] for item in eda_summary["original_columns"]]
    raw_report = validate_raw_report(
        raw_files=[
            {
                "source_filename": path.name,
                "row_count_observed": file_reports.get(path.name, {}).get("rows_seen", 0),
            }
            for path in raw_paths
        ],
        expected_header=original_names,
        observed_headers=observed_headers,
        observed_labels=dict(observed_labels),
        label_mapping=label_mapping,
        row_length_mismatches=row_length_mismatches,
        duplicate_header_names=eda_summary["duplicate_original_column_names"],
        duplicate_trimmed_header_names=eda_summary["duplicate_trimmed_column_names"],
        duplicate_column_equivalence={
            "name": config["duplicate_column_policy"]["trimmed_name"],
            "keep_original_index": duplicate_keep,
            "drop_original_index": duplicate_drop,
            "equivalent": duplicate_equivalent,
            "mismatch_examples": duplicate_mismatch_examples,
        },
    )
    fail_if_needed(raw_report, report_dir / RAW_VALIDATION_FILENAME)

    feature_schema = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "original_label_column": ORIGINAL_LABEL_COLUMN,
        "provenance_columns": PROVENANCE_COLUMNS,
        "excluded_from_model_features": [ORIGINAL_LABEL_COLUMN, TARGET_COLUMN, *PROVENANCE_COLUMNS],
        "removed_features": [
            item for item in column_plan if item["action"] == "drop"
        ],
        "retained_review_features": [
            {
                "feature": "destination_port",
                "reason": config["destination_port_policy"]["warning"],
            },
            *[
                {
                    "feature": item.get("trimmed_name", item["name"].strip()),
                    "reason": "near-constant feature retained for initial baseline",
                }
                for item in eda_summary["overall"]["near_constant_features"]
            ],
        ],
        "column_mapping": column_plan,
    }
    write_report(report_dir / FEATURES_FILENAME, feature_schema)

    processed_report = validate_processed_partitions(
        partition_paths=output_paths,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        provenance_columns=PROVENANCE_COLUMNS,
    )
    fail_if_needed(processed_report, report_dir / PROCESSED_VALIDATION_FILENAME)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "config_path": config["_config_path"],
        "label_mapping_path": label_mapping["_mapping_path"],
        "eda_summary_path": str(eda_summary_path),
        "output_format": "parquet",
        "outputs": {partition: str(path) for partition, path in output_paths.items()},
        "split_strategy": config["split_strategy"],
        "raw_files_unchanged": raw_unchanged,
        "raw_file_fingerprints_before": raw_before,
        "raw_file_fingerprints_after": raw_after,
        "row_removals": {key: int(value) for key, value in row_removals.items()},
        "files": file_reports,
        "partitions": processed_report["partitions"],
        "feature_schema_path": str(report_dir / FEATURES_FILENAME),
        "raw_validation_report": str(report_dir / RAW_VALIDATION_FILENAME),
        "processed_validation_report": str(report_dir / PROCESSED_VALIDATION_FILENAME),
        "preprocessing_decisions": [
            "Trimmed/canonicalized raw column names and stored original-to-canonical mapping.",
            "Verified duplicated Fwd Header Length columns are equivalent before dropping original index 55.",
            "Dropped globally constant features identified by Phase 1 EDA.",
            "Retained near-constant features for initial baseline.",
            "Retained Destination Port with documented leakage/scenario-identity warning.",
            "Dropped rows with unresolved missing or positive/negative infinite feature values.",
            "Removed exact duplicate raw observations globally before partition assignment.",
            "Mapped BENIGN to 0 and every known non-BENIGN label to 1.",
            "Preserved source-file/day provenance columns while excluding them from feature schema.",
        ],
    }
    write_report(report_dir / "preprocessing_report.json", manifest)
    write_markdown_report(manifest, feature_schema, report_dir / "phase1_data_foundation.md")
    return manifest


def write_markdown_report(
    manifest: dict[str, Any], feature_schema: dict[str, Any], path: Path
) -> None:
    def distribution(partition: str) -> str:
        stats = manifest["partitions"][partition]
        counts = stats["target_distribution"]
        benign = int(counts.get("0", 0))
        attack = int(counts.get("1", 0))
        return f"{stats['row_count']:,} rows; BENIGN={benign:,}, ATTACK={attack:,}"

    lines = [
        "# Phase 1 Data Foundation Report",
        "",
        f"Generated at `{manifest['generated_at_utc']}`.",
        "",
        "## Outputs",
        *[f"- `{partition}`: `{output}`" for partition, output in manifest["outputs"].items()],
        "",
        "## Row Counts and Binary Distributions",
        *[f"- {partition}: {distribution(partition)}" for partition in PARTITIONS],
        "",
        "## Preprocessing Decisions",
        *[f"- {item}" for item in manifest["preprocessing_decisions"]],
        "",
        "## Removed Features",
        *[
            f"- `{item['canonical_name']}` from raw index {item['original_index']} "
            f"(`{item['original_name']}`): {item['reason']}"
            for item in feature_schema["removed_features"]
        ],
        "",
        "## Retained Review Features",
        *[
            f"- `{item['feature']}`: {item['reason']}"
            for item in feature_schema["retained_review_features"]
        ],
        "",
        "## Final Feature List",
        *[f"- `{feature}`" for feature in feature_schema["feature_columns"]],
        "",
        "## Validation Reports",
        f"- Raw validation: `{manifest['raw_validation_report']}`",
        f"- Processed validation: `{manifest['processed_validation_report']}`",
        f"- Feature schema: `{manifest['feature_schema_path']}`",
        "",
        "## Warnings",
        "- The approved split intentionally leaves several attack families unseen in training; this is a temporal generalization trade-off, not a random IID benchmark.",
        "- Destination Port remains available for the first baseline but should be tested later as a leakage/scenario-identity ablation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
