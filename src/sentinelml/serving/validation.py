"""Strict request validation against the Phase 1 feature schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelml.tracking.mlflow import sha256_file
from sentinelml.training.data import load_feature_schema


@dataclass(frozen=True)
class FeatureSchema:
    feature_columns: list[str]
    feature_count: int
    target_column: str
    fingerprint: str
    source_path: Path
    generated_at_utc: str | None = None


class FeatureValidationError(ValueError):
    """Raised when an inference request violates the serving feature schema."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.details = details or {}

    def to_error_payload(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": str(self),
            **self.details,
        }


def load_serving_feature_schema(path: Path) -> FeatureSchema:
    raw = load_feature_schema(path)
    feature_columns = [str(column) for column in raw["feature_columns"]]
    return FeatureSchema(
        feature_columns=feature_columns,
        feature_count=len(feature_columns),
        target_column=str(raw["target_column"]),
        fingerprint=sha256_file(path),
        source_path=path,
        generated_at_utc=raw.get("generated_at_utc"),
    )


def validate_feature_record(
    record: Any,
    schema: FeatureSchema,
) -> dict[str, float]:
    if not isinstance(record, dict):
        raise FeatureValidationError(
            "prediction input must be a JSON object",
            category="invalid_record",
            details={"expected": "object"},
        )

    expected = set(schema.feature_columns)
    actual = set(record)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    invalid_fields: list[dict[str, str]] = []

    for field in schema.feature_columns:
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, bool):
            invalid_fields.append({"field": field, "reason": "boolean_not_numeric"})
            continue
        if not isinstance(value, int | float):
            invalid_fields.append({"field": field, "reason": "not_numeric"})
            continue
        if not math.isfinite(float(value)):
            invalid_fields.append({"field": field, "reason": "not_finite"})

    if missing or unexpected or invalid_fields:
        category = "schema_mismatch" if missing or unexpected else "invalid_value"
        raise FeatureValidationError(
            "prediction input does not match the serving feature schema",
            category=category,
            details={
                "missing_fields": missing,
                "unexpected_fields": unexpected,
                "invalid_fields": invalid_fields,
            },
        )

    return {field: float(record[field]) for field in schema.feature_columns}


def validate_feature_batch(
    records: Any,
    schema: FeatureSchema,
    *,
    max_batch_size: int,
) -> list[dict[str, float]]:
    if not isinstance(records, list):
        raise FeatureValidationError(
            "batch prediction input must be a JSON array",
            category="invalid_batch",
            details={"expected": "array"},
        )
    if not records:
        raise FeatureValidationError(
            "batch prediction input must contain at least one record",
            category="empty_batch",
            details={"min_batch_size": 1},
        )
    if len(records) > max_batch_size:
        raise FeatureValidationError(
            "batch prediction input exceeds configured maximum batch size",
            category="batch_too_large",
            details={
                "max_batch_size": max_batch_size,
                "actual_batch_size": len(records),
            },
        )

    ordered: list[dict[str, float]] = []
    row_errors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            ordered.append(validate_feature_record(record, schema))
        except FeatureValidationError as exc:
            row_errors.append({"index": index, "error": exc.to_error_payload()})

    if row_errors:
        raise FeatureValidationError(
            "one or more batch records do not match the serving feature schema",
            category="batch_validation_failed",
            details={"invalid_rows": row_errors},
        )
    return ordered
