"""Reference and current-window data access for monitoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from sentinelml.monitoring.config import MonitoringConfig
from sentinelml.serving.validation import FeatureSchema, load_serving_feature_schema
from sentinelml.tracking.mlflow import sha256_file


class MonitoringDataError(RuntimeError):
    """Raised when monitoring inputs cannot be loaded or validated."""


@dataclass(frozen=True)
class ReferenceDataset:
    frame: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CurrentWindow:
    frame: pd.DataFrame
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def load_reference_dataset(
    config: MonitoringConfig,
    schema: FeatureSchema | None = None,
) -> ReferenceDataset:
    schema = schema or load_serving_feature_schema(config.feature_schema_path)
    reject_test_reference(config.reference_path)
    columns = list(dict.fromkeys([*config.monitored_features, schema.target_column]))
    available = pd.read_parquet(config.reference_path, engine="pyarrow").columns
    missing = sorted(set(config.monitored_features) - set(available))
    if missing:
        raise MonitoringDataError(f"reference data is missing features: {missing}")
    readable_columns = [column for column in columns if column in available]
    raw = pd.read_parquet(
        config.reference_path,
        columns=readable_columns,
        engine="pyarrow",
    )
    row_count = int(len(raw))
    selected = raw
    if config.reference_sample_size and len(raw) > config.reference_sample_size:
        selected = raw.sample(
            n=config.reference_sample_size,
            random_state=config.reference_sample_seed,
        )
    frame = selected.loc[:, config.monitored_features].reset_index(drop=True)
    validate_numeric_frame(frame, config.monitored_features, source="reference")
    metadata = {
        "path": str(config.reference_path),
        "fingerprint": sha256_file(config.reference_path),
        "row_count": row_count,
        "selected_row_count": int(len(frame)),
        "sample_seed": config.reference_sample_seed,
        "sample_size": config.reference_sample_size,
        "schema_fingerprint": schema.fingerprint,
        "provenance": (
            "Phase 1 non-test reference partition at data/reference/reference.parquet"
        ),
    }
    return ReferenceDataset(frame=frame, metadata=metadata)


def load_current_window(
    config: MonitoringConfig,
    schema: FeatureSchema | None = None,
) -> CurrentWindow:
    schema = schema or load_serving_feature_schema(config.feature_schema_path)
    if not config.database_url:
        raise MonitoringDataError(f"{config.database_url_env} is not configured")
    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,
        connect_args=_connect_args(config),
    )
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT prediction_id, timestamp, model_name, model_version,
                           model_family, execution_mode, demo_model, features,
                           prediction, probability, latency_ms, ground_truth,
                           ground_truth_received_at
                    FROM predictions
                    ORDER BY timestamp DESC, prediction_id DESC
                    LIMIT :limit
                    """
                ),
                {"limit": int(config.window_size)},
            ).mappings().all()
    finally:
        engine.dispose()
    decoded = [_decode_row(dict(row)) for row in rows]
    decoded = list(reversed(decoded))
    records = [
        validate_feature_payload(row["features"], schema.feature_columns)
        for row in decoded
    ]
    frame = pd.DataFrame(records, columns=schema.feature_columns)
    if decoded:
        frame = frame.loc[:, config.monitored_features]
        validate_numeric_frame(frame, config.monitored_features, source="current")
    metadata = {
        "requested_window_size": config.window_size,
        "minimum_window_size": config.minimum_window_size,
        "actual_window_size": int(len(decoded)),
        "window_start_timestamp": decoded[0]["timestamp"] if decoded else None,
        "window_end_timestamp": decoded[-1]["timestamp"] if decoded else None,
        "schema_fingerprint": schema.fingerprint,
        "source": "PostgreSQL predictions table ordered by latest production timestamp",
    }
    return CurrentWindow(frame=frame, rows=decoded, metadata=metadata)


def validate_feature_payload(
    payload: Any,
    feature_columns: list[str],
) -> dict[str, float]:
    features = _decode_json(payload)
    if not isinstance(features, dict):
        raise MonitoringDataError("stored prediction features are not a JSON object")
    expected = set(feature_columns)
    actual = set(features)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise MonitoringDataError(
            "stored prediction feature schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordered: dict[str, float] = {}
    for feature in feature_columns:
        value = features[feature]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MonitoringDataError(f"stored feature is not numeric: {feature}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise MonitoringDataError(f"stored feature is not finite: {feature}")
        ordered[feature] = numeric
    return ordered


def validate_numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MonitoringDataError(f"{source} frame is missing features: {missing}")
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.isna().any():
            raise MonitoringDataError(
                f"{source} feature has non-numeric values: {column}"
            )
        if not series.map(math.isfinite).all():
            raise MonitoringDataError(
                f"{source} feature has non-finite values: {column}"
            )


def reject_test_reference(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if normalized.endswith("data/processed/test.parquet"):
        raise MonitoringDataError(
            "Phase 7 monitoring reference must not use data/processed/test.parquet"
        )


def _connect_args(config: MonitoringConfig) -> dict[str, Any]:
    if not config.database_url or not config.database_url.startswith("postgresql"):
        return {}
    return {
        "connect_timeout": config.database_connect_timeout_seconds,
        "options": f"-c statement_timeout={config.database_statement_timeout_ms}",
    }


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    row["features"] = _decode_json(row["features"])
    return {key: _jsonable_value(value) for key, value in row.items()}


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _jsonable_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
