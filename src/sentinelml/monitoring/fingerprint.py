"""Deterministic identity for Phase 7 monitoring inputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sentinelml.monitoring.config import MonitoringConfig
from sentinelml.monitoring.data import CurrentWindow, ReferenceDataset
from sentinelml.serving.validation import FeatureSchema


def monitoring_input_fingerprint(
    *,
    config: MonitoringConfig,
    schema: FeatureSchema,
    reference: ReferenceDataset,
    current: CurrentWindow,
) -> dict[str, Any]:
    """Hash the effective input used by Phase 7 monitoring calculations."""

    identity = {
        "schema_version": "1.0",
        "current_window": {
            "metadata": {
                "requested_window_size": current.metadata["requested_window_size"],
                "actual_window_size": current.metadata["actual_window_size"],
                "window_start_timestamp": current.metadata["window_start_timestamp"],
                "window_end_timestamp": current.metadata["window_end_timestamp"],
            },
            "rows": [
                _row_identity(row, schema.feature_columns) for row in current.rows
            ],
        },
        "reference": {
            "fingerprint": reference.metadata["fingerprint"],
            "selected_row_count": reference.metadata["selected_row_count"],
            "sample_seed": reference.metadata["sample_seed"],
            "sample_size": reference.metadata["sample_size"],
        },
        "schema": {
            "fingerprint": schema.fingerprint,
            "feature_columns": list(schema.feature_columns),
            "target_column": schema.target_column,
        },
        "monitoring_config": {
            "window_size": config.window_size,
            "minimum_window_size": config.minimum_window_size,
            "monitored_features": list(config.monitored_features),
            "drift_share_threshold": config.drift_share_threshold,
            "psi_threshold": config.psi_threshold,
            "evidently_method": config.evidently_method,
            "evidently_include_html": config.evidently_include_html,
            "minimum_labelled_rows": config.minimum_labelled_rows,
            "min_attack_support": config.min_attack_support,
            "min_benign_support": config.min_benign_support,
            "reference_sample_size": config.reference_sample_size,
            "reference_sample_seed": config.reference_sample_seed,
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(encoded.encode()).hexdigest(),
        "included_fields": [
            "ordered prediction_id/timestamp/current-window rows",
            "persisted feature vectors in canonical feature order",
            "prediction/probability/model version",
            "ground_truth and ground_truth_received_at",
            "feature schema fingerprint and columns",
            "reference dataset fingerprint and sampling settings",
            "window/drift/performance monitoring config",
        ],
    }


def _row_identity(row: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    features = row["features"]
    return {
        "prediction_id": row.get("prediction_id"),
        "timestamp": row.get("timestamp"),
        "features": {feature: float(features[feature]) for feature in feature_columns},
        "prediction": row.get("prediction"),
        "probability": row.get("probability"),
        "model_version": row.get("model_version"),
        "ground_truth": row.get("ground_truth"),
        "ground_truth_received_at": row.get("ground_truth_received_at"),
    }
