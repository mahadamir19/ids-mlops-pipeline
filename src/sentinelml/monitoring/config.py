"""Configuration loading for Phase 7 monitoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinelml.data.config import PROJECT_ROOT, resolve_project_path
from sentinelml.serving.validation import load_serving_feature_schema

DEFAULT_MONITORING_CONFIG_PATH = PROJECT_ROOT / "configs" / "monitoring_config.yaml"


@dataclass(frozen=True)
class MonitoringConfig:
    window_size: int
    minimum_window_size: int
    interval_seconds: float
    reference_path: Path
    reference_sample_size: int | None
    reference_sample_seed: int
    feature_schema_path: Path
    monitored_features: list[str]
    drift_share_threshold: float
    psi_threshold: float
    evidently_method: str
    evidently_include_html: bool
    minimum_labelled_rows: int
    min_attack_support: int
    min_benign_support: int
    database_url_env: str
    database_connect_timeout_seconds: int
    database_statement_timeout_ms: int
    reports_dir: Path
    host: str
    port: int
    config_path: Path

    @property
    def database_url(self) -> str | None:
        value = os.environ.get(self.database_url_env)
        return value if value else None


def load_monitoring_config(
    path: Path = DEFAULT_MONITORING_CONFIG_PATH,
    *,
    window_size: int | None = None,
) -> MonitoringConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    monitoring = raw.get("monitoring", {})
    reference = raw.get("reference", {})
    drift = raw.get("drift", {})
    performance = raw.get("performance", {})
    database = raw.get("database", {})
    prometheus = raw.get("prometheus", {})
    reports = raw.get("reports", {})
    paths = raw.get("paths", {})

    configured_features = drift.get("monitored_features") or []
    feature_schema_path = resolve_project_path(
        paths.get("feature_schema", "reports/data/feature_schema.json")
    )
    schema = load_serving_feature_schema(feature_schema_path)
    monitored_features = (
        [str(feature) for feature in configured_features]
        if configured_features
        else list(schema.feature_columns)
    )

    config = MonitoringConfig(
        window_size=int(window_size or monitoring.get("window_size", 500)),
        minimum_window_size=int(monitoring.get("minimum_window_size", 50)),
        interval_seconds=float(monitoring.get("interval_seconds", 60)),
        reference_path=resolve_project_path(
            reference.get("path", "data/reference/reference.parquet")
        ),
        reference_sample_size=_optional_int(reference.get("sample_size")),
        reference_sample_seed=int(reference.get("sample_seed", 42)),
        feature_schema_path=feature_schema_path,
        monitored_features=monitored_features,
        drift_share_threshold=float(drift.get("drift_share_threshold", 0.30)),
        psi_threshold=float(drift.get("psi_threshold", 0.20)),
        evidently_method=str(drift.get("evidently_method", "psi")),
        evidently_include_html=bool(drift.get("evidently_include_html", False)),
        minimum_labelled_rows=int(performance.get("minimum_labelled_rows", 20)),
        min_attack_support=int(performance.get("min_attack_support", 1)),
        min_benign_support=int(performance.get("min_benign_support", 1)),
        database_url_env=str(database.get("url_env", "SENTINELML_DATABASE_URL")),
        database_connect_timeout_seconds=int(
            database.get("connect_timeout_seconds", 5)
        ),
        database_statement_timeout_ms=int(database.get("statement_timeout_ms", 5000)),
        reports_dir=resolve_project_path(
            reports.get("output_dir", "reports/monitoring")
        ),
        host=str(prometheus.get("host", "0.0.0.0")),
        port=int(prometheus.get("port", 9101)),
        config_path=path,
    )
    validate_monitoring_config(config, schema.feature_columns)
    return config


def validate_monitoring_config(
    config: MonitoringConfig,
    feature_columns: list[str],
) -> None:
    if config.window_size < 1:
        raise ValueError("monitoring window_size must be at least 1")
    if config.minimum_window_size < 1:
        raise ValueError("monitoring minimum_window_size must be at least 1")
    if config.minimum_window_size > config.window_size:
        raise ValueError("monitoring minimum_window_size cannot exceed window_size")
    if config.interval_seconds <= 0:
        raise ValueError("monitoring interval_seconds must be positive")
    if not 0 < config.drift_share_threshold <= 1:
        raise ValueError("drift_share_threshold must be in (0, 1]")
    if config.psi_threshold <= 0:
        raise ValueError("psi_threshold must be positive")
    if not config.reference_path.exists():
        raise ValueError(f"reference data does not exist: {config.reference_path}")
    unknown = sorted(set(config.monitored_features) - set(feature_columns))
    if unknown:
        raise ValueError(f"monitoring config contains unknown features: {unknown}")


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None
