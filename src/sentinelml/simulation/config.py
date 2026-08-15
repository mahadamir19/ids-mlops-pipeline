"""Configuration for Phase 6 production traffic simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinelml.data.config import PROJECT_ROOT, resolve_project_path

DEFAULT_SIMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation_config.yaml"


@dataclass(frozen=True)
class SimulationConfig:
    api_base_url: str
    default_seed: int
    default_request_count: int
    request_interval_seconds: float
    timeout_seconds: float
    source_data_path: Path
    feature_schema_path: Path
    reports_dir: Path
    randomized_delay: dict[str, Any]
    batch_delivery: dict[str, Any]
    retry: dict[str, Any]
    scenarios: dict[str, Any]
    custom: dict[str, Any]
    config_path: Path


def load_simulation_config(
    path: Path = DEFAULT_SIMULATION_CONFIG_PATH,
) -> SimulationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    api = raw.get("api", {})
    simulation = raw.get("simulation", {})
    labels = raw.get("labels", {})
    paths = raw.get("paths", {})

    config = SimulationConfig(
        api_base_url=str(api.get("base_url", "http://127.0.0.1:8000")).rstrip("/"),
        default_seed=int(simulation.get("default_seed", 42)),
        default_request_count=int(simulation.get("default_request_count", 100)),
        request_interval_seconds=float(simulation.get("request_interval_seconds", 0)),
        timeout_seconds=float(simulation.get("timeout_seconds", 5)),
        source_data_path=resolve_project_path(
            simulation.get("source_data", "data/reference/reference.parquet")
        ),
        feature_schema_path=resolve_project_path(
            paths.get("feature_schema", "reports/data/feature_schema.json")
        ),
        reports_dir=resolve_project_path(
            paths.get("reports_dir", "reports/simulation")
        ),
        randomized_delay=dict(labels.get("randomized_delay", {})),
        batch_delivery=dict(labels.get("batch_delivery", {})),
        retry=dict(labels.get("retry", {})),
        scenarios=dict(raw.get("scenarios", {})),
        custom=dict(raw.get("custom", {})),
        config_path=path,
    )
    validate_simulation_config(config)
    return config


def validate_simulation_config(config: SimulationConfig) -> None:
    if config.default_request_count < 1:
        raise ValueError("simulation default_request_count must be at least 1")
    if config.request_interval_seconds < 0:
        raise ValueError("simulation request_interval_seconds must be non-negative")
    if config.timeout_seconds <= 0:
        raise ValueError("simulation timeout_seconds must be positive")
    if not config.feature_schema_path.exists():
        raise ValueError(f"feature schema does not exist: {config.feature_schema_path}")
    reject_test_source(config.source_data_path)

    min_delay = float(config.randomized_delay.get("min_delay_seconds", 0))
    max_delay = float(config.randomized_delay.get("max_delay_seconds", 0))
    if min_delay < 0 or max_delay < min_delay:
        raise ValueError("randomized label delay bounds are invalid")
    batch_size = int(config.batch_delivery.get("batch_size", 10))
    if batch_size < 1:
        raise ValueError("batch label delivery size must be at least 1")
    for key in ["max_attempts", "retry_delay_seconds"]:
        if key in config.retry and not math.isfinite(float(config.retry[key])):
            raise ValueError(f"label retry value is not finite: {key}")


def reject_test_source(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if normalized.endswith("data/processed/test.parquet"):
        raise ValueError(
            "Phase 6 simulation must not use data/processed/test.parquet as source"
        )
