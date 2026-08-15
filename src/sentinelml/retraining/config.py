"""Configuration loading for Phase 8 continuous retraining."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinelml.data.config import PROJECT_ROOT, resolve_project_path

DEFAULT_RETRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "retraining_config.yaml"


@dataclass(frozen=True)
class RetrainingConfig:
    enabled: bool
    poll_interval_seconds: float
    cooldown_seconds: int
    execution_mode: str
    minimum_approved_production_rows: int
    drift_enabled: bool
    performance_enabled: bool
    performance_minimum_labelled_rows: int
    performance_minimum_attack_support: int
    performance_minimum_benign_support: int
    min_attack_recall: float | None
    max_false_positive_rate: float | None
    max_f1_drop: float
    historical_source: Path
    max_historical_rows_smoke: int | None
    max_production_rows_smoke: int | None
    sampling_seed: int
    deduplication: str
    production_weighting_ratio: float
    candidate_strategy: str
    training_device: str
    force_cpu: bool
    auto_register: bool
    auto_evaluate: bool
    auto_promote: bool
    monitoring_reports_dir: Path
    retraining_reports_dir: Path
    feature_schema_path: Path
    lifecycle_config_path: Path
    training_config_path: Path
    database_url_env: str
    database_connect_timeout_seconds: int
    database_statement_timeout_ms: int
    config_path: Path

    @property
    def database_url(self) -> str | None:
        value = os.environ.get(self.database_url_env)
        return value if value else None


def load_retraining_config(
    path: Path = DEFAULT_RETRAINING_CONFIG_PATH,
    *,
    validate_data_paths: bool = True,
) -> RetrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    retraining = raw.get("retraining", {})
    trigger = raw.get("trigger", {})
    performance = raw.get("performance", {})
    dataset = raw.get("dataset", {})
    training = raw.get("training", {})
    lifecycle = raw.get("lifecycle", {})
    paths = raw.get("paths", {})
    database = raw.get("database", {})

    enabled = _env_bool(
        "SENTINELML_RETRAINING_ENABLED",
        retraining.get("enabled", False),
    )
    execution_mode = str(
        os.environ.get(
            "SENTINELML_RETRAINING_EXECUTION_MODE",
            retraining.get("execution_mode", "smoke"),
        )
    )
    raw_training_device = training.get("device")
    legacy_force_cpu = bool(training.get("force_cpu", raw_training_device is None))
    training_device = str(
        os.environ.get(
            "SENTINELML_RETRAINING_DEVICE",
            raw_training_device or ("cpu" if legacy_force_cpu else "cuda"),
        )
    ).strip().lower()
    config = RetrainingConfig(
        enabled=enabled,
        poll_interval_seconds=float(retraining.get("poll_interval_seconds", 60)),
        cooldown_seconds=int(retraining.get("cooldown_seconds", 300)),
        execution_mode=execution_mode,
        minimum_approved_production_rows=int(
            retraining.get("minimum_approved_production_rows", 10)
        ),
        drift_enabled=bool(trigger.get("drift_enabled", True)),
        performance_enabled=bool(trigger.get("performance_enabled", True)),
        performance_minimum_labelled_rows=int(
            performance.get("minimum_labelled_rows", 20)
        ),
        performance_minimum_attack_support=int(
            performance.get("minimum_attack_support", 1)
        ),
        performance_minimum_benign_support=int(
            performance.get("minimum_benign_support", 1)
        ),
        min_attack_recall=_optional_float(performance.get("min_attack_recall")),
        max_false_positive_rate=_optional_float(
            performance.get("max_false_positive_rate")
        ),
        max_f1_drop=float(performance.get("max_f1_drop", 0.10)),
        historical_source=resolve_project_path(
            dataset.get("historical_source", "data/processed/train.parquet")
        ),
        max_historical_rows_smoke=_optional_int(
            dataset.get("max_historical_rows_smoke", 20000)
        ),
        max_production_rows_smoke=_optional_int(
            dataset.get("max_production_rows_smoke", 10000)
        ),
        sampling_seed=int(dataset.get("sampling_seed", 42)),
        deduplication=str(
            dataset.get("deduplication", "canonical_feature_target_sha256")
        ),
        production_weighting_ratio=float(
            dataset.get("production_weighting_ratio", 1.0)
        ),
        candidate_strategy=str(
            training.get("candidate_strategy", "fresh_champion_family_retrain")
        ),
        training_device=training_device,
        force_cpu=training_device == "cpu",
        auto_register=bool(lifecycle.get("auto_register", True)),
        auto_evaluate=bool(lifecycle.get("auto_evaluate", True)),
        auto_promote=bool(lifecycle.get("auto_promote", True)),
        monitoring_reports_dir=resolve_project_path(
            paths.get("monitoring_reports", "reports/monitoring")
        ),
        retraining_reports_dir=resolve_project_path(
            paths.get("retraining_reports", "reports/retraining")
        ),
        feature_schema_path=resolve_project_path(
            paths.get("feature_schema", "reports/data/feature_schema.json")
        ),
        lifecycle_config_path=resolve_project_path(
            paths.get("lifecycle_config", "configs/lifecycle_config.yaml")
        ),
        training_config_path=resolve_project_path(
            paths.get("training_config", "configs/training_config.yaml")
        ),
        database_url_env=str(database.get("url_env", "SENTINELML_DATABASE_URL")),
        database_connect_timeout_seconds=int(
            database.get("connect_timeout_seconds", 5)
        ),
        database_statement_timeout_ms=int(database.get("statement_timeout_ms", 5000)),
        config_path=path,
    )
    validate_retraining_config(config, validate_data_paths=validate_data_paths)
    return config


def validate_retraining_config(
    config: RetrainingConfig,
    *,
    validate_data_paths: bool = True,
) -> None:
    if config.execution_mode not in {"smoke", "full"}:
        raise ValueError("retraining execution_mode must be smoke or full")
    if config.poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if config.cooldown_seconds < 0:
        raise ValueError("cooldown_seconds cannot be negative")
    if config.minimum_approved_production_rows < 1:
        raise ValueError("minimum_approved_production_rows must be at least 1")
    if config.candidate_strategy != "fresh_champion_family_retrain":
        raise ValueError("only fresh_champion_family_retrain is supported in Phase 8")
    if config.training_device not in {"cpu", "cuda"}:
        raise ValueError("retraining training.device must be cpu or cuda")
    if validate_data_paths and not config.historical_source.exists():
        raise ValueError(
            f"historical train source does not exist: {config.historical_source}"
        )
    normalized = str(config.historical_source).replace("\\", "/").lower()
    if normalized.endswith("data/processed/test.parquet"):
        raise ValueError(
            "Phase 8 retraining must never use data/processed/test.parquet"
        )


def _optional_float(value: Any) -> float | None:
    if value in {None, "", "null", "none"}:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in {None, "", "null", "none"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _env_bool(name: str, default: Any) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}
