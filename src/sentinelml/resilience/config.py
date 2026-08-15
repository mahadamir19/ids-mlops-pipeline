"""Configuration loading for Phase 9 rollback and resilience."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sentinelml.data.config import PROJECT_ROOT, resolve_project_path

DEFAULT_RESILIENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "resilience_config.yaml"


@dataclass(frozen=True)
class SevereGuardrails:
    min_attack_recall: float | None
    min_f1: float | None
    max_false_positive_rate: float | None
    threshold_source: str


@dataclass(frozen=True)
class ResilienceConfig:
    poll_interval_seconds: float
    probation_enabled: bool
    probation_duration_seconds: int
    probation_minimum_labelled_rows: int
    probation_minimum_attack_support: int
    probation_minimum_benign_support: int
    automatic_rollback_enabled: bool
    severe_guardrails: SevereGuardrails
    monitoring_maximum_heartbeat_age_seconds: int
    promotion_retry_enabled: bool
    promotion_retry_interval_seconds: int
    promotion_retry_max_attempts: int
    resilience_reports_dir: Path
    database_url_env: str
    database_connect_timeout_seconds: int
    database_statement_timeout_ms: int
    config_path: Path

    @property
    def database_url(self) -> str | None:
        value = os.environ.get(self.database_url_env)
        return value if value else None


def load_resilience_config(
    path: Path = DEFAULT_RESILIENCE_CONFIG_PATH,
) -> ResilienceConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resilience = raw.get("resilience", {})
    probation = raw.get("probation", {})
    rollback = raw.get("automatic_rollback", {})
    monitoring = raw.get("monitoring", {})
    retry = raw.get("promotion_retry", {})
    paths = raw.get("paths", {})
    database = raw.get("database", {})
    guardrails = rollback.get("severe_guardrails", {})

    config = ResilienceConfig(
        poll_interval_seconds=float(resilience.get("poll_interval_seconds", 30)),
        probation_enabled=bool(probation.get("enabled", True)),
        probation_duration_seconds=int(probation.get("duration_seconds", 300)),
        probation_minimum_labelled_rows=int(
            probation.get("minimum_labelled_rows", 20)
        ),
        probation_minimum_attack_support=int(
            probation.get("minimum_attack_support", 1)
        ),
        probation_minimum_benign_support=int(
            probation.get("minimum_benign_support", 1)
        ),
        automatic_rollback_enabled=bool(rollback.get("enabled", True)),
        severe_guardrails=SevereGuardrails(
            min_attack_recall=_optional_float(guardrails.get("min_attack_recall")),
            min_f1=_optional_float(guardrails.get("min_f1")),
            max_false_positive_rate=_optional_float(
                guardrails.get("max_false_positive_rate")
            ),
            threshold_source=str(
                guardrails.get("threshold_source", "resilience_config")
            ),
        ),
        monitoring_maximum_heartbeat_age_seconds=int(
            monitoring.get("maximum_heartbeat_age_seconds", 180)
        ),
        promotion_retry_enabled=bool(retry.get("enabled", True)),
        promotion_retry_interval_seconds=int(retry.get("retry_interval_seconds", 60)),
        promotion_retry_max_attempts=int(retry.get("max_attempts", 5)),
        resilience_reports_dir=resolve_project_path(
            paths.get("resilience_reports", "reports/resilience")
        ),
        database_url_env=str(database.get("url_env", "SENTINELML_DATABASE_URL")),
        database_connect_timeout_seconds=int(
            database.get("connect_timeout_seconds", 5)
        ),
        database_statement_timeout_ms=int(database.get("statement_timeout_ms", 5000)),
        config_path=path,
    )
    validate_resilience_config(config)
    return config


def validate_resilience_config(config: ResilienceConfig) -> None:
    if config.poll_interval_seconds <= 0:
        raise ValueError("resilience poll_interval_seconds must be positive")
    if config.probation_duration_seconds <= 0:
        raise ValueError("probation duration_seconds must be positive")
    if config.probation_minimum_labelled_rows < 1:
        raise ValueError("probation minimum_labelled_rows must be at least 1")
    if config.probation_minimum_attack_support < 0:
        raise ValueError("probation minimum_attack_support cannot be negative")
    if config.probation_minimum_benign_support < 0:
        raise ValueError("probation minimum_benign_support cannot be negative")
    if config.monitoring_maximum_heartbeat_age_seconds <= 0:
        raise ValueError("monitoring maximum_heartbeat_age_seconds must be positive")
    if config.promotion_retry_interval_seconds <= 0:
        raise ValueError("promotion retry_interval_seconds must be positive")
    if config.promotion_retry_max_attempts < 1:
        raise ValueError("promotion max_attempts must be at least 1")


def _optional_float(value: Any) -> float | None:
    if value in {None, "", "null", "none"}:
        return None
    return float(value)
