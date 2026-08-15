"""Configuration helpers for Phase 5 model serving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT, resolve_project_path
from sentinelml.lifecycle.config import load_simple_yaml

DEFAULT_SERVING_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving_config.yaml"


@dataclass(frozen=True)
class ServingConfig:
    model_name: str
    champion_alias: str
    host: str
    port: int
    max_batch_size: int
    schema_path: Path
    queue_path: Path
    queue_flush_interval_seconds: float
    rejection_log_path: Path
    database_url_env: str
    database_connect_timeout_seconds: int
    database_statement_timeout_ms: int
    reload_endpoint_enabled: bool
    reload_notification_timeout_seconds: float
    mlflow_tracking_uri: str | None
    config_path: Path
    cors_allowed_origins: tuple[str, ...] = ("http://localhost:5173",)

    @property
    def database_url(self) -> str | None:
        value = os.environ.get(self.database_url_env)
        return value if value else None


def load_serving_config(path: Path = DEFAULT_SERVING_CONFIG_PATH) -> ServingConfig:
    raw = load_simple_yaml(path)
    model = raw.get("model", {})
    api = raw.get("api", {})
    request = raw.get("request", {})
    paths = raw.get("paths", {})
    queue = raw.get("queue", {})
    database = raw.get("database", {})
    reload = raw.get("reload", {})
    mlflow = raw.get("mlflow", {})
    cors = raw.get("cors", {})
    allowed_origins = os.environ.get(
        "SENTINELML_CORS_ALLOWED_ORIGINS",
        str(cors.get("allowed_origins", "http://localhost:5173")),
    )

    config = ServingConfig(
        model_name=str(model.get("registered_model_name", "sentinelml-ids")),
        champion_alias=str(model.get("champion_alias", "champion")),
        host=str(api.get("host", "127.0.0.1")),
        port=int(api.get("port", 8000)),
        max_batch_size=int(request.get("max_batch_size", 128)),
        schema_path=resolve_project_path(paths.get("feature_schema", "")),
        queue_path=resolve_project_path(
            queue.get("path", ".tmp/serving/predictions.jsonl")
        ),
        queue_flush_interval_seconds=float(queue.get("flush_interval_seconds", 10)),
        rejection_log_path=resolve_project_path(
            paths.get("rejection_log", ".tmp/serving/rejections.jsonl")
        ),
        database_url_env=str(database.get("url_env", "SENTINELML_DATABASE_URL")),
        database_connect_timeout_seconds=int(
            database.get("connect_timeout_seconds", 5)
        ),
        database_statement_timeout_ms=int(database.get("statement_timeout_ms", 5000)),
        reload_endpoint_enabled=bool(reload.get("internal_endpoint_enabled", True)),
        reload_notification_timeout_seconds=float(
            reload.get("notification_timeout_seconds", 2)
        ),
        cors_allowed_origins=tuple(
            origin.strip() for origin in allowed_origins.split(",") if origin.strip()
        ),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            str(mlflow.get("tracking_uri", "")) or None,
        ),
        config_path=path,
    )
    validate_serving_config(config)
    return config


def validate_serving_config(config: ServingConfig) -> None:
    if not config.model_name:
        raise ValueError("serving config requires a registered model name")
    if not config.champion_alias:
        raise ValueError("serving config requires a champion alias")
    if config.max_batch_size < 1:
        raise ValueError("serving max_batch_size must be at least 1")
    if not config.schema_path.exists():
        raise ValueError(f"feature schema does not exist: {config.schema_path}")


def as_public_config(config: ServingConfig) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "champion_alias": config.champion_alias,
        "host": config.host,
        "port": config.port,
        "max_batch_size": config.max_batch_size,
        "schema_path": str(config.schema_path),
        "queue_path": str(config.queue_path),
        "queue_flush_interval_seconds": config.queue_flush_interval_seconds,
        "rejection_log_path": str(config.rejection_log_path),
        "database_url_env": config.database_url_env,
        "database_configured": config.database_url is not None,
        "reload_endpoint_enabled": config.reload_endpoint_enabled,
        "cors_allowed_origins": list(config.cors_allowed_origins),
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
    }
