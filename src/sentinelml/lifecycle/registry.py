"""MLflow Model Registry operations for Phase 4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlflow.exceptions import MlflowException

CHAMPION_ALIAS_NOT_FOUND_FRAGMENTS = (
    "Registered model alias champion not found",
    "alias champion not found",
    "RESOURCE_DOES_NOT_EXIST",
    "does not exist",
)


@dataclass(frozen=True)
class ModelVersionInfo:
    name: str
    version: str
    source: str | None
    run_id: str | None
    tags: dict[str, str]
    current_stage: str | None = None

    @property
    def lifecycle_state(self) -> str | None:
        return self.tags.get("lifecycle_state")


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def as_model_version_info(value: Any) -> ModelVersionInfo:
    return ModelVersionInfo(
        name=str(_get_attr(value, "name")),
        version=str(_get_attr(value, "version")),
        source=_get_attr(value, "source"),
        run_id=_get_attr(value, "run_id"),
        tags=dict(_get_attr(value, "tags", {}) or {}),
        current_stage=_get_attr(value, "current_stage"),
    )


def tag_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def ensure_registered_model(client: Any, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        try:
            client.create_registered_model(model_name)
        except Exception:
            client.get_registered_model(model_name)


def search_model_versions(client: Any, model_name: str) -> list[ModelVersionInfo]:
    versions = client.search_model_versions(f"name = '{model_name}'")
    return [as_model_version_info(version) for version in versions]


def find_duplicate_source_version(
    client: Any,
    *,
    model_name: str,
    source_run_id: str,
    source_model_uri: str,
) -> ModelVersionInfo | None:
    for version in search_model_versions(client, model_name):
        if (
            version.tags.get("source_run_id") == source_run_id
            and version.tags.get("source_model_uri") == source_model_uri
        ):
            return version
    return None


def get_champion(
    client: Any,
    *,
    model_name: str,
    alias: str,
) -> ModelVersionInfo | None:
    try:
        version = client.get_model_version_by_alias(model_name, alias)
        return as_model_version_info(version)
    except MlflowException as exc:
        message = str(exc)
        if any(fragment in message for fragment in CHAMPION_ALIAS_NOT_FOUND_FRAGMENTS):
            return None
        raise
    except Exception as exc:
        message = str(exc)
        if any(fragment in message for fragment in CHAMPION_ALIAS_NOT_FOUND_FRAGMENTS):
            return None
        raise


def set_version_tags(
    client: Any,
    *,
    model_name: str,
    version: str | int,
    tags: dict[str, Any],
) -> None:
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, str(value))
