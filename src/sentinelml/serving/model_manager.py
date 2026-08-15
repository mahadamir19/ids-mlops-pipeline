"""MLflow champion loading and safe in-memory model reloads."""

from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.lifecycle.registry import ModelVersionInfo, get_champion
from sentinelml.serving.config import ServingConfig
from sentinelml.serving.validation import FeatureSchema, load_serving_feature_schema
from sentinelml.tracking.mlflow import configure_mlflow_runtime_environment
from sentinelml.training.gpu import force_estimator_cpu


class ModelLoadError(RuntimeError):
    """Raised when the serving process cannot load the registry champion."""


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    model_name: str
    model_version: str
    model_family: str | None
    execution_mode: str | None
    demo_model: bool | None
    source_run_id: str | None
    source_model_uri: str | None
    loaded_at: str
    feature_schema: FeatureSchema

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("model", None)
        payload["feature_schema"] = {
            "source_path": str(self.feature_schema.source_path),
            "fingerprint": self.feature_schema.fingerprint,
            "feature_count": self.feature_schema.feature_count,
            "generated_at_utc": self.feature_schema.generated_at_utc,
        }
        return payload


class ModelManager:
    def __init__(
        self,
        config: ServingConfig,
        *,
        mlflow_module: Any | None = None,
        client: Any | None = None,
        model_loader: Any | None = None,
        schema_path: Path | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._active: LoadedModel | None = None
        self.mlflow = mlflow_module
        if self.mlflow is None:
            configure_mlflow_runtime_environment(PROJECT_ROOT)
            import mlflow as mlflow_module_real

            self.mlflow = mlflow_module_real
        if config.mlflow_tracking_uri:
            self.mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self.client = client or self.mlflow.MlflowClient()
        self.model_loader = model_loader or self._load_model_artifact
        self.schema_path = schema_path or config.schema_path
        self.last_registry_error: str | None = None
        self.last_reload_error: str | None = None
        self.last_resolved_registry_version: str | None = None

    def load_startup(self) -> LoadedModel:
        loaded = self._load_champion()
        with self._lock:
            self._active = loaded
        return loaded

    def current(self) -> LoadedModel:
        with self._lock:
            if self._active is None:
                raise ModelLoadError("no champion is loaded")
            return self._active

    def is_ready(self) -> bool:
        with self._lock:
            return self._active is not None

    def reload_champion(self, *, force: bool = False) -> dict[str, Any]:
        previous = self.current()
        try:
            champion = self._resolve_champion()
        except Exception as exc:
            self.last_reload_error = str(exc)
            return {
                "reloaded": False,
                "success": False,
                "previous_model_version": previous.model_version,
                "active_model_version": previous.model_version,
                "registry_champion_version": None,
                "message": "registry champion could not be resolved",
                "error": str(exc),
            }
        if not force and champion.version == previous.model_version:
            self.last_reload_error = None
            return {
                "reloaded": False,
                "success": True,
                "previous_model_version": previous.model_version,
                "active_model_version": previous.model_version,
                "registry_champion_version": champion.version,
                "message": "loaded champion already matches registry alias",
            }
        try:
            loaded = self._build_loaded_model(champion)
        except Exception as exc:
            self.last_reload_error = str(exc)
            return {
                "reloaded": False,
                "success": False,
                "previous_model_version": previous.model_version,
                "active_model_version": previous.model_version,
                "registry_champion_version": champion.version,
                "message": "champion reload failed; previous model remains active",
                "error": str(exc),
            }
        with self._lock:
            self._active = loaded
        self.last_reload_error = None
        return {
            "reloaded": True,
            "success": True,
            "previous_model_version": previous.model_version,
            "active_model_version": loaded.model_version,
            "registry_champion_version": champion.version,
            "message": "champion reloaded successfully",
        }

    def registry_status(self) -> dict[str, Any]:
        active_version = self.current().model_version if self.is_ready() else None
        try:
            champion = self._resolve_champion()
        except Exception as exc:
            self.last_registry_error = str(exc)
            return {
                "connectivity": "unavailable",
                "registry_champion_version": self.last_resolved_registry_version,
                "loaded_model_version": active_version,
                "divergent": False,
                "error": str(exc),
                "last_reload_error": self.last_reload_error,
            }
        divergent = active_version is not None and champion.version != active_version
        return {
            "connectivity": "available",
            "registry_champion_version": champion.version,
            "loaded_model_version": active_version,
            "divergent": divergent,
            "error": None,
            "last_reload_error": self.last_reload_error,
        }

    def _load_champion(self) -> LoadedModel:
        return self._build_loaded_model(self._resolve_champion())

    def _resolve_champion(self) -> ModelVersionInfo:
        champion = get_champion(
            self.client,
            model_name=self.config.model_name,
            alias=self.config.champion_alias,
        )
        if champion is None:
            raise ModelLoadError(
                f"no MLflow champion alias {self.config.champion_alias!r} "
                f"exists for model {self.config.model_name!r}"
            )
        if champion.lifecycle_state and champion.lifecycle_state != "champion":
            raise ModelLoadError(
                f"registry alias resolved to version {champion.version} with "
                f"lifecycle_state={champion.lifecycle_state!r}, not 'champion'"
            )
        self.last_registry_error = None
        self.last_resolved_registry_version = champion.version
        return champion

    def _build_loaded_model(self, version: ModelVersionInfo) -> LoadedModel:
        source_uri = (
            version.tags.get("source_model_uri")
            or version.source
            or f"models:/{version.name}/{version.version}"
        )
        model = self.model_loader(source_uri)
        feature_schema = load_serving_feature_schema(self.schema_path)
        self._validate_model_compatibility(model, feature_schema)
        return LoadedModel(
            model=model,
            model_name=version.name,
            model_version=version.version,
            model_family=version.tags.get("model_family"),
            execution_mode=version.tags.get("execution_mode"),
            demo_model=_parse_bool_tag(version.tags.get("demo_model")),
            source_run_id=version.tags.get("source_run_id") or version.run_id,
            source_model_uri=source_uri,
            loaded_at=datetime.now(UTC).isoformat(),
            feature_schema=feature_schema,
        )

    def _load_model_artifact(self, model_uri: str) -> Any:
        for flavor in ["xgboost", "sklearn", "pyfunc"]:
            loader = getattr(getattr(self.mlflow, flavor, None), "load_model", None)
            if loader is None:
                continue
            try:
                with _patched_accessible_temporary_directory():
                    return force_estimator_cpu(loader(model_uri))
            except Exception:
                continue
        raise ModelLoadError(f"unable to load model artifact from MLflow: {model_uri}")

    def _validate_model_compatibility(
        self,
        model: Any,
        feature_schema: FeatureSchema,
    ) -> None:
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != feature_schema.feature_count:
            raise ModelLoadError(
                f"loaded model expects {n_features} features but schema has "
                f"{feature_schema.feature_count}"
            )


def _parse_bool_tag(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _accessible_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
) -> str:
    root = Path(dir or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
    path = root / name
    path.mkdir()
    return str(path)


class _AccessibleTemporaryDirectory:
    """TemporaryDirectory replacement for managed Windows runs with bad ACLs."""

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
        delete: bool = True,
    ) -> None:
        self.name = _accessible_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self.ignore_cleanup_errors = ignore_cleanup_errors
        self.delete = delete

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if not self.delete:
            return
        shutil.rmtree(self.name, ignore_errors=self.ignore_cleanup_errors)


@contextmanager
def _patched_accessible_temporary_directory() -> Iterator[None]:
    original_temporary_directory = tempfile.TemporaryDirectory
    original_mkdtemp = tempfile.mkdtemp
    tempfile.TemporaryDirectory = _AccessibleTemporaryDirectory
    tempfile.mkdtemp = _accessible_mkdtemp
    try:
        yield
    finally:
        tempfile.TemporaryDirectory = original_temporary_directory
        tempfile.mkdtemp = original_mkdtemp
