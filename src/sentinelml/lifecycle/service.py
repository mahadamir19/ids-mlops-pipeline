"""Phase 4 model registration, gated promotion, retry, and rollback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.lifecycle.audit import utc_now, write_json
from sentinelml.lifecycle.candidate_evaluation import CandidateEvaluator
from sentinelml.lifecycle.config import configured_path, load_lifecycle_config
from sentinelml.lifecycle.errors import LifecycleError
from sentinelml.lifecycle.evaluation import (
    evaluate_model_on_promotion_slice,
    load_promotion_validation_dataset,
)
from sentinelml.lifecycle.notifier import ServingNotifier
from sentinelml.lifecycle.promotion import PromotionManager
from sentinelml.lifecycle.registration import CandidateRegistrar
from sentinelml.lifecycle.registry import (
    ModelVersionInfo,
    ensure_registered_model,
    get_champion,
    search_model_versions,
    set_version_tags,
    tag_json,
)
from sentinelml.lifecycle.rollback import RollbackManager
from sentinelml.lifecycle.thresholds import derive_thresholds, load_json
from sentinelml.tracking.mlflow import configure_mlflow_runtime_environment
from sentinelml.training.data import load_feature_schema, load_partition
from sentinelml.training.gpu import (
    fit_estimator_on_configured_device,
    force_estimator_cpu,
    normalize_xgboost_execution_params,
)
from sentinelml.training.models import build_baseline_model_spec, load_training_config


class LifecycleService:
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        mlflow_module: Any | None = None,
        client: Any | None = None,
        repo_root: Path = PROJECT_ROOT,
        validate_model_uri: bool = True,
        model_loader: Any | None = None,
        evaluator: Any | None = None,
        promotion_dataset_loader: Any | None = None,
        baseline_reference_evaluator: Any | None = None,
    ) -> None:
        self.config = config or load_lifecycle_config()
        self.repo_root = repo_root
        self.validate_model_uri = validate_model_uri
        self.mlflow = mlflow_module
        if self.mlflow is None:
            configure_mlflow_runtime_environment(repo_root)
            import mlflow as mlflow_module_real

            self.mlflow = mlflow_module_real
        self.client = client or self.mlflow.MlflowClient()
        self.model_name = str(self.config["registered_model_name"])
        self.alias = str(self.config["champion_alias"])
        self.model_loader = model_loader or self._load_model_from_mlflow
        self.evaluator = evaluator or evaluate_model_on_promotion_slice
        self.promotion_dataset_loader = promotion_dataset_loader
        self.baseline_reference_evaluator = baseline_reference_evaluator
        self.serving_notifier = ServingNotifier(config=self.config)
        self.registrar = CandidateRegistrar(
            client=self.client,
            mlflow=self.mlflow,
            model_name=self.model_name,
            report_root=self.report_root,
            load_manifest=self.load_final_candidate_manifest,
            validate_model_uri=self.validate_source_model_uri,
            execution_mode_from_manifest=self.execution_mode_from_manifest,
            demo_model_for_mode=self.demo_model_for_mode,
        )
        self.candidate_evaluator = CandidateEvaluator(
            client=self.client,
            model_name=self.model_name,
            config=self.config,
            report_root=self.report_root,
            get_version=self._get_version,
            derive_thresholds=self.derive_thresholds,
            metrics_for_version=self.metrics_for_version,
            get_champion=self.get_champion,
        )
        self.promotion_manager = PromotionManager(
            client=self.client,
            model_name=self.model_name,
            alias=self.alias,
            report_root=self.report_root,
            pending_dir=self.pending_dir,
            get_version=self._get_version,
            evaluate_candidate=self.evaluate_candidate,
            get_champion=self.get_champion,
            notify_serving_reload=self._notify_serving_reload,
            start_probation_after_promotion=self._start_probation_after_promotion,
        )
        self.rollback_manager = RollbackManager(
            client=self.client,
            model_name=self.model_name,
            alias=self.alias,
            report_root=self.report_root,
            get_version=self._get_version,
            validate_model_uri=self.validate_source_model_uri,
            get_champion=self.get_champion,
            notify_serving_reload=self._notify_serving_reload,
        )

    @property
    def report_root(self) -> Path:
        return configured_path(self.config, "lifecycle_reports")

    @property
    def pending_dir(self) -> Path:
        return configured_path(self.config, "pending_dir")

    def load_final_candidate_manifest(
        self,
        mode: str | None = None,
        *,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        path = manifest_path or configured_path(self.config, "final_candidate_manifest")
        manifest = load_json(path)
        self._validate_manifest(manifest, mode=mode)
        return manifest

    def _validate_manifest(self, manifest: dict[str, Any], mode: str | None) -> None:
        if manifest.get("model_name") != self.model_name:
            raise LifecycleError("final-candidate manifest targets a different model")
        mlflow = manifest.get("mlflow", {})
        if not mlflow.get("final_candidate_run_id"):
            raise LifecycleError("manifest missing final MLflow run id")
        if not mlflow.get("logged_model_uri"):
            raise LifecycleError("manifest missing logged model URI")
        if not manifest.get("model_family"):
            raise LifecycleError("manifest missing model family")
        continuous_retraining = manifest.get("candidate_source") == (
            "continuous_retraining"
        )
        source_optimization = manifest.get("source_optimization", {})
        if (
            not continuous_retraining
            and not source_optimization.get("best_trial_mlflow_run_id")
        ):
            raise LifecycleError("manifest missing source Optuna run id")
        for section in ["git", "dvc", "configuration", "dependencies", "evaluation"]:
            if not isinstance(manifest.get(section), dict):
                raise LifecycleError(f"manifest missing {section} lineage")
        evaluation = manifest["evaluation"]
        if "validation_metrics" not in evaluation:
            raise LifecycleError("manifest missing validation metrics")
        if not continuous_retraining and "test_metrics" not in evaluation:
            raise LifecycleError("manifest missing reporting test metrics")
        if continuous_retraining and evaluation.get("test_data_consulted") is not False:
            raise LifecycleError("continuous retraining manifest must not consult TEST")
        actual_mode = self.execution_mode_from_manifest(manifest)
        if mode is not None and actual_mode != mode:
            raise LifecycleError(
                f"manifest mode {actual_mode!r} does not match {mode!r}"
            )

    def execution_mode_from_manifest(self, manifest: dict[str, Any]) -> str:
        source = manifest.get("source_optimization", {})
        mode = source.get("execution_mode")
        if mode in {"smoke", "full"}:
            return str(mode)
        # Phase 3E v1 manifests did not persist execution_mode directly; infer from
        # the consumed smoke/full sample policy without changing the manifest schema.
        sample_rows = manifest.get("evaluation", {}).get("test_metrics", {}).get("rows")
        if sample_rows is not None and int(sample_rows) <= 10000:
            return "smoke"
        return "full"

    def demo_model_for_mode(self, mode: str) -> str:
        return "true" if mode == "smoke" else "false"

    def validate_source_model_uri(self, model_uri: str) -> None:
        if not self.validate_model_uri:
            return
        try:
            self.mlflow.models.get_model_info(model_uri)
        except Exception as exc:
            message = f"logged model URI does not resolve: {model_uri}"
            raise LifecycleError(message) from exc

    def register_candidate(
        self,
        *,
        mode: str = "smoke",
        force_new_version: bool = False,
        manifest_path: Path | None = None,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.registrar.register_candidate(
            mode=mode,
            force_new_version=force_new_version,
            manifest_path=manifest_path,
            extra_tags=extra_tags,
        )

    def derive_thresholds(self) -> dict[str, Any]:
        selected_baseline_path = configured_path(
            self.config,
            "smoke_selected_baseline",
        )
        baseline_manifest_path = configured_path(self.config, "smoke_baseline_manifest")
        output_path = configured_path(self.config, "threshold_report")
        selected_baseline = load_json(selected_baseline_path)
        baseline_evaluation = self.evaluate_baseline_reference(selected_baseline)
        dataset_metadata = baseline_evaluation["canonical_dataset"]
        return derive_thresholds(
            config=self.config,
            baseline_metrics=baseline_evaluation["metrics"],
            baseline_metrics_path=baseline_evaluation["report_path"],
            baseline_manifest_path=baseline_manifest_path,
            selected_baseline=selected_baseline,
            selected_baseline_path=selected_baseline_path,
            output_path=output_path,
            canonical_dataset=dataset_metadata,
            baseline_evaluation=baseline_evaluation,
        )

    def _feature_schema(self) -> dict[str, Any]:
        return load_feature_schema(configured_path(self.config, "feature_schema"))

    def promotion_dataset(self) -> Any:
        if self.promotion_dataset_loader is not None:
            return self.promotion_dataset_loader()
        feature_schema_path = configured_path(self.config, "feature_schema")
        return load_promotion_validation_dataset(
            validation_path=configured_path(self.config, "validation_partition"),
            feature_schema=load_feature_schema(feature_schema_path),
            feature_schema_path=feature_schema_path,
            config=self.config,
        )

    def evaluate_baseline_reference(
        self,
        selected_baseline: dict[str, Any],
    ) -> dict[str, Any]:
        if self.baseline_reference_evaluator is not None:
            return self.baseline_reference_evaluator(selected_baseline)
        selected_model = str(selected_baseline["recommended_baseline"])
        dataset = self.promotion_dataset()
        schema = self._feature_schema()
        policy = self.config["promotion_evaluation"]
        training_config = load_training_config()
        train = load_partition(
            configured_path(self.config, "train_partition"),
            feature_columns=schema["feature_columns"],
            target_column=schema["target_column"],
            sample_size=int(policy["baseline_train_sample_size"]),
            seed=int(policy["random_seed"])
            + int(policy.get("baseline_train_seed_offset", 0)),
        )
        spec = build_baseline_model_spec(
            selected_model,
            train.target,
            config=_lifecycle_training_config(
                training_config,
                selected_model,
                requested_device=str(policy.get("device", "cpu")),
            ),
        )
        fit_estimator_on_configured_device(
            spec.estimator,
            train.features,
            train.target,
        )
        metrics = self.evaluator(model=spec.estimator, dataset=dataset)
        payload = {
            "event": "baseline_evaluated",
            "evaluation_type": "lifecycle_promotion_validation",
            "model_name": self.model_name,
            "model_family": selected_model,
            "execution_mode": policy["mode"],
            "baseline_source": "phase2_selected_baseline_smoke_refit",
            "selected_baseline": selected_baseline,
            "canonical_dataset": dataset.metadata,
            "metrics": metrics,
            "evaluated_at": utc_now(),
        }
        path = self.report_root / "baseline_reference.json"
        write_json(path, payload)
        payload["report_path"] = path
        return payload

    def get_champion(self) -> ModelVersionInfo | None:
        return get_champion(self.client, model_name=self.model_name, alias=self.alias)

    def _get_version(self, version: str | int) -> ModelVersionInfo:
        model_version = self.client.get_model_version(
            self.model_name,
            str(version),
        )
        return self._as_info(model_version)

    def _as_info(self, value: Any) -> ModelVersionInfo:
        from sentinelml.lifecycle.registry import as_model_version_info

        return as_model_version_info(value)

    def metrics_for_version(self, version: ModelVersionInfo) -> dict[str, Any]:
        dataset = self.promotion_dataset()
        fingerprint = dataset.metadata["selected_row_fingerprint"]
        encoded = version.tags.get("lifecycle.validation_metrics_json")
        cached_fingerprint = version.tags.get(
            "lifecycle.validation_dataset_fingerprint"
        )
        if encoded and cached_fingerprint == fingerprint:
            return json.loads(encoded)
        if encoded and cached_fingerprint and cached_fingerprint != fingerprint:
            encoded = None
        if not encoded:
            encoded = self._validation_metrics_from_matching_manifest(version)
        if (
            encoded
            and version.tags.get("lifecycle.validation_dataset_fingerprint")
            == fingerprint
        ):
            return json.loads(encoded)
        evaluation = self.evaluate_model_version_on_promotion_slice(version, dataset)
        return evaluation["metrics"]

    def evaluate_model_version_on_promotion_slice(
        self,
        version: ModelVersionInfo,
        dataset: Any | None = None,
    ) -> dict[str, Any]:
        dataset = dataset or self.promotion_dataset()
        source_uri = version.tags.get("source_model_uri") or version.source
        if not source_uri:
            raise LifecycleError(f"model version {version.version} has no model URI")
        model = self.model_loader(source_uri)
        metrics = self.evaluator(model=model, dataset=dataset)
        payload = {
            "event": "evaluated",
            "evaluation_type": "lifecycle_promotion_validation",
            "model_name": self.model_name,
            "model_version": version.version,
            "model_family": version.tags.get("model_family"),
            "execution_mode": version.tags.get("execution_mode"),
            "source_model_uri": source_uri,
            "source_run_id": version.tags.get("source_run_id") or version.run_id,
            "canonical_dataset": dataset.metadata,
            "metrics": metrics,
            "evaluated_at": utc_now(),
        }
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=version.version,
            tags={
                "lifecycle.validation_metrics_json": tag_json(metrics),
                "lifecycle.metrics_json": tag_json(metrics),
                "lifecycle.validation_dataset_fingerprint": dataset.metadata[
                    "selected_row_fingerprint"
                ],
                "lifecycle.validation_evaluated_at": payload["evaluated_at"],
            },
        )
        path = self.report_root / "evaluations" / f"version_{version.version}.json"
        write_json(path, payload)
        return payload

    def _validation_metrics_from_matching_manifest(
        self,
        version: ModelVersionInfo,
    ) -> str | None:
        # Historical compatibility only: these metrics are usable only if a
        # matching canonical fingerprint tag exists, which older versions lack.
        manifest_path = configured_path(self.config, "final_candidate_manifest")
        if not manifest_path.exists():
            return None
        manifest = load_json(manifest_path)
        source_run_id = manifest.get("mlflow", {}).get("final_candidate_run_id")
        source_model_uri = manifest.get("mlflow", {}).get("logged_model_uri")
        if (
            version.tags.get("source_run_id") != source_run_id
            or version.tags.get("source_model_uri") != source_model_uri
        ):
            return None
        validation_metrics = manifest.get("evaluation", {}).get("validation_metrics")
        if validation_metrics is None:
            validation_path = (
                self.repo_root
                / "reports"
                / "final_candidate"
                / "validation_metrics.json"
            )
            validation_metrics = load_json(validation_path)
        return tag_json(validation_metrics)

    def _load_model_from_mlflow(self, model_uri: str) -> Any:
        for flavor in ["xgboost", "sklearn", "pyfunc"]:
            loader = getattr(getattr(self.mlflow, flavor, None), "load_model", None)
            if loader is None:
                continue
            try:
                return force_estimator_cpu(loader(model_uri))
            except Exception:
                continue
        raise LifecycleError(f"unable to load model artifact from MLflow: {model_uri}")

    def evaluate_candidate(self, *, version: str | int) -> dict[str, Any]:
        return self.candidate_evaluator.evaluate_candidate(version=version)

    def promote_or_reject(self, *, version: str | int) -> dict[str, Any]:
        return self.promotion_manager.promote_or_reject(version=version)

    def mark_candidate_failed(
        self,
        *,
        version: str | int,
        error: str,
        retraining_run_id: str | None = None,
        monitoring_run_id: str | None = None,
    ) -> dict[str, Any]:
        return self.promotion_manager.mark_candidate_failed(
            version=version,
            error=error,
            retraining_run_id=retraining_run_id,
            monitoring_run_id=monitoring_run_id,
        )

    def retry_pending(self) -> list[dict[str, Any]]:
        return self.promotion_manager.retry_pending()

    def rollback(
        self,
        *,
        version: str | int,
        reason: str | None = None,
        source: str = "manual",
        probation_id: str | None = None,
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.rollback_manager.rollback(
            version=version,
            reason=reason,
            source=source,
            probation_id=probation_id,
            trigger_metrics=trigger_metrics,
        )

    def register_and_promote(self, *, mode: str = "smoke") -> dict[str, Any]:
        registration = self.register_candidate(mode=mode)
        version = registration["model_version"]
        outcome = self.promote_or_reject(version=version)
        return {"registration": registration, "outcome": outcome}

    def _notify_serving_reload(self) -> dict[str, Any]:
        return self.serving_notifier.notify_reload()

    def _start_probation_after_promotion(
        self,
        *,
        promoted_version: str,
        previous_champion_version: str | None,
        promotion_timestamp: str,
    ) -> dict[str, Any]:
        try:
            from sentinelml.resilience.config import load_resilience_config
            from sentinelml.resilience.repository import ResilienceRepository
            from sentinelml.resilience.service import ResilienceService

            resilience_config = load_resilience_config()
            if resilience_config.database_url is None:
                return {
                    "started": False,
                    "reason": "resilience_database_unconfigured",
                }
            service = ResilienceService(
                config=resilience_config,
                repository=ResilienceRepository(resilience_config),
                lifecycle_service=self,
            )
            return service.start_probation_after_promotion(
                promoted_version=promoted_version,
                previous_champion_version=previous_champion_version,
                promotion_timestamp=promotion_timestamp,
            )
        except Exception as exc:
            return {
                "started": False,
                "reason": "probation_start_failed",
                "error": str(exc),
            }

    def status(self) -> dict[str, Any]:
        ensure_registered_model(self.client, self.model_name)
        champion = self.get_champion()
        versions = search_model_versions(self.client, self.model_name)
        return {
            "registered_model": self.model_name,
            "champion": champion.__dict__ if champion else None,
            "versions": [version.__dict__ for version in versions],
        }


def _lifecycle_training_config(
    training_config: dict[str, Any],
    model_family: str,
    *,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    """Normalize Phase 4 baseline refits for the local lifecycle environment."""

    import copy

    updated = copy.deepcopy(training_config)
    if model_family != "xgboost":
        return updated
    family_config = updated["baseline"][model_family]
    normalized, _runtime_config = normalize_xgboost_execution_params(
        family_config,
        requested_device=requested_device,
    )
    updated["baseline"][model_family] = normalized
    return updated
