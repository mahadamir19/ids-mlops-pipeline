"""Train and log Phase 8 continuous-retraining candidates."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from mlflow.models import infer_signature

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.final_candidate.dependencies import (
    dependency_lock_metadata,
    ensure_dependency_lock,
    write_pip_freeze_snapshot,
)
from sentinelml.final_candidate.training import log_model_artifact
from sentinelml.lifecycle.audit import utc_now, write_json
from sentinelml.lifecycle.registry import ModelVersionInfo
from sentinelml.optimization.config import overlay_model_params
from sentinelml.retraining.dataset import RetrainingDataset
from sentinelml.tracking.mlflow import (
    collect_reproducibility_lineage,
    configure_mlflow_runtime_environment,
    flatten_mlflow_metrics,
    lineage_tags,
    log_input_artifacts,
    log_lineage_artifacts,
    sha256_file,
)
from sentinelml.training.evaluate import evaluate_binary_classifier
from sentinelml.training.gpu import (
    estimator_effective_device,
    fit_estimator_on_configured_device,
    normalize_xgboost_execution_params,
    validate_xgboost_runtime_available,
)
from sentinelml.training.models import build_baseline_model_spec, load_training_config

MLFLOW_EXPERIMENT_NAME = "sentinelml-retraining"


class RetrainingTrainer:
    def __init__(
        self,
        *,
        config: Any,
        lifecycle_service: Any,
        mlflow_module: Any | None = None,
        python_executable: str = sys.executable,
    ) -> None:
        self.config = config
        self.lifecycle_service = lifecycle_service
        self.mlflow = mlflow_module
        self.python_executable = python_executable

    def train_candidate(
        self,
        *,
        retraining_run_id: str,
        monitoring_run_id: str,
        trigger_decision: dict[str, Any],
        dataset: RetrainingDataset,
        output_dir: Path,
    ) -> dict[str, Any]:
        champion = self.lifecycle_service.get_champion()
        if champion is None:
            raise RuntimeError("continuous retraining requires an existing champion")
        model_family = _champion_family(champion)
        training_config, execution_config = self._candidate_training_config(
            champion,
            model_family,
        )
        validate_xgboost_runtime_available(str(execution_config["requested_device"]))
        model_spec = build_baseline_model_spec(
            model_family,
            dataset.bundle.target,
            config=training_config,
        )

        start = perf_counter()
        fit_estimator_on_configured_device(
            model_spec.estimator,
            dataset.bundle.features,
            dataset.bundle.target,
        )
        training_seconds = perf_counter() - start
        effective_device = estimator_effective_device(model_spec.estimator)
        promotion_dataset = self.lifecycle_service.promotion_dataset()
        validation_metrics = self.lifecycle_service.evaluator(
            model=model_spec.estimator,
            dataset=promotion_dataset,
        )
        training_metrics = evaluate_binary_classifier(
            model_spec.estimator,
            dataset.bundle.features,
            dataset.bundle.target,
        )
        input_example = dataset.bundle.features.head(5).copy()
        signature = infer_signature(
            input_example,
            model_spec.estimator.predict(input_example),
        )

        mlflow = self.mlflow
        if mlflow is None:
            tracking_uri = configure_mlflow_runtime_environment()
            import mlflow as mlflow_module

            mlflow = mlflow_module
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        dependency_lock_path = ensure_dependency_lock(
            python_executable=self.python_executable,
            repo_root=PROJECT_ROOT,
        )
        freeze_path = write_pip_freeze_snapshot(
            python_executable=self.python_executable,
            output_path=output_dir / "environment" / "pip_freeze.txt",
            repo_root=PROJECT_ROOT,
        )
        run_context = mlflow.start_run(run_name=f"retraining-{retraining_run_id}")
        with run_context as run:
            run_id = run.info.run_id
            mlflow.set_tags(
                {
                    "project": "SentinelML",
                    "run_type": "continuous_retraining",
                    "pipeline_stage": "phase8_retraining",
                    "candidate_source": "continuous_retraining",
                    "model_family": model_family,
                    "execution_mode": self.config.execution_mode,
                    "source_champion_version": champion.version,
                    "trigger_monitoring_run_id": monitoring_run_id,
                    "retraining_run_id": retraining_run_id,
                    "registry_status": "unregistered",
                }
            )
            mlflow.log_params(
                {
                    "random_seed": self.config.sampling_seed,
                    "feature_count": len(dataset.bundle.features.columns),
                    "source_champion_version": champion.version,
                    "source_champion_family": model_family,
                    "requested_device": execution_config["requested_device"],
                    "effective_device": execution_config["effective_device"],
                    "effective_tree_method": execution_config[
                        "effective_tree_method"
                    ],
                    "xgboost_version": execution_config["xgboost_version"],
                    "removed_gpu_params": ",".join(
                        execution_config["removed_gpu_params"]
                    ),
                    "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
                    "historical_rows": dataset.manifest["historical"][
                        "row_count_after_sampling"
                    ],
                    "production_rows": dataset.manifest["production"][
                        "row_count_after_sampling"
                    ],
                }
            )
            mlflow.log_params(
                {f"candidate.{k}": v for k, v in model_spec.parameters.items()}
            )
            mlflow.log_metrics(flatten_mlflow_metrics("train", training_metrics))
            mlflow.log_metrics(flatten_mlflow_metrics("validation", validation_metrics))
            mlflow.log_metric("training_time_seconds", float(training_seconds))
            if effective_device is not None:
                mlflow.set_tag("effective_device", effective_device)
            mlflow.set_tags(
                {
                    "requested_device": str(execution_config["requested_device"]),
                    "effective_tree_method": str(
                        execution_config["effective_tree_method"]
                    ),
                    "xgboost_version": str(execution_config["xgboost_version"]),
                }
            )
            model_info = log_model_artifact(
                mlflow,
                model_family=model_family,
                model=model_spec.estimator,
                signature=signature,
                input_example=input_example,
            )
            lineage = collect_reproducibility_lineage(
                mlflow_parent_run_id=str(run_id),
                training_config_path=self.config.training_config_path,
                feature_schema_path=self.config.feature_schema_path,
                dependency_lock_path=dependency_lock_path,
                strict_repository_tools=False,
            )
            dep_metadata = dependency_lock_metadata(dependency_lock_path)
            manifest = {
                "schema_version": "1.0",
                "model_name": self.lifecycle_service.model_name,
                "model_family": model_family,
                "candidate_source": "continuous_retraining",
                "created_at_utc": utc_now(),
                "random_seed": self.config.sampling_seed,
                "retraining": {
                    "retraining_run_id": retraining_run_id,
                    "trigger_monitoring_run_id": monitoring_run_id,
                    "trigger_reasons": trigger_decision.get("trigger_reasons", []),
                    "trigger_decision": trigger_decision,
                    "source_champion_version": champion.version,
                    "source_champion_run_id": champion.tags.get("source_run_id")
                    or champion.run_id,
                    "source_champion_model_uri": champion.tags.get("source_model_uri")
                    or champion.source,
                    "dataset_manifest_path": str(dataset.manifest_path),
                    "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
                },
                "source_optimization": {
                    "execution_mode": self.config.execution_mode,
                    "best_trial_mlflow_run_id": "not_used_continuous_retraining",
                    "note": (
                        "Optuna is intentionally not run during Phase 8 "
                        "automatic retraining."
                    ),
                },
                "execution_runtime": execution_config,
                "winning_hyperparameters": _effective_hyperparameters(
                    model_spec.parameters
                ),
                "git": {
                    "commit": lineage["git_commit"],
                    "branch": lineage["git_branch"],
                    "dirty": lineage["git_dirty"],
                },
                "dvc": {
                    "dvc_lock_sha256": lineage["dvc_lock_sha256"],
                    "dvc_status_clean": lineage["dvc_status_clean"],
                    "raw_dvc_sha256": (
                        sha256_file(PROJECT_ROOT / "data" / "raw.dvc")
                        if (PROJECT_ROOT / "data" / "raw.dvc").exists()
                        else None
                    ),
                },
                "configuration": {
                    "training_config_sha256": lineage["training_config_sha256"],
                    "optimization_config_sha256": "not_used_continuous_retraining",
                    "feature_schema_sha256": lineage["feature_schema_sha256"],
                },
                "dependencies": {
                    **dep_metadata,
                    "python_version": lineage["python_version"],
                    "platform": lineage["platform"],
                },
                "mlflow": {
                    "experiment_name": MLFLOW_EXPERIMENT_NAME,
                    "final_candidate_run_id": run_id,
                    "logged_model_uri": model_info["model_uri"],
                    "model_flavor": model_info["flavor"],
                },
                "evaluation": {
                    "training_time_seconds": float(training_seconds),
                    "effective_device": effective_device,
                    "requested_device": execution_config["requested_device"],
                    "effective_tree_method": execution_config[
                        "effective_tree_method"
                    ],
                    "training_metrics": training_metrics,
                    "validation_metrics": validation_metrics,
                    "test_metrics": None,
                    "test_data_consulted": False,
                    "promotion_validation_metadata": promotion_dataset.metadata,
                },
                "registry_state": {"registered": False, "model_version": None},
            }
            manifest_path = output_dir / "model_manifest.json"
            write_json(manifest_path, manifest)
            mlflow.set_tags(lineage_tags(lineage))
            log_input_artifacts(
                mlflow,
                [
                    self.config.training_config_path,
                    self.config.feature_schema_path,
                    self.config.config_path,
                    dataset.manifest_path,
                    manifest_path,
                ],
            )
            log_lineage_artifacts(
                mlflow,
                lineage,
                dependency_lock_path=dependency_lock_path,
            )
            mlflow.log_artifact(str(freeze_path), artifact_path="environment")
            mlflow.log_artifact(str(dataset.manifest_path), artifact_path="retraining")
            mlflow.log_artifact(str(manifest_path), artifact_path="retraining")

        return {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "mlflow_run_id": run_id,
            "model_family": model_family,
            "training_seconds": float(training_seconds),
            "validation_metrics": validation_metrics,
        }

    def _candidate_training_config(
        self,
        champion: ModelVersionInfo,
        model_family: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        training_config = load_training_config(self.config.training_config_path)
        recovered = self._recover_champion_hyperparameters(champion)
        if recovered:
            training_config = overlay_model_params(
                training_config,
                model_family,
                recovered,
            )
        requested_device = str(
            getattr(
                self.config,
                "training_device",
                "cpu" if bool(getattr(self.config, "force_cpu", True)) else "cuda",
            )
        ).lower()
        return _apply_xgboost_execution_policy(
            training_config,
            model_family,
            requested_device=requested_device,
        )

    def _recover_champion_hyperparameters(
        self,
        champion: ModelVersionInfo,
    ) -> dict[str, Any]:
        source_run_id = champion.tags.get("source_run_id") or champion.run_id
        if not source_run_id:
            return {}
        client = self.lifecycle_service.client
        try:
            run = client.get_run(source_run_id)
        except Exception:
            return {}
        params = getattr(getattr(run, "data", None), "params", {}) or {}
        recovered: dict[str, Any] = {}
        for key, value in params.items():
            if key.startswith("winning."):
                recovered[key.removeprefix("winning.")] = _parse_mlflow_param(value)
            elif key.startswith("candidate."):
                recovered[key.removeprefix("candidate.")] = _parse_mlflow_param(value)
        return recovered


def _champion_family(champion: ModelVersionInfo) -> str:
    family = champion.tags.get("model_family")
    if not family:
        raise RuntimeError("champion model version is missing model_family tag")
    return family


def _parse_mlflow_param(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _effective_hyperparameters(parameters: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key, value in parameters.items():
        if isinstance(value, str | int | float | bool) or value is None:
            keep[key] = value
    return keep


def _force_cpu_training_config(
    training_config: dict[str, Any],
    model_family: str,
) -> dict[str, Any]:
    """Make Phase 8 smoke retraining safe in Docker without CUDA drivers."""

    return _apply_xgboost_execution_policy(
        training_config,
        model_family,
        requested_device="cpu",
    )[0]


def _apply_xgboost_execution_policy(
    training_config: dict[str, Any],
    model_family: str,
    *,
    requested_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply SentinelML's environment-specific XGBoost runtime policy."""

    import copy

    updated = copy.deepcopy(training_config)
    family_config = updated["baseline"][model_family]
    normalized, runtime_config = normalize_xgboost_execution_params(
        family_config,
        requested_device=requested_device,
    )
    updated["baseline"][model_family] = normalized
    return updated, runtime_config
