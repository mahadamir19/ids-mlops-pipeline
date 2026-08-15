"""Phase 4 model registration, gated promotion, retry, and rollback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.lifecycle.audit import audit_path, utc_now, write_json
from sentinelml.lifecycle.config import configured_path, load_lifecycle_config
from sentinelml.lifecycle.evaluation import (
    evaluate_model_on_promotion_slice,
    load_promotion_validation_dataset,
)
from sentinelml.lifecycle.candidate_evaluation import CandidateEvaluator
from sentinelml.lifecycle.notifier import ServingNotifier
from sentinelml.lifecycle.promotion import PromotionManager
from sentinelml.lifecycle.registration import CandidateRegistrar
from sentinelml.lifecycle.registry import (
    ModelVersionInfo,
    ensure_registered_model,
    find_duplicate_source_version,
    get_champion,
    search_model_versions,
    set_version_tags,
    tag_json,
)
from sentinelml.lifecycle.rollback import RollbackManager
from sentinelml.lifecycle.thresholds import (
    composite_score,
    derive_thresholds,
    evaluate_absolute_gates,
    load_json,
)
from sentinelml.tracking.mlflow import configure_mlflow_runtime_environment
from sentinelml.training.data import load_feature_schema, load_partition
from sentinelml.training.gpu import (
    fit_estimator_on_configured_device,
    force_estimator_cpu,
    normalize_xgboost_execution_params,
)
from sentinelml.training.models import build_baseline_model_spec, load_training_config


class LifecycleError(RuntimeError):
    """Raised when a Phase 4 lifecycle operation cannot be completed."""


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
        self.registrar = CandidateRegistrar(self)
        self.candidate_evaluator = CandidateEvaluator(self)
        self.promotion_manager = PromotionManager(self)
        self.rollback_manager = RollbackManager(self)
        self.serving_notifier = ServingNotifier(self)

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

    def _register_candidate(
        self,
        *,
        mode: str = "smoke",
        force_new_version: bool = False,
        manifest_path: Path | None = None,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self.load_final_candidate_manifest(
            mode=mode,
            manifest_path=manifest_path,
        )
        source_run_id = str(manifest["mlflow"]["final_candidate_run_id"])
        source_model_uri = str(manifest["mlflow"]["logged_model_uri"])
        ensure_registered_model(self.client, self.model_name)
        duplicate = None
        if not force_new_version:
            duplicate = find_duplicate_source_version(
                self.client,
                model_name=self.model_name,
                source_run_id=source_run_id,
                source_model_uri=source_model_uri,
            )
        if duplicate is not None:
            return {
                "event": "registered",
                "registered": False,
                "reused": True,
                "model_version": duplicate.version,
                "model_name": self.model_name,
            }

        self.validate_source_model_uri(source_model_uri)
        created = self.mlflow.register_model(source_model_uri, self.model_name)
        version = str(created.version)
        execution_mode = self.execution_mode_from_manifest(manifest)
        tags = self._registration_tags(manifest, execution_mode)
        if extra_tags:
            tags.update(extra_tags)
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=version,
            tags=tags,
        )
        payload = {
            "event": "registered",
            "registered": True,
            "reused": False,
            "model_name": self.model_name,
            "model_version": version,
            "source_model_uri": source_model_uri,
            "source_run_id": source_run_id,
            "tags": tags,
        }
        write_json(audit_path(self.report_root, "registered", version), payload)
        return payload

    def _registration_tags(
        self,
        manifest: dict[str, Any],
        execution_mode: str,
    ) -> dict[str, Any]:
        validation_metrics = manifest["evaluation"]["validation_metrics"]
        test_metrics = manifest["evaluation"].get("test_metrics")
        tags = {
            "lifecycle_state": "candidate",
            "execution_mode": execution_mode,
            "demo_model": self.demo_model_for_mode(execution_mode),
            "model_family": manifest["model_family"],
            "source_run_id": manifest["mlflow"]["final_candidate_run_id"],
            "source_model_uri": manifest["mlflow"]["logged_model_uri"],
            "candidate_source": manifest.get("candidate_source", "final_candidate"),
            "source_optuna_run_id": manifest.get("source_optimization", {}).get(
                "best_trial_mlflow_run_id",
                "",
            ),
            "git_commit": manifest["git"]["commit"],
            "dvc_lock_sha256": manifest["dvc"]["dvc_lock_sha256"],
            "training_config_sha256": manifest["configuration"][
                "training_config_sha256"
            ],
            "optimization_config_sha256": manifest["configuration"][
                "optimization_config_sha256"
            ],
            "dependency_lock_sha256": manifest["dependencies"][
                "dependency_lock_sha256"
            ],
            "registered_at": utc_now(),
            "evaluation_source": "reports/final_candidate/validation_metrics.json",
            "lifecycle.metrics_json": tag_json(validation_metrics),
            "lifecycle.validation_metrics_json": tag_json(validation_metrics),
        }
        if manifest.get("candidate_source") == "continuous_retraining":
            tags.update(
                {
                    "evaluation_source": "phase8_canonical_promotion_validation",
                    "retraining_run_id": manifest["retraining"]["retraining_run_id"],
                    "trigger_monitoring_run_id": manifest["retraining"][
                        "trigger_monitoring_run_id"
                    ],
                    "dataset_fingerprint": manifest["retraining"][
                        "dataset_fingerprint"
                    ],
                    "test_data_consulted": "false",
                }
            )
        elif test_metrics is not None:
            tags["lifecycle.test_metrics_json"] = tag_json(test_metrics)
        return tags

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

    def _evaluate_candidate(self, *, version: str | int) -> dict[str, Any]:
        candidate = self._get_version(version)
        thresholds = self.derive_thresholds()
        candidate_metrics = self.metrics_for_version(candidate)
        absolute = evaluate_absolute_gates(
            candidate_metrics=candidate_metrics,
            thresholds=thresholds["thresholds"],
            threshold_policy=self.config["threshold_policy"],
        )
        champion = self.get_champion()
        candidate_score = composite_score(
            candidate_metrics,
            thresholds=thresholds["thresholds"],
            config=self.config,
        )
        comparison: dict[str, Any] = {
            "has_champion": champion is not None,
            "candidate_composite_score": candidate_score,
        }
        relative_passed = True
        if champion is not None and champion.version != candidate.version:
            if not self._modes_compatible(candidate, champion):
                relative_passed = False
                comparison["mode_mismatch"] = {
                    "candidate": candidate.tags.get("execution_mode"),
                    "champion": champion.tags.get("execution_mode"),
                }
            else:
                champion_metrics = self.metrics_for_version(champion)
                champion_score = composite_score(
                    champion_metrics,
                    thresholds=thresholds["thresholds"],
                    config=self.config,
                )
                min_improvement = float(
                    self.config["composite_score"]["minimum_relative_improvement"]
                )
                delta = candidate_score["score"] - champion_score["score"]
                relative_passed = delta >= min_improvement
                comparison.update(
                    {
                        "champion_version": champion.version,
                        "champion_composite_score": champion_score,
                        "minimum_relative_improvement": min_improvement,
                        "score_delta": delta,
                        "relative_passed": relative_passed,
                    }
                )
        elif champion is not None:
            comparison["already_champion"] = True

        result = {
            "event": "evaluated",
            "model_name": self.model_name,
            "candidate_version": candidate.version,
            "passed": bool(absolute["passed"] and relative_passed),
            "absolute_gates": absolute,
            "comparison": comparison,
            "threshold_report": thresholds,
            "evaluated_at": utc_now(),
        }
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags={
                "lifecycle.last_evaluated_at": result["evaluated_at"],
                "lifecycle.last_evaluation_passed": str(result["passed"]).lower(),
                "lifecycle.composite_score": candidate_score["score"],
            },
        )
        evaluation_path = (
            self.report_root / "evaluations" / f"version_{candidate.version}.json"
        )
        write_json(evaluation_path, result)
        return result

    def _modes_compatible(
        self,
        candidate: ModelVersionInfo,
        champion: ModelVersionInfo,
    ) -> bool:
        if bool(self.config.get("mode_policy", {}).get("allow_cross_mode_promotion")):
            return True
        return candidate.tags.get("execution_mode") == champion.tags.get(
            "execution_mode"
        )

    def promote_or_reject(self, *, version: str | int) -> dict[str, Any]:
        return self.promotion_manager.promote_or_reject(version=version)

    def _promote_or_reject(self, *, version: str | int) -> dict[str, Any]:
        candidate = self._get_version(version)
        evaluation = self.evaluate_candidate(version=version)
        champion = self.get_champion()
        if champion is not None and champion.version == candidate.version:
            return {
                "event": "promoted",
                "status": "already_champion",
                "model_name": self.model_name,
                "model_version": candidate.version,
                "evaluation": evaluation,
            }
        if not evaluation["passed"]:
            return self.reject_candidate(candidate, evaluation)
        return self._promote_candidate(candidate, evaluation, champion)

    def reject_candidate(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        failed = evaluation["absolute_gates"]["failed_gates"]
        if evaluation["comparison"].get("mode_mismatch"):
            failed = [*failed, "execution_mode"]
        elif not evaluation["comparison"].get("relative_passed", True):
            failed = [*failed, "composite_improvement"]
        rejected_at = utc_now()
        tags = {
            "lifecycle_state": "rejected",
            "lifecycle.rejected_at": rejected_at,
            "lifecycle.failed_gates": ",".join(failed),
            "lifecycle.gate_result_json": tag_json(
                {"passed": False, "failed_gates": failed}
            ),
        }
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags=tags,
        )
        payload = {
            "event": "rejected",
            "model_name": self.model_name,
            "model_version": candidate.version,
            "rejected_at": rejected_at,
            "failed_gates": failed,
            "evaluation": evaluation,
        }
        write_json(audit_path(self.report_root, "rejected", candidate.version), payload)
        return payload

    def mark_candidate_failed(
        self,
        *,
        version: str | int,
        error: str,
        retraining_run_id: str | None = None,
        monitoring_run_id: str | None = None,
    ) -> dict[str, Any]:
        candidate = self._get_version(version)
        failed_at = utc_now()
        payload = {
            "event": "failed",
            "model_name": self.model_name,
            "model_version": candidate.version,
            "failed_at": failed_at,
            "error": error,
            "retraining_run_id": retraining_run_id,
            "monitoring_run_id": monitoring_run_id,
        }
        tags = {
            "lifecycle_state": "failed",
            "lifecycle.failed_at": failed_at,
            "lifecycle.failure_reason": error[:1000],
        }
        if retraining_run_id:
            tags["retraining_run_id"] = retraining_run_id
        if monitoring_run_id:
            tags["trigger_monitoring_run_id"] = monitoring_run_id
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags=tags,
        )
        write_json(audit_path(self.report_root, "failed", candidate.version), payload)
        return payload

    def _promote_candidate(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
        previous_champion: ModelVersionInfo | None,
    ) -> dict[str, Any]:
        promoted_at = utc_now()
        previous_version = previous_champion.version if previous_champion else None
        try:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "promotion_pending",
                    "lifecycle.promotion_attempted_at": promoted_at,
                    "lifecycle.previous_champion_version": previous_version or "",
                },
            )
            self.client.set_registered_model_alias(
                self.model_name,
                self.alias,
                candidate.version,
            )
            verified = self.get_champion()
            if verified is None or verified.version != candidate.version:
                raise LifecycleError("champion alias verification failed")
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "champion",
                    "lifecycle.promoted_at": promoted_at,
                    "lifecycle.previous_champion_version": previous_version or "",
                    "lifecycle.evaluated_against_version": previous_version or "",
                },
            )
            if previous_champion is not None:
                set_version_tags(
                    self.client,
                    model_name=self.model_name,
                    version=previous_champion.version,
                    tags={
                        "lifecycle_state": "superseded",
                        "lifecycle.superseded_at": promoted_at,
                        "lifecycle.superseded_by_version": candidate.version,
                    },
                )
            payload = {
                "event": "promoted",
                "model_name": self.model_name,
                "model_version": candidate.version,
                "previous_champion_version": previous_version,
                "promoted_at": promoted_at,
                "evaluation": evaluation,
            }
            payload["serving_reload_notification"] = self._notify_serving_reload()
            payload["probation"] = self._start_probation_after_promotion(
                promoted_version=candidate.version,
                previous_champion_version=previous_version,
                promotion_timestamp=promoted_at,
            )
            write_json(
                audit_path(self.report_root, "promoted", candidate.version),
                payload,
            )
            return payload
        except Exception as exc:
            pending = self._write_pending_promotion(
                candidate,
                evaluation,
                previous_champion,
                str(exc),
            )
            return pending

    def _write_pending_promotion(
        self,
        candidate: ModelVersionInfo,
        evaluation: dict[str, Any],
        previous_champion: ModelVersionInfo | None,
        error: str,
    ) -> dict[str, Any]:
        created_at = utc_now()
        try:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=candidate.version,
                tags={
                    "lifecycle_state": "promotion_pending",
                    "lifecycle.promotion_pending_at": created_at,
                },
            )
        except Exception:
            pass
        payload = {
            "event": "promotion_pending",
            "operation_state": "promotion_pending",
            "model_name": self.model_name,
            "candidate_version": candidate.version,
            "candidate_source_run_id": candidate.tags.get("source_run_id"),
            "candidate_source_model_uri": candidate.tags.get("source_model_uri"),
            "previous_champion_version": previous_champion.version
            if previous_champion
            else None,
            "created_at": created_at,
            "registry_error": error,
            "original_gate_evaluation": evaluation,
        }
        path = self.pending_dir / f"{self.model_name}_version_{candidate.version}.json"
        write_json(path, payload)
        write_json(
            audit_path(self.report_root, "promotion_pending", candidate.version),
            payload,
        )
        return payload

    def retry_pending(self) -> list[dict[str, Any]]:
        return self.promotion_manager.retry_pending()

    def _retry_pending(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.pending_dir.glob("*.json")):
            pending = load_json(path)
            version = pending["candidate_version"]
            champion_before = self.get_champion()
            outcome = self.promote_or_reject(version=version)
            archive_payload = {
                "event": "promotion_retry",
                "retried_at": utc_now(),
                "pending": pending,
                "current_champion_version": champion_before.version
                if champion_before
                else None,
                "outcome": outcome,
            }
            still_pending = (
                outcome.get("event") == "promotion_pending"
                or outcome.get("operation_state") == "promotion_pending"
            )
            if still_pending:
                retry_history = pending.get("retry_history", [])
                pending["retry_history"] = [*retry_history, archive_payload]
                pending["last_retry_at"] = archive_payload["retried_at"]
                pending["last_retry_outcome"] = outcome
                write_json(path, pending)
            else:
                archive = self.pending_dir.parent / "resolved" / path.name
                write_json(archive, archive_payload)
                path.unlink()
            write_json(
                audit_path(self.report_root, "promotion_retry", version),
                archive_payload,
            )
            results.append(archive_payload)
        return results

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

    def _rollback(
        self,
        *,
        version: str | int,
        reason: str | None = None,
        source: str = "manual",
        probation_id: str | None = None,
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._get_version(version)
        state = target.tags.get("lifecycle_state")
        if state not in {"champion", "superseded"}:
            raise LifecycleError(
                "rollback target must be a current or superseded approved champion; "
                "rejected, failed, and candidate versions are not valid targets"
            )
        required = ["execution_mode", "demo_model", "model_family", "source_run_id"]
        missing = [key for key in required if key not in target.tags]
        if missing:
            raise LifecycleError(
                f"rollback target missing lifecycle metadata: {missing}"
            )
        source_model_uri = target.source or target.tags.get("source_model_uri")
        if not source_model_uri:
            raise LifecycleError("rollback target has no source model URI")
        self.validate_source_model_uri(source_model_uri)
        previous = self.get_champion()
        if previous is not None and previous.version == target.version:
            payload = {
                "event": "rollback",
                "status": "already_current",
                "model_name": self.model_name,
                "target_version": target.version,
                "previous_champion_version": previous.version,
                "rolled_back_at": utc_now(),
                "reason": reason,
                "source": source,
                "probation_id": probation_id,
                "trigger_metrics": trigger_metrics or {},
                "serving_reload_notification": {
                    "attempted": False,
                    "reason": "target already active champion",
                },
            }
            write_json(
                audit_path(self.report_root, "rollback", target.version),
                payload,
            )
            return payload
        rolled_back_at = utc_now()
        self.client.set_registered_model_alias(
            self.model_name,
            self.alias,
            target.version,
        )
        verified = self.get_champion()
        if verified is None or verified.version != target.version:
            raise LifecycleError("rollback alias verification failed")
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=target.version,
            tags={
                "lifecycle_state": "champion",
                "lifecycle.rollback_at": rolled_back_at,
                "lifecycle.rollback_reason": reason or "",
                "lifecycle.rollback_source": source,
                "lifecycle.rollback_probation_id": probation_id or "",
                "lifecycle.rollback_previous_champion_version": previous.version
                if previous
                else "",
            },
        )
        if previous is not None:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=previous.version,
                tags={
                    "lifecycle_state": "rolled_back",
                    "lifecycle.rolled_back_at": rolled_back_at,
                    "lifecycle.rolled_back_to_version": target.version,
                    "lifecycle.rollback_source": source,
                    "lifecycle.rollback_probation_id": probation_id or "",
                },
            )
        payload = {
            "event": "rollback",
            "status": "succeeded",
            "model_name": self.model_name,
            "target_version": target.version,
            "previous_champion_version": previous.version if previous else None,
            "rolled_back_at": rolled_back_at,
            "reason": reason,
            "source": source,
            "probation_id": probation_id,
            "trigger_metrics": trigger_metrics or {},
        }
        payload["serving_reload_notification"] = self._notify_serving_reload()
        write_json(audit_path(self.report_root, "rollback", target.version), payload)
        return payload

    def register_and_promote(self, *, mode: str = "smoke") -> dict[str, Any]:
        registration = self.register_candidate(mode=mode)
        version = registration["model_version"]
        outcome = self.promote_or_reject(version=version)
        return {"registration": registration, "outcome": outcome}

    def _notify_serving_reload(self) -> dict[str, Any]:
        return self.serving_notifier.notify_reload()

    def _notify_serving_reload_impl(self) -> dict[str, Any]:
        reload_config = self.config.get("serving_reload", {})
        url_env = str(reload_config.get("url_env", "SENTINELML_SERVING_RELOAD_URL"))
        timeout_env = str(
            reload_config.get(
                "timeout_seconds_env",
                "SENTINELML_SERVING_RELOAD_TIMEOUT_SECONDS",
            )
        )
        url = os.environ.get(url_env)
        if not url:
            return {"attempted": False, "reason": f"{url_env} is not set"}
        timeout = float(os.environ.get(timeout_env, "2"))
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "attempted": True,
                    "success": 200 <= response.status < 300,
                    "status_code": response.status,
                    "body": body[:500],
                }
        except (OSError, urllib.error.URLError) as exc:
            return {
                "attempted": True,
                "success": False,
                "error": str(exc),
            }

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
