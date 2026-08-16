"""Candidate registration worker for the LifecycleService facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sentinelml.lifecycle.audit import audit_path, utc_now, write_json
from sentinelml.lifecycle.registry import (
    ensure_registered_model,
    find_duplicate_source_version,
    set_version_tags,
    tag_json,
)


class CandidateRegistrar:
    def __init__(
        self,
        *,
        client: Any,
        mlflow: Any,
        model_name: str,
        report_root: Path,
        load_manifest: Callable[..., dict[str, Any]],
        validate_model_uri: Callable[[str], None],
        execution_mode_from_manifest: Callable[[dict[str, Any]], str],
        demo_model_for_mode: Callable[[str], str],
    ) -> None:
        self.client = client
        self.mlflow = mlflow
        self.model_name = model_name
        self.report_root = report_root
        self.load_manifest = load_manifest
        self.validate_model_uri = validate_model_uri
        self.execution_mode_from_manifest = execution_mode_from_manifest
        self.demo_model_for_mode = demo_model_for_mode

    def register_candidate(
        self,
        *,
        mode: str = "smoke",
        force_new_version: bool = False,
        manifest_path: Path | None = None,
        extra_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = self.load_manifest(mode=mode, manifest_path=manifest_path)
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

        self.validate_model_uri(source_model_uri)
        created = self.mlflow.register_model(source_model_uri, self.model_name)
        version = str(created.version)
        execution_mode = self.execution_mode_from_manifest(manifest)
        tags = self.registration_tags(manifest, execution_mode)
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

    def registration_tags(
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
