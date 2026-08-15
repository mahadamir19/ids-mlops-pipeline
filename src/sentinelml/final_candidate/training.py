"""Phase 3E final-candidate training and MLflow logging."""

from __future__ import annotations

import sys
from contextlib import nullcontext
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
from sentinelml.final_candidate.selection import (
    DEFAULT_OPTIMIZATION_REPORT_DIR,
    select_optimization_winner,
)
from sentinelml.optimization.config import (
    DEFAULT_OPTIMIZATION_CONFIG_PATH,
    load_optimization_config,
    mode_sample_sizes,
    overlay_model_params,
)
from sentinelml.tracking.mlflow import (
    collect_reproducibility_lineage,
    configure_mlflow_runtime_environment,
    flatten_mlflow_metrics,
    lineage_tags,
    log_input_artifacts,
    log_lineage_artifacts,
    sha256_file,
)
from sentinelml.training.compare import write_json
from sentinelml.training.data import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    load_feature_schema,
    load_partition,
)
from sentinelml.training.evaluate import (
    destination_port_summary,
    evaluate_binary_classifier,
    feature_importance_table,
)
from sentinelml.training.gpu import (
    estimator_effective_device,
    fit_estimator_on_configured_device,
)
from sentinelml.training.models import (
    DEFAULT_TRAINING_CONFIG_PATH,
    build_baseline_model_spec,
    load_training_config,
)

MLFLOW_EXPERIMENT_NAME = "sentinelml-final-training"
DEFAULT_FINAL_CANDIDATE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "final_candidate"


def log_model_artifact(
    mlflow: Any,
    *,
    model_family: str,
    model: Any,
    signature: Any,
    input_example: Any,
) -> dict[str, Any]:
    """Log the final candidate model with the correct MLflow flavor."""

    if model_family == "xgboost":
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        flavor = "xgboost"
    else:
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        flavor = "sklearn"
    return {
        "flavor": flavor,
        "model_uri": getattr(model_info, "model_uri", None),
        "artifact_path": getattr(model_info, "artifact_path", "model"),
        "model_uuid": getattr(model_info, "model_uuid", None),
    }


def _write_final_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    selected_optimization: dict[str, Any],
    feature_importance: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_manifest": output_dir / "model_manifest.json",
        "validation_metrics": output_dir / "validation_metrics.json",
        "test_metrics": output_dir / "test_metrics.json",
        "selected_optimization": output_dir / "selected_optimization.json",
        "feature_importance": output_dir / "feature_importance.json",
    }
    write_json(paths["model_manifest"], manifest)
    write_json(paths["validation_metrics"], validation_metrics)
    write_json(paths["test_metrics"], test_metrics)
    write_json(paths["selected_optimization"], selected_optimization)
    write_json(paths["feature_importance"], feature_importance)
    return paths


def run_phase3_final_candidate(
    *,
    mode: str,
    enable_mlflow: bool = False,
    output_dir: Path = DEFAULT_FINAL_CANDIDATE_OUTPUT_DIR,
    optimization_report_dir: Path = DEFAULT_OPTIMIZATION_REPORT_DIR,
    optimization_config_path: Path = DEFAULT_OPTIMIZATION_CONFIG_PATH,
    training_config_path: Path = DEFAULT_TRAINING_CONFIG_PATH,
    feature_schema_path: Path = DEFAULT_FEATURE_SCHEMA_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Train one final candidate from Phase 3D results and evaluate test once."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")

    selected = select_optimization_winner(
        mode=mode,
        optimization_report_dir=optimization_report_dir,
    )
    dependency_lock_path = ensure_dependency_lock(
        python_executable=python_executable,
        repo_root=PROJECT_ROOT,
    )
    freeze_path = write_pip_freeze_snapshot(
        python_executable=python_executable,
        output_path=output_dir / "environment" / "pip_freeze.txt",
        repo_root=PROJECT_ROOT,
    )

    optimization_config = load_optimization_config(optimization_config_path)
    training_config = load_training_config(training_config_path)
    seed = int(training_config["random_seed"])
    sample_sizes = mode_sample_sizes(optimization_config, mode)
    test_sample_size = sample_sizes["validation"] if mode == "smoke" else None

    schema = load_feature_schema(feature_schema_path)
    feature_columns = schema["feature_columns"]
    target_column = schema["target_column"]
    train = load_partition(
        train_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_size=sample_sizes["train"],
        seed=seed,
    )
    validation = load_partition(
        validation_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_size=sample_sizes["validation"],
        seed=seed + 10,
    )

    candidate_config = overlay_model_params(
        training_config,
        selected["model_family"],
        selected["best_params"],
    )
    model_spec = build_baseline_model_spec(
        selected["model_family"],
        train.target,
        config=candidate_config,
    )

    start = perf_counter()
    fit_estimator_on_configured_device(
        model_spec.estimator,
        train.features,
        train.target,
    )
    training_seconds = perf_counter() - start
    effective_device = estimator_effective_device(model_spec.estimator)
    validation_metrics = evaluate_binary_classifier(
        model_spec.estimator,
        validation.features,
        validation.target,
    )

    test = load_partition(
        test_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_size=test_sample_size,
        seed=seed + 20,
    )
    test_metrics = evaluate_binary_classifier(
        model_spec.estimator,
        test.features,
        test.target,
    )
    importance_rows, unsupported_reason = feature_importance_table(
        model_name=selected["model_family"],
        model=model_spec.estimator,
        feature_columns=feature_columns,
    )
    feature_importance = {
        "rows": importance_rows,
        "unsupported_reason": unsupported_reason,
        "destination_port": destination_port_summary(importance_rows),
    }

    input_example = train.features.head(5).copy()
    predictions = model_spec.estimator.predict(input_example)
    signature = infer_signature(input_example, predictions)

    mlflow = None
    if enable_mlflow:
        tracking_uri = configure_mlflow_runtime_environment()
        import mlflow as mlflow_module

        mlflow = mlflow_module
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    run_context = (
        mlflow.start_run(run_name="final-candidate")
        if mlflow is not None
        else nullcontext()
    )
    with run_context as run:
        run_id = run.info.run_id if run is not None else None
        model_info: dict[str, Any] = {
            "flavor": None,
            "model_uri": None,
            "artifact_path": "model",
            "model_uuid": None,
        }
        if mlflow is not None:
            mlflow.set_tags(
                {
                    "project": "SentinelML",
                    "run_type": "final_candidate",
                    "pipeline_stage": "final_training",
                    "model_family": selected["model_family"],
                    "execution_mode": mode,
                    "candidate_source": "optuna_winner",
                    "registry_status": "unregistered",
                    "source_optuna_mlflow_run_id": selected["best_mlflow_run_id"],
                }
            )
            mlflow.log_params(
                {
                    "random_seed": seed,
                    "feature_count": len(feature_columns),
                    "selected_model_family": selected["model_family"],
                    "source_study_name": selected["study_name"],
                    "source_best_trial_number": selected["best_trial_number"],
                    "source_objective_metric": selected["objective_metric"],
                    "source_objective_value": selected["best_objective_value"],
                }
            )
            mlflow.log_params(
                {
                    f"winning.{key}": value
                    for key, value in selected["best_params"].items()
                }
            )
            mlflow.log_metric("training_time_seconds", float(training_seconds))
            if effective_device is not None:
                mlflow.set_tag("effective_device", effective_device)
            mlflow.log_metrics(flatten_mlflow_metrics("val", validation_metrics))
            mlflow.log_metrics(flatten_mlflow_metrics("test", test_metrics))
            model_info = log_model_artifact(
                mlflow,
                model_family=selected["model_family"],
                model=model_spec.estimator,
                signature=signature,
                input_example=input_example,
            )
            lineage = collect_reproducibility_lineage(
                mlflow_parent_run_id=str(run_id),
                training_config_path=training_config_path,
                feature_schema_path=feature_schema_path,
                optimization_config_path=optimization_config_path,
                dependency_lock_path=dependency_lock_path,
            )
        else:
            lineage = collect_reproducibility_lineage(
                mlflow_parent_run_id="disabled",
                training_config_path=training_config_path,
                feature_schema_path=feature_schema_path,
                optimization_config_path=optimization_config_path,
                dependency_lock_path=dependency_lock_path,
            )

        dep_metadata = dependency_lock_metadata(dependency_lock_path)
        manifest = {
            "schema_version": "1.0",
            "model_name": "sentinelml-ids",
            "model_family": selected["model_family"],
            "created_at_utc": None,
            "random_seed": seed,
            "source_optimization": {
                "study_name": selected["study_name"],
                "best_trial_number": selected["best_trial_number"],
                "best_trial_mlflow_run_id": selected["best_mlflow_run_id"],
                "objective_metric": selected["objective_metric"],
                "objective_value": selected["best_objective_value"],
                "direction": selected["direction"],
            },
            "winning_hyperparameters": selected["best_params"],
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
                "optimization_config_sha256": lineage["optimization_config_sha256"],
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
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
            "registry_state": {
                "registered": False,
                "model_version": None,
            },
        }

        import datetime as _datetime

        manifest["created_at_utc"] = _datetime.datetime.now(
            _datetime.UTC
        ).isoformat()
        output_paths = _write_final_outputs(
            output_dir=output_dir,
            manifest=manifest,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            selected_optimization=selected,
            feature_importance=feature_importance,
        )
        if mlflow is not None:
            mlflow.set_tags(lineage_tags(lineage))
            log_input_artifacts(
                mlflow,
                [training_config_path, optimization_config_path, feature_schema_path],
            )
            log_lineage_artifacts(
                mlflow,
                lineage,
                dependency_lock_path=dependency_lock_path,
            )
            mlflow.log_artifact(str(freeze_path), artifact_path="environment")
            for path in output_paths.values():
                mlflow.log_artifact(str(path), artifact_path="final_candidate_outputs")

    return manifest
