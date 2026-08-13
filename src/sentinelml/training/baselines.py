"""Phase 2 baseline training workflow."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.tracking.mlflow import (
    collect_reproducibility_lineage,
    configure_mlflow_runtime_environment,
    dvc_status_is_clean,
    flatten_mlflow_metrics,
    lineage_tags,
    log_input_artifacts,
    log_lineage_artifacts,
    run_repository_command,
    sha256_file,
)
from sentinelml.training.compare import (
    comparison_rows,
    select_best_baseline,
    write_comparison_csv,
    write_json,
    write_markdown_report,
)
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
    build_baseline_model_specs,
    load_training_config,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "models" / "phase2_baseline"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "models" / "phase2_smoke"
MLFLOW_EXPERIMENT_NAME = "sentinelml-baselines"
__all__ = [
    "collect_reproducibility_lineage",
    "configure_mlflow_runtime_environment",
    "dvc_status_is_clean",
    "run_phase2_baselines",
    "run_repository_command",
    "sha256_file",
]


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def assert_files_unchanged(before: dict[str, dict[str, Any]]) -> None:
    after = {
        name: file_fingerprint(Path(info["path"])) for name, info in before.items()
    }
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(
            f"Phase 1 dataset files changed during Phase 2 run: {changed}"
        )


def _sample_sizes(
    *,
    mode: str,
    train_sample_size: int,
    validation_sample_size: int,
    test_sample_size: int,
) -> dict[str, int | None]:
    if mode == "full":
        return {"train": None, "validation": None, "test": None}
    return {
        "train": train_sample_size,
        "validation": validation_sample_size,
        "test": test_sample_size,
    }


def _flatten_mlflow_metrics(
    prefix: str,
    values: dict[str, Any],
) -> dict[str, float]:
    return flatten_mlflow_metrics(prefix, values)


def _log_parent_lineage_artifacts(mlflow: Any, lineage: dict[str, Any]) -> None:
    log_lineage_artifacts(mlflow, lineage)


def _log_mlflow_dict(
    mlflow: Any,
    dictionary: dict[str, Any],
    artifact_file: str,
) -> None:
    local_path = PROJECT_ROOT / ".tmp" / "mlflow-artifacts" / artifact_file
    local_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(local_path, dictionary)
    artifact_dir = str(Path(artifact_file).parent).replace("\\", "/")
    _log_mlflow_artifact(mlflow, local_path, artifact_path=artifact_dir)


def _log_mlflow_artifact(
    mlflow: Any,
    local_path: Path,
    *,
    artifact_path: str,
) -> None:
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


def _log_mlflow_artifacts(
    mlflow: Any,
    local_dir: Path,
    *,
    artifact_path: str,
) -> None:
    mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)


def run_phase2_baselines(
    *,
    mode: str,
    output_dir: Path | None = None,
    feature_schema_path: Path = DEFAULT_FEATURE_SCHEMA_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    train_sample_size: int = 20_000,
    validation_sample_size: int = 10_000,
    test_sample_size: int = 10_000,
    training_config_path: Path = DEFAULT_TRAINING_CONFIG_PATH,
    enable_mlflow: bool = False,
) -> dict[str, Any]:
    """Train, evaluate, compare, and report the four Phase 2 baselines."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be either 'smoke' or 'full'")
    output_dir = output_dir or (
        SMOKE_OUTPUT_DIR if mode == "smoke" else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    protected_files = {
        "train": file_fingerprint(train_path),
        "validation": file_fingerprint(validation_path),
        "test": file_fingerprint(test_path),
    }
    schema = load_feature_schema(feature_schema_path)
    training_config = load_training_config(training_config_path)
    seed = int(training_config["random_seed"])
    feature_columns = schema["feature_columns"]
    target_column = schema["target_column"]
    sizes = _sample_sizes(
        mode=mode,
        train_sample_size=train_sample_size,
        validation_sample_size=validation_sample_size,
        test_sample_size=test_sample_size,
    )

    train = load_partition(
        train_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_size=sizes["train"],
        seed=seed,
    )
    validation = load_partition(
        validation_path,
        feature_columns=feature_columns,
        target_column=target_column,
        sample_size=sizes["validation"],
        seed=seed + 10,
    )

    model_specs = build_baseline_model_specs(train.target, config=training_config)
    validation_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    feature_importance: dict[str, Any] = {}

    mlflow = None
    mlflow_client = None
    child_run_ids: dict[str, str] = {}

    if enable_mlflow:
        tracking_uri = configure_mlflow_runtime_environment()

        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        parent_run = mlflow.start_run(run_name="baseline-suite")
        mlflow_client = MlflowClient()
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow parent run ID: {parent_run.info.run_id}")

        try:
            mlflow.set_tags(
                {
                    "project": "SentinelML",
                    "run_type": "baseline_suite",
                    "pipeline_stage": "baseline_training",
                    "dataset": "CICIDS2017",
                    "task": "binary_intrusion_detection",
                    "execution_mode": mode,
                }
            )
            mlflow.log_params(
                {
                    "seed": seed,
                    "feature_count": len(feature_columns),
                    "training_config_path": str(training_config_path),
                    "feature_schema_path": str(feature_schema_path),
                }
            )
            lineage = collect_reproducibility_lineage(
                mlflow_parent_run_id=parent_run.info.run_id,
                training_config_path=training_config_path,
                feature_schema_path=feature_schema_path,
            )
            mlflow.set_tags(lineage_tags(lineage))
            _log_parent_lineage_artifacts(mlflow, lineage)
        except Exception:
            mlflow.end_run()
            raise

    try:
        for spec in model_specs:
            child_context = (
                mlflow.start_run(run_name=spec.name, nested=True)
                if mlflow is not None
                else nullcontext()
            )
            with child_context as child_run:
                if child_run is not None:
                    child_run_ids[spec.name] = child_run.info.run_id
                    print(f"MLflow child run: {spec.name} ({child_run.info.run_id})")
                    mlflow.set_tags(
                        {
                            "project": "SentinelML",
                            "run_type": "baseline_model",
                            "model_family": spec.name,
                            "execution_mode": mode,
                        }
                    )

                start = perf_counter()
                fit_estimator_on_configured_device(
                    spec.estimator,
                    train.features,
                    train.target,
                )
                training_seconds = perf_counter() - start
                effective_device = estimator_effective_device(spec.estimator)
                metrics = evaluate_binary_classifier(
                    spec.estimator,
                    validation.features,
                    validation.target,
                )
                importance_rows, unsupported_reason = feature_importance_table(
                    model_name=spec.name,
                    model=spec.estimator,
                    feature_columns=feature_columns,
                )
                validation_results[spec.name] = {
                    "validation_metrics": metrics,
                    "training_time_seconds": float(training_seconds),
                    "effective_device": effective_device,
                }
                fitted_models[spec.name] = spec.estimator
                parameters[spec.name] = spec.parameters
                feature_importance[spec.name] = {
                    "rows": importance_rows,
                    "unsupported_reason": unsupported_reason,
                    "destination_port": destination_port_summary(importance_rows),
                }

                if mlflow is not None:
                    mlflow.log_params(
                        {
                            f"model_param.{key}": value
                            for key, value in spec.parameters.items()
                            if isinstance(value, str | int | float | bool)
                            or value is None
                        }
                    )
                    mlflow.log_metric("training_time_seconds", float(training_seconds))
                    if effective_device is not None:
                        mlflow.set_tag("effective_device", effective_device)
                    mlflow.log_metrics(_flatten_mlflow_metrics("", metrics))
                    destination_port = feature_importance[spec.name][
                        "destination_port"
                    ]
                    if destination_port is not None:
                        mlflow.log_metric(
                            "destination_port.rank",
                            float(destination_port["rank"]),
                        )
                        mlflow.log_metric(
                            "destination_port.importance",
                            float(destination_port["importance"]),
                        )

        comparison = comparison_rows(validation_results)
        selected = select_best_baseline(comparison)
        selected_model_name = selected["recommended_baseline"]
        if mlflow is not None:
            mlflow.set_tag("selected_baseline", selected_model_name)

        test = load_partition(
            test_path,
            feature_columns=feature_columns,
            target_column=target_column,
            sample_size=sizes["test"],
            seed=seed + 20,
        )
        selected_test_metrics = evaluate_binary_classifier(
            fitted_models[selected_model_name],
            test.features,
            test.target,
        )

        if mlflow_client is not None:
            selected_child_run_id = child_run_ids[selected_model_name]
            for metric_name, metric_value in _flatten_mlflow_metrics(
                "selected_test",
                selected_test_metrics,
            ).items():
                mlflow_client.log_metric(
                    selected_child_run_id,
                    metric_name,
                    float(metric_value),
                )
            mlflow_client.set_tag(
                selected_child_run_id,
                "selected_baseline",
                "true",
            )

        sample_metadata = {
            "train": train.metadata,
            "validation": validation.metadata,
            "test": test.metadata,
        }
        report = {
            "mode": mode,
            "seed": seed,
            "training_config_path": str(training_config_path),
            "feature_schema_path": str(feature_schema_path),
            "feature_count": len(feature_columns),
            "sample_metadata": sample_metadata,
            "parameters": parameters,
            "validation_results": validation_results,
            "comparison": comparison,
            "selected": selected,
            "selected_test_metrics": selected_test_metrics,
            "feature_importance": feature_importance,
            "phase1_dataset_fingerprints_before": protected_files,
        }

        write_json(output_dir / "baseline_parameters.json", parameters)
        write_json(output_dir / "validation_metrics.json", validation_results)
        write_json(output_dir / "comparison.json", comparison)
        write_comparison_csv(output_dir / "comparison.csv", comparison)
        write_json(output_dir / "selected_baseline.json", selected)
        write_json(output_dir / "selected_test_metrics.json", selected_test_metrics)
        write_json(output_dir / "feature_importance.json", feature_importance)
        write_markdown_report(
            path=output_dir / "phase2_baseline_report.md",
            mode=mode,
            sample_metadata=sample_metadata,
            comparison=comparison,
            selected=selected,
            validation_results=validation_results,
            selected_test_metrics=selected_test_metrics,
            feature_importance=feature_importance,
        )
        write_json(output_dir / "run_manifest.json", report)

        if mlflow is not None:
            log_input_artifacts(
                mlflow,
                [training_config_path, feature_schema_path],
            )
            _log_mlflow_artifacts(
                mlflow,
                output_dir,
                artifact_path="phase2_baseline_outputs",
            )

        assert_files_unchanged(protected_files)
        return report
    finally:
        if mlflow is not None:
            mlflow.end_run()
