"""Study orchestration and report serialization for Phase 3D."""

from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.optimization.config import (
    DEFAULT_OPTIMIZATION_CONFIG_PATH,
    enabled_model_families,
    load_optimization_config,
    mode_sample_sizes,
    trial_count,
)
from sentinelml.optimization.objective import OptimizationObjective
from sentinelml.optimization.search_spaces import search_space_summary
from sentinelml.tracking.mlflow import (
    collect_reproducibility_lineage,
    configure_mlflow_runtime_environment,
    lineage_tags,
    log_input_artifacts,
    log_lineage_artifacts,
)
from sentinelml.training.compare import write_json
from sentinelml.training.data import (
    DEFAULT_FEATURE_SCHEMA_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    load_feature_schema,
    load_partition,
)
from sentinelml.training.models import (
    DEFAULT_TRAINING_CONFIG_PATH,
    MODEL_FAMILIES,
    load_training_config,
)

MLFLOW_EXPERIMENT_NAME = "sentinelml-optimization"
DEFAULT_OPTIMIZATION_OUTPUT_DIR = PROJECT_ROOT / "reports" / "optimization"


@dataclass(frozen=True)
class StudyArtifacts:
    trials_csv: Path
    best_params_json: Path
    study_summary_json: Path


def create_sampler(config: dict[str, Any]) -> Any:
    import optuna

    return optuna.samplers.TPESampler(seed=int(config["sampler"]["seed"]))


def create_pruner(config: dict[str, Any], model_family: str) -> Any:
    import optuna

    model_config = config["models"][model_family]
    if not bool(model_config.get("pruning_enabled", False)):
        return optuna.pruners.NopPruner()
    pruner_config = config["pruner"]
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_config["n_startup_trials"]),
        n_warmup_steps=int(pruner_config["n_warmup_steps"]),
        interval_steps=int(pruner_config["interval_steps"]),
    )


def trial_rows(study: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": int(trial.number),
            "state": trial.state.name,
            "value": trial.value,
            "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
        }
        for key, value in trial.params.items():
            row[f"param.{key}"] = value
        for key, value in trial.user_attrs.items():
            if key != "mlflow_run_id":
                row[f"user_attr.{key}"] = value
        rows.append(row)
    return rows


def _write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["number", "state", "value", "mlflow_run_id"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trial_state_counts(study: Any) -> dict[str, int]:
    counts = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0}
    for trial in study.trials:
        if trial.state.name in counts:
            counts[trial.state.name] += 1
    return counts


def study_summary(
    study: Any,
    *,
    model_family: str,
    mode: str,
    requested_n_trials: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    complete_trials = [
        trial for trial in study.trials if trial.state.name == "COMPLETE"
    ]
    best_trial = study.best_trial if complete_trials else None
    model_config = config["models"][model_family]
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "model_family": model_family,
        "execution_mode": mode,
        "requested_n_trials": int(requested_n_trials),
        "trial_state_counts": _trial_state_counts(study),
        "objective_metric": config["objective_metric"],
        "direction": config["direction"],
        "sampler": config["sampler"]["name"],
        "seed": int(config["sampler"]["seed"]),
        "pruning_enabled": bool(model_config.get("pruning_enabled", False)),
        "pruning_strategy": str(model_config.get("pruning_strategy", "unsupported")),
        "search_space": search_space_summary(model_config["search_space"]),
        "best_trial_number": None,
        "best_objective_value": None,
        "best_params": {},
        "best_mlflow_run_id": None,
        "best_effective_device": None,
    }
    if best_trial is not None:
        summary.update(
            {
                "best_trial_number": int(best_trial.number),
                "best_objective_value": best_trial.value,
                "best_params": dict(best_trial.params),
                "best_mlflow_run_id": best_trial.user_attrs.get("mlflow_run_id"),
                "best_effective_device": best_trial.user_attrs.get(
                    "effective_device"
                ),
            }
        )
    return summary


def write_study_outputs(
    study: Any,
    *,
    model_family: str,
    mode: str,
    requested_n_trials: int,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[StudyArtifacts, dict[str, Any]]:
    model_output_dir = output_dir / model_family
    rows = trial_rows(study)
    summary = study_summary(
        study,
        model_family=model_family,
        mode=mode,
        requested_n_trials=requested_n_trials,
        config=config,
    )
    artifacts = StudyArtifacts(
        trials_csv=model_output_dir / "trials.csv",
        best_params_json=model_output_dir / "best_params.json",
        study_summary_json=model_output_dir / "study_summary.json",
    )
    _write_trials_csv(artifacts.trials_csv, rows)
    write_json(artifacts.best_params_json, summary["best_params"])
    write_json(artifacts.study_summary_json, summary)
    return artifacts, summary


def _log_parent_summary(
    mlflow: Any,
    summary: dict[str, Any],
    artifacts: StudyArtifacts,
) -> None:
    if summary["best_objective_value"] is not None:
        mlflow.log_metric("best_objective_value", summary["best_objective_value"])
    if summary["best_trial_number"] is not None:
        mlflow.log_param("best_trial_number", summary["best_trial_number"])
    if summary["best_mlflow_run_id"]:
        mlflow.set_tag("best_mlflow_run_id", summary["best_mlflow_run_id"])
    for key, value in summary["best_params"].items():
        mlflow.log_param(f"best.{key}", value)
    for path in [
        artifacts.trials_csv,
        artifacts.best_params_json,
        artifacts.study_summary_json,
    ]:
        mlflow.log_artifact(str(path), artifact_path="optimization")


def _selected_model_families(
    config: dict[str, Any],
    model: str,
) -> list[str]:
    if model == "all":
        return enabled_model_families(config)
    if model not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family: {model}")
    if not bool(config["models"][model].get("enabled", False)):
        raise ValueError(f"model family is disabled in optimization config: {model}")
    return [model]


def run_phase3_optimization(
    *,
    mode: str,
    model: str = "all",
    enable_mlflow: bool = False,
    output_dir: Path = DEFAULT_OPTIMIZATION_OUTPUT_DIR,
    optimization_config_path: Path = DEFAULT_OPTIMIZATION_CONFIG_PATH,
    training_config_path: Path = DEFAULT_TRAINING_CONFIG_PATH,
    feature_schema_path: Path = DEFAULT_FEATURE_SCHEMA_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
) -> dict[str, Any]:
    """Run validation-only Optuna optimization studies."""

    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")

    import optuna

    config = load_optimization_config(optimization_config_path)
    training_config = load_training_config(training_config_path)
    model_families = _selected_model_families(config, model)
    sample_sizes = mode_sample_sizes(config, mode)
    seed = int(training_config["random_seed"])

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

    mlflow = None
    if enable_mlflow:
        tracking_uri = configure_mlflow_runtime_environment()
        import mlflow as mlflow_module

        mlflow = mlflow_module
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    summaries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_family in model_families:
        n_trials = trial_count(config, model_family, mode)
        study_name = f"optuna-study-{model_family}"
        study = optuna.create_study(
            study_name=study_name,
            direction=config["direction"],
            sampler=create_sampler(config),
            pruner=create_pruner(config, model_family),
        )
        objective = OptimizationObjective(
            model_family=model_family,
            mode=mode,
            training_config=training_config,
            optimization_config=config,
            train=train,
            validation=validation,
            enable_mlflow=enable_mlflow,
            mlflow=mlflow,
        )

        parent_context = (
            mlflow.start_run(run_name=study_name)
            if mlflow is not None
            else nullcontext()
        )
        with parent_context as parent_run:
            if parent_run is not None:
                mlflow.set_tags(
                    {
                        "project": "SentinelML",
                        "run_type": "optuna_study",
                        "pipeline_stage": "optimization",
                        "model_family": model_family,
                        "execution_mode": mode,
                        "objective_metric": config["objective_metric"],
                        "direction": config["direction"],
                        "sampler": config["sampler"]["name"],
                        "seed": str(config["sampler"]["seed"]),
                        "pruning_enabled": str(
                            config["models"][model_family]["pruning_enabled"]
                        ).lower(),
                        "pruning_strategy": str(
                            config["models"][model_family]["pruning_strategy"]
                        ),
                    }
                )
                mlflow.log_params(
                    {
                        "requested_n_trials": n_trials,
                        "objective_metric": config["objective_metric"],
                        "direction": config["direction"],
                        "sampler": config["sampler"]["name"],
                        "seed": int(config["sampler"]["seed"]),
                    }
                )
                lineage = collect_reproducibility_lineage(
                    mlflow_parent_run_id=parent_run.info.run_id,
                    training_config_path=training_config_path,
                    feature_schema_path=feature_schema_path,
                    optimization_config_path=optimization_config_path,
                )
                mlflow.set_tags(lineage_tags(lineage))
                log_lineage_artifacts(mlflow, lineage)
                log_input_artifacts(
                    mlflow,
                    [
                        training_config_path,
                        optimization_config_path,
                        feature_schema_path,
                    ],
                )

            study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))
            artifacts, summary = write_study_outputs(
                study,
                model_family=model_family,
                mode=mode,
                requested_n_trials=n_trials,
                config=config,
                output_dir=output_dir,
            )
            if mlflow is not None:
                _log_parent_summary(mlflow, summary, artifacts)
            summaries.append(summary)

    optimization_summary = {
        "execution_mode": mode,
        "objective_metric": config["objective_metric"],
        "direction": config["direction"],
        "model_families": model_families,
        "studies": summaries,
        "validation_only_comparison": sorted(
            [
                {
                    "model_family": summary["model_family"],
                    "best_objective_value": summary["best_objective_value"],
                    "best_trial_number": summary["best_trial_number"],
                    "best_mlflow_run_id": summary["best_mlflow_run_id"],
                }
                for summary in summaries
                if summary["best_objective_value"] is not None
            ],
            key=lambda row: row["best_objective_value"],
            reverse=config["direction"] == "maximize",
        ),
        "data_usage": {
            "train": train.metadata,
            "validation": validation.metadata,
            "test": "not_loaded_or_used",
        },
    }
    write_json(output_dir / "optimization_summary.json", optimization_summary)
    return optimization_summary
