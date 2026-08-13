"""Optuna objective functions for Phase 3D optimization."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sentinelml.optimization.config import overlay_model_params
from sentinelml.optimization.search_spaces import suggest_hyperparameters
from sentinelml.tracking.mlflow import flatten_mlflow_metrics
from sentinelml.training.data import DatasetBundle
from sentinelml.training.evaluate import evaluate_binary_classifier
from sentinelml.training.gpu import (
    estimator_effective_device,
    fit_estimator_on_configured_device,
)
from sentinelml.training.models import build_baseline_model_spec


def objective_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    """Extract the configured scalar objective from validation metrics."""

    value: Any = metrics
    for part in metric_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(
                f"objective metric not found in validation metrics: {metric_name}"
            )
        value = value[part]
    if not isinstance(value, int | float) or value is None:
        raise TypeError(f"objective metric is not numeric: {metric_name}")
    return float(value)


def _flatten_params(prefix: str, values: Any) -> dict[str, Any]:
    if isinstance(values, dict):
        params: dict[str, Any] = {}
        for key, value in values.items():
            metric_key = f"{prefix}.{key}" if prefix else str(key)
            params.update(_flatten_params(metric_key, value))
        return params
    if isinstance(values, list):
        return {prefix: ",".join(str(value) for value in values)}
    if isinstance(values, str | int | float | bool) or values is None:
        return {prefix: values}
    return {}


def _xgboost_pruning_callback(trial: Any) -> Any:
    try:
        from optuna_integration.xgboost import XGBoostPruningCallback
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "XGBoost pruning requires optuna-integration[xgboost]. "
            "Install the project tracking extra."
        ) from exc
    return XGBoostPruningCallback(trial, "validation_0-aucpr")


@dataclass
class TrialResult:
    model_family: str
    trial_number: int
    objective_metric: str
    objective_value: float
    sampled_params: dict[str, Any]
    validation_metrics: dict[str, Any]
    training_time_seconds: float
    effective_device: str | None
    mlflow_run_id: str | None
    state: str


@dataclass
class OptimizationObjective:
    model_family: str
    mode: str
    training_config: dict[str, Any]
    optimization_config: dict[str, Any]
    train: DatasetBundle
    validation: DatasetBundle
    enable_mlflow: bool = False
    mlflow: Any | None = None

    def __post_init__(self) -> None:
        self.results: list[TrialResult] = []

    @property
    def objective_metric(self) -> str:
        return str(self.optimization_config["objective_metric"])

    @property
    def pruning_enabled(self) -> bool:
        model_config = self.optimization_config["models"][self.model_family]
        return bool(model_config.get("pruning_enabled", False))

    def _fit_model(self, model: Any, trial: Any) -> float:
        callbacks = []
        if self.model_family == "xgboost" and self.pruning_enabled:
            callbacks.append(_xgboost_pruning_callback(trial))
            model.set_params(callbacks=callbacks)

        start = perf_counter()
        if self.model_family == "xgboost" and callbacks:
            fit_estimator_on_configured_device(
                model,
                self.train.features,
                self.train.target,
                eval_set=[(self.validation.features, self.validation.target)],
                verbose=False,
            )
        else:
            fit_estimator_on_configured_device(
                model,
                self.train.features,
                self.train.target,
            )
        return perf_counter() - start

    def __call__(self, trial: Any) -> float:
        model_config = self.optimization_config["models"][self.model_family]
        sampled_params = suggest_hyperparameters(
            trial,
            self.model_family,
            model_config["search_space"],
        )
        trial_config = overlay_model_params(
            self.training_config,
            self.model_family,
            sampled_params,
        )
        model_spec = build_baseline_model_spec(
            self.model_family,
            self.train.target,
            config=trial_config,
        )

        child_context = (
            self.mlflow.start_run(
                run_name=f"trial-{trial.number:03d}",
                nested=True,
            )
            if self.enable_mlflow and self.mlflow is not None
            else nullcontext()
        )
        mlflow_run_id = None
        with child_context as child_run:
            if child_run is not None:
                mlflow_run_id = child_run.info.run_id
                trial.set_user_attr("mlflow_run_id", mlflow_run_id)
                self.mlflow.set_tags(
                    {
                        "project": "SentinelML",
                        "run_type": "optuna_trial",
                        "model_family": self.model_family,
                        "trial_number": str(trial.number),
                        "execution_mode": self.mode,
                        "trial_state": "running",
                    }
                )
                self.mlflow.log_params(
                    {f"sampled.{key}": value for key, value in sampled_params.items()}
                )
                self.mlflow.log_params(
                    {
                        f"model_param.{key}": value
                        for key, value in _flatten_params(
                            "",
                            model_spec.parameters,
                        ).items()
                    }
                )

            try:
                training_seconds = self._fit_model(model_spec.estimator, trial)
                effective_device = estimator_effective_device(model_spec.estimator)
                metrics = evaluate_binary_classifier(
                    model_spec.estimator,
                    self.validation.features,
                    self.validation.target,
                )
                objective_value = objective_metric_value(metrics, self.objective_metric)
            except Exception as exc:
                if self.mlflow is not None and child_run is not None:
                    trial_state = (
                        "pruned"
                        if exc.__class__.__name__ == "TrialPruned"
                        else "failed"
                    )
                    self.mlflow.set_tag("trial_state", trial_state)
                raise

            trial.set_user_attr("mlflow_run_id", mlflow_run_id)
            trial.set_user_attr("objective_metric", self.objective_metric)
            trial.set_user_attr("objective_value", objective_value)
            trial.set_user_attr("effective_device", effective_device)
            if mlflow_run_id is not None:
                self.mlflow.log_metric("objective_value", objective_value)
                self.mlflow.log_metric(
                    "training_time_seconds",
                    float(training_seconds),
                )
                self.mlflow.log_metrics(flatten_mlflow_metrics("", metrics))
                if effective_device is not None:
                    self.mlflow.set_tag("effective_device", effective_device)
                self.mlflow.set_tag("trial_state", "complete")

        self.results.append(
            TrialResult(
                model_family=self.model_family,
                trial_number=int(trial.number),
                objective_metric=self.objective_metric,
                objective_value=objective_value,
                sampled_params=sampled_params,
                validation_metrics=metrics,
                training_time_seconds=float(training_seconds),
                effective_device=effective_device,
                mlflow_run_id=mlflow_run_id,
                state="COMPLETE",
            )
        )
        return objective_value
