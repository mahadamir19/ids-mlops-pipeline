"""Configuration loading for Phase 3D Optuna optimization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.training.models import MODEL_FAMILIES

DEFAULT_OPTIMIZATION_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "optimization_config.yaml"
)
VALID_DIRECTIONS = {"maximize", "minimize"}


def _parse_scalar(value: str) -> int | float | bool | None | str:
    value = value.strip()
    lowered = value.lower()
    if lowered == "":
        raise ValueError("empty scalar values are not supported")
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_optimization_config(
    path: Path = DEFAULT_OPTIMIZATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the simple nested YAML optimization configuration."""

    config: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, config)]
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        indent = len(content) - len(content.lstrip(" "))
        if indent % 2:
            raise ValueError(f"invalid indentation on line {line_number}")
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"expected key/value pair on line {line_number}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value.strip():
            parent[key] = _parse_scalar(raw_value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))

    validate_optimization_config(config)
    return config


def validate_optimization_config(config: dict[str, Any]) -> None:
    if config.get("objective_metric") != "pr_auc":
        objective = config.get("objective_metric")
        if not isinstance(objective, str) or not objective:
            raise ValueError("optimization config requires objective_metric")
    if config.get("direction") not in VALID_DIRECTIONS:
        raise ValueError("optimization direction must be maximize or minimize")
    if int(config.get("sampler", {}).get("seed", -1)) < 0:
        raise ValueError("sampler.seed must be non-negative")

    modes = config.get("modes")
    if not isinstance(modes, dict) or not {"smoke", "full"}.issubset(modes):
        raise ValueError("optimization config requires smoke and full modes")
    for mode, mode_config in modes.items():
        for key in ["train_sample_size", "validation_sample_size"]:
            value = mode_config.get(key)
            if value is not None and int(value) < 2:
                raise ValueError(f"{mode}.{key} must be null or at least 2")

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("optimization config requires models")
    missing_models = sorted(set(MODEL_FAMILIES) - set(models))
    if missing_models:
        raise ValueError(f"optimization config is missing models: {missing_models}")
    for model_family in MODEL_FAMILIES:
        model_config = models[model_family]
        for key in ["enabled", "smoke_trials", "full_trials", "search_space"]:
            if key not in model_config:
                raise ValueError(f"{model_family} is missing {key}")
        smoke_trials = int(model_config["smoke_trials"])
        full_trials = int(model_config["full_trials"])
        if smoke_trials < 0 or full_trials < 0:
            raise ValueError(f"{model_family} trial counts must be non-negative")


def enabled_model_families(config: dict[str, Any]) -> list[str]:
    return [
        model_family
        for model_family in MODEL_FAMILIES
        if bool(config["models"][model_family].get("enabled", False))
    ]


def trial_count(config: dict[str, Any], model_family: str, mode: str) -> int:
    return int(config["models"][model_family][f"{mode}_trials"])


def mode_sample_sizes(config: dict[str, Any], mode: str) -> dict[str, int | None]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    mode_config = config["modes"][mode]
    return {
        "train": mode_config["train_sample_size"],
        "validation": mode_config["validation_sample_size"],
    }


def overlay_model_params(
    training_config: dict[str, Any],
    model_family: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    trial_config = deepcopy(training_config)
    trial_config["baseline"][model_family].update(params)
    if model_family == "xgboost":
        trial_config["baseline"][model_family]["eval_metric"] = "aucpr"
    return trial_config
