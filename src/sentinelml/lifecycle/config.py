"""Configuration loading for Phase 4 model lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentinelml.config import load_yaml_mapping
from sentinelml.data.config import PROJECT_ROOT, resolve_project_path

DEFAULT_LIFECYCLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "lifecycle_config.yaml"


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load a nested YAML config mapping."""

    return load_yaml_mapping(path)


def load_lifecycle_config(path: Path = DEFAULT_LIFECYCLE_CONFIG_PATH) -> dict[str, Any]:
    config = load_simple_yaml(path)
    validate_lifecycle_config(config)
    return config


def validate_lifecycle_config(config: dict[str, Any]) -> None:
    if not config.get("registered_model_name"):
        raise ValueError("lifecycle config requires registered_model_name")
    if not config.get("champion_alias"):
        raise ValueError("lifecycle config requires champion_alias")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("lifecycle config requires paths")
    for key in [
        "final_candidate_manifest",
        "smoke_baseline_metrics",
        "smoke_selected_baseline",
        "feature_schema",
        "train_partition",
        "validation_partition",
        "threshold_report",
        "lifecycle_reports",
        "pending_dir",
    ]:
        if key not in paths:
            raise ValueError(f"lifecycle config paths missing {key}")
    metrics = config.get("threshold_policy", {}).get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("lifecycle config requires threshold metrics")
    promotion_evaluation = config.get("promotion_evaluation")
    if not isinstance(promotion_evaluation, dict):
        raise ValueError("lifecycle config requires promotion_evaluation")
    for key in [
        "validation_sample_size",
        "baseline_train_sample_size",
        "random_seed",
        "sampling_strategy",
        "min_positive_rows",
        "max_positive_fraction",
    ]:
        if key not in promotion_evaluation:
            raise ValueError(f"promotion_evaluation missing {key}")
    weights = config.get("composite_score", {}).get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("lifecycle config requires composite weights")


def configured_path(config: dict[str, Any], key: str) -> Path:
    return resolve_project_path(config["paths"][key])
