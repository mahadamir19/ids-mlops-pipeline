"""Selection of the Phase 3E final candidate from Phase 3D outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT

DEFAULT_OPTIMIZATION_REPORT_DIR = PROJECT_ROOT / "reports" / "optimization"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_optimization_winner(
    *,
    mode: str,
    optimization_report_dir: Path = DEFAULT_OPTIMIZATION_REPORT_DIR,
) -> dict[str, Any]:
    """Select the validation-only Optuna winner for the requested mode."""

    summary_path = optimization_report_dir / "optimization_summary.json"
    summary = load_json(summary_path)
    summary_mode = summary.get("execution_mode")
    if summary_mode != mode:
        raise ValueError(
            "optimization mode mismatch: "
            f"requested {mode}, found {summary_mode} in {summary_path}"
        )

    comparison = summary.get("validation_only_comparison")
    if not isinstance(comparison, list) or not comparison:
        raise ValueError("optimization summary has no validation-only comparison")
    direction = summary.get("direction")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("optimization summary has invalid direction")

    winner_row = sorted(
        comparison,
        key=lambda row: row["best_objective_value"],
        reverse=direction == "maximize",
    )[0]
    model_family = winner_row["model_family"]
    study_path = optimization_report_dir / model_family / "study_summary.json"
    params_path = optimization_report_dir / model_family / "best_params.json"
    study_summary = load_json(study_path)
    best_params = load_json(params_path)

    if study_summary.get("execution_mode") != mode:
        raise ValueError(
            "selected study mode mismatch: "
            f"requested {mode}, found {study_summary.get('execution_mode')}"
        )
    expected = {
        "best_trial_number": winner_row["best_trial_number"],
        "best_objective_value": winner_row["best_objective_value"],
        "best_mlflow_run_id": winner_row["best_mlflow_run_id"],
    }
    for key, value in expected.items():
        if study_summary.get(key) != value:
            raise ValueError(f"selected study has inconsistent {key}")
    if study_summary.get("best_params") != best_params:
        raise ValueError("best_params.json does not match selected study summary")
    if not best_params:
        raise ValueError("selected study does not contain winning hyperparameters")

    return {
        "source_summary_path": str(summary_path),
        "source_study_path": str(study_path),
        "source_best_params_path": str(params_path),
        "study_name": study_summary["study_name"],
        "model_family": model_family,
        "execution_mode": mode,
        "best_trial_number": study_summary["best_trial_number"],
        "best_objective_value": study_summary["best_objective_value"],
        "best_mlflow_run_id": study_summary["best_mlflow_run_id"],
        "objective_metric": study_summary["objective_metric"],
        "direction": study_summary["direction"],
        "best_params": best_params,
    }

