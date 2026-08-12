"""Phase 2 baseline training workflow."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sentinelml.data.config import PROJECT_ROOT
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
from sentinelml.training.models import (
    DEFAULT_TRAINING_CONFIG_PATH,
    build_baseline_model_specs,
    load_training_config,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "models" / "phase2_baseline"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "models" / "phase2_smoke"


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

    for spec in model_specs:
        start = perf_counter()
        spec.estimator.fit(train.features, train.target)
        training_seconds = perf_counter() - start
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
        }
        fitted_models[spec.name] = spec.estimator
        parameters[spec.name] = spec.parameters
        feature_importance[spec.name] = {
            "rows": importance_rows,
            "unsupported_reason": unsupported_reason,
            "destination_port": destination_port_summary(importance_rows),
        }

    comparison = comparison_rows(validation_results)
    selected = select_best_baseline(comparison)
    selected_model_name = selected["recommended_baseline"]

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
    assert_files_unchanged(protected_files)
    return report
