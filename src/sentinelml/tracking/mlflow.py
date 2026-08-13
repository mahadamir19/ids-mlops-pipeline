"""Shared MLflow runtime and reproducibility lineage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from sentinelml.data.config import PROJECT_ROOT

DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def configure_mlflow_runtime_environment(repo_root: Path = PROJECT_ROOT) -> str:
    mlflow_temp_dir = repo_root / ".tmp" / "mlflow-temp"
    mlflow_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(mlflow_temp_dir)
    os.environ["TMP"] = str(mlflow_temp_dir)
    os.environ["TEMP"] = str(mlflow_temp_dir)

    env_values = load_env_file(repo_root / "infra" / "mlflow" / ".env")
    if "MINIO_ROOT_USER" in env_values:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", env_values["MINIO_ROOT_USER"])
    if "MINIO_ROOT_PASSWORD" in env_values:
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY",
            env_values["MINIO_ROOT_PASSWORD"],
        )
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")

    return os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )


def run_repository_command(args: list[str], repo_root: Path = PROJECT_ROOT) -> str:
    env = os.environ.copy()
    if args and args[0] == "dvc":
        dvc_site_cache = repo_root / ".tmp" / "dvc-site-cache"
        dvc_site_cache.mkdir(parents=True, exist_ok=True)
        env.setdefault("DVC_SITE_CACHE_DIR", str(dvc_site_cache))
    completed = subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dvc_status_is_clean(status: str) -> bool:
    stripped = status.strip()
    return stripped in {"", "Data and pipelines are up to date."}


def collect_reproducibility_lineage(
    *,
    mlflow_parent_run_id: str,
    training_config_path: Path,
    feature_schema_path: Path,
    optimization_config_path: Path | None = None,
    dependency_lock_path: Path | None = None,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    git_commit = run_repository_command(["git", "rev-parse", "HEAD"], repo_root).strip()
    git_branch = run_repository_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        repo_root,
    ).strip()
    git_status_porcelain = run_repository_command(
        ["git", "status", "--porcelain=v1"],
        repo_root,
    )
    dvc_status = run_repository_command(["dvc", "status"], repo_root)
    dvc_lock_path = repo_root / "dvc.lock"

    lineage: dict[str, Any] = {
        "mlflow_parent_run_id": mlflow_parent_run_id,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": bool(git_status_porcelain.strip()),
        "git_status_porcelain": git_status_porcelain,
        "dvc_lock_sha256": sha256_file(dvc_lock_path),
        "dvc_status_clean": dvc_status_is_clean(dvc_status),
        "dvc_status": dvc_status,
        "training_config_sha256": sha256_file(training_config_path),
        "feature_schema_sha256": sha256_file(feature_schema_path),
    }
    if optimization_config_path is not None:
        lineage["optimization_config_sha256"] = sha256_file(optimization_config_path)
    if dependency_lock_path is not None:
        lineage["dependency_lock_path"] = str(dependency_lock_path)
        lineage["dependency_lock_sha256"] = sha256_file(dependency_lock_path)
    lineage["python_version"] = sys.version
    lineage["platform"] = platform.platform()
    return lineage


def lineage_tags(lineage: dict[str, Any]) -> dict[str, str]:
    tags = {
        "lineage.git_commit": str(lineage["git_commit"]),
        "lineage.git_branch": str(lineage["git_branch"]),
        "lineage.git_dirty": str(lineage["git_dirty"]).lower(),
        "lineage.dvc_lock_sha256": str(lineage["dvc_lock_sha256"]),
        "lineage.dvc_status_clean": str(lineage["dvc_status_clean"]).lower(),
        "lineage.training_config_sha256": str(lineage["training_config_sha256"]),
        "lineage.feature_schema_sha256": str(lineage["feature_schema_sha256"]),
    }
    if "optimization_config_sha256" in lineage:
        tags["lineage.optimization_config_sha256"] = str(
            lineage["optimization_config_sha256"]
        )
    if "dependency_lock_sha256" in lineage:
        tags["lineage.dependency_lock_sha256"] = str(
            lineage["dependency_lock_sha256"]
        )
    return tags


def flatten_mlflow_metrics(
    prefix: str,
    values: dict[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            metrics.update(flatten_mlflow_metrics(metric_key, value))
        elif isinstance(value, bool):
            metrics[metric_key] = float(value)
        elif isinstance(value, int | float) and value is not None:
            metrics[metric_key] = float(value)
    return metrics


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json_artifact(
    mlflow: Any,
    payload: dict[str, Any] | list[dict[str, Any]],
    artifact_file: str,
    *,
    repo_root: Path = PROJECT_ROOT,
) -> None:
    local_path = repo_root / ".tmp" / "mlflow-artifacts" / artifact_file
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )
    artifact_dir = str(Path(artifact_file).parent).replace("\\", "/")
    mlflow.log_artifact(str(local_path), artifact_path=artifact_dir)


def log_lineage_artifacts(
    mlflow: Any,
    lineage: dict[str, Any],
    *,
    repo_root: Path = PROJECT_ROOT,
    dependency_lock_path: Path | None = None,
) -> None:
    write_json_artifact(
        mlflow,
        lineage,
        "lineage/lineage.json",
        repo_root=repo_root,
    )
    for artifact_path in [
        repo_root / "dvc.yaml",
        repo_root / "dvc.lock",
        repo_root / "data" / "raw.dvc",
    ]:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="lineage")
    if dependency_lock_path is not None and dependency_lock_path.exists():
        mlflow.log_artifact(str(dependency_lock_path), artifact_path="lineage")


def log_input_artifacts(
    mlflow: Any,
    paths: list[Path],
    *,
    artifact_path: str = "inputs",
) -> None:
    for path in paths:
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
