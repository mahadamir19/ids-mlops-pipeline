"""Dependency-lock and environment snapshot helpers for Phase 3E."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sentinelml.data.config import PROJECT_ROOT
from sentinelml.tracking.mlflow import sha256_file

DEFAULT_DEPENDENCY_LOCK_PATH = PROJECT_ROOT / "pylock.toml"


def existing_dependency_lock(repo_root: Path = PROJECT_ROOT) -> Path | None:
    for name in ["pylock.toml", "uv.lock", "poetry.lock", "pdm.lock"]:
        path = repo_root / name
        if path.exists():
            return path
    return None


def ensure_dependency_lock(
    *,
    python_executable: str,
    repo_root: Path = PROJECT_ROOT,
) -> Path:
    """Ensure a real resolver lockfile exists for the project extras."""

    existing = existing_dependency_lock(repo_root)
    if existing is not None:
        return existing

    lock_path = repo_root / "pylock.toml"
    command = [
        python_executable,
        "-m",
        "pip",
        "lock",
        "--output",
        str(lock_path),
        f"{repo_root}[ml,tracking]",
    ]
    try:
        subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "failed to generate pylock.toml with python -m pip lock; "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc
    if not lock_path.exists():
        raise FileNotFoundError("dependency lock was not created: pylock.toml")
    return lock_path


def dependency_lock_metadata(lock_path: Path) -> dict[str, str | int]:
    return {
        "dependency_lock_file": str(lock_path),
        "dependency_lock_name": lock_path.name,
        "dependency_lock_sha256": sha256_file(lock_path),
        "dependency_lock_size_bytes": lock_path.stat().st_size,
    }


def write_pip_freeze_snapshot(
    *,
    python_executable: str,
    output_path: Path,
    repo_root: Path = PROJECT_ROOT,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [python_executable, "-m", "pip", "freeze", "--all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    return output_path

