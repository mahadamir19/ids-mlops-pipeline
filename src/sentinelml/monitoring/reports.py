"""Monitoring report serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sentinelml.training.compare import json_default


def write_monitoring_report(report: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(report["monitoring_run_id"])
    run_path = reports_dir / f"{run_id}.json"
    latest_path = reports_dir / "latest.json"
    encoded = json.dumps(report, indent=2, sort_keys=True, default=json_default)
    run_path.write_text(encoded + "\n", encoding="utf-8")
    tmp_latest = latest_path.with_suffix(".json.tmp")
    tmp_latest.write_text(encoded + "\n", encoding="utf-8")
    os.replace(tmp_latest, latest_path)
    return run_path


def latest_monitoring_report(reports_dir: Path) -> dict[str, Any] | None:
    latest_path = reports_dir / "latest.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def write_monitoring_status(status: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    status_path = reports_dir / "status.json"
    encoded = json.dumps(status, indent=2, sort_keys=True, default=json_default)
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.write_text(encoded + "\n", encoding="utf-8")
    os.replace(tmp_path, status_path)
    return status_path
