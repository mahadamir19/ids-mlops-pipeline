"""Simulation report serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_simulation_report(report: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(report["simulation_run_id"])
    run_path = reports_dir / f"{run_id}.json"
    latest_path = reports_dir / "latest.json"
    encoded = json.dumps(report, indent=2, sort_keys=True)
    run_path.write_text(encoded + "\n", encoding="utf-8")
    tmp_latest = latest_path.with_suffix(".json.tmp")
    tmp_latest.write_text(encoded + "\n", encoding="utf-8")
    os.replace(tmp_latest, latest_path)
    return run_path
