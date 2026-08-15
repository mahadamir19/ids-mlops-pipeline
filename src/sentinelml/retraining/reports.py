"""Phase 8 report serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sentinelml.training.compare import json_default


def write_retraining_report(
    payload: dict[str, Any],
    reports_dir: Path,
    *,
    filename: str,
    latest_name: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    path.write_text(encoded + "\n", encoding="utf-8")
    if latest_name is not None:
        latest_path = reports_dir / latest_name
        tmp_latest = latest_path.with_suffix(latest_path.suffix + ".tmp")
        tmp_latest.write_text(encoded + "\n", encoding="utf-8")
        os.replace(tmp_latest, latest_path)
    return path
