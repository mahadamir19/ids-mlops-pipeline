"""Machine-readable reports for Phase 9 resilience operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentinelml.training.compare import json_default


def write_resilience_report(
    payload: dict[str, Any],
    reports_dir: Path,
    *,
    category: str,
    name: str,
    latest: bool = True,
) -> Path:
    target_dir = reports_dir / category
    target_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    path = target_dir / name
    path.write_text(encoded + "\n", encoding="utf-8")
    if latest:
        latest_payload = {
            "category": category,
            "report_path": str(path),
            "payload": payload,
        }
        (reports_dir / "latest.json").parent.mkdir(parents=True, exist_ok=True)
        (reports_dir / "latest.json").write_text(
            json.dumps(
                latest_payload,
                indent=2,
                sort_keys=True,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
    return path
