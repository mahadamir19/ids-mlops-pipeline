"""Machine-readable audit output for Phase 4 lifecycle operations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinelml.training.compare import json_default


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default, sort_keys=True),
        encoding="utf-8",
    )
    return path


def audit_path(root: Path, event_type: str, version: str | int | None) -> Path:
    stamp = utc_now().replace(":", "").replace("+", "Z")
    suffix = f"version_{version}" if version is not None else "registry"
    return root / f"{event_type}s" / f"{suffix}_{stamp}.json"
