"""Structured, payload-safe logging for rejected inference requests."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class StructuredRejectionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(
        self,
        *,
        endpoint: str,
        request_id: str,
        schema_fingerprint: str | None,
        error: dict[str, Any],
        http_status: int = 422,
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "request_id": request_id,
            "expected_schema_fingerprint": schema_fingerprint,
            "error_category": error.get("category"),
            "http_status": http_status,
            "missing_fields": error.get("missing_fields", []),
            "unexpected_fields": error.get("unexpected_fields", []),
            "invalid_fields": error.get("invalid_fields", []),
            "invalid_rows": error.get("invalid_rows", []),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
