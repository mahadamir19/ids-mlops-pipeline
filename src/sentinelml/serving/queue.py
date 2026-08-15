"""Durable JSONL queue for prediction records that could not reach PostgreSQL."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueueSnapshot:
    records: list[dict[str, Any]]
    malformed_lines: list[str]


class DurableJsonlQueue:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> bool:
        prediction_id = str(record["prediction_id"])
        if prediction_id in self.prediction_ids():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def snapshot(self) -> QueueSnapshot:
        if not self.path.exists():
            return QueueSnapshot(records=[], malformed_lines=[])
        records: list[dict[str, Any]] = []
        malformed: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line)
                continue
            if isinstance(value, dict) and value.get("prediction_id"):
                records.append(value)
            else:
                malformed.append(line)
        return QueueSnapshot(records=records, malformed_lines=malformed)

    def prediction_ids(self) -> set[str]:
        return {str(record["prediction_id"]) for record in self.snapshot().records}

    def depth(self) -> int:
        return len(self.snapshot().records)

    def malformed_count(self) -> int:
        return len(self.snapshot().malformed_lines)

    def replace_records(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp_path = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        tmp_path = Path(raw_tmp_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)

    def archive_malformed(self, malformed_lines: list[str]) -> Path | None:
        if not malformed_lines:
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive_path = self.path.with_suffix(
            f"{self.path.suffix}.malformed.{timestamp}"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as handle:
            for line in malformed_lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return archive_path
