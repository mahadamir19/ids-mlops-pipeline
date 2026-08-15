"""Cross-process retraining lock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy.engine import Connection

from sentinelml.retraining.repository import RetrainingRepository

LOCK_KEY = 876_800_008


@dataclass
class RetrainingLock:
    repository: RetrainingRepository
    owner: str | None = None
    acquired: bool = False
    _connection: Connection | None = None

    def acquire(self) -> bool:
        self.owner = self.owner or str(uuid4())
        if self.repository.engine is None:
            self.acquired = False
            return False
        if self.repository.dialect_name == "postgresql":
            self._connection = self.repository.engine.connect()
            self.acquired = self.repository.try_postgres_advisory_lock(
                self._connection,
                LOCK_KEY,
            )
            if not self.acquired:
                self._connection.close()
                self._connection = None
            return self.acquired
        self.acquired = self.repository.acquire_table_lock(owner=self.owner)
        return self.acquired

    def release(self) -> None:
        if not self.acquired or self.repository.engine is None:
            return
        try:
            if (
                self.repository.dialect_name == "postgresql"
                and self._connection is not None
            ):
                self.repository.release_postgres_advisory_lock(
                    self._connection,
                    LOCK_KEY,
                )
            elif self.owner is not None:
                self.repository.release_table_lock(owner=self.owner)
        finally:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self.acquired = False

    def __enter__(self) -> RetrainingLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()
