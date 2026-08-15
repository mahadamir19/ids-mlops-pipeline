"""Rollback worker for the LifecycleService facade."""

from __future__ import annotations

from typing import Any


class RollbackManager:
    def __init__(self, service: Any) -> None:
        self.service = service

    def rollback(self, **kwargs: Any) -> dict[str, Any]:
        return self.service._rollback(**kwargs)
