"""Promotion worker for the LifecycleService facade."""

from __future__ import annotations

from typing import Any


class PromotionManager:
    def __init__(self, service: Any) -> None:
        self.service = service

    def promote_or_reject(self, **kwargs: Any) -> dict[str, Any]:
        return self.service._promote_or_reject(**kwargs)

    def retry_pending(self) -> list[dict[str, Any]]:
        return self.service._retry_pending()
