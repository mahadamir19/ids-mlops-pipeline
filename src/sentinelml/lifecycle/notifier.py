"""Serving notification worker for lifecycle promotions."""

from __future__ import annotations

from typing import Any


class ServingNotifier:
    def __init__(self, service: Any) -> None:
        self.service = service

    def notify_reload(self) -> dict[str, Any]:
        return self.service._notify_serving_reload_impl()
