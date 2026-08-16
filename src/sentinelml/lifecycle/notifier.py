"""Serving notification worker for lifecycle promotions."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Any


class ServingNotifier:
    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = config

    def notify_reload(self) -> dict[str, Any]:
        reload_config = self.config.get("serving_reload", {})
        url_env = str(reload_config.get("url_env", "SENTINELML_SERVING_RELOAD_URL"))
        timeout_env = str(
            reload_config.get(
                "timeout_seconds_env",
                "SENTINELML_SERVING_RELOAD_TIMEOUT_SECONDS",
            )
        )
        url = os.environ.get(url_env)
        if not url:
            return {"attempted": False, "reason": f"{url_env} is not set"}
        timeout = float(os.environ.get(timeout_env, "2"))
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "attempted": True,
                    "success": 200 <= response.status < 300,
                    "status_code": response.status,
                    "body": body[:500],
                }
        except (OSError, urllib.error.URLError) as exc:
            return {
                "attempted": True,
                "success": False,
                "error": str(exc),
            }
