"""HTTP client for the Phase 6 simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int | None
    message: str
    payload: dict[str, Any] | None = None
    retryable: bool = False

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"HTTP {self.status_code}: {self.message}"


class SimulationHttpClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        return self._post("/predict", features)

    def predict_batch(self, records: list[dict[str, float]]) -> dict[str, Any]:
        return self._post("/predict/batch", records)

    def submit_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
    ) -> dict[str, Any]:
        return self._post(
            "/ground-truth",
            {"prediction_id": prediction_id, "ground_truth": ground_truth},
        )

    def submit_ground_truth_batch(
        self,
        labels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._post("/ground-truth/batch", labels)

    def _post(self, path: str, payload: Any) -> dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            payload = _load_error_payload(exc)
            retryable = bool(payload.get("retryable", exc.code >= 500))
            raise ApiError(
                status_code=exc.code,
                message=str(payload.get("error") or payload.get("detail") or exc),
                payload=payload,
                retryable=retryable,
            ) from exc
        except URLError as exc:
            raise ApiError(
                status_code=None,
                message=f"API unavailable: {exc.reason}",
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise ApiError(
                status_code=None,
                message="API request timed out",
                retryable=True,
            ) from exc
        return json.loads(raw)


def _load_error_payload(exc: HTTPError) -> dict[str, Any]:
    raw = exc.read().decode("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw}
    return payload if isinstance(payload, dict) else {"error": payload}
