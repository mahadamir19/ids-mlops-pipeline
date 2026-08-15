"""Candidate evaluation worker for the LifecycleService facade."""

from __future__ import annotations

from typing import Any


class CandidateEvaluator:
    def __init__(self, service: Any) -> None:
        self.service = service

    def evaluate_candidate(self, **kwargs: Any) -> dict[str, Any]:
        return self.service._evaluate_candidate(**kwargs)
