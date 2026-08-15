"""Candidate registration worker for the LifecycleService facade."""

from __future__ import annotations

from typing import Any


class CandidateRegistrar:
    def __init__(self, service: Any) -> None:
        self.service = service

    def register_candidate(self, **kwargs: Any) -> dict[str, Any]:
        return self.service._register_candidate(**kwargs)
