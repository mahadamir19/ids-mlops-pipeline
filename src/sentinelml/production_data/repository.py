"""Read boundary for Phase 8 retraining-eligible production observations."""

from __future__ import annotations

from typing import Any

from sentinelml.serving.database import PredictionDatabase


class ProductionDataRepository:
    def __init__(self, database: PredictionDatabase) -> None:
        self.database = database

    def get_approved_production_observations(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.database.get_approved_production_observations(limit=limit)
