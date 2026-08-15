"""Service facade for approved labelled production observations."""

from __future__ import annotations

from typing import Any

from sentinelml.production_data.repository import ProductionDataRepository


class ProductionDataService:
    def __init__(self, repository: ProductionDataRepository) -> None:
        self.repository = repository

    def get_approved_production_observations(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.repository.get_approved_production_observations(limit=limit)
