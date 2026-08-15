"""Prediction persistence orchestration with DB-first, queue-on-failure behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinelml.production_data.validation import validate_prediction_for_production
from sentinelml.serving.database import (
    ConflictingGroundTruthError,
    GroundTruthUpdateResult,
    PredictionDatabase,
    PredictionNotFoundError,
    PredictionRecord,
)
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.validation import FeatureSchema


@dataclass(frozen=True)
class FlushResult:
    attempted: int
    persisted: int
    remaining: int
    malformed: int
    database_logging: str


class PredictionRepository:
    def __init__(self, database: PredictionDatabase, queue: DurableJsonlQueue) -> None:
        self.database = database
        self.queue = queue
        self.last_database_error: str | None = None

    def initialize(self) -> None:
        self.database.initialize()

    def database_status(self) -> str:
        if not self.database.configured:
            return "unconfigured"
        try:
            self.database.check()
        except Exception as exc:
            self.last_database_error = str(exc)
            return "unhealthy"
        self.last_database_error = None
        return "healthy"

    def persist_prediction(self, record: PredictionRecord) -> bool:
        try:
            self.database.insert_prediction(record)
        except Exception as exc:
            self.last_database_error = str(exc)
            self.queue.append(record.to_jsonable())
            return False
        self.last_database_error = None
        return True

    def flush_queue(self) -> FlushResult:
        snapshot = self.queue.snapshot()
        attempted = len(snapshot.records)
        persisted = 0
        remaining: list[dict[str, Any]] = []
        database_logging = "healthy"
        for record in snapshot.records:
            try:
                self.database.insert_prediction(record)
            except Exception as exc:
                self.last_database_error = str(exc)
                database_logging = "unhealthy"
                remaining.append(record)
            else:
                persisted += 1
        self.queue.archive_malformed(snapshot.malformed_lines)
        self.queue.replace_records(remaining)
        if database_logging == "healthy":
            self.last_database_error = None
        return FlushResult(
            attempted=attempted,
            persisted=persisted,
            remaining=len(remaining),
            malformed=len(snapshot.malformed_lines),
            database_logging=database_logging,
        )

    def record_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
        feature_schema: FeatureSchema,
    ) -> GroundTruthUpdateResult:
        prediction = self.database.get_prediction(prediction_id)
        if prediction is None:
            raise PredictionNotFoundError(prediction_id)
        validation = validate_prediction_for_production(
            prediction=prediction,
            ground_truth=ground_truth,
            schema=feature_schema,
        )
        return self.database.record_ground_truth(
            prediction_id=prediction_id,
            ground_truth=ground_truth,
            schema_fingerprint=feature_schema.fingerprint,
            validation_status=validation.status,
            validation_errors=validation.errors,
        )

    def record_ground_truth_batch(
        self,
        labels: list[dict[str, Any]],
        *,
        feature_schema: FeatureSchema,
    ) -> list[GroundTruthUpdateResult]:
        normalized = _normalize_label_batch(labels)
        validations: dict[str, tuple[int, str, list[dict[str, Any]]]] = {}
        for prediction_id, ground_truth in normalized:
            prediction = self.database.get_prediction(prediction_id)
            if prediction is None:
                raise PredictionNotFoundError(prediction_id)
            validation = validate_prediction_for_production(
                prediction=prediction,
                ground_truth=ground_truth,
                schema=feature_schema,
            )
            existing_truth = prediction.get("ground_truth")
            if existing_truth is not None and int(existing_truth) != ground_truth:
                raise ConflictingGroundTruthError(
                    f"prediction {prediction_id} already has ground_truth "
                    f"{existing_truth}"
                )
            validations[prediction_id] = (
                ground_truth,
                validation.status,
                validation.errors,
            )

        results: list[GroundTruthUpdateResult] = []
        for prediction_id, ground_truth in normalized:
            _, status, errors = validations[prediction_id]
            results.append(
                self.database.record_ground_truth(
                    prediction_id=prediction_id,
                    ground_truth=ground_truth,
                    schema_fingerprint=feature_schema.fingerprint,
                    validation_status=status,
                    validation_errors=errors,
                )
            )
        return results


def _normalize_label_batch(labels: list[dict[str, Any]]) -> list[tuple[str, int]]:
    seen: dict[str, int] = {}
    normalized: list[tuple[str, int]] = []
    for item in labels:
        prediction_id = str(item["prediction_id"])
        ground_truth = int(item["ground_truth"])
        if prediction_id in seen:
            if seen[prediction_id] != ground_truth:
                raise ConflictingGroundTruthError(
                    f"batch contains conflicting labels for {prediction_id}"
                )
            continue
        seen[prediction_id] = ground_truth
        normalized.append((prediction_id, ground_truth))
    return normalized
