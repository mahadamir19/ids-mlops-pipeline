"""Pydantic response schemas for the Phase 5 FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    process_alive: bool
    model_ready: bool
    model_version: str | None
    database_logging: str
    queue_depth: int
    malformed_queue_records: int


class ModelResponse(BaseModel):
    model_name: str
    model_version: str
    model_family: str | None
    execution_mode: str | None
    demo_model: bool | None
    source_run_id: str | None
    source_model_uri: str | None
    loaded_at: str
    feature_schema_path: str
    feature_schema_fingerprint: str
    feature_count: int


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: Literal["BENIGN", "ATTACK"]
    probability: float = Field(ge=0.0, le=1.0)
    model_version: str
    latency_ms: float = Field(ge=0.0)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    model_version: str
    count: int


class GroundTruthRequest(BaseModel):
    prediction_id: str = Field(min_length=1)
    ground_truth: int = Field(ge=0, le=1)


class GroundTruthResponse(BaseModel):
    prediction_id: str
    ground_truth: int
    status: Literal["recorded", "idempotent"]
    ground_truth_received_at: str
    production_observation_status: Literal["approved", "rejected"]
    validation_errors: list[dict[str, Any]]
    retryable: bool = False


class BatchGroundTruthResponse(BaseModel):
    labels: list[GroundTruthResponse]
    count: int
    atomic: bool


class ReloadResponse(BaseModel):
    reloaded: bool
    previous_model_version: str | None
    active_model_version: str
    message: str


class QueueStatusResponse(BaseModel):
    queue_depth: int
    malformed_queue_records: int
    queue_path: str


class QueueFlushResponse(BaseModel):
    attempted: int
    persisted: int
    remaining: int
    malformed: int
    database_logging: str


class ValidationErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    error: dict[str, Any]
