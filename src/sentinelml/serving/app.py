"""FastAPI application for Phase 5 production-style inference."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sentinelml.serving.config import ServingConfig, load_serving_config
from sentinelml.serving.database import (
    ConflictingGroundTruthError,
    PredictionDatabase,
    PredictionNotFoundError,
)
from sentinelml.serving.model_manager import ModelManager
from sentinelml.serving.prediction_service import PredictionService
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.rejection_logging import StructuredRejectionLogger
from sentinelml.serving.repository import PredictionRepository
from sentinelml.serving.schemas import (
    BatchGroundTruthResponse,
    BatchPredictionResponse,
    GroundTruthRequest,
    GroundTruthResponse,
    HealthResponse,
    ModelResponse,
    PredictionResponse,
    QueueFlushResponse,
    QueueStatusResponse,
    ReloadResponse,
)
from sentinelml.serving.validation import (
    FeatureValidationError,
    validate_feature_batch,
    validate_feature_record,
)

REQUEST_BODY = Body(...)


def create_app(
    config: ServingConfig | None = None,
    *,
    model_manager_factory: Callable[[ServingConfig], ModelManager] | None = None,
    repository_factory: Callable[[ServingConfig], PredictionRepository] | None = None,
    rejection_logger_factory: Callable[[ServingConfig], StructuredRejectionLogger]
    | None = None,
) -> FastAPI:
    holder: dict[str, Any] = {"config": config}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        serving_config = holder["config"] or load_serving_config()
        holder["config"] = serving_config
        if repository_factory is None:
            database = PredictionDatabase(
                serving_config.database_url,
                connect_timeout_seconds=serving_config.database_connect_timeout_seconds,
                statement_timeout_ms=serving_config.database_statement_timeout_ms,
            )
            queue = DurableJsonlQueue(serving_config.queue_path)
            repository = PredictionRepository(database, queue)
        else:
            repository = repository_factory(serving_config)
        repository.initialize()
        model_manager = (
            ModelManager(serving_config)
            if model_manager_factory is None
            else model_manager_factory(serving_config)
        )
        model_manager.load_startup()
        app.state.config = serving_config
        app.state.repository = repository
        app.state.model_manager = model_manager
        app.state.prediction_service = PredictionService(model_manager, repository)
        app.state.rejection_logger = (
            StructuredRejectionLogger(serving_config.rejection_log_path)
            if rejection_logger_factory is None
            else rejection_logger_factory(serving_config)
        )
        stop_event = asyncio.Event()
        flush_task = asyncio.create_task(_periodic_flush(app, stop_event))
        app.state.queue_flush_task = flush_task
        try:
            yield
        finally:
            stop_event.set()
            flush_task.cancel()
            try:
                await flush_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="SentinelML Phase 5 Serving API",
        version="0.5.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def malformed_request_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        error = {
            "category": "malformed_json",
            "message": "request body is malformed or does not match endpoint shape",
            "details": exc.errors(),
        }
        _log_rejection(request, request_id, error)
        return JSONResponse(
            status_code=422,
            content={"request_id": request_id, "error": error},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        repository: PredictionRepository = app.state.repository
        model_manager: ModelManager = app.state.model_manager
        model_ready = model_manager.is_ready()
        active = model_manager.current() if model_ready else None
        db_status = repository.database_status()
        queue_depth = repository.queue.depth()
        status = "healthy" if model_ready and db_status == "healthy" else "degraded"
        if not model_ready:
            status = "unhealthy"
        return {
            "status": status,
            "process_alive": True,
            "model_ready": model_ready,
            "model_version": active.model_version if active else None,
            "database_logging": db_status,
            "queue_depth": queue_depth,
            "malformed_queue_records": repository.queue.malformed_count(),
        }

    @app.get("/model", response_model=ModelResponse)
    def model() -> dict[str, Any]:
        active = app.state.model_manager.current()
        return _model_payload(active)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, body: Any = REQUEST_BODY) -> dict[str, Any]:
        try:
            ordered = validate_feature_record(
                body,
                app.state.model_manager.current().feature_schema,
            )
        except FeatureValidationError as exc:
            return _validation_error_response(request, exc)
        return app.state.prediction_service.predict_one(ordered)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: Request, body: Any = REQUEST_BODY) -> dict[str, Any]:
        config: ServingConfig = app.state.config
        try:
            ordered = validate_feature_batch(
                body,
                app.state.model_manager.current().feature_schema,
                max_batch_size=config.max_batch_size,
            )
        except FeatureValidationError as exc:
            return _validation_error_response(request, exc)
        predictions = app.state.prediction_service.predict_batch(ordered)
        return {
            "predictions": predictions,
            "model_version": predictions[0]["model_version"],
            "count": len(predictions),
        }

    @app.post("/ground-truth", response_model=GroundTruthResponse)
    def ground_truth(body: GroundTruthRequest) -> dict[str, Any] | JSONResponse:
        repository: PredictionRepository = app.state.repository
        feature_schema = app.state.model_manager.current().feature_schema
        try:
            result = repository.record_ground_truth(
                prediction_id=body.prediction_id,
                ground_truth=body.ground_truth,
                feature_schema=feature_schema,
            )
        except PredictionNotFoundError:
            return JSONResponse(
                status_code=404,
                content={
                    "prediction_id": body.prediction_id,
                    "error": "prediction_id was not found",
                    "retryable": True,
                },
            )
        except ConflictingGroundTruthError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "prediction_id": body.prediction_id,
                    "error": str(exc),
                    "retryable": False,
                },
            )
        return result.__dict__

    @app.post("/ground-truth/batch", response_model=BatchGroundTruthResponse)
    def ground_truth_batch(
        body: list[GroundTruthRequest],
    ) -> dict[str, Any] | JSONResponse:
        config: ServingConfig = app.state.config
        if not body:
            return JSONResponse(
                status_code=422,
                content={"error": "batch ground-truth input must not be empty"},
            )
        if len(body) > config.max_batch_size:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "batch ground-truth input exceeds maximum batch size",
                    "max_batch_size": config.max_batch_size,
                    "actual_batch_size": len(body),
                },
            )
        repository: PredictionRepository = app.state.repository
        feature_schema = app.state.model_manager.current().feature_schema
        payload = [item.model_dump() for item in body]
        try:
            results = repository.record_ground_truth_batch(
                payload,
                feature_schema=feature_schema,
            )
        except PredictionNotFoundError as exc:
            return JSONResponse(
                status_code=404,
                content={
                    "prediction_id": str(exc),
                    "error": "one or more prediction_ids were not found",
                    "retryable": True,
                    "atomic": True,
                },
            )
        except ConflictingGroundTruthError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": str(exc), "retryable": False, "atomic": True},
            )
        return {
            "labels": [result.__dict__ for result in results],
            "count": len(results),
            "atomic": True,
        }

    @app.post("/internal/reload", response_model=ReloadResponse)
    def reload_champion() -> dict[str, Any]:
        config: ServingConfig = app.state.config
        if not config.reload_endpoint_enabled:
            return JSONResponse(
                status_code=404,
                content={"detail": "internal reload endpoint is disabled"},
            )
        return app.state.model_manager.reload_champion()

    @app.get("/internal/queue", response_model=QueueStatusResponse)
    def queue_status() -> dict[str, Any]:
        repository: PredictionRepository = app.state.repository
        return {
            "queue_depth": repository.queue.depth(),
            "malformed_queue_records": repository.queue.malformed_count(),
            "queue_path": str(repository.queue.path),
        }

    @app.post("/internal/queue/flush", response_model=QueueFlushResponse)
    def flush_queue() -> dict[str, Any]:
        return app.state.repository.flush_queue().__dict__

    return app


async def _periodic_flush(app: FastAPI, stop_event: asyncio.Event) -> None:
    config: ServingConfig = app.state.config
    if config.queue_flush_interval_seconds <= 0:
        return
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=config.queue_flush_interval_seconds,
            )
        except TimeoutError:
            app.state.repository.flush_queue()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid4())


def _validation_error_response(
    request: Request,
    exc: FeatureValidationError,
) -> JSONResponse:
    request_id = _request_id(request)
    error = exc.to_error_payload()
    _log_rejection(request, request_id, error)
    return JSONResponse(
        status_code=422,
        content={"request_id": request_id, "error": error},
    )


def _log_rejection(request: Request, request_id: str, error: dict[str, Any]) -> None:
    logger: StructuredRejectionLogger | None = getattr(
        request.app.state,
        "rejection_logger",
        None,
    )
    model_manager: ModelManager | None = getattr(
        request.app.state,
        "model_manager",
        None,
    )
    schema_fingerprint = None
    if model_manager is not None and model_manager.is_ready():
        schema_fingerprint = model_manager.current().feature_schema.fingerprint
    if logger is not None:
        logger.log(
            endpoint=request.url.path,
            request_id=request_id,
            schema_fingerprint=schema_fingerprint,
            error=error,
            http_status=422,
        )


def _model_payload(active: Any) -> dict[str, Any]:
    return {
        "model_name": active.model_name,
        "model_version": active.model_version,
        "model_family": active.model_family,
        "execution_mode": active.execution_mode,
        "demo_model": active.demo_model,
        "source_run_id": active.source_run_id,
        "source_model_uri": active.source_model_uri,
        "loaded_at": active.loaded_at,
        "feature_schema_path": str(active.feature_schema.source_path),
        "feature_schema_fingerprint": active.feature_schema.fingerprint,
        "feature_count": active.feature_schema.feature_count,
    }


app = create_app()
