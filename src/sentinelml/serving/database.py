"""Application PostgreSQL persistence for Phase 5 prediction logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    timestamp: str
    model_name: str
    model_version: str
    model_family: str | None
    execution_mode: str | None
    demo_model: bool | None
    features: dict[str, float]
    prediction: str
    probability: float
    latency_ms: float
    ground_truth: int | None = None
    ground_truth_received_at: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthUpdateResult:
    prediction_id: str
    ground_truth: int
    status: str
    ground_truth_received_at: str
    production_observation_status: str
    validation_errors: list[dict[str, Any]]
    retryable: bool = False


class PredictionNotFoundError(LookupError):
    """Raised when delayed ground truth arrives before a prediction is durable."""

    retryable = True


class ConflictingGroundTruthError(ValueError):
    """Raised when an existing prediction already has a different label."""


class PredictionDatabase:
    def __init__(
        self,
        database_url: str | None,
        *,
        connect_timeout_seconds: int = 5,
        statement_timeout_ms: int = 5000,
        engine: Engine | None = None,
    ) -> None:
        self.database_url = database_url
        self.connect_timeout_seconds = connect_timeout_seconds
        self.statement_timeout_ms = statement_timeout_ms
        self.engine = engine or self._create_engine(database_url)

    def _create_engine(self, database_url: str | None) -> Engine | None:
        if not database_url:
            return None
        connect_args: dict[str, Any] = {}
        if database_url.startswith("postgresql"):
            connect_args["connect_timeout"] = self.connect_timeout_seconds
            connect_args["options"] = (
                f"-c statement_timeout={self.statement_timeout_ms}"
            )
        return create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

    @property
    def configured(self) -> bool:
        return self.engine is not None

    @property
    def dialect_name(self) -> str | None:
        return self.engine.dialect.name if self.engine is not None else None

    def initialize(self) -> None:
        if self.engine is None:
            return
        features_type = "JSONB" if self.dialect_name == "postgresql" else "TEXT"
        errors_type = "JSONB" if self.dialect_name == "postgresql" else "TEXT"
        ddl = f"""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id VARCHAR(64) PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            model_name VARCHAR(255) NOT NULL,
            model_version VARCHAR(64) NOT NULL,
            model_family VARCHAR(128),
            execution_mode VARCHAR(64),
            demo_model BOOLEAN,
            features {features_type} NOT NULL,
            prediction VARCHAR(16) NOT NULL,
            probability DOUBLE PRECISION NOT NULL,
            latency_ms DOUBLE PRECISION NOT NULL,
            ground_truth INTEGER,
            ground_truth_received_at TIMESTAMP WITH TIME ZONE
        )
        """
        production_ddl = f"""
        CREATE TABLE IF NOT EXISTS production_labelled_observations (
            prediction_id VARCHAR(64) PRIMARY KEY,
            features {features_type} NOT NULL,
            ground_truth INTEGER NOT NULL,
            staged_at TIMESTAMP WITH TIME ZONE NOT NULL,
            validation_status VARCHAR(32) NOT NULL,
            validation_errors {errors_type} NOT NULL,
            approved_at TIMESTAMP WITH TIME ZONE,
            source_model_version VARCHAR(64) NOT NULL,
            schema_fingerprint VARCHAR(64) NOT NULL
        )
        """
        with self.engine.begin() as connection:
            connection.execute(text(ddl))
            connection.execute(text(production_ddl))

    def check(self) -> bool:
        if self.engine is None:
            return False
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def insert_prediction(self, record: PredictionRecord | dict[str, Any]) -> None:
        if self.engine is None:
            raise RuntimeError("prediction database is not configured")
        payload = (
            record.to_jsonable()
            if isinstance(record, PredictionRecord)
            else dict(record)
        )
        params = {
            **payload,
            "features": json.dumps(payload["features"], sort_keys=True),
        }
        if self.dialect_name == "postgresql":
            sql = """
            INSERT INTO predictions (
                prediction_id, timestamp, model_name, model_version, model_family,
                execution_mode, demo_model, features, prediction, probability,
                latency_ms, ground_truth, ground_truth_received_at
            )
            VALUES (
                :prediction_id, :timestamp, :model_name, :model_version, :model_family,
                :execution_mode, :demo_model, CAST(:features AS JSONB), :prediction,
                :probability, :latency_ms, :ground_truth, :ground_truth_received_at
            )
            ON CONFLICT (prediction_id) DO NOTHING
            """
        else:
            sql = """
            INSERT OR IGNORE INTO predictions (
                prediction_id, timestamp, model_name, model_version, model_family,
                execution_mode, demo_model, features, prediction, probability,
                latency_ms, ground_truth, ground_truth_received_at
            )
            VALUES (
                :prediction_id, :timestamp, :model_name, :model_version, :model_family,
                :execution_mode, :demo_model, :features, :prediction, :probability,
                :latency_ms, :ground_truth, :ground_truth_received_at
            )
            """
        with self.engine.begin() as connection:
            connection.execute(text(sql), params)

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        if self.engine is None:
            raise RuntimeError("prediction database is not configured")
        sql = """
        SELECT prediction_id, timestamp, model_name, model_version, model_family,
               execution_mode, demo_model, features, prediction, probability,
               latency_ms, ground_truth, ground_truth_received_at
        FROM predictions
        WHERE prediction_id = :prediction_id
        """
        with self.engine.connect() as connection:
            row = connection.execute(
                text(sql),
                {"prediction_id": prediction_id},
            ).mappings().first()
        if row is None:
            return None
        return _decode_prediction_row(dict(row))

    def record_ground_truth(
        self,
        *,
        prediction_id: str,
        ground_truth: int,
        schema_fingerprint: str,
        validation_status: str,
        validation_errors: list[dict[str, Any]],
    ) -> GroundTruthUpdateResult:
        if self.engine is None:
            raise RuntimeError("prediction database is not configured")
        if ground_truth not in {0, 1}:
            raise ValueError("ground_truth must be 0 or 1")

        received_at = datetime.now(UTC).isoformat()
        approved_at = received_at if validation_status == "approved" else None
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT prediction_id, features, model_version, ground_truth,
                           ground_truth_received_at
                    FROM predictions
                    WHERE prediction_id = :prediction_id
                    """
                ),
                {"prediction_id": prediction_id},
            ).mappings().first()
            if row is None:
                raise PredictionNotFoundError(prediction_id)

            current_truth = row["ground_truth"]
            if current_truth is not None and int(current_truth) != ground_truth:
                raise ConflictingGroundTruthError(
                    f"prediction {prediction_id} already has ground_truth "
                    f"{current_truth}"
                )

            status = "recorded"
            if current_truth is None:
                connection.execute(
                    text(
                        """
                        UPDATE predictions
                        SET ground_truth = :ground_truth,
                            ground_truth_received_at = :ground_truth_received_at
                        WHERE prediction_id = :prediction_id
                        """
                    ),
                    {
                        "prediction_id": prediction_id,
                        "ground_truth": ground_truth,
                        "ground_truth_received_at": received_at,
                    },
                )
            else:
                status = "idempotent"
                received_at = str(row["ground_truth_received_at"])

            features = _decode_json(row["features"])
            observation_params = {
                "prediction_id": prediction_id,
                "features": json.dumps(features, sort_keys=True),
                "ground_truth": ground_truth,
                "staged_at": received_at,
                "validation_status": validation_status,
                "validation_errors": json.dumps(validation_errors, sort_keys=True),
                "approved_at": approved_at,
                "source_model_version": str(row["model_version"]),
                "schema_fingerprint": schema_fingerprint,
            }
            if self.dialect_name == "postgresql":
                upsert_sql = """
                INSERT INTO production_labelled_observations (
                    prediction_id, features, ground_truth, staged_at,
                    validation_status, validation_errors, approved_at,
                    source_model_version, schema_fingerprint
                )
                VALUES (
                    :prediction_id, CAST(:features AS JSONB), :ground_truth,
                    :staged_at, :validation_status,
                    CAST(:validation_errors AS JSONB), :approved_at,
                    :source_model_version, :schema_fingerprint
                )
                ON CONFLICT (prediction_id) DO UPDATE SET
                    features = EXCLUDED.features,
                    ground_truth = EXCLUDED.ground_truth,
                    validation_status = EXCLUDED.validation_status,
                    validation_errors = EXCLUDED.validation_errors,
                    approved_at = EXCLUDED.approved_at,
                    source_model_version = EXCLUDED.source_model_version,
                    schema_fingerprint = EXCLUDED.schema_fingerprint
                """
            else:
                upsert_sql = """
                INSERT INTO production_labelled_observations (
                    prediction_id, features, ground_truth, staged_at,
                    validation_status, validation_errors, approved_at,
                    source_model_version, schema_fingerprint
                )
                VALUES (
                    :prediction_id, :features, :ground_truth, :staged_at,
                    :validation_status, :validation_errors, :approved_at,
                    :source_model_version, :schema_fingerprint
                )
                ON CONFLICT(prediction_id) DO UPDATE SET
                    features = excluded.features,
                    ground_truth = excluded.ground_truth,
                    validation_status = excluded.validation_status,
                    validation_errors = excluded.validation_errors,
                    approved_at = excluded.approved_at,
                    source_model_version = excluded.source_model_version,
                    schema_fingerprint = excluded.schema_fingerprint
                """
            connection.execute(text(upsert_sql), observation_params)

        return GroundTruthUpdateResult(
            prediction_id=prediction_id,
            ground_truth=ground_truth,
            status=status,
            ground_truth_received_at=received_at,
            production_observation_status=validation_status,
            validation_errors=validation_errors,
        )

    def get_approved_production_observations(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("prediction database is not configured")
        params: dict[str, Any] = {}
        sql = """
        SELECT prediction_id, features, ground_truth, staged_at, validation_status,
               validation_errors, approved_at, source_model_version,
               schema_fingerprint
        FROM production_labelled_observations
        WHERE validation_status = 'approved'
        ORDER BY staged_at ASC, prediction_id ASC
        """
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = int(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(text(sql), params).mappings().all()
        return [_decode_production_row(dict(row)) for row in rows]


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    return value


def _decode_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    row["features"] = _decode_json(row["features"])
    return {key: _jsonable_value(value) for key, value in row.items()}


def _decode_production_row(row: dict[str, Any]) -> dict[str, Any]:
    row["features"] = _decode_json(row["features"])
    row["validation_errors"] = _decode_json(row["validation_errors"])
    return {key: _jsonable_value(value) for key, value in row.items()}
