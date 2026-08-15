"""Durable Phase 9 probation, rollback, and retry state."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sentinelml.resilience.config import ResilienceConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ResilienceRepository:
    def __init__(
        self,
        config: ResilienceConfig,
        *,
        engine: Engine | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or self._create_engine(config.database_url)

    def _create_engine(self, database_url: str | None) -> Engine | None:
        if not database_url:
            return None
        connect_args: dict[str, Any] = {}
        if database_url.startswith("postgresql"):
            connect_args["connect_timeout"] = (
                self.config.database_connect_timeout_seconds
            )
            connect_args["options"] = (
                f"-c statement_timeout={self.config.database_statement_timeout_ms}"
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
        json_type = "JSONB" if self.dialect_name == "postgresql" else "TEXT"
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS model_probations (
                        probation_id VARCHAR(64) PRIMARY KEY,
                        promoted_version VARCHAR(64) NOT NULL,
                        previous_champion_version VARCHAR(64) NOT NULL,
                        promotion_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        ends_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        status VARCHAR(64) NOT NULL,
                        minimum_labelled_rows INTEGER NOT NULL,
                        minimum_attack_support INTEGER NOT NULL,
                        minimum_benign_support INTEGER NOT NULL,
                        latest_evaluation_at TIMESTAMP WITH TIME ZONE,
                        labelled_rows INTEGER NOT NULL DEFAULT 0,
                        attack_support INTEGER NOT NULL DEFAULT 0,
                        benign_support INTEGER NOT NULL DEFAULT 0,
                        performance_metrics {json_type} NOT NULL,
                        severe_violation_reasons {json_type} NOT NULL,
                        rollback_attempt_id VARCHAR(64),
                        completed_at TIMESTAMP WITH TIME ZONE
                    )
                    """
                )
            )
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS rollback_events (
                        rollback_id VARCHAR(64) PRIMARY KEY,
                        source VARCHAR(32) NOT NULL,
                        probation_id VARCHAR(64),
                        from_version VARCHAR(64),
                        to_version VARCHAR(64) NOT NULL,
                        trigger_metrics {json_type} NOT NULL,
                        reason TEXT,
                        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        status VARCHAR(64) NOT NULL,
                        registry_result {json_type} NOT NULL,
                        reload_result {json_type} NOT NULL,
                        error TEXT
                    )
                    """
                )
            )
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS promotion_retry_attempts (
                        retry_id VARCHAR(64) PRIMARY KEY,
                        candidate_version VARCHAR(64) NOT NULL,
                        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        completed_at TIMESTAMP WITH TIME ZONE,
                        status VARCHAR(64) NOT NULL,
                        current_champion_version VARCHAR(64),
                        gate_evaluation {json_type} NOT NULL,
                        outcome {json_type} NOT NULL,
                        error TEXT
                    )
                    """
                )
            )

    def create_probation(self, payload: dict[str, Any]) -> None:
        if self.engine is None:
            return
        params = {
            **payload,
            "performance_metrics": self._json(
                payload.get("performance_metrics", {})
            ),
            "severe_violation_reasons": self._json(
                payload.get("severe_violation_reasons", [])
            ),
        }
        sql = (
            """
            INSERT INTO model_probations (
                probation_id, promoted_version, previous_champion_version,
                promotion_timestamp, started_at, ends_at, status,
                minimum_labelled_rows, minimum_attack_support,
                minimum_benign_support, latest_evaluation_at, labelled_rows,
                attack_support, benign_support, performance_metrics,
                severe_violation_reasons, rollback_attempt_id, completed_at
            )
            VALUES (
                :probation_id, :promoted_version, :previous_champion_version,
                :promotion_timestamp, :started_at, :ends_at, :status,
                :minimum_labelled_rows, :minimum_attack_support,
                :minimum_benign_support, :latest_evaluation_at, :labelled_rows,
                :attack_support, :benign_support, :performance_metrics,
                :severe_violation_reasons, :rollback_attempt_id, :completed_at
            )
            ON CONFLICT (probation_id) DO NOTHING
            """
            if self.dialect_name == "postgresql"
            else """
            INSERT OR IGNORE INTO model_probations (
                probation_id, promoted_version, previous_champion_version,
                promotion_timestamp, started_at, ends_at, status,
                minimum_labelled_rows, minimum_attack_support,
                minimum_benign_support, latest_evaluation_at, labelled_rows,
                attack_support, benign_support, performance_metrics,
                severe_violation_reasons, rollback_attempt_id, completed_at
            )
            VALUES (
                :probation_id, :promoted_version, :previous_champion_version,
                :promotion_timestamp, :started_at, :ends_at, :status,
                :minimum_labelled_rows, :minimum_attack_support,
                :minimum_benign_support, :latest_evaluation_at, :labelled_rows,
                :attack_support, :benign_support, :performance_metrics,
                :severe_violation_reasons, :rollback_attempt_id, :completed_at
            )
            """
        )
        with self.engine.begin() as connection:
            connection.execute(text(sql), params)

    def update_probation(self, probation_id: str, **fields: Any) -> None:
        if self.engine is None or not fields:
            return
        json_fields = {"performance_metrics", "severe_violation_reasons"}
        params: dict[str, Any] = {"probation_id": probation_id}
        assignments: list[str] = []
        for key, value in fields.items():
            assignments.append(f"{key} = :{key}")
            params[key] = self._json(value) if key in json_fields else value
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    UPDATE model_probations
                    SET {", ".join(assignments)}
                    WHERE probation_id = :probation_id
                    """
                ),
                params,
            )

    def active_probations(self) -> list[dict[str, Any]]:
        if self.engine is None:
            return []
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT *
                    FROM model_probations
                    WHERE status IN ('active', 'awaiting_evidence')
                    ORDER BY started_at ASC
                    """
                )
            ).mappings().all()
        return [self._decode_row(dict(row)) for row in rows]

    def latest_probation(self) -> dict[str, Any] | None:
        if self.engine is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT *
                    FROM model_probations
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        return self._decode_row(dict(row)) if row else None

    def insert_rollback_event(self, payload: dict[str, Any]) -> None:
        if self.engine is None:
            return
        params = {
            **payload,
            "trigger_metrics": self._json(payload.get("trigger_metrics", {})),
            "registry_result": self._json(payload.get("registry_result", {})),
            "reload_result": self._json(payload.get("reload_result", {})),
        }
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO rollback_events (
                        rollback_id, source, probation_id, from_version, to_version,
                        trigger_metrics, reason, started_at, completed_at, status,
                        registry_result, reload_result, error
                    )
                    VALUES (
                        :rollback_id, :source, :probation_id, :from_version,
                        :to_version, :trigger_metrics, :reason, :started_at,
                        :completed_at, :status, :registry_result, :reload_result,
                        :error
                    )
                    """
                ),
                params,
            )

    def update_rollback_event(self, rollback_id: str, **fields: Any) -> None:
        if self.engine is None or not fields:
            return
        json_fields = {"trigger_metrics", "registry_result", "reload_result"}
        params: dict[str, Any] = {"rollback_id": rollback_id}
        assignments: list[str] = []
        for key, value in fields.items():
            assignments.append(f"{key} = :{key}")
            params[key] = self._json(value) if key in json_fields else value
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    UPDATE rollback_events
                    SET {", ".join(assignments)}
                    WHERE rollback_id = :rollback_id
                    """
                ),
                params,
            )

    def latest_rollback(self) -> dict[str, Any] | None:
        if self.engine is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT *
                    FROM rollback_events
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        return self._decode_row(dict(row)) if row else None

    def rollback_for_probation(self, probation_id: str) -> dict[str, Any] | None:
        if self.engine is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT *
                    FROM rollback_events
                    WHERE probation_id = :probation_id
                      AND status IN ('succeeded', 'already_rolled_back')
                    ORDER BY started_at ASC
                    LIMIT 1
                    """
                ),
                {"probation_id": probation_id},
            ).mappings().first()
        return self._decode_row(dict(row)) if row else None

    def insert_retry_attempt(self, payload: dict[str, Any]) -> None:
        if self.engine is None:
            return
        params = {
            **payload,
            "gate_evaluation": self._json(payload.get("gate_evaluation", {})),
            "outcome": self._json(payload.get("outcome", {})),
        }
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO promotion_retry_attempts (
                        retry_id, candidate_version, started_at, completed_at,
                        status, current_champion_version, gate_evaluation,
                        outcome, error
                    )
                    VALUES (
                        :retry_id, :candidate_version, :started_at, :completed_at,
                        :status, :current_champion_version, :gate_evaluation,
                        :outcome, :error
                    )
                    """
                ),
                params,
            )

    def probation_observations(
        self,
        *,
        model_version: str,
        started_at: str,
        ends_at: str,
    ) -> list[dict[str, Any]]:
        if self.engine is None:
            return []
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT p.prediction_id, p.timestamp, p.model_version,
                           p.prediction, p.ground_truth,
                           p.ground_truth_received_at
                    FROM predictions AS p
                    INNER JOIN production_labelled_observations AS plo
                      ON plo.prediction_id = p.prediction_id
                    WHERE p.model_version = :model_version
                      AND p.timestamp >= :started_at
                      AND p.timestamp < :ends_at
                      AND p.ground_truth IS NOT NULL
                      AND plo.validation_status = 'approved'
                    ORDER BY p.timestamp ASC, p.prediction_id ASC
                    """
                ),
                {
                    "model_version": str(model_version),
                    "started_at": started_at,
                    "ends_at": ends_at,
                },
            ).mappings().all()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        active = self.active_probations()
        latest = self.latest_probation()
        latest_rollback = self.latest_rollback()
        return {
            "configured": self.configured,
            "active_probation": active[0] if active else None,
            "active_probation_count": len(active),
            "latest_probation": latest,
            "latest_rollback": latest_rollback,
        }

    def _json(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for key in [
            "performance_metrics",
            "severe_violation_reasons",
            "trigger_metrics",
            "registry_result",
            "reload_result",
            "gate_evaluation",
            "outcome",
        ]:
            if key in row and isinstance(row[key], str):
                row[key] = json.loads(row[key])
        for key, value in list(row.items()):
            if isinstance(value, datetime):
                row[key] = value.isoformat()
        return row
