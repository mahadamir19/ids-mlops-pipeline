"""Durable Phase 8 retraining state in the application database."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine

from sentinelml.retraining.config import RetrainingConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RetrainingRepository:
    def __init__(
        self,
        config: RetrainingConfig,
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

    def schema_available(self) -> bool:
        if self.engine is None:
            return False
        inspector = inspect(self.engine)
        return all(
            inspector.has_table(table_name)
            for table_name in [
                "retraining_runs",
                "retraining_monitoring_processed",
            ]
        )

    def initialize(self) -> None:
        if self.engine is None:
            return
        json_type = "JSONB" if self.dialect_name == "postgresql" else "TEXT"
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS retraining_runs (
                        retraining_run_id VARCHAR(64) PRIMARY KEY,
                        trigger_monitoring_run_id VARCHAR(128),
                        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        finished_at TIMESTAMP WITH TIME ZONE,
                        status VARCHAR(64) NOT NULL,
                        trigger_reasons {json_type} NOT NULL,
                        drift_share DOUBLE PRECISION,
                        performance_metrics {json_type} NOT NULL,
                        historical_rows INTEGER,
                        production_rows INTEGER,
                        deduplicated_rows INTEGER,
                        dataset_fingerprint VARCHAR(128),
                        mlflow_run_id VARCHAR(128),
                        registered_model_version VARCHAR(64),
                        candidate_evaluation {json_type},
                        promotion_result {json_type},
                        error TEXT,
                        cooldown_until TIMESTAMP WITH TIME ZONE
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS retraining_monitoring_processed (
                        monitoring_run_id VARCHAR(128) PRIMARY KEY,
                        processed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        decision VARCHAR(64) NOT NULL,
                        action_taken VARCHAR(64) NOT NULL,
                        report_path TEXT
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS retraining_consumed_observations (
                        prediction_id VARCHAR(64) PRIMARY KEY,
                        retraining_run_id VARCHAR(64) NOT NULL,
                        consumed_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS retraining_locks (
                        lock_name VARCHAR(64) PRIMARY KEY,
                        owner VARCHAR(128) NOT NULL,
                        acquired_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """
                )
            )

    def already_processed(self, monitoring_run_id: str) -> bool:
        if self.engine is None or not monitoring_run_id:
            return False
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT monitoring_run_id
                    FROM retraining_monitoring_processed
                    WHERE monitoring_run_id = :monitoring_run_id
                      AND action_taken <> 'failed'
                    """
                ),
                {"monitoring_run_id": monitoring_run_id},
            ).first()
        return row is not None

    def mark_processed(
        self,
        *,
        monitoring_run_id: str,
        decision: str,
        action_taken: str,
        report_path: str | None,
    ) -> None:
        if self.engine is None or not monitoring_run_id:
            return
        params = {
            "monitoring_run_id": monitoring_run_id,
            "processed_at": utc_now(),
            "decision": decision,
            "action_taken": action_taken,
            "report_path": report_path,
        }
        if self.dialect_name == "postgresql":
            sql = """
            INSERT INTO retraining_monitoring_processed (
                monitoring_run_id, processed_at, decision, action_taken, report_path
            )
            VALUES (
                :monitoring_run_id, :processed_at, :decision, :action_taken,
                :report_path
            )
            ON CONFLICT (monitoring_run_id) DO NOTHING
            """
        else:
            sql = """
            INSERT OR IGNORE INTO retraining_monitoring_processed (
                monitoring_run_id, processed_at, decision, action_taken, report_path
            )
            VALUES (
                :monitoring_run_id, :processed_at, :decision, :action_taken,
                :report_path
            )
            """
        with self.engine.begin() as connection:
            connection.execute(text(sql), params)

    def claim_monitoring_run(
        self,
        *,
        monitoring_run_id: str,
        decision: str,
        report_path: str | None,
    ) -> bool:
        """Atomically claim a monitoring run for execution processing."""

        if self.engine is None or not monitoring_run_id:
            return False
        params = {
            "monitoring_run_id": monitoring_run_id,
            "processed_at": utc_now(),
            "decision": decision,
            "action_taken": "claimed",
            "report_path": report_path,
        }
        if self.dialect_name == "postgresql":
            sql = """
            INSERT INTO retraining_monitoring_processed (
                monitoring_run_id, processed_at, decision, action_taken, report_path
            )
            VALUES (
                :monitoring_run_id, :processed_at, :decision, :action_taken,
                :report_path
            )
            ON CONFLICT (monitoring_run_id) DO UPDATE
            SET processed_at = EXCLUDED.processed_at,
                decision = EXCLUDED.decision,
                action_taken = EXCLUDED.action_taken,
                report_path = EXCLUDED.report_path
            WHERE retraining_monitoring_processed.action_taken = 'failed'
            """
        else:
            sql = """
            INSERT INTO retraining_monitoring_processed (
                monitoring_run_id, processed_at, decision, action_taken, report_path
            )
            VALUES (
                :monitoring_run_id, :processed_at, :decision, :action_taken,
                :report_path
            )
            ON CONFLICT(monitoring_run_id) DO UPDATE
            SET processed_at = excluded.processed_at,
                decision = excluded.decision,
                action_taken = excluded.action_taken,
                report_path = excluded.report_path
            WHERE retraining_monitoring_processed.action_taken = 'failed'
            """
        with self.engine.begin() as connection:
            result = connection.execute(text(sql), params)
        return int(result.rowcount or 0) == 1

    def update_processed_action(
        self,
        *,
        monitoring_run_id: str,
        decision: str,
        action_taken: str,
        report_path: str | None,
    ) -> None:
        if self.engine is None or not monitoring_run_id:
            return
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE retraining_monitoring_processed
                    SET processed_at = :processed_at,
                        decision = :decision,
                        action_taken = :action_taken,
                        report_path = :report_path
                    WHERE monitoring_run_id = :monitoring_run_id
                    """
                ),
                {
                    "monitoring_run_id": monitoring_run_id,
                    "processed_at": utc_now(),
                    "decision": decision,
                    "action_taken": action_taken,
                    "report_path": report_path,
                },
            )

    def create_run(
        self,
        *,
        retraining_run_id: str,
        monitoring_run_id: str,
        trigger_reasons: list[str],
        drift_share: float | None,
        performance_metrics: dict[str, Any],
        status: str = "triggered",
    ) -> None:
        if self.engine is None:
            return
        trigger_expr = (
            "CAST(:trigger_reasons AS JSONB)"
            if self.dialect_name == "postgresql"
            else ":trigger_reasons"
        )
        performance_expr = (
            "CAST(:performance_metrics AS JSONB)"
            if self.dialect_name == "postgresql"
            else ":performance_metrics"
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO retraining_runs (
                        retraining_run_id, trigger_monitoring_run_id, started_at,
                        status, trigger_reasons, drift_share, performance_metrics
                    )
                    VALUES (
                        :retraining_run_id, :trigger_monitoring_run_id, :started_at,
                        :status, {trigger_expr}, :drift_share, {performance_expr}
                    )
                    """
                ),
                {
                    "retraining_run_id": retraining_run_id,
                    "trigger_monitoring_run_id": monitoring_run_id,
                    "started_at": utc_now(),
                    "status": status,
                    "trigger_reasons": self._json(trigger_reasons),
                    "drift_share": drift_share,
                    "performance_metrics": self._json(performance_metrics),
                },
            )

    def update_run(self, retraining_run_id: str, **fields: Any) -> None:
        if self.engine is None or not fields:
            return
        assignments = []
        params: dict[str, Any] = {"retraining_run_id": retraining_run_id}
        json_fields = {"candidate_evaluation", "promotion_result"}
        for key, value in fields.items():
            assignment_value = (
                f"CAST(:{key} AS JSONB)"
                if self.dialect_name == "postgresql" and key in json_fields
                else f":{key}"
            )
            assignments.append(f"{key} = {assignment_value}")
            params[key] = self._json(value) if key in json_fields else value
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    UPDATE retraining_runs
                    SET {", ".join(assignments)}
                    WHERE retraining_run_id = :retraining_run_id
                    """
                ),
                params,
            )

    def finish_run(
        self,
        retraining_run_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> str:
        cooldown_until = (
            datetime.now(UTC) + timedelta(seconds=self.config.cooldown_seconds)
        ).isoformat()
        self.update_run(
            retraining_run_id,
            status=status,
            finished_at=utc_now(),
            error=error,
            cooldown_until=cooldown_until,
        )
        return cooldown_until

    def cooldown_status(self) -> dict[str, Any]:
        if self.engine is None:
            return {
                "active": False,
                "last_cycle_finished_at": None,
                "cooldown_until": None,
            }
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT finished_at, cooldown_until, status
                    FROM retraining_runs
                    WHERE finished_at IS NOT NULL
                    ORDER BY finished_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        if row is None:
            return {
                "active": False,
                "last_cycle_finished_at": None,
                "cooldown_until": None,
            }
        cooldown_until = _parse_datetime(row["cooldown_until"])
        now = datetime.now(UTC)
        active = cooldown_until is not None and now < cooldown_until
        remaining = (cooldown_until - now).total_seconds() if active else 0.0
        return {
            "active": active,
            "last_cycle_finished_at": _jsonable_time(row["finished_at"]),
            "cooldown_until": _jsonable_time(row["cooldown_until"]),
            "remaining_seconds": max(0.0, float(remaining)),
            "last_status": row["status"],
        }

    def consumed_prediction_ids(self) -> set[str]:
        if self.engine is None:
            return set()
        with self.engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT consumed.prediction_id
                    FROM retraining_consumed_observations AS consumed
                    LEFT JOIN retraining_runs AS runs
                      ON runs.retraining_run_id = consumed.retraining_run_id
                    WHERE runs.status IS NULL OR runs.status <> 'failed'
                    """
                )
            ).all()
        return {str(row[0]) for row in rows}

    def mark_observations_consumed(
        self,
        *,
        retraining_run_id: str,
        prediction_ids: list[str],
    ) -> None:
        if self.engine is None:
            return
        if not prediction_ids:
            return
        sql = (
            """
            INSERT INTO retraining_consumed_observations (
                prediction_id, retraining_run_id, consumed_at
            )
            VALUES (:prediction_id, :retraining_run_id, :consumed_at)
            ON CONFLICT (prediction_id) DO NOTHING
            """
            if self.dialect_name == "postgresql"
            else """
            INSERT OR IGNORE INTO retraining_consumed_observations (
                prediction_id, retraining_run_id, consumed_at
            )
            VALUES (:prediction_id, :retraining_run_id, :consumed_at)
            """
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(sql),
                [
                    {
                        "prediction_id": prediction_id,
                        "retraining_run_id": retraining_run_id,
                        "consumed_at": utc_now(),
                    }
                    for prediction_id in prediction_ids
                ],
            )

    def current_state(self) -> dict[str, Any]:
        if self.engine is None:
            return {"state": "idle", "configured": False}
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT retraining_run_id, status, started_at, finished_at,
                           cooldown_until, registered_model_version, error
                    FROM retraining_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        if row is None:
            return {"state": "idle", "configured": True}
        cooldown = self.cooldown_status()
        state = "cooldown" if cooldown.get("active") else str(row["status"])
        terminal = row["status"] in {"promoted", "rejected", "promotion_pending"}
        if terminal and not cooldown.get("active"):
            state = "idle"
        return {
            "state": state,
            "configured": True,
            "latest_run": {
                key: _jsonable_time(value) for key, value in dict(row).items()
            },
            "cooldown": cooldown,
        }

    def latest_run(self) -> dict[str, Any] | None:
        """Return the latest retraining run without changing retraining state."""

        if self.engine is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT retraining_run_id, trigger_monitoring_run_id, started_at,
                           finished_at, status, trigger_reasons, drift_share,
                           performance_metrics, historical_rows, production_rows,
                           deduplicated_rows, dataset_fingerprint, mlflow_run_id,
                           registered_model_version, candidate_evaluation,
                           promotion_result, error, cooldown_until
                    FROM retraining_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        return _jsonable_mapping(row)

    def latest_processed_monitoring(self) -> dict[str, Any] | None:
        """Return the latest processed monitoring decision, if one exists."""

        if self.engine is None:
            return None
        with self.engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT monitoring_run_id, processed_at, decision, action_taken,
                           report_path
                    FROM retraining_monitoring_processed
                    ORDER BY processed_at DESC
                    LIMIT 1
                    """
                )
            ).mappings().first()
        return _jsonable_mapping(row)

    def acquire_table_lock(self, *, owner: str, lock_name: str = "phase8") -> bool:
        if self.engine is None:
            return False
        sql = (
            """
            INSERT INTO retraining_locks (lock_name, owner, acquired_at)
            VALUES (:lock_name, :owner, :acquired_at)
            ON CONFLICT (lock_name) DO NOTHING
            """
            if self.dialect_name == "postgresql"
            else """
            INSERT OR IGNORE INTO retraining_locks (lock_name, owner, acquired_at)
            VALUES (:lock_name, :owner, :acquired_at)
            """
        )
        with self.engine.begin() as connection:
            result = connection.execute(
                text(sql),
                {"lock_name": lock_name, "owner": owner, "acquired_at": utc_now()},
            )
            return int(result.rowcount or 0) == 1

    def release_table_lock(self, *, owner: str, lock_name: str = "phase8") -> None:
        if self.engine is None:
            return
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    DELETE FROM retraining_locks
                    WHERE lock_name = :lock_name AND owner = :owner
                    """
                ),
                {"lock_name": lock_name, "owner": owner},
            )

    def try_postgres_advisory_lock(self, connection: Connection, key: int) -> bool:
        row = connection.execute(
            text("SELECT pg_try_advisory_lock(:key)"),
            {"key": key},
        ).first()
        return bool(row and row[0])

    def release_postgres_advisory_lock(self, connection: Connection, key: int) -> None:
        connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})

    def _json(self, value: Any) -> str:
        return json.dumps(value, sort_keys=True)


def latest_retraining_state(config: RetrainingConfig) -> dict[str, Any]:
    repository = RetrainingRepository(config)
    repository.initialize()
    return repository.current_state()


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _jsonable_time(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _jsonable_mapping(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = {key: _jsonable_time(value) for key, value in dict(row).items()}
    for key in [
        "trigger_reasons",
        "performance_metrics",
        "candidate_evaluation",
        "promotion_result",
    ]:
        value = payload.get(key)
        if isinstance(value, str):
            try:
                payload[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return payload
