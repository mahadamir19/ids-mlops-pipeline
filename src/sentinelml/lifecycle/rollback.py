"""Rollback worker for the LifecycleService facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sentinelml.lifecycle.audit import audit_path, utc_now, write_json
from sentinelml.lifecycle.errors import LifecycleError
from sentinelml.lifecycle.registry import ModelVersionInfo, set_version_tags


class RollbackManager:
    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        alias: str,
        report_root: Path,
        get_version: Callable[[str | int], ModelVersionInfo],
        validate_model_uri: Callable[[str], None],
        get_champion: Callable[[], ModelVersionInfo | None],
        notify_serving_reload: Callable[[], dict[str, Any]],
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.alias = alias
        self.report_root = report_root
        self.get_version = get_version
        self.validate_model_uri = validate_model_uri
        self.get_champion = get_champion
        self.notify_serving_reload = notify_serving_reload

    def rollback(
        self,
        *,
        version: str | int,
        reason: str | None = None,
        source: str = "manual",
        probation_id: str | None = None,
        trigger_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self.get_version(version)
        state = target.tags.get("lifecycle_state")
        if state not in {"champion", "superseded"}:
            raise LifecycleError(
                "rollback target must be a current or superseded approved champion; "
                "rejected, failed, and candidate versions are not valid targets"
            )
        required = ["execution_mode", "demo_model", "model_family", "source_run_id"]
        missing = [key for key in required if key not in target.tags]
        if missing:
            raise LifecycleError(
                f"rollback target missing lifecycle metadata: {missing}"
            )
        source_model_uri = target.source or target.tags.get("source_model_uri")
        if not source_model_uri:
            raise LifecycleError("rollback target has no source model URI")
        self.validate_model_uri(source_model_uri)
        previous = self.get_champion()
        if previous is not None and previous.version == target.version:
            payload = {
                "event": "rollback",
                "status": "already_current",
                "model_name": self.model_name,
                "target_version": target.version,
                "previous_champion_version": previous.version,
                "rolled_back_at": utc_now(),
                "reason": reason,
                "source": source,
                "probation_id": probation_id,
                "trigger_metrics": trigger_metrics or {},
                "serving_reload_notification": {
                    "attempted": False,
                    "reason": "target already active champion",
                },
            }
            write_json(
                audit_path(self.report_root, "rollback", target.version),
                payload,
            )
            return payload
        rolled_back_at = utc_now()
        self.client.set_registered_model_alias(
            self.model_name,
            self.alias,
            target.version,
        )
        verified = self.get_champion()
        if verified is None or verified.version != target.version:
            raise LifecycleError("rollback alias verification failed")
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=target.version,
            tags={
                "lifecycle_state": "champion",
                "lifecycle.rollback_at": rolled_back_at,
                "lifecycle.rollback_reason": reason or "",
                "lifecycle.rollback_source": source,
                "lifecycle.rollback_probation_id": probation_id or "",
                "lifecycle.rollback_previous_champion_version": previous.version
                if previous
                else "",
            },
        )
        if previous is not None:
            set_version_tags(
                self.client,
                model_name=self.model_name,
                version=previous.version,
                tags={
                    "lifecycle_state": "rolled_back",
                    "lifecycle.rolled_back_at": rolled_back_at,
                    "lifecycle.rolled_back_to_version": target.version,
                    "lifecycle.rollback_source": source,
                    "lifecycle.rollback_probation_id": probation_id or "",
                },
            )
        payload = {
            "event": "rollback",
            "status": "succeeded",
            "model_name": self.model_name,
            "target_version": target.version,
            "previous_champion_version": previous.version if previous else None,
            "rolled_back_at": rolled_back_at,
            "reason": reason,
            "source": source,
            "probation_id": probation_id,
            "trigger_metrics": trigger_metrics or {},
        }
        payload["serving_reload_notification"] = self.notify_serving_reload()
        write_json(audit_path(self.report_root, "rollback", target.version), payload)
        return payload
