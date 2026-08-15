from __future__ import annotations

import os
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.retraining.config import load_retraining_config
from sentinelml.retraining.repository import RetrainingRepository
from sentinelml.serving.ops import (
    OpsConfig,
    ops_monitoring,
    ops_overview,
    ops_resilience,
    ops_retraining,
)


class Phase95OpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / f"{self._testMethodName}_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_monitoring_healthy_live_payload_populates_metrics(self) -> None:
        payload = {
            "status": "healthy",
            "monitoring_health": "healthy",
            "last_check_timestamp": "2026-08-16T00:00:00+00:00",
            "window_size": 100,
            "drift_share": 0.2,
            "drifting_feature_count": 3,
            "prediction_distribution": {"0": 95, "1": 5},
            "labelled_row_count": 20,
            "performance": {
                "attack_recall": 0.8,
                "f1": 0.7,
                "false_positive_rate": None,
                "support": {"labelled_rows": 20},
            },
        }
        config = OpsConfig("http://monitor:9101/status", None, 2)
        with patch(
            "sentinelml.serving.ops._get_json",
            return_value={"ok": True, "payload": payload},
        ):
            result = ops_monitoring(config)

        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["latest"]["drift_share"], 0.2)
        self.assertEqual(result["heartbeat"]["last_check_timestamp"], payload["last_check_timestamp"])

    def test_monitoring_warming_up_stale_and_unreachable_are_preserved(self) -> None:
        config = OpsConfig("http://monitor:9101/status", None, 2)
        cases = [
            ({"status": "warming_up", "monitoring_health": "warming_up"}, "warming_up"),
            ({"status": "stale", "monitoring_health": "stale"}, "stale"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected), patch(
                "sentinelml.serving.ops._get_json",
                return_value={"ok": True, "payload": payload},
            ):
                self.assertEqual(ops_monitoring(config)["status"], expected)

        with patch(
            "sentinelml.serving.ops._get_json",
            return_value={"ok": False, "error": "timeout"},
        ):
            result = ops_monitoring(config)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"], "timeout")

    def test_resilience_active_inactive_and_unreachable(self) -> None:
        config = OpsConfig(None, "http://resilience:9201/status", 2)
        with patch(
            "sentinelml.serving.ops._get_json",
            return_value={
                "ok": True,
                "payload": {"status": "healthy", "active_probation_count": 1},
            },
        ):
            self.assertEqual(ops_resilience(config)["status"], "healthy")

        with patch(
            "sentinelml.serving.ops._get_json",
            return_value={
                "ok": True,
                "payload": {"status": "healthy", "active_probation_count": 0},
            },
        ):
            self.assertEqual(ops_resilience(config)["status"], "inactive")

        with patch(
            "sentinelml.serving.ops._get_json",
            return_value={"ok": False, "error": "connection refused"},
        ):
            self.assertEqual(ops_resilience(config)["status"], "unavailable")

    def test_retraining_disabled_no_run_and_latest_completed_run(self) -> None:
        db_path = self.root / "retraining.sqlite"
        env = {
            "SENTINELML_DATABASE_URL": f"sqlite:///{db_path}",
            "SENTINELML_RETRAINING_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            result = ops_retraining()
            self.assertEqual(result["status"], "inactive")
            self.assertEqual(result["latest_status"], "no_data")

            config = load_retraining_config(validate_data_paths=False)
            repository = RetrainingRepository(config)
            repository.initialize()
            repository.create_run(
                retraining_run_id="run-1",
                monitoring_run_id="mon-1",
                trigger_reasons=["data_drift"],
                drift_share=0.5,
                performance_metrics={"f1": 0.7},
            )
            repository.finish_run("run-1", status="rejected")
            repository.mark_processed(
                monitoring_run_id="mon-1",
                decision="triggered",
                action_taken="rejected",
                report_path=None,
            )
            if repository.engine is not None:
                repository.engine.dispose()
            result = ops_retraining()

        self.assertEqual(result["latest"]["retraining_run_id"], "run-1")
        self.assertEqual(result["latest_status"], "rejected")
        self.assertEqual(result["latest_processed_monitoring"]["decision"], "triggered")
        self.assertIn("cooldown", result)

    def test_overview_keeps_inference_available_when_registry_unavailable(self) -> None:
        app_state = types.SimpleNamespace(
            model_manager=FakeModelManager(registry_available=False),
            repository=types.SimpleNamespace(database_status=lambda: "healthy"),
        )
        config = OpsConfig("http://monitor/status", "http://resilience/status", 2)

        def fake_get_json(url: str, *, timeout: float) -> dict[str, object]:
            if "monitor" in url:
                return {
                    "ok": True,
                    "payload": {
                        "status": "healthy",
                        "monitoring_health": "healthy",
                        "drift_share": 0.1,
                        "performance": {"attack_recall": None, "f1": 0.5},
                    },
                }
            return {"ok": False, "error": "down"}

        with (
            patch("sentinelml.serving.ops._get_json", side_effect=fake_get_json),
            patch("sentinelml.serving.ops.ops_retraining", return_value={"status": "no_data"}),
        ):
            result = ops_overview(app_state, config)

        self.assertEqual(result["health"]["inference"]["status"], "available")
        self.assertEqual(result["health"]["registry"]["status"], "unavailable")
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["metrics"]["drift_share"], 0.1)
        self.assertIsNone(result["metrics"]["attack_recall"])


class FakeModel:
    model_name = "sentinelml-ids"
    model_version = "2"
    model_family = "xgboost"
    execution_mode = "smoke"
    demo_model = True
    loaded_at = "2026-08-16T00:00:00+00:00"
    source_run_id = "run-id"


class FakeModelManager:
    def __init__(self, *, registry_available: bool) -> None:
        self.registry_available = registry_available

    def current(self) -> FakeModel:
        return FakeModel()

    def registry_status(self) -> dict[str, object]:
        if self.registry_available:
            return {"connectivity": "available", "registry_champion_version": "2"}
        return {
            "connectivity": "unavailable",
            "registry_champion_version": None,
            "error": "mlflow down",
        }


if __name__ == "__main__":
    unittest.main()
