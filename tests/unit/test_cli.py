from __future__ import annotations

import json
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml import cli


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class UnifiedCliTests(unittest.TestCase):
    def test_data_prepare_dispatches_phase1_preprocessing_with_options(self) -> None:
        with patch(
            "sentinelml.data.preprocess.generate_phase1_datasets",
            return_value={"partitions": {}},
        ) as generate:
            code, stdout, stderr = run_cli(
                [
                    "data",
                    "prepare",
                    "--config",
                    "configs/custom_data.json",
                    "--label-mapping",
                    "configs/custom_labels.json",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), {"partitions": {}})
        generate.assert_called_once_with(
            config_path=Path("configs/custom_data.json"),
            label_mapping_path=Path("configs/custom_labels.json"),
        )

    def test_data_prepare_defaults_dispatch_to_phase1_preprocessing(self) -> None:
        with patch(
            "sentinelml.data.preprocess.generate_phase1_datasets",
            return_value={"partitions": {}},
        ) as generate:
            code, stdout, stderr = run_cli(["data", "prepare"])

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), {"partitions": {}})
        generate.assert_called_once_with(config_path=None, label_mapping_path=None)

    def test_data_eda_dispatches_existing_phase1_eda_entrypoint(self) -> None:
        with patch("scripts.phase1_eda.main") as eda_main:
            code, stdout, stderr = run_cli(["data", "eda"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        eda_main.assert_called_once_with()

    def test_phase_commands_dispatch_and_print_json(self) -> None:
        commands = [
            (
                ["train", "baselines", "--mode", "full", "--mlflow"],
                "_run_phase2",
                {"args": ("full", True)},
            ),
            (
                ["optimize", "--mode", "full", "--model", "xgboost", "--mlflow"],
                "_run_phase3",
                {"args": ("full", "xgboost", True)},
            ),
            (
                ["candidate", "build", "--mode", "full", "--mlflow"],
                "_run_candidate",
                {"args": ("full", True)},
            ),
        ]
        for argv, target, expectation in commands:
            with self.subTest(argv=argv), patch.object(cli, target) as patched:
                patched.return_value = {"command": target}

                code, stdout, stderr = run_cli(argv)

                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                patched.assert_called_once_with(*expectation.get("args", ()))
                self.assertEqual(json.loads(stdout), {"command": target})

    def test_model_commands_dispatch_to_lifecycle_facade(self) -> None:
        service = Mock()
        service.status.return_value = {"status": "ok"}
        service.evaluate_candidate.return_value = {"event": "evaluated"}
        service.promote_or_reject.return_value = {"event": "promoted"}
        service.rollback.return_value = {"event": "rollback"}
        service.retry_pending.return_value = [{"event": "promotion_retry"}]

        cases = [
            (["model", "status"], service.status, (), {}),
            (
                ["model", "evaluate", "--version", "7"],
                service.evaluate_candidate,
                (),
                {"version": "7"},
            ),
            (
                ["model", "promote", "--version", "7"],
                service.promote_or_reject,
                (),
                {"version": "7"},
            ),
            (
                ["model", "rollback", "--version", "2"],
                service.rollback,
                (),
                {"version": "2", "reason": "manual_cli"},
            ),
            (["model", "retry-pending"], service.retry_pending, (), {}),
        ]
        with patch.object(cli, "_lifecycle_service", return_value=service):
            for argv, method, args, kwargs in cases:
                with self.subTest(argv=argv):
                    code, stdout, stderr = run_cli(argv)

                    self.assertEqual(code, 0)
                    self.assertEqual(stderr, "")
                    method.assert_called_once_with(*args, **kwargs)
                    self.assertTrue(json.loads(stdout))
                    method.reset_mock()

    def test_retrain_commands_dispatch_to_real_service_api(self) -> None:
        service = Mock()
        service.status.return_value = {"enabled": True}
        service.evaluate_latest_trigger.return_value = {"action_taken": "evaluate_only"}
        service.process_once.return_value = {"action_taken": "skipped"}
        service_cls = Mock(return_value=service)

        with patch("sentinelml.retraining.service.RetrainingService", service_cls):
            code, stdout, _stderr = run_cli(["retrain", "status"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), {"enabled": True})
            service.status.assert_called_once_with()

            code, stdout, _stderr = run_cli(["retrain", "evaluate-trigger"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["action_taken"], "evaluate_only")
            service.evaluate_latest_trigger.assert_called_once_with(
                force_recheck=False
            )

            code, _stdout, _stderr = run_cli(
                ["retrain", "evaluate-trigger", "--force-recheck"]
            )
            self.assertEqual(code, 0)
            service.evaluate_latest_trigger.assert_called_with(force_recheck=True)

            code, stdout, _stderr = run_cli(["retrain", "once"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["action_taken"], "skipped")
            service.process_once.assert_called_once_with()
            self.assertFalse(service.evaluate_trigger.called)
            self.assertFalse(service.run_once.called)

    def test_simulate_dispatches_supported_orchestration(self) -> None:
        with patch(
            "sentinelml.simulation.simulator.run_simulation_from_config",
            return_value={"simulation_run_id": "sim-1"},
        ) as run_simulation:
            code, stdout, stderr = run_cli(
                [
                    "simulate",
                    "--scenario",
                    "normal",
                    "--requests",
                    "3",
                    "--seed",
                    "99",
                    "--label-mode",
                    "none",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), {"simulation_run_id": "sim-1"})
        kwargs = run_simulation.call_args.kwargs
        self.assertEqual(kwargs["scenario_name"], "normal")
        self.assertEqual(kwargs["request_count"], 3)
        self.assertEqual(kwargs["seed"], 99)
        self.assertEqual(kwargs["label_mode"], "none")

    def test_monitor_serve_and_resilience_dispatch(self) -> None:
        uvicorn_run = Mock()
        service = Mock()
        service.status.return_value = {"resilience": "ok"}
        service.evaluate_probation.return_value = []
        service.rollback.return_value = {"event": "rollback"}
        service_cls = Mock(return_value=service)
        modules = {"uvicorn": types.SimpleNamespace(run=uvicorn_run)}

        with (
            patch.dict(sys.modules, modules),
            patch(
                "sentinelml.monitoring.service.run_monitoring_once",
                return_value={"monitoring": "ok"},
            ) as monitor_once,
            patch("sentinelml.serving.ops.ops_monitoring", return_value={"ops": "ok"}),
            patch("sentinelml.resilience.service.ResilienceService", service_cls),
        ):
            code, _stdout, _stderr = run_cli(["serve"])
            self.assertEqual(code, 0)
            uvicorn_run.assert_called_once_with(
                "sentinelml.serving.app:app",
                host="0.0.0.0",
                port=8000,
            )

            code, stdout, _stderr = run_cli(["monitor", "once"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), {"monitoring": "ok"})
            monitor_once.assert_called_once_with()

            code, stdout, _stderr = run_cli(["monitor", "status"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), {"ops": "ok"})

            code, stdout, _stderr = run_cli(["resilience", "status"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), {"resilience": "ok"})
            service.status.assert_called_once_with()

            code, stdout, _stderr = run_cli(["resilience", "evaluate-probation"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), [])
            service.evaluate_probation.assert_called_once_with()

            code, stdout, _stderr = run_cli(
                ["resilience", "rollback", "--version", "4"]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), {"event": "rollback"})
            service.rollback.assert_called_once_with(
                to_version="4",
                reason="manual_cli",
            )

    def test_help_smoke_and_invalid_options_use_argparse(self) -> None:
        for argv in [
            ["--help"],
            ["data", "--help"],
            ["data", "prepare", "--help"],
            ["data", "eda", "--help"],
            ["model", "--help"],
            ["retrain", "--help"],
            ["resilience", "--help"],
        ]:
            with self.subTest(argv=argv):
                stdout = StringIO()
                with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                    cli.main(argv)
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", stdout.getvalue())

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main(["retrain", "evaluate-trigger", "--not-real"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
