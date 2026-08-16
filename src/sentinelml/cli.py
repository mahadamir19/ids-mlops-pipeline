"""Unified SentinelML command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinelml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_data(subparsers)
    _add_train(subparsers)
    _add_optimize(subparsers)
    _add_candidate(subparsers)
    _add_model(subparsers)
    _add_serve(subparsers)
    _add_simulate(subparsers)
    _add_monitor(subparsers)
    _add_retrain(subparsers)
    _add_resilience(subparsers)
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0


def _add_data(subparsers: Any) -> None:
    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare")
    prepare.set_defaults(func=lambda _args: _run_phase1())


def _add_train(subparsers: Any) -> None:
    train = subparsers.add_parser("train")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    baselines = train_sub.add_parser("baselines")
    baselines.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    baselines.add_argument("--mlflow", action="store_true")
    baselines.set_defaults(func=lambda args: _run_phase2(args.mode, args.mlflow))


def _add_optimize(subparsers: Any) -> None:
    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    optimize.add_argument("--model", default="all")
    optimize.add_argument("--mlflow", action="store_true")
    optimize.set_defaults(
        func=lambda args: _run_phase3(args.mode, args.model, args.mlflow)
    )


def _add_candidate(subparsers: Any) -> None:
    candidate = subparsers.add_parser("candidate")
    candidate_sub = candidate.add_subparsers(dest="candidate_command", required=True)
    build = candidate_sub.add_parser("build")
    build.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    build.add_argument("--mlflow", action="store_true")
    build.set_defaults(func=lambda args: _run_candidate(args.mode, args.mlflow))


def _add_model(subparsers: Any) -> None:
    model = subparsers.add_parser("model")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    for name in ["status", "retry-pending"]:
        command = model_sub.add_parser(name)
        command.set_defaults(func=lambda args, n=name: _run_lifecycle(n, args))
    for name in ["evaluate", "promote", "rollback"]:
        command = model_sub.add_parser(name)
        command.add_argument("--version", required=True)
        command.set_defaults(func=lambda args, n=name: _run_lifecycle(n, args))


def _add_serve(subparsers: Any) -> None:
    serve = subparsers.add_parser("serve")
    serve.set_defaults(func=lambda _args: _serve())


def _add_simulate(subparsers: Any) -> None:
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--scenario", default="normal")
    simulate.add_argument("--requests", type=int, default=10)
    simulate.add_argument("--seed", type=int, default=None)
    simulate.add_argument(
        "--label-mode",
        choices=["randomized", "batch", "none"],
        default="randomized",
    )
    simulate.set_defaults(
        func=lambda args: _simulate(
            args.scenario,
            args.requests,
            seed=args.seed,
            label_mode=args.label_mode,
        )
    )


def _add_monitor(subparsers: Any) -> None:
    monitor = subparsers.add_parser("monitor")
    monitor_sub = monitor.add_subparsers(dest="monitor_command", required=True)
    once = monitor_sub.add_parser("once")
    once.set_defaults(func=lambda _args: _monitor_once())
    status = monitor_sub.add_parser("status")
    status.set_defaults(func=lambda _args: _read_ops("monitoring"))


def _add_retrain(subparsers: Any) -> None:
    retrain = subparsers.add_parser("retrain")
    retrain_sub = retrain.add_subparsers(dest="retrain_command", required=True)
    status = retrain_sub.add_parser("status")
    status.set_defaults(func=lambda args: _retrain("status", args))
    evaluate = retrain_sub.add_parser("evaluate-trigger")
    evaluate.add_argument("--force-recheck", action="store_true")
    evaluate.set_defaults(func=lambda args: _retrain("evaluate-trigger", args))
    for name in ["once"]:
        command = retrain_sub.add_parser(name)
        command.set_defaults(func=lambda args, n=name: _retrain(n, args))


def _add_resilience(subparsers: Any) -> None:
    resilience = subparsers.add_parser("resilience")
    resilience_sub = resilience.add_subparsers(
        dest="resilience_command",
        required=True,
    )
    for name in ["status", "evaluate-probation"]:
        command = resilience_sub.add_parser(name)
        command.set_defaults(func=lambda args, n=name: _resilience(n, args))
    rollback = resilience_sub.add_parser("rollback")
    rollback.add_argument("--version", required=True)
    rollback.set_defaults(func=lambda args: _resilience("rollback", args))


def _run_phase1() -> None:
    from scripts.phase1_eda import main as phase1_main

    phase1_main()


def _run_phase2(mode: str, enable_mlflow: bool) -> dict[str, Any]:
    from sentinelml.training.baselines import run_phase2_baselines

    return run_phase2_baselines(mode=mode, enable_mlflow=enable_mlflow)


def _run_phase3(mode: str, model: str, enable_mlflow: bool) -> dict[str, Any]:
    from sentinelml.optimization.study import run_phase3_optimization

    return run_phase3_optimization(
        mode=mode,
        model=model,
        enable_mlflow=enable_mlflow,
    )


def _run_candidate(mode: str, enable_mlflow: bool) -> dict[str, Any]:
    from sentinelml.final_candidate.training import run_phase3_final_candidate

    return run_phase3_final_candidate(mode=mode, enable_mlflow=enable_mlflow)


def _lifecycle_service() -> Any:
    from sentinelml.lifecycle.service import LifecycleService

    return LifecycleService()


def _run_lifecycle(name: str, args: Any) -> Any:
    service = _lifecycle_service()
    if name == "status":
        return service.status()
    if name == "evaluate":
        return service.evaluate_candidate(version=args.version)
    if name == "promote":
        return service.promote_or_reject(version=args.version)
    if name == "rollback":
        return service.rollback(version=args.version, reason="manual_cli")
    if name == "retry-pending":
        return service.retry_pending()
    raise ValueError(name)


def _serve() -> None:
    import uvicorn

    uvicorn.run("sentinelml.serving.app:app", host="0.0.0.0", port=8000)


def _simulate(
    scenario: str,
    requests: int,
    *,
    seed: int | None = None,
    label_mode: str = "randomized",
) -> Any:
    from sentinelml.simulation.config import DEFAULT_SIMULATION_CONFIG_PATH
    from sentinelml.simulation.simulator import run_simulation_from_config

    return run_simulation_from_config(
        config_path=Path(DEFAULT_SIMULATION_CONFIG_PATH),
        scenario_name=scenario,
        request_count=requests,
        seed=seed,
        label_mode=label_mode,
    )


def _monitor_once() -> Any:
    from sentinelml.monitoring.service import run_monitoring_once

    return run_monitoring_once()


def _read_ops(name: str) -> Any:
    if name == "monitoring":
        from sentinelml.serving.ops import ops_monitoring

        return ops_monitoring()
    return {}


def _retrain(name: str, args: Any) -> Any:
    from sentinelml.retraining.service import RetrainingService

    service = RetrainingService()
    if name == "status":
        return service.status()
    if name == "evaluate-trigger":
        return service.evaluate_latest_trigger(
            force_recheck=bool(getattr(args, "force_recheck", False))
        )
    if name == "once":
        return service.process_once()
    raise ValueError(name)


def _resilience(name: str, args: Any) -> Any:
    from sentinelml.resilience.service import ResilienceService

    service = ResilienceService()
    if name == "status":
        return service.status()
    if name == "evaluate-probation":
        return service.evaluate_probation()
    if name == "rollback":
        return service.rollback(to_version=args.version, reason="manual_cli")
    raise ValueError(name)


if __name__ == "__main__":
    raise SystemExit(main())
