"""Run SentinelML Phase 9 rollback and resilience controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.resilience.config import (  # noqa: E402
    DEFAULT_RESILIENCE_CONFIG_PATH,
    load_resilience_config,
)
from sentinelml.resilience.metrics import resilience_metrics  # noqa: E402
from sentinelml.resilience.service import ResilienceService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SentinelML Phase 9 rollback and resilience controller."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RESILIENCE_CONFIG_PATH)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("status")
    subcommands.add_parser("evaluate-probation")
    subcommands.add_parser("retry-pending")
    subcommands.add_parser("metrics")
    subcommands.add_parser("watch")
    rollback = subcommands.add_parser("rollback")
    rollback.add_argument("--to-version", default=None)
    rollback.add_argument(
        "--reason",
        default="manual operator rollback",
        help="Audit reason recorded with the rollback event.",
    )
    args = parser.parse_args()

    service = ResilienceService(config=load_resilience_config(args.config))
    if args.command == "status":
        print(json.dumps(service.status(), indent=2, sort_keys=True, default=str))
    elif args.command == "evaluate-probation":
        print(
            json.dumps(
                service.evaluate_probation(),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    elif args.command == "retry-pending":
        print(
            json.dumps(
                service.retry_pending(),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    elif args.command == "rollback":
        print(
            json.dumps(
                service.rollback(to_version=args.to_version, reason=args.reason),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    elif args.command == "metrics":
        service.status()
        print(resilience_metrics.render())
    elif args.command == "watch":
        service.watch()


if __name__ == "__main__":
    main()
