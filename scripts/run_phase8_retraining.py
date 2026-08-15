"""Run SentinelML Phase 8 continuous retraining."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.retraining.config import (
    DEFAULT_RETRAINING_CONFIG_PATH,
    load_retraining_config,
)
from sentinelml.retraining.service import RetrainingService


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelML Phase 8 retraining CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RETRAINING_CONFIG_PATH,
        help="Retraining configuration file.",
    )
    parser.add_argument("--status", action="store_true", help="Print retraining state.")
    parser.add_argument(
        "--evaluate-trigger",
        action="store_true",
        help="Evaluate latest Phase 7 report without training.",
    )
    parser.add_argument(
        "--force-recheck",
        action="store_true",
        help="Ignore processed-monitoring idempotency for evaluation only.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the latest unprocessed monitoring report once.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll for new monitoring reports forever.",
    )
    args = parser.parse_args()
    selected = [args.status, args.evaluate_trigger, args.once, args.watch]
    if sum(bool(value) for value in selected) != 1:
        parser.error(
            "choose exactly one of --status, --evaluate-trigger, --once, --watch"
        )

    service = RetrainingService(config=load_retraining_config(args.config))
    if args.status:
        result = service.status()
    elif args.evaluate_trigger:
        result = service.evaluate_latest_trigger(force_recheck=args.force_recheck)
    elif args.once:
        result = service.process_once()
    elif args.watch:
        service.watch()
        return
    else:
        raise AssertionError("unreachable")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
