"""Run SentinelML Phase 7 monitoring once or in a safe watch loop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.monitoring.config import DEFAULT_MONITORING_CONFIG_PATH
from sentinelml.monitoring.service import run_monitoring_once


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelML Phase 7 monitoring")
    parser.add_argument("--config", type=Path, default=DEFAULT_MONITORING_CONFIG_PATH)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=None)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Generate a new monitoring report even if the input fingerprint matches.",
    )
    args = parser.parse_args()

    if args.status:
        latest = (
            args.config.parents[0] / ".." / "reports" / "monitoring" / "latest.json"
        )
        latest = latest.resolve()
        if not latest.exists():
            print(json.dumps({"status": "missing", "path": str(latest)}, indent=2))
            return
        print(latest.read_text(encoding="utf-8"))
        return

    if args.watch:
        interval = float(args.interval_seconds or 60)
        while True:
            print(json.dumps(_run(args), indent=2, sort_keys=True))
            time.sleep(interval)

    print(json.dumps(_run(args), indent=2, sort_keys=True))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    return run_monitoring_once(
        config_path=args.config,
        window_size=args.window_size,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
