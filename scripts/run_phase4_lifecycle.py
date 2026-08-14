"""Run SentinelML Phase 4 model lifecycle operations."""

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

from sentinelml.lifecycle.config import (
    DEFAULT_LIFECYCLE_CONFIG_PATH,
    load_lifecycle_config,
)
from sentinelml.lifecycle.service import LifecycleService


def print_status(payload: dict[str, object]) -> None:
    print(f"Registered model: {payload['registered_model']}")
    champion = payload.get("champion")
    if champion:
        tags = champion["tags"]
        print(f"Champion: version {champion['version']}")
        print(f"Champion family: {tags.get('model_family')}")
        print(f"Execution mode: {tags.get('execution_mode')}")
        print(f"Demo model: {tags.get('demo_model')}")
    else:
        print("Champion: none")
    print("\nVersions:")
    for version in payload["versions"]:
        tags = version["tags"]
        state = tags.get("lifecycle_state", "unknown")
        suffix = ""
        if state == "rejected" and tags.get("lifecycle.failed_gates"):
            suffix = f" - {tags['lifecycle.failed_gates']}"
        marker = (
            " champion"
            if champion and version["version"] == champion["version"]
            else ""
        )
        print(f"v{version['version']} {state}{marker}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelML Phase 4 lifecycle CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_LIFECYCLE_CONFIG_PATH,
        help="Lifecycle configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")

    register = subparsers.add_parser("register")
    register.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    register.add_argument("--force-new-version", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--version", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--version", required=True)

    rap = subparsers.add_parser("register-and-promote")
    rap.add_argument("--mode", choices=["smoke", "full"], default="smoke")

    subparsers.add_parser("retry-pending")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--version", required=True)
    rollback.add_argument("--reason", default=None)
    rollback.add_argument("--allow-rejected", action="store_true")

    args = parser.parse_args()
    service = LifecycleService(config=load_lifecycle_config(args.config))

    if args.command == "status":
        print_status(service.status())
        return
    if args.command == "register":
        result = service.register_candidate(
            mode=args.mode,
            force_new_version=args.force_new_version,
        )
    elif args.command == "evaluate":
        result = service.evaluate_candidate(version=args.version)
    elif args.command == "promote":
        result = service.promote_or_reject(version=args.version)
    elif args.command == "register-and-promote":
        result = service.register_and_promote(mode=args.mode)
    elif args.command == "retry-pending":
        result = {"results": service.retry_pending()}
    elif args.command == "rollback":
        result = service.rollback(
            version=args.version,
            reason=args.reason,
            allow_rejected=args.allow_rejected,
        )
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
