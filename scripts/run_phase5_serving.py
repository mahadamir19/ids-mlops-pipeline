"""Operational helpers for SentinelML Phase 5 serving."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.serving.config import DEFAULT_SERVING_CONFIG_PATH, load_serving_config
from sentinelml.serving.database import PredictionDatabase
from sentinelml.serving.queue import DurableJsonlQueue
from sentinelml.serving.repository import PredictionRepository


def repository_from_config(config_path: Path) -> PredictionRepository:
    config = load_serving_config(config_path)
    return PredictionRepository(
        PredictionDatabase(
            config.database_url,
            connect_timeout_seconds=config.database_connect_timeout_seconds,
            statement_timeout_ms=config.database_statement_timeout_ms,
        ),
        DurableJsonlQueue(config.queue_path),
    )


def post_json(url: str, *, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelML Phase 5 serving ops")
    parser.add_argument("--config", type=Path, default=DEFAULT_SERVING_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")
    subparsers.add_parser("db-check")
    subparsers.add_parser("queue-depth")
    subparsers.add_parser("flush-queue")
    reload_parser = subparsers.add_parser("reload")
    reload_parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/internal/reload",
    )
    reload_parser.add_argument("--timeout", type=float, default=2)

    args = parser.parse_args()
    if args.command == "reload":
        print(json.dumps(post_json(args.url, timeout=args.timeout), indent=2))
        return

    repository = repository_from_config(args.config)
    if args.command == "init-db":
        repository.initialize()
        print(json.dumps({"initialized": True}, indent=2))
    elif args.command == "db-check":
        print(json.dumps({"database_logging": repository.database_status()}, indent=2))
    elif args.command == "queue-depth":
        print(
            json.dumps(
                {
                    "queue_depth": repository.queue.depth(),
                    "malformed_queue_records": repository.queue.malformed_count(),
                    "queue_path": str(repository.queue.path),
                },
                indent=2,
            )
        )
    elif args.command == "flush-queue":
        print(json.dumps(repository.flush_queue().__dict__, indent=2))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
