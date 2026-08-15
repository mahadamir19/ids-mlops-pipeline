"""Run SentinelML Phase 6 HTTP traffic and delayed-label simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.production_data.repository import ProductionDataRepository
from sentinelml.serving.config import DEFAULT_SERVING_CONFIG_PATH, load_serving_config
from sentinelml.serving.database import PredictionDatabase
from sentinelml.simulation.config import DEFAULT_SIMULATION_CONFIG_PATH
from sentinelml.simulation.simulator import run_simulation_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SentinelML Phase 6 production traffic simulator"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument(
        "--scenario",
        choices=[
            "normal",
            "gradual_drift",
            "sudden_drift",
            "attack_rate_spike",
            "packet_size_shift",
            "flow_duration_shift",
            "multi_feature_shift",
            "custom",
        ],
        default="normal",
    )
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--label-mode",
        choices=["randomized", "batch", "none"],
        default="randomized",
    )
    parser.add_argument(
        "--custom-json",
        help="Custom scenario JSON object or path to a JSON file.",
    )
    parser.add_argument(
        "--inspect-approved",
        action="store_true",
        help="Print approved production observations from the app database.",
    )
    parser.add_argument("--approved-limit", type=int, default=10)
    parser.add_argument(
        "--serving-config",
        type=Path,
        default=DEFAULT_SERVING_CONFIG_PATH,
        help="Serving config used only with --inspect-approved.",
    )
    args = parser.parse_args()

    if args.inspect_approved:
        approved = inspect_approved(args.serving_config, args.approved_limit)
        print(json.dumps(approved, indent=2, default=str))
        return

    report = run_simulation_from_config(
        config_path=args.config,
        scenario_name=args.scenario,
        request_count=args.requests,
        seed=args.seed,
        label_mode=args.label_mode,
        custom=load_custom_json(args.custom_json),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def load_custom_json(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("--custom-json must resolve to a JSON object")
    return loaded


def inspect_approved(serving_config_path: Path, limit: int) -> dict[str, Any]:
    config = load_serving_config(serving_config_path)
    if not os.environ.get(config.database_url_env):
        raise RuntimeError(
            f"{config.database_url_env} must be set to inspect approved observations"
        )
    database = PredictionDatabase(
        config.database_url,
        connect_timeout_seconds=config.database_connect_timeout_seconds,
        statement_timeout_ms=config.database_statement_timeout_ms,
    )
    repository = ProductionDataRepository(database)
    rows = repository.get_approved_production_observations(limit=limit)
    return {"count": len(rows), "observations": rows}


if __name__ == "__main__":
    main()
