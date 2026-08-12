"""Configuration helpers for Phase 1 data foundation work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_data_config.json"
DEFAULT_LABEL_MAPPING_PATH = PROJECT_ROOT / "configs" / "label_mapping.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_phase1_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_json(path)
    config["_config_path"] = str(path)
    return config


def load_label_mapping(path: Path = DEFAULT_LABEL_MAPPING_PATH) -> dict[str, Any]:
    mapping = load_json(path)
    mapping["_mapping_path"] = str(path)
    return mapping


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
