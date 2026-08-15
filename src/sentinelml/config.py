"""Shared YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path, *, label: str = "config") -> dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return raw
