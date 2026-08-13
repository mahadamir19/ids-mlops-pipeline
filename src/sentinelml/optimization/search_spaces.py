"""Optuna search-space helpers for SentinelML model families."""

from __future__ import annotations

from typing import Any


def suggest_hyperparameters(
    trial: Any,
    model_family: str,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Sample model-family hyperparameters from config-defined bounds."""

    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "int":
            kwargs: dict[str, Any] = {}
            if "step" in spec:
                kwargs["step"] = int(spec["step"])
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                **kwargs,
            )
        elif kind == "float":
            kwargs = {}
            if "step" in spec and spec["step"] is not None:
                kwargs["step"] = float(spec["step"])
            if bool(spec.get("log", False)):
                kwargs["log"] = True
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                **kwargs,
            )
        else:
            raise ValueError(
                f"unsupported search-space type for {model_family}.{name}: {kind}"
            )
    return params


def search_space_summary(search_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a JSON-friendly copy of configured search-space bounds."""

    return {name: dict(spec) for name, spec in search_space.items()}

