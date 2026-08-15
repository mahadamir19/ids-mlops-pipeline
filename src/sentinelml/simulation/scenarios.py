"""Deterministic Phase 6 traffic scenario definitions and transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

ScenarioCategory = Literal[
    "healthy_baseline",
    "feature_drift",
    "class_prevalence_change",
    "custom",
]


@dataclass(frozen=True)
class FeatureTransform:
    feature: str
    additive_shift: float = 0.0
    multiplicative_scale: float = 1.0


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    category: ScenarioCategory
    schedule: Literal["none", "gradual", "sudden"]
    transforms: list[FeatureTransform] = field(default_factory=list)
    change_point_fraction: float = 0.5
    target_attack_fraction: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def affected_features(self) -> list[str]:
        return [transform.feature for transform in self.transforms]


def build_scenario(
    name: str,
    *,
    feature_columns: list[str],
    configured: dict[str, Any],
    custom: dict[str, Any] | None = None,
) -> ScenarioDefinition:
    if name == "custom":
        return build_custom_scenario(
            feature_columns=feature_columns,
            config=custom or configured.get("custom", {}),
        )
    if name == "normal":
        return ScenarioDefinition(
            name="normal",
            category="healthy_baseline",
            schedule="none",
            parameters=dict(configured.get("normal", {})),
        )
    if name == "attack_rate_spike":
        params = dict(configured.get(name, {}))
        target = float(params.get("target_attack_fraction", 0.75))
        validate_fraction(target, "target_attack_fraction")
        return ScenarioDefinition(
            name=name,
            category="class_prevalence_change",
            schedule="none",
            target_attack_fraction=target,
            parameters=params,
        )

    params = dict(configured.get(name, {}))
    if name == "gradual_drift":
        schedule = "gradual"
        category: ScenarioCategory = "feature_drift"
    elif name == "sudden_drift":
        schedule = "sudden"
        category = "feature_drift"
    elif name in predefined_scenario_names():
        params = predefined_scenarios()[name] | params
        schedule = str(params.get("schedule", "gradual"))
        category = "feature_drift"
    else:
        raise ValueError(f"unknown simulation scenario: {name}")

    transforms = parse_transforms(params, feature_columns=feature_columns)
    change_point = float(params.get("change_point_fraction", 0.5))
    validate_fraction(change_point, "change_point_fraction", allow_zero=False)
    if schedule not in {"gradual", "sudden"}:
        raise ValueError(f"unsupported scenario schedule: {schedule}")
    return ScenarioDefinition(
        name=name,
        category=category,
        schedule=schedule,
        transforms=transforms,
        change_point_fraction=change_point,
        parameters=params,
    )


def predefined_scenario_names() -> set[str]:
    return set(predefined_scenarios())


def predefined_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "packet_size_shift": {
            "schedule": "gradual",
            "affected_features": [
                "packet_length_mean",
                "average_packet_size",
                "max_packet_length",
            ],
            "additive_shift": 75.0,
            "multiplicative_scale": 1.2,
        },
        "flow_duration_shift": {
            "schedule": "sudden",
            "change_point_fraction": 0.5,
            "affected_features": ["flow_duration", "flow_iat_mean", "flow_iat_max"],
            "additive_shift": 5000.0,
            "multiplicative_scale": 1.1,
        },
        "multi_feature_shift": {
            "schedule": "gradual",
            "affected_features": [
                "flow_duration",
                "packet_length_mean",
                "total_fwd_packets",
            ],
            "additive_shift": 25.0,
            "multiplicative_scale": 1.15,
        },
    }


def build_custom_scenario(
    *,
    feature_columns: list[str],
    config: dict[str, Any],
) -> ScenarioDefinition:
    schedule = str(config.get("schedule", "gradual"))
    if schedule not in {"gradual", "sudden", "none"}:
        raise ValueError("custom schedule must be gradual, sudden, or none")
    target_attack_fraction = config.get("target_attack_fraction")
    if target_attack_fraction is not None:
        target_attack_fraction = float(target_attack_fraction)
        validate_fraction(target_attack_fraction, "target_attack_fraction")

    transforms = parse_transforms(config, feature_columns=feature_columns)
    if not transforms and target_attack_fraction is None:
        raise ValueError("custom scenario requires a transform or attack fraction")

    category: ScenarioCategory
    if transforms and target_attack_fraction is not None:
        category = "custom"
    elif transforms:
        category = "feature_drift"
    else:
        category = "class_prevalence_change"

    change_point = float(config.get("change_point_fraction", 0.5))
    validate_fraction(change_point, "change_point_fraction", allow_zero=False)
    return ScenarioDefinition(
        name="custom",
        category=category,
        schedule=schedule,
        transforms=transforms,
        change_point_fraction=change_point,
        target_attack_fraction=target_attack_fraction,
        parameters=dict(config),
    )


def parse_transforms(
    config: dict[str, Any],
    *,
    feature_columns: list[str],
) -> list[FeatureTransform]:
    features = list(config.get("affected_features", []))
    unknown = sorted(set(features) - set(feature_columns))
    if unknown:
        raise ValueError(f"unknown scenario features: {unknown}")
    additive = float(config.get("additive_shift", 0.0))
    scale = float(config.get("multiplicative_scale", 1.0))
    if not math.isfinite(additive) or not math.isfinite(scale):
        raise ValueError("feature transform parameters must be finite")
    if scale < 0:
        raise ValueError("multiplicative_scale must be non-negative")
    return [
        FeatureTransform(
            feature=feature,
            additive_shift=additive,
            multiplicative_scale=scale,
        )
        for feature in features
    ]


def apply_scenario(
    features: dict[str, float],
    *,
    definition: ScenarioDefinition,
    index: int,
    total: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if definition.schedule == "none" or not definition.transforms:
        return dict(features), {"progress": 0.0, "applied": False}

    progress = scenario_progress(
        schedule=definition.schedule,
        index=index,
        total=total,
        change_point_fraction=definition.change_point_fraction,
    )
    transformed = dict(features)
    changes: dict[str, dict[str, float]] = {}
    for transform in definition.transforms:
        before = float(transformed[transform.feature])
        scale_delta = 1.0 + ((transform.multiplicative_scale - 1.0) * progress)
        after = (before * scale_delta) + (transform.additive_shift * progress)
        if not math.isfinite(after):
            raise ValueError(
                f"scenario produced non-finite value for {transform.feature}"
            )
        transformed[transform.feature] = float(after)
        changes[transform.feature] = {
            "before": before,
            "after": float(after),
            "progress": progress,
        }
    return transformed, {
        "progress": progress,
        "applied": progress > 0,
        "changes": changes,
    }


def scenario_progress(
    *,
    schedule: str,
    index: int,
    total: int,
    change_point_fraction: float,
) -> float:
    if schedule == "gradual":
        return 0.0 if total <= 1 else index / (total - 1)
    if schedule == "sudden":
        change_index = max(0, min(total - 1, int(total * change_point_fraction)))
        return 1.0 if index >= change_index else 0.0
    return 0.0


def validate_fraction(
    value: float,
    name: str,
    *,
    allow_zero: bool = True,
) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite fraction in (0, 1]")
