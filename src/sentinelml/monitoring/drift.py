"""Evidently-backed drift execution plus compact PSI signal extraction."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sentinelml.monitoring.config import MonitoringConfig


@dataclass(frozen=True)
class DriftResult:
    drifting_features: list[str]
    feature_scores: dict[str, float]
    drifting_feature_count: int
    monitored_feature_count: int
    drift_share: float
    data_drift_detected: bool
    zero_variance_reference_features: list[str]
    zero_variance_current_features: list[str]
    evidently_summary: dict[str, Any]
    evidently_payload: dict[str, Any]
    html: str | None = None


class EvidentlyDriftRunner:
    """Run the pinned Evidently report API lazily so inference never imports it."""

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        config: MonitoringConfig,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            report = Report(
                [
                    DataDriftPreset(
                        drift_share=config.drift_share_threshold,
                        method=config.evidently_method,
                    )
                ],
                include_tests=True,
            )
            snapshot = _run_with_expected_numpy_warning_filter(
                lambda: report.run(current, reference)
            )
        except TypeError:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            _run_with_expected_numpy_warning_filter(
                lambda: report.run(current_data=current, reference_data=reference)
            )
            snapshot = report

        payload = _snapshot_to_dict(snapshot)
        html = None
        if config.evidently_include_html and hasattr(snapshot, "save_html"):
            html = ""
        return payload, html


def calculate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: MonitoringConfig,
    *,
    evidently_runner: EvidentlyDriftRunner | None = None,
) -> DriftResult:
    monitored = config.monitored_features
    evidently_runner = evidently_runner or EvidentlyDriftRunner()
    zero_variance_reference = zero_variance_features(reference, monitored)
    zero_variance_current = zero_variance_features(current, monitored)
    evidently_payload, html = evidently_runner.run(
        reference.loc[:, monitored],
        current.loc[:, monitored],
        config,
    )
    feature_scores = {
        feature: population_stability_index(reference[feature], current[feature])
        for feature in monitored
    }
    drifting_features = sorted(
        feature
        for feature, score in feature_scores.items()
        if score >= config.psi_threshold
    )
    monitored_count = len(monitored)
    drift_share = len(drifting_features) / monitored_count if monitored_count else 0.0
    data_drift_detected = drift_share >= config.drift_share_threshold
    evidently_summary = summarize_evidently_payload(evidently_payload)
    evidently_summary.update(
        {
            "api": "evidently.Report + evidently.presets.DataDriftPreset",
            "method": config.evidently_method,
            "compact_signal_source": (
                "PSI scores computed over the same reference/current frames"
            ),
            "zero_variance_reference_feature_count": len(zero_variance_reference),
            "zero_variance_current_feature_count": len(zero_variance_current),
        }
    )
    return DriftResult(
        drifting_features=drifting_features,
        feature_scores=feature_scores,
        drifting_feature_count=len(drifting_features),
        monitored_feature_count=monitored_count,
        drift_share=drift_share,
        data_drift_detected=data_drift_detected,
        zero_variance_reference_features=zero_variance_reference,
        zero_variance_current_features=zero_variance_current,
        evidently_summary=evidently_summary,
        evidently_payload=compact_evidently_payload(evidently_payload),
        html=html,
    )


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("inf")
    if np.nanmin(ref) == np.nanmax(ref):
        return 0.0 if np.nanmin(cur) == np.nanmax(cur) == np.nanmin(ref) else 1.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        edges = np.linspace(
            float(np.min(ref)),
            float(np.max(ref)),
            min(bins, len(ref)) + 1,
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    epsilon = 1e-6
    ref_pct = np.maximum(ref_counts / max(len(ref), 1), epsilon)
    cur_pct = np.maximum(cur_counts / max(len(cur), 1), epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def zero_variance_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    constant: list[str] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.nunique(dropna=True) <= 1:
            constant.append(feature)
    return constant


def _run_with_expected_numpy_warning_filter(action: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"numpy\.lib\._function_base_impl",
        )
        return action()


def summarize_evidently_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, default=str)
    return {
        "payload_available": bool(payload),
        "payload_bytes": len(encoded.encode("utf-8")),
        "top_level_keys": sorted(payload.keys())[:20],
    }


def compact_evidently_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, default=str)
    if len(encoded) <= 20000:
        return payload
    return {
        "truncated": True,
        "payload_bytes": len(encoded.encode("utf-8")),
        "top_level_keys": sorted(payload.keys())[:20],
    }


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    if hasattr(snapshot, "dict"):
        value = snapshot.dict()
        if isinstance(value, dict):
            return value
    if hasattr(snapshot, "json"):
        return json.loads(snapshot.json())
    return {"repr": repr(snapshot)}
