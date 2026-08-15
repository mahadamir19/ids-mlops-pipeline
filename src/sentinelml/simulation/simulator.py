"""Phase 6 simulator orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sentinelml.serving.validation import load_serving_feature_schema
from sentinelml.simulation.client import ApiError, SimulationHttpClient
from sentinelml.simulation.config import SimulationConfig, load_simulation_config
from sentinelml.simulation.data import deterministic_sample, load_simulation_frame
from sentinelml.simulation.labels import (
    BatchLabeler,
    NoopLabeler,
    RandomizedDelayLabeler,
)
from sentinelml.simulation.reports import write_simulation_report
from sentinelml.simulation.scenarios import (
    ScenarioDefinition,
    apply_scenario,
    build_scenario,
)


class TrafficSimulator:
    def __init__(
        self,
        config: SimulationConfig,
        *,
        client: SimulationHttpClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or SimulationHttpClient(
            config.api_base_url,
            timeout_seconds=config.timeout_seconds,
        )

    def run(
        self,
        *,
        scenario_name: str,
        request_count: int | None = None,
        seed: int | None = None,
        label_mode: str = "randomized",
        custom: dict[str, Any] | None = None,
        write_report: bool = True,
        wait_for_labels: bool = True,
    ) -> dict[str, Any]:
        seed = self.config.default_seed if seed is None else seed
        request_count = (
            self.config.default_request_count
            if request_count is None
            else request_count
        )
        schema = load_serving_feature_schema(self.config.feature_schema_path)
        scenario = build_scenario(
            scenario_name,
            feature_columns=schema.feature_columns,
            configured=self.config.scenarios,
            custom=custom or self.config.custom,
        )
        frame, source = load_simulation_frame(
            self.config.source_data_path,
            feature_columns=schema.feature_columns,
            target_column=schema.target_column,
        )
        records = deterministic_sample(
            frame,
            feature_columns=schema.feature_columns,
            target_column=schema.target_column,
            request_count=request_count,
            seed=seed,
            attack_fraction=scenario.target_attack_fraction,
        )
        labeler = self._labeler(label_mode=label_mode, seed=seed)
        run_id = str(uuid4())
        started_at = datetime.now(UTC).isoformat()
        predictions: Counter[str] = Counter()
        true_labels: Counter[str] = Counter()
        model_versions: Counter[str] = Counter()
        errors: list[dict[str, Any]] = []
        selected_fingerprints = [record.row_fingerprint for record in records]
        transformed_features: set[str] = set()
        succeeded = 0

        for index, record in enumerate(records):
            true_labels[str(record.true_label)] += 1
            try:
                features, transform_metadata = apply_scenario(
                    record.features,
                    definition=scenario,
                    index=index,
                    total=len(records),
                )
                transformed_features.update(
                    transform_metadata.get("changes", {}).keys()
                )
                response = self.client.predict(features)
            except (ApiError, ValueError) as exc:
                errors.append({"index": index, "error": str(exc)})
                continue

            succeeded += 1
            predictions[str(response["prediction"])] += 1
            model_versions[str(response["model_version"])] += 1
            labeler.schedule(
                prediction_id=str(response["prediction_id"]),
                ground_truth=record.true_label,
                now=time.monotonic(),
            )
            labeler.deliver_due(now=time.monotonic())
            if self.config.request_interval_seconds > 0:
                time.sleep(self.config.request_interval_seconds)

        labeler.drain(wait=wait_for_labels)
        finished_at = datetime.now(UTC).isoformat()
        report = self._report(
            run_id=run_id,
            scenario=scenario,
            seed=seed,
            source={**source.__dict__, "path": str(source.path)},
            request_count=request_count,
            selected_fingerprints=selected_fingerprints,
            succeeded=succeeded,
            errors=errors,
            predictions=predictions,
            true_labels=true_labels,
            model_versions=model_versions,
            transformed_features=sorted(transformed_features),
            label_mode=label_mode,
            delivered_count=len(labeler.delivered),
            undelivered=[label.__dict__ for label in labeler.pending],
            failed_labels=[label.__dict__ for label in labeler.failed],
            started_at=started_at,
            finished_at=finished_at,
        )
        if write_report:
            report_path = write_simulation_report(report, self.config.reports_dir)
            report["report_path"] = str(report_path)
        return report

    def _labeler(
        self,
        *,
        label_mode: str,
        seed: int,
    ) -> RandomizedDelayLabeler | BatchLabeler | NoopLabeler:
        retry = self.config.retry
        max_attempts = int(retry.get("max_attempts", 3))
        retry_delay = float(retry.get("retry_delay_seconds", 1))
        if label_mode == "randomized":
            delay = self.config.randomized_delay
            return RandomizedDelayLabeler(
                self.client,
                seed=seed + 1009,
                min_delay_seconds=float(delay.get("min_delay_seconds", 0.1)),
                max_delay_seconds=float(delay.get("max_delay_seconds", 1.0)),
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay,
            )
        if label_mode == "batch":
            batch = self.config.batch_delivery
            return BatchLabeler(
                self.client,
                batch_size=int(batch.get("batch_size", 10)),
                max_attempts=max_attempts,
                retry_delay_seconds=retry_delay,
            )
        if label_mode == "none":
            return NoopLabeler()
        raise ValueError("label_mode must be randomized, batch, or none")

    def _report(
        self,
        *,
        run_id: str,
        scenario: ScenarioDefinition,
        seed: int,
        source: dict[str, Any],
        request_count: int,
        selected_fingerprints: list[str],
        succeeded: int,
        errors: list[dict[str, Any]],
        predictions: Counter[str],
        true_labels: Counter[str],
        model_versions: Counter[str],
        transformed_features: list[str],
        label_mode: str,
        delivered_count: int,
        undelivered: list[dict[str, Any]],
        failed_labels: list[dict[str, Any]],
        started_at: str,
        finished_at: str,
    ) -> dict[str, Any]:
        return {
            "simulation_run_id": run_id,
            "simulation_fingerprint": simulation_fingerprint(
                scenario=scenario,
                seed=seed,
                source_fingerprint=str(source["fingerprint"]),
                request_count=request_count,
                selected_fingerprints=selected_fingerprints,
            ),
            "scenario": scenario.name,
            "scenario_category": scenario.category,
            "seed": seed,
            "source": source,
            "request_count_attempted": request_count,
            "requests_succeeded": succeeded,
            "requests_failed": len(errors),
            "prediction_class_distribution": dict(predictions),
            "true_class_distribution": dict(true_labels),
            "model_versions_observed": dict(model_versions),
            "affected_features": transformed_features or scenario.affected_features,
            "scenario_parameters": scenario.parameters,
            "label_delivery_mode": label_mode,
            "labels_delivered": delivered_count,
            "labels_pending": len(undelivered),
            "labels_failed": len(failed_labels),
            "undelivered_labels": undelivered,
            "failed_labels": failed_labels,
            "started_at": started_at,
            "finished_at": finished_at,
            "errors_summary": errors[:20],
        }


def simulation_fingerprint(
    *,
    scenario: ScenarioDefinition,
    seed: int,
    source_fingerprint: str,
    request_count: int,
    selected_fingerprints: list[str],
) -> str:
    payload = {
        "scenario": scenario.name,
        "scenario_category": scenario.category,
        "seed": seed,
        "source_fingerprint": source_fingerprint,
        "request_count": request_count,
        "selected_fingerprints": selected_fingerprints,
        "scenario_parameters": scenario.parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_simulation_from_config(
    *,
    config_path: Path,
    scenario_name: str,
    request_count: int | None,
    seed: int | None,
    label_mode: str,
    custom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_simulation_config(config_path)
    simulator = TrafficSimulator(config)
    return simulator.run(
        scenario_name=scenario_name,
        request_count=request_count,
        seed=seed,
        label_mode=label_mode,
        custom=custom,
    )
