"""Candidate evaluation worker for the LifecycleService facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sentinelml.lifecycle.audit import utc_now, write_json
from sentinelml.lifecycle.registry import ModelVersionInfo, set_version_tags
from sentinelml.lifecycle.thresholds import composite_score, evaluate_absolute_gates


class CandidateEvaluator:
    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        config: dict[str, Any],
        report_root: Path,
        get_version: Callable[[str | int], ModelVersionInfo],
        derive_thresholds: Callable[[], dict[str, Any]],
        metrics_for_version: Callable[[ModelVersionInfo], dict[str, Any]],
        get_champion: Callable[[], ModelVersionInfo | None],
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.config = config
        self.report_root = report_root
        self.get_version = get_version
        self.derive_thresholds = derive_thresholds
        self.metrics_for_version = metrics_for_version
        self.get_champion = get_champion

    def evaluate_candidate(self, *, version: str | int) -> dict[str, Any]:
        candidate = self.get_version(version)
        thresholds = self.derive_thresholds()
        candidate_metrics = self.metrics_for_version(candidate)
        absolute = evaluate_absolute_gates(
            candidate_metrics=candidate_metrics,
            thresholds=thresholds["thresholds"],
            threshold_policy=self.config["threshold_policy"],
        )
        champion = self.get_champion()
        candidate_score = composite_score(
            candidate_metrics,
            thresholds=thresholds["thresholds"],
            config=self.config,
        )
        comparison: dict[str, Any] = {
            "has_champion": champion is not None,
            "candidate_composite_score": candidate_score,
        }
        relative_passed = True
        if champion is not None and champion.version != candidate.version:
            if not self.modes_compatible(candidate, champion):
                relative_passed = False
                comparison["mode_mismatch"] = {
                    "candidate": candidate.tags.get("execution_mode"),
                    "champion": champion.tags.get("execution_mode"),
                }
            else:
                champion_metrics = self.metrics_for_version(champion)
                champion_score = composite_score(
                    champion_metrics,
                    thresholds=thresholds["thresholds"],
                    config=self.config,
                )
                min_improvement = float(
                    self.config["composite_score"]["minimum_relative_improvement"]
                )
                delta = candidate_score["score"] - champion_score["score"]
                relative_passed = delta >= min_improvement
                comparison.update(
                    {
                        "champion_version": champion.version,
                        "champion_composite_score": champion_score,
                        "minimum_relative_improvement": min_improvement,
                        "score_delta": delta,
                        "relative_passed": relative_passed,
                    }
                )
        elif champion is not None:
            comparison["already_champion"] = True

        result = {
            "event": "evaluated",
            "model_name": self.model_name,
            "candidate_version": candidate.version,
            "passed": bool(absolute["passed"] and relative_passed),
            "absolute_gates": absolute,
            "comparison": comparison,
            "threshold_report": thresholds,
            "evaluated_at": utc_now(),
        }
        set_version_tags(
            self.client,
            model_name=self.model_name,
            version=candidate.version,
            tags={
                "lifecycle.last_evaluated_at": result["evaluated_at"],
                "lifecycle.last_evaluation_passed": str(result["passed"]).lower(),
                "lifecycle.composite_score": candidate_score["score"],
            },
        )
        evaluation_path = (
            self.report_root / "evaluations" / f"version_{candidate.version}.json"
        )
        write_json(evaluation_path, result)
        return result

    def modes_compatible(
        self,
        candidate: ModelVersionInfo,
        champion: ModelVersionInfo,
    ) -> bool:
        if bool(self.config.get("mode_policy", {}).get("allow_cross_mode_promotion")):
            return True
        return candidate.tags.get("execution_mode") == champion.tags.get(
            "execution_mode"
        )
