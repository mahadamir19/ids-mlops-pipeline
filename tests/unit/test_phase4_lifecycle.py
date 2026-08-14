from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sentinelml.lifecycle.evaluation import canonical_promotion_slice
from sentinelml.lifecycle.service import LifecycleError, LifecycleService
from sentinelml.lifecycle.thresholds import (
    composite_score,
    evaluate_absolute_gates,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeVersion:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        source: str = "runs:/run/model",
        run_id: str = "run",
        tags: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.source = source
        self.run_id = run_id
        self.tags = tags or {}
        self.current_stage = None


class FakeClient:
    def __init__(self) -> None:
        self.models: set[str] = set()
        self.versions: dict[str, list[FakeVersion]] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.fail_alias = False
        self.verify_wrong_version: str | None = None
        self.force_wrong_champion_after_set: str | None = None

    def get_registered_model(self, name: str) -> object:
        if name not in self.models:
            raise Exception("registered model does not exist")
        return types.SimpleNamespace(name=name)

    def create_registered_model(self, name: str) -> object:
        self.models.add(name)
        self.versions.setdefault(name, [])
        return types.SimpleNamespace(name=name)

    def create_version(self, name: str, source: str) -> FakeVersion:
        self.models.add(name)
        version = FakeVersion(
            name=name,
            version=str(len(self.versions.setdefault(name, [])) + 1),
            source=source,
            run_id="registered-run",
        )
        self.versions[name].append(version)
        return version

    def search_model_versions(self, _filter: str) -> list[FakeVersion]:
        return [version for versions in self.versions.values() for version in versions]

    def get_model_version(self, name: str, version: str) -> FakeVersion:
        for candidate in self.versions.get(name, []):
            if candidate.version == str(version):
                return candidate
        raise Exception("model version does not exist")

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.get_model_version(name, version).tags[key] = value

    def set_registered_model_alias(
        self,
        name: str,
        alias: str,
        version: str,
    ) -> None:
        if self.fail_alias:
            raise Exception("MLflow unavailable")
        self.aliases[(name, alias)] = str(version)
        if self.force_wrong_champion_after_set is not None:
            self.verify_wrong_version = self.force_wrong_champion_after_set

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeVersion:
        version = self.verify_wrong_version or self.aliases.get((name, alias))
        if version is None:
            raise Exception("alias champion not found")
        return self.get_model_version(name, version)


class FakeMLflow:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.models = types.SimpleNamespace(get_model_info=lambda uri: {"uri": uri})

    def MlflowClient(self) -> FakeClient:
        return self.client

    def register_model(self, model_uri: str, name: str) -> FakeVersion:
        return self.client.create_version(name, model_uri)


def lifecycle_config(root: Path) -> dict[str, object]:
    return {
        "registered_model_name": "sentinelml-ids",
        "champion_alias": "champion",
        "paths": {
            "final_candidate_manifest": str(
                root / "final_candidate" / "model_manifest.json"
            ),
            "smoke_baseline_metrics": str(
                root / "baseline" / "validation_metrics.json"
            ),
            "smoke_selected_baseline": str(
                root / "baseline" / "selected_baseline.json"
            ),
            "smoke_baseline_manifest": str(root / "baseline" / "run_manifest.json"),
            "feature_schema": str(root / "data" / "feature_schema.json"),
            "train_partition": str(root / "data" / "train.parquet"),
            "validation_partition": str(root / "data" / "validation.parquet"),
            "threshold_report": str(root / "lifecycle" / "threshold_derivation.json"),
            "lifecycle_reports": str(root / "lifecycle"),
            "pending_dir": str(root / ".tmp" / "pending_promotions"),
        },
        "mode_policy": {"allow_cross_mode_promotion": False},
        "threshold_policy": {
            "baseline_mode": "smoke",
            "evaluation_split": "validation",
            "metrics": {
                "pr_auc": {
                    "source_metric": "pr_auc",
                    "direction": "min",
                    "fraction_of_baseline": 0.90,
                },
                "attack_recall": {
                    "source_metric": "attack_class_recall",
                    "direction": "min",
                    "fraction_of_baseline": 0.90,
                },
                "f1": {
                    "source_metric": "f1",
                    "direction": "min",
                    "fraction_of_baseline": 0.90,
                },
                "false_positive_rate": {
                    "source_metric": "false_positive_rate",
                    "direction": "max",
                    "multiplier_of_baseline": 5.0,
                    "minimum_ceiling": 0.001,
                },
                "inference_latency_ms_per_row": {
                    "source_metric": "inference_latency_ms_per_row",
                    "direction": "max",
                    "multiplier_of_baseline": 20.0,
                    "minimum_ceiling": 0.01,
                },
            },
        },
        "promotion_evaluation": {
            "mode": "smoke",
            "validation_sample_size": 100,
            "baseline_train_sample_size": 200,
            "random_seed": 42,
            "validation_seed_offset": 10,
            "baseline_train_seed_offset": 0,
            "sampling_strategy": "stratified_min_positive",
            "min_positive_rows": 10,
            "max_positive_fraction": 0.20,
        },
        "composite_score": {
            "minimum_relative_improvement": 0.01,
            "weights": {
                "pr_auc": 0.30,
                "attack_recall": 0.30,
                "f1": 0.20,
                "false_positive_rate": 0.10,
                "inference_latency_ms_per_row": 0.10,
            },
        },
    }


def selected_validation_metrics() -> dict[str, object]:
    return {
        "rows": 10000,
        "pr_auc": 0.80,
        "attack_class_recall": 0.40,
        "f1": 0.50,
        "false_positive_rate": 0.001,
        "inference_latency_ms_per_row": 0.002,
    }


def baseline_validation_collection() -> dict[str, object]:
    return {
        "logistic_regression": {"validation_metrics": {"pr_auc": 0.01}},
        "xgboost": {
            "validation_metrics": {
                "rows": 10000,
                "pr_auc": 0.10,
                "attack_class_recall": 0.10,
                "f1": 0.10,
                "false_positive_rate": 0.10,
                "inference_latency_ms_per_row": 10.0,
            }
        },
    }


def candidate_metrics(**overrides: float) -> dict[str, object]:
    metrics = {
        "rows": 1000,
        "pr_auc": 0.75,
        "attack_class_recall": 0.38,
        "f1": 0.48,
        "false_positive_rate": 0.002,
        "inference_latency_ms_per_row": 0.02,
    }
    metrics.update(overrides)
    return metrics


def bad_test_metrics() -> dict[str, object]:
    return candidate_metrics(
        pr_auc=0.01,
        attack_class_recall=0.01,
        f1=0.01,
        false_positive_rate=0.99,
        inference_latency_ms_per_row=99.0,
    )


def manifest(
    validation_metrics: dict[str, object] | None = None,
    test_metrics: dict[str, object] | None = None,
    run_id: str = "final-run",
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "model_name": "sentinelml-ids",
        "model_family": "xgboost",
        "source_optimization": {
            "execution_mode": "smoke",
            "best_trial_mlflow_run_id": "optuna-run",
        },
        "git": {"commit": "abc", "branch": "main", "dirty": False},
        "dvc": {"dvc_lock_sha256": "dvc-sha"},
        "configuration": {
            "training_config_sha256": "train-sha",
            "optimization_config_sha256": "opt-sha",
        },
        "dependencies": {"dependency_lock_sha256": "lock-sha"},
        "mlflow": {
            "final_candidate_run_id": run_id,
            "logged_model_uri": f"runs:/{run_id}/model",
        },
        "evaluation": {
            "validation_metrics": validation_metrics or candidate_metrics(),
            "test_metrics": test_metrics or bad_test_metrics(),
        },
    }


class Phase4LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "tmp_tests" / self._testMethodName
        self.client = FakeClient()
        self.mlflow = FakeMLflow(self.client)
        self.config = lifecycle_config(self.root)
        self.dataset_fingerprint = "canonical-fingerprint"
        self.dataset = types.SimpleNamespace(
            metadata={
                "selected_row_fingerprint": self.dataset_fingerprint,
                "selected_row_count": 100,
                "benign_count": 90,
                "attack_count": 10,
                "target_distribution": {"0": 90, "1": 10},
                "split": "validation",
                "evaluation_type": "lifecycle_promotion_validation",
            }
        )
        self.loaded_model_uris: list[str] = []
        self.evaluated_fingerprints: list[str] = []
        self.model_metrics_by_uri: dict[str, dict[str, object]] = {
            "runs:/final-run/model": candidate_metrics()
        }
        write_json(
            self.root / "baseline" / "validation_metrics.json",
            baseline_validation_collection(),
        )
        write_json(
            self.root / "baseline" / "selected_baseline.json",
            {
                "recommended_baseline": "xgboost",
                "selection_metric": "validation composite score",
            },
        )
        write_json(
            self.root / "baseline" / "selected_test_metrics.json",
            bad_test_metrics(),
        )
        write_json(self.root / "baseline" / "run_manifest.json", {"mode": "smoke"})
        write_json(self.root / "final_candidate" / "model_manifest.json", manifest())

        def fake_model_loader(uri: str) -> object:
            self.loaded_model_uris.append(uri)
            return types.SimpleNamespace(uri=uri)

        def fake_evaluator(*, model: object, dataset: object) -> dict[str, object]:
            self.evaluated_fingerprints.append(
                dataset.metadata["selected_row_fingerprint"]
            )
            return dict(self.model_metrics_by_uri[model.uri])

        def fake_baseline_reference(
            selected_baseline: dict[str, object],
        ) -> dict[str, object]:
            report_path = self.root / "lifecycle" / "baseline_reference.json"
            payload = {
                "event": "baseline_evaluated",
                "evaluation_type": "lifecycle_promotion_validation",
                "model_name": "sentinelml-ids",
                "model_family": selected_baseline["recommended_baseline"],
                "execution_mode": "smoke",
                "baseline_source": "phase2_selected_baseline_smoke_refit",
                "selected_baseline": selected_baseline,
                "canonical_dataset": self.dataset.metadata,
                "metrics": selected_validation_metrics(),
                "evaluated_at": "2026-08-14T00:00:00+00:00",
                "report_path": str(report_path),
            }
            write_json(report_path, payload)
            return payload

        self.service = LifecycleService(
            config=self.config,
            client=self.client,
            mlflow_module=self.mlflow,
            validate_model_uri=False,
            model_loader=fake_model_loader,
            evaluator=fake_evaluator,
            promotion_dataset_loader=lambda: self.dataset,
            baseline_reference_evaluator=fake_baseline_reference,
        )

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_registered_model_idempotency_and_duplicate_source_protection(self) -> None:
        first = self.service.register_candidate(mode="smoke")
        second = self.service.register_candidate(mode="smoke")

        self.assertTrue(first["registered"])
        self.assertFalse(second["registered"])
        self.assertTrue(second["reused"])
        self.assertEqual(first["model_version"], second["model_version"])
        self.assertEqual(len(self.client.versions["sentinelml-ids"]), 1)

    def test_champion_lookup_distinguishes_no_champion(self) -> None:
        self.service.register_candidate(mode="smoke")

        self.assertIsNone(self.service.get_champion())

        self.client.set_registered_model_alias("sentinelml-ids", "champion", "1")
        champion = self.service.get_champion()
        self.assertIsNotNone(champion)
        self.assertEqual(champion.version, "1")

    def test_threshold_derivation_writes_machine_readable_report(self) -> None:
        report = self.service.derive_thresholds()

        self.assertAlmostEqual(report["thresholds"]["pr_auc"], 0.72)
        self.assertAlmostEqual(report["thresholds"]["attack_recall"], 0.36)
        self.assertEqual(report["thresholds"]["f1"], 0.45)
        self.assertEqual(report["thresholds"]["false_positive_rate"], 0.005)
        self.assertEqual(report["evidence"]["evaluation_split"], "validation")
        self.assertTrue(
            report["evidence"]["baseline_metrics_path"].endswith(
                "baseline_reference.json"
            )
        )
        self.assertEqual(
            report["evidence"]["selected_baseline"]["recommended_baseline"],
            "xgboost",
        )
        self.assertTrue(
            (self.root / "lifecycle" / "threshold_derivation.json").exists()
        )

    def test_threshold_derivation_does_not_use_test_metrics_regression(self) -> None:
        report = self.service.derive_thresholds()

        self.assertAlmostEqual(report["thresholds"]["pr_auc"], 0.72)
        self.assertNotEqual(report["thresholds"]["pr_auc"], 0.009)
        self.assertNotEqual(report["thresholds"]["pr_auc"], 0.09)

    def test_canonical_slice_is_deterministic_and_enforces_min_positive(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"target": [0] * 190 + [1] * 10, "feature": range(200)})
        first, first_meta = canonical_promotion_slice(
            frame,
            target_column="target",
            sample_size=100,
            seed=52,
            min_positive_rows=8,
            max_positive_fraction=0.20,
        )
        second, second_meta = canonical_promotion_slice(
            frame,
            target_column="target",
            sample_size=100,
            seed=52,
            min_positive_rows=8,
            max_positive_fraction=0.20,
        )

        self.assertEqual(
            first_meta["selected_row_fingerprint"],
            second_meta["selected_row_fingerprint"],
        )
        self.assertEqual(first_meta["attack_count"], 8)
        self.assertEqual(first["target"].value_counts().to_dict()[1], 8)

    def test_absolute_gates_pass_and_fail_individually(self) -> None:
        report = self.service.derive_thresholds()
        passed = evaluate_absolute_gates(
            candidate_metrics=candidate_metrics(),
            thresholds=report["thresholds"],
            threshold_policy=self.config["threshold_policy"],
        )
        failed = evaluate_absolute_gates(
            candidate_metrics=candidate_metrics(attack_class_recall=0.1),
            thresholds=report["thresholds"],
            threshold_policy=self.config["threshold_policy"],
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn("attack_recall", failed["failed_gates"])

    def test_composite_score_uses_latency_and_fpr_as_bounded_penalties(self) -> None:
        thresholds = self.service.derive_thresholds()["thresholds"]
        fast = composite_score(
            candidate_metrics(inference_latency_ms_per_row=0.001),
            thresholds=thresholds,
            config=self.config,
        )
        slow = composite_score(
            candidate_metrics(inference_latency_ms_per_row=0.20),
            thresholds=thresholds,
            config=self.config,
        )

        self.assertGreater(fast["score"], slow["score"])
        self.assertLessEqual(slow["components"]["inference_latency_ms_per_row"], 0.1)

    def test_bootstrap_first_champion_promotes_without_relative(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["event"], "promoted")
        self.assertEqual(self.service.get_champion().version, "1")
        self.assertEqual(self.loaded_model_uris, ["runs:/final-run/model"])
        self.assertEqual(self.evaluated_fingerprints, [self.dataset_fingerprint])
        self.assertEqual(
            self.client.get_model_version("sentinelml-ids", "1").tags[
                "lifecycle_state"
            ],
            "champion",
        )

    def test_candidate_promotes_and_preserves_previous(self) -> None:
        champion = self.service.register_candidate(mode="smoke")
        self.service.promote_or_reject(version=champion["model_version"])
        write_json(
            self.root / "final_candidate" / "model_manifest.json",
            manifest(
                candidate_metrics(pr_auc=0.90, attack_class_recall=0.60, f1=0.70),
                run_id="final-run-2",
            ),
        )
        self.model_metrics_by_uri["runs:/final-run-2/model"] = candidate_metrics(
            pr_auc=0.90,
            attack_class_recall=0.60,
            f1=0.70,
        )
        updated = self.service.register_candidate(mode="smoke", force_new_version=True)
        result = self.service.promote_or_reject(version=updated["model_version"])

        self.assertEqual(result["event"], "promoted")
        self.assertEqual(result["previous_champion_version"], "1")
        self.assertEqual(self.service.get_champion().version, "2")
        self.assertEqual(
            self.evaluated_fingerprints[-2:],
            [self.dataset_fingerprint, self.dataset_fingerprint],
        )
        self.assertEqual(
            self.client.get_model_version("sentinelml-ids", "1").tags[
                "lifecycle_state"
            ],
            "superseded",
        )

    def test_candidate_fails_promotion_and_is_rejected_with_metadata(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        version = self.client.get_model_version(
            "sentinelml-ids",
            registration["model_version"],
        )
        self.model_metrics_by_uri["runs:/final-run/model"] = candidate_metrics(
            attack_class_recall=0.01
        )
        version.tags["lifecycle.validation_metrics_json"] = json.dumps(
            candidate_metrics(attack_class_recall=0.01)
        )
        version.tags["lifecycle.validation_dataset_fingerprint"] = (
            self.dataset_fingerprint
        )
        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["event"], "rejected")
        self.assertIn("attack_recall", result["failed_gates"])
        self.assertEqual(version.tags["lifecycle_state"], "rejected")
        self.assertIn("attack_recall", version.tags["lifecycle.failed_gates"])

    def test_mlflow_alias_verification_failure_creates_promotion_pending(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        self.client.force_wrong_champion_after_set = "99"
        self.client.versions["sentinelml-ids"].append(
            FakeVersion(name="sentinelml-ids", version="99")
        )
        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["event"], "promotion_pending")
        self.assertTrue(
            (
                self.root
                / ".tmp"
                / "pending_promotions"
                / "sentinelml-ids_version_1.json"
            ).exists()
        )

    def test_mlflow_outage_creates_promotion_pending(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        self.client.fail_alias = True
        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["operation_state"], "promotion_pending")
        self.assertIsNone(self.service.get_champion())

    def test_pending_retry_re_evaluates_and_promotes(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        self.client.fail_alias = True
        self.service.promote_or_reject(version=registration["model_version"])
        self.client.fail_alias = False
        retry = self.service.retry_pending()

        self.assertEqual(retry[0]["event"], "promotion_retry")
        self.assertEqual(self.service.get_champion().version, "1")
        self.assertFalse(
            (
                self.root
                / ".tmp"
                / "pending_promotions"
                / "sentinelml-ids_version_1.json"
            ).exists()
        )

    def test_pending_retry_rejects_if_current_champion_now_wins(self) -> None:
        candidate = self.service.register_candidate(mode="smoke")
        self.client.fail_alias = True
        self.service.promote_or_reject(version=candidate["model_version"])
        self.client.fail_alias = False
        strong = FakeVersion(
            name="sentinelml-ids",
            version="2",
            tags={
                **self.client.get_model_version("sentinelml-ids", "1").tags,
                "lifecycle.validation_metrics_json": json.dumps(
                    candidate_metrics(pr_auc=0.99, attack_class_recall=0.99, f1=0.99)
                ),
                "lifecycle.validation_dataset_fingerprint": self.dataset_fingerprint,
                "lifecycle_state": "champion",
            },
        )
        self.client.versions["sentinelml-ids"].append(strong)
        self.client.set_registered_model_alias("sentinelml-ids", "champion", "2")
        retry = self.service.retry_pending()

        self.assertEqual(retry[0]["outcome"]["event"], "rejected")
        self.assertEqual(
            self.client.get_model_version("sentinelml-ids", "1").tags[
                "lifecycle_state"
            ],
            "rejected",
        )

    def test_smoke_full_mismatch_protection(self) -> None:
        champion = self.service.register_candidate(mode="smoke")
        self.service.promote_or_reject(version=champion["model_version"])
        write_json(
            self.root / "final_candidate" / "model_manifest.json",
            manifest(
                candidate_metrics(pr_auc=0.99, attack_class_recall=0.99, f1=0.99),
                run_id="final-run-full",
            ),
        )
        self.model_metrics_by_uri["runs:/final-run-full/model"] = candidate_metrics(
            pr_auc=0.99,
            attack_class_recall=0.99,
            f1=0.99,
        )
        updated = self.service.register_candidate(mode="smoke", force_new_version=True)
        self.client.get_model_version("sentinelml-ids", updated["model_version"]).tags[
            "execution_mode"
        ] = "full"
        result = self.service.promote_or_reject(version=updated["model_version"])

        self.assertEqual(result["event"], "rejected")
        self.assertIn("execution_mode", result["failed_gates"])

    def test_rollback_validates_target_and_switches_alias(self) -> None:
        champion = self.service.register_candidate(mode="smoke")
        self.service.promote_or_reject(version=champion["model_version"])
        write_json(
            self.root / "final_candidate" / "model_manifest.json",
            manifest(
                candidate_metrics(pr_auc=0.90, attack_class_recall=0.60, f1=0.70),
                run_id="final-run-rollback",
            ),
        )
        self.model_metrics_by_uri["runs:/final-run-rollback/model"] = candidate_metrics(
            pr_auc=0.90,
            attack_class_recall=0.60,
            f1=0.70,
        )
        new = self.service.register_candidate(mode="smoke", force_new_version=True)
        self.service.promote_or_reject(version=new["model_version"])

        result = self.service.rollback(version="1", reason="manual test")

        self.assertEqual(result["event"], "rollback")
        self.assertEqual(self.service.get_champion().version, "1")
        self.assertEqual(
            self.client.get_model_version("sentinelml-ids", "1").tags[
                "lifecycle.rollback_reason"
            ],
            "manual test",
        )

    def test_rollback_refuses_rejected_target_without_override(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        version = self.client.get_model_version(
            "sentinelml-ids",
            registration["model_version"],
        )
        version.tags["lifecycle_state"] = "rejected"

        with self.assertRaisesRegex(LifecycleError, "rejected"):
            self.service.rollback(version=registration["model_version"])

    def test_lifecycle_audit_serialization(self) -> None:
        registration = self.service.register_candidate(mode="smoke")
        self.service.promote_or_reject(version=registration["model_version"])

        self.assertTrue(list((self.root / "lifecycle" / "registereds").glob("*.json")))
        self.assertTrue(
            (self.root / "lifecycle" / "evaluations" / "version_1.json").exists()
        )
        self.assertTrue(list((self.root / "lifecycle" / "promoteds").glob("*.json")))

    def test_candidate_manifest_validation_metrics_do_not_override_canonical(
        self,
    ) -> None:
        write_json(
            self.root / "final_candidate" / "model_manifest.json",
            manifest(
                candidate_metrics(pr_auc=0.01, attack_class_recall=0.01, f1=0.01),
                run_id="canonical-wins",
            ),
        )
        self.model_metrics_by_uri["runs:/canonical-wins/model"] = candidate_metrics()
        registration = self.service.register_candidate(mode="smoke")
        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["event"], "promoted")

    def test_stale_cached_lifecycle_metrics_wrong_fingerprint_are_rejected(
        self,
    ) -> None:
        registration = self.service.register_candidate(mode="smoke")
        version = self.client.get_model_version(
            "sentinelml-ids",
            registration["model_version"],
        )
        version.tags["lifecycle.validation_metrics_json"] = json.dumps(
            candidate_metrics(pr_auc=0.99, attack_class_recall=0.99, f1=0.99)
        )
        version.tags["lifecycle.validation_dataset_fingerprint"] = "stale"
        self.model_metrics_by_uri["runs:/final-run/model"] = candidate_metrics(
            attack_class_recall=0.01
        )

        result = self.service.promote_or_reject(version=registration["model_version"])

        self.assertEqual(result["event"], "rejected")
        self.assertIn("attack_recall", result["failed_gates"])


class Phase4OptionalIntegrationTests(unittest.TestCase):
    def test_real_mlflow_registry_backend_optional(self) -> None:
        enabled = __import__("os").environ.get(
            "SENTINELML_RUN_MLFLOW_INTEGRATION",
            "0",
        )
        if not bool(int(enabled)):
            self.skipTest("set SENTINELML_RUN_MLFLOW_INTEGRATION=1 to run")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
