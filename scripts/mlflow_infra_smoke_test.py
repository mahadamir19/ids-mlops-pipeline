# ruff: noqa: I001
import sys
from pathlib import Path

import mlflow


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelml.training.baselines import configure_mlflow_runtime_environment  # noqa: E402


EXPERIMENT_NAME = "sentinelml-infra-smoke-test"


tracking_uri = configure_mlflow_runtime_environment()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)
print("Tracking URI:", mlflow.get_tracking_uri())

artifact_path = Path("mlflow-smoke-artifact.txt")
artifact_path.write_text(
    "SentinelML MLflow infrastructure smoke test.",
    encoding="utf-8",
)

with mlflow.start_run(run_name="infra-smoke-run") as run:
    mlflow.log_param("model_family", "smoke_test")
    mlflow.log_param("random_seed", 42)

    mlflow.log_metric("dummy_accuracy", 0.95)

    mlflow.set_tag("purpose", "infrastructure_validation")
    mlflow.set_tag("project", "SentinelML")

    mlflow.log_artifact(str(artifact_path))

    print("Run ID:", run.info.run_id)
    print("Experiment ID:", run.info.experiment_id)
    print("Artifact URI:", run.info.artifact_uri)

artifact_path.unlink()
