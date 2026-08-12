from pathlib import Path

import mlflow


EXPERIMENT_NAME = "sentinelml-infra-smoke-test"


mlflow.set_experiment(EXPERIMENT_NAME)

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