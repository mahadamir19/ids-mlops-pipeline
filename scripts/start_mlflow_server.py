"""Start the SentinelML MLflow server with MinIO artifact storage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "infra" / "mlflow" / ".env"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def ensure_minio_bucket(*, bucket: str, endpoint_url: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchBucket"}:
            raise
        client.create_bucket(Bucket=bucket)


def main() -> None:
    env_values = load_env_file(ENV_PATH)
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
    bucket = env_values["MINIO_BUCKET"]
    temp_dir = PROJECT_ROOT / ".tmp" / "mlflow-server-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AWS_ACCESS_KEY_ID", env_values["MINIO_ROOT_USER"])
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", env_values["MINIO_ROOT_PASSWORD"])
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", endpoint_url)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)

    ensure_minio_bucket(bucket=bucket, endpoint_url=endpoint_url)

    backend_store_uri = (
        "postgresql+psycopg2://"
        f"{env_values['POSTGRES_USER']}:{env_values['POSTGRES_PASSWORD']}"
        f"@127.0.0.1:5433/{env_values['POSTGRES_DB']}"
    )
    host = os.environ.get("MLFLOW_HOST", "127.0.0.1")
    port = os.environ.get("MLFLOW_PORT", "5000")

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--host",
        host,
        "--port",
        port,
        "--workers",
        os.environ.get("MLFLOW_WORKERS", "1"),
        "--serve-artifacts",
        "--artifacts-destination",
        f"s3://{bucket}",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
