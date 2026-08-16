# Immutable Docker Stack

This runbook covers startup for an existing initialized SentinelML workspace. It is intentionally limited to the Docker stack and does not document the complete fresh-clone bootstrap procedure.

## Compose Files

`docker/docker-compose.yml` is the base, production-like stack. It builds application code into Docker images, so "immutable" means the application code used by the containers is baked into those images.

The base stack still uses host mounts and named volumes for runtime state, prepared data, reports, Prometheus configuration, and Grafana provisioning. Immutable mode does not mean there are no mounts.

`docker/docker-compose.dev.yml` is a development override. It bind-mounts local source and configuration into application containers so local edits become visible without rebuilding the corresponding image.

Base immutable mode uses:

```text
docker-compose.yml
```

Development mode uses:

```text
docker-compose.yml
docker-compose.dev.yml
```

## Compose Discovery

Compose file discovery is directory-sensitive. From the repository root, the Compose file lives under `docker/`, so use an explicit `-f` path. From inside `docker/`, plain `docker compose up -d` discovers `docker-compose.yml`.

`docker-compose.dev.yml` is not included automatically merely because it exists. Development mode requires explicitly specifying both Compose files.

## Prerequisites

Install Docker Desktop or Docker Engine with Docker Compose v2.

Create a local environment file:

```powershell
Copy-Item docker\.env.example docker\.env
```

Replace every `REPLACE_WITH_SECURE_PASSWORD` value in `docker\.env` with local passwords. Do not commit `docker\.env`.

An existing initialized workspace must have the runtime artifacts expected by the base stack:

- `data\reference\reference.parquet`, mounted into the monitor service.
- `data\processed\train.parquet`, mounted into the retrainer and resilience services.
- `data\processed\validation.parquet`, mounted into the retrainer and resilience services.
- `reports\data\feature_schema.json`, copied into application images during builds.
- `reports\models\phase2_smoke\`, copied into the retrainer/resilience image during builds.
- `reports\lifecycle\threshold_derivation.json`, copied into the retrainer/resilience image during builds.
- `data\raw.dvc`, `dvc.yaml`, `dvc.lock`, and `pylock.toml`, copied into the retrainer/resilience image during builds.
- Docker monitoring assets under `docker\prometheus\` and `docker\grafana\`, mounted into Prometheus and Grafana.
- Existing MLflow/PostgreSQL/MinIO registry state containing a champion model if the API is expected to serve predictions immediately.

A completely fresh clone can start infrastructure containers, but a fresh empty MLflow registry has no champion. In that case the serving service may remain unavailable or unhealthy until lifecycle bootstrap/model registration has occurred. The complete fresh-environment reproduction/bootstrap procedure will be documented separately.

Retraining defaults to disabled unless explicitly enabled:

```text
SENTINELML_RETRAINING_ENABLED=false
```

## Immutable Startup

From the repository root, start only the base immutable stack:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build
```

Check container state:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  ps
```

Source-code changes are not visible automatically in immutable mode. Rebuild the affected image before recreating the service.

## Service Endpoints

Default local endpoints from `docker/docker-compose.yml` are:

| Service | URL |
| --- | --- |
| SentinelML frontend | `http://localhost:8080` |
| Inference/API | `http://localhost:8000` |
| MLflow | `http://localhost:5000` |
| Monitoring service | `http://localhost:9101` |
| Resilience service | `http://localhost:9201` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |
| MinIO API | `http://localhost:9000` |
| MinIO console | `http://localhost:9001` |
| MLflow PostgreSQL | `localhost:5433` |
| Application PostgreSQL | `localhost:5434` |

The retrainer service does not expose an HTTP port.

## Health Verification

Use `ps` first:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  ps
```

Then verify HTTP services:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:9101/health
Invoke-RestMethod http://localhost:9201/health
```

Services may be warming or degraded while upstream dependencies initialize. The API can also remain unhealthy if no MLflow champion is available yet.

## Logs

Follow full stack logs:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f
```

Inspect targeted services:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f api

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f monitor

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f retrainer

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f resilience

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  logs -f mlflow-server
```

## Rebuilding After Code Changes

Immutable mode does not see host source edits automatically.

Full-stack safe rebuild:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build
```

Targeted rebuild examples:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build api

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build monitor

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build retrainer resilience

docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  up -d --build frontend
```

Do not delete persistent volumes as part of a normal rebuild.

## Stopping The Stack

Safe shutdown:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  down
```

`docker compose down` does not remove named volumes by default.

Avoid `docker compose down -v` unless you intentionally want a full reset. The `-v` flag deletes persistent Docker volumes and may destroy MLflow PostgreSQL metadata, MinIO artifacts, application PostgreSQL state, Prometheus data, and Grafana state.

## Persistent State

Important named volumes from the base Compose file:

| Volume | Preserves |
| --- | --- |
| `mlflow_mlflow_postgres_data` | MLflow PostgreSQL metadata, including experiments, runs, model registry records, and aliases. |
| `mlflow_mlflow_minio_data` | MinIO object storage for MLflow artifacts. |
| `sentinelml_app_postgres_data` | SentinelML application PostgreSQL state for serving, monitoring, retraining, and resilience workflows. |
| `sentinelml_prometheus_data` | Prometheus time-series data. |
| `sentinelml_grafana_data` | Grafana local state. |

`docker compose up --build` does not reset these volumes.

## Development Mode

Development mode is separate from immutable demo/final validation mode. It uses the base file plus the development override:

```powershell
docker compose --env-file docker\.env `
  -f docker\docker-compose.yml `
  -f docker\docker-compose.dev.yml `
  up -d --build
```

Use this only when you want local source/configuration edits bind-mounted into containers.
