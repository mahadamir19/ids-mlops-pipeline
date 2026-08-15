# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV LOKY_MAX_CPU_COUNT=1

WORKDIR /app

COPY pyproject.toml pylock.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; lock=tomllib.load(open('pylock.toml','rb')); print('\n'.join(f\"{p['name']}=={p['version']}\" for p in lock.get('packages', []) if 'name' in p and 'version' in p))" > /tmp/constraints.txt \
    && python -c "import tomllib; project=tomllib.load(open('pyproject.toml','rb')); deps=project['project']['optional-dependencies']; print('\n'.join(project['build-system']['requires'] + deps['retrainer-runtime']))" > /tmp/requirements-retrainer.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements-retrainer.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY reports/data/feature_schema.json ./reports/data/feature_schema.json
COPY reports/models/phase2_smoke ./reports/models/phase2_smoke
COPY reports/lifecycle/threshold_derivation.json ./reports/lifecycle/threshold_derivation.json
COPY dvc.yaml dvc.lock pylock.toml ./
COPY data/raw.dvc ./data/raw.dvc

RUN pip install --no-build-isolation --no-deps -e . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.tmp /app/reports/retraining /app/reports/lifecycle /app/data/processed \
    && chown -R appuser:appuser /app/.tmp /app/reports /app/data

USER appuser

CMD ["python", "scripts/run_phase8_retraining.py", "--watch"]
