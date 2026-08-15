# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml pylock.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; lock=tomllib.load(open('pylock.toml','rb')); print('\n'.join(f\"{p['name']}=={p['version']}\" for p in lock.get('packages', []) if 'name' in p and 'version' in p))" > /tmp/constraints.txt \
    && python -c "import tomllib; project=tomllib.load(open('pyproject.toml','rb')); deps=project['project']['optional-dependencies']; print('\n'.join(project['build-system']['requires'] + deps['monitoring']))" > /tmp/requirements-monitor.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements-monitor.txt

COPY src ./src
COPY configs ./configs
COPY reports/data/feature_schema.json ./reports/data/feature_schema.json

RUN pip install --no-build-isolation --no-deps -e . \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/reports/monitoring /app/reports/resilience /app/data/reference \
    && chown -R appuser:appuser /app/reports /app/data

USER appuser

EXPOSE 9101

CMD ["uvicorn", "sentinelml.monitoring.app:app", "--host", "0.0.0.0", "--port", "9101"]
