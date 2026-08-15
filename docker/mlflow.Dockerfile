# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml pylock.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; lock=tomllib.load(open('pylock.toml','rb')); print('\n'.join(f\"{p['name']}=={p['version']}\" for p in lock.get('packages', []) if 'name' in p and 'version' in p))" > /tmp/constraints.txt \
    && python -c "import tomllib; project=tomllib.load(open('pyproject.toml','rb')); deps=project['project']['optional-dependencies']; print('\n'.join(deps['mlflow-runtime']))" > /tmp/requirements-mlflow.txt \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements-mlflow.txt

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 5000
