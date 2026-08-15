# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml pylock.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; project=tomllib.load(open('pyproject.toml','rb')); deps=project['project']['optional-dependencies']; print('\n'.join(project['build-system']['requires'] + deps['serving'] + deps['tracking'] + deps['ml']))" > /tmp/requirements-api.txt \
    && pip install -r /tmp/requirements-api.txt

COPY src ./src
COPY configs ./configs
COPY reports/data/feature_schema.json ./reports/data/feature_schema.json

RUN pip install --no-build-isolation --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "sentinelml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
