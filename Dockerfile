# Image of ml-service (T-601). Build from the repository root: docker build -t inspecto/ml-service .
# The model is not in the image: mount the directory with model.onnx and model.json at
# /models/current (ML_MODEL_DIR). Lint and tests run in CI, not here.
FROM python:3.15.0-slim AS build
RUN pip install --no-cache-dir uv==0.13.7
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/app/.venv
COPY service/pyproject.toml service/uv.lock service/.python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY service/src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.15.0-slim
# Security fixes of Debian packages released after the base image (Trivy blocks High/Critical).
RUN apt-get update && apt-get -y --no-install-recommends upgrade && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ML_MODEL_DIR=/models/current
USER 10001:10001
EXPOSE 8084
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8084/health/live', timeout=2)"]
ENTRYPOINT ["python", "-m", "ml_service"]
