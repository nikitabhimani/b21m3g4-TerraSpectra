# syntax=docker/dockerfile:1.7
# TerraSpectra P2 training image. Build context: repository root.
#   docker build -f model/Dockerfile -t terraspectra-model .
#   docker run --gpus all -v "$PWD/data:/data" -v "$PWD/models:/models" terraspectra-model \
#       train -c configs/default.yaml -s data.data_dir=/data/raw
# The default PyPI torch wheel bundles the CUDA runtime, so this slim image trains on GPU
# hosts via the NVIDIA container toolkit and falls back to CPU elsewhere.
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/model/.venv

WORKDIR /app/model

# 1) Dependencies only (cached unless the lockfile or contracts change).
COPY contracts/python /app/contracts/python
COPY model/pyproject.toml model/uv.lock model/README.md /app/model/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Project source.
COPY model/ /app/model/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
 && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data /models /app/model/runs \
 && chown -R app:app /data /models /app/model/runs

ENV PATH="/app/model/.venv/bin:${PATH}" \
    TS_MODEL_PATH=/models/model.pt

USER app
ENTRYPOINT ["terraspectra-model"]
CMD ["--help"]
