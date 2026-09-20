# syntax=docker/dockerfile:1.7
# TerraSpectra inference API + worker. Build context: repository root.
#   docker build -f api/Dockerfile .
# The default PyPI torch wheel bundles the CUDA runtime, so this slim image runs on
# GPU hosts via the NVIDIA container toolkit (docker-compose.gpu.yml) and on CPU.
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/api/.venv

WORKDIR /app/api

# 1) Dependencies only (cached unless the lockfile or contracts change).
COPY contracts/python /app/contracts/python
COPY api/pyproject.toml api/uv.lock api/README.md /app/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Project source.
COPY api/ /app/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
 && chmod +x /app/api/docker-entrypoint.sh \
 && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data /models \
 && chown -R app:app /data

ENV PATH="/app/api/.venv/bin:${PATH}" \
    TS_ENV=production \
    TS_STORAGE_DIR=/data \
    TS_MODEL_PATH=/models/model.pt \
    TS_HOST=0.0.0.0 \
    TS_PORT=8000

USER app
EXPOSE 8000
ENTRYPOINT ["/app/api/docker-entrypoint.sh"]
CMD ["api"]
