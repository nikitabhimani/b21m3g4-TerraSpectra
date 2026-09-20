# syntax=docker/dockerfile:1.7
# TerraSpectra geospatial pipeline CLI. Build context: repository root.
#   docker build -f pipeline/Dockerfile .
#   docker compose --profile tools run --rm pipeline process /data/scene /data/cube.tif
# Rasterio wheels bundle GDAL/PROJ, so no system GDAL packages are needed.
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/pipeline/.venv

WORKDIR /app/pipeline

# 1) Dependencies only (cached unless the lockfile or contracts change).
COPY contracts/python /app/contracts/python
COPY pipeline/pyproject.toml pipeline/uv.lock pipeline/README.md /app/pipeline/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-install-project

# 2) Project source.
COPY pipeline/ /app/pipeline/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras \
 && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data \
 && chown -R app:app /data

ENV PATH="/app/pipeline/.venv/bin:${PATH}" \
    TS_PIPELINE_DATA_DIR=/data \
    GDAL_CACHEMAX=512 \
    GDAL_NUM_THREADS=ALL_CPUS \
    CPL_TMPDIR=/tmp

USER app
WORKDIR /data
ENTRYPOINT ["terraspectra-pipeline"]
CMD ["--help"]
