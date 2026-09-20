#!/bin/sh
# Dispatch: api | worker | migrate | <any command>
set -eu

ALEMBIC_CFG=/app/api/alembic.ini

case "${1:-api}" in
  api)
    shift || true
    alembic -c "$ALEMBIC_CFG" upgrade head
    exec uvicorn terraspectra_api.main:create_app --factory \
      --host "${TS_HOST:-0.0.0.0}" --port "${TS_PORT:-8000}" \
      --workers "${TS_WEB_CONCURRENCY:-2}" \
      --proxy-headers --forwarded-allow-ips "${TS_FORWARDED_ALLOW_IPS:-127.0.0.1}" \
      --no-server-header "$@"
    ;;
  worker)
    shift || true
    exec python -m terraspectra_api.worker "$@"
    ;;
  migrate)
    shift || true
    exec alembic -c "$ALEMBIC_CFG" upgrade "${1:-head}"
    ;;
  *)
    exec "$@"
    ;;
esac
