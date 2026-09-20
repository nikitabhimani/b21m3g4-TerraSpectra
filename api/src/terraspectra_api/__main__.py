"""``terraspectra-api`` / ``python -m terraspectra_api``: run the API with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from terraspectra_api.settings import get_settings


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="TerraSpectra inference API")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    uvicorn.run(
        "terraspectra_api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_config=None,
        proxy_headers=True,
        server_header=False,
    )


if __name__ == "__main__":
    main()
