"""Consistent error model ``{"detail": str, "code": str}`` and exception handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ApiError(Exception):
    """An error that maps directly onto an HTTP response."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        code: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.headers = headers


def not_found(kind: str, ident: str) -> ApiError:
    """404 helper, e.g. ``not_found("scene", "scn_x")``."""
    return ApiError(404, f"{kind} {ident!r} not found", f"{kind}_not_found")


def _code_for(status: int) -> str:
    try:
        return HTTPStatus(status).phrase.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        return "error"


def error_response(
    status: int, detail: str, code: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Build a JSON error response following the contract ``Error`` schema."""
    return JSONResponse({"detail": detail, "code": code}, status_code=status, headers=headers)


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers so every error uses the ``{detail, code}`` shape."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.detail, exc.code, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(
            exc.status_code, str(exc.detail), _code_for(exc.status_code), exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            parts.append(f"{loc}: {err.get('msg', 'invalid')}")
        return error_response(422, "; ".join(parts) or "invalid request", "validation_error")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return error_response(500, "internal server error", "internal_error")
