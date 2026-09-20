"""API-key authentication and a small in-memory rate limiter."""

from __future__ import annotations

import hmac
import math
import threading
import time

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from terraspectra_api.errors import ApiError
from terraspectra_api.settings import Settings, parse_rate_limit

api_key_header = APIKeyHeader(name="X-API-Key", scheme_name="ApiKeyAuth", auto_error=False)


def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    """Reject requests whose ``X-API-Key`` is not in ``TS_API_KEYS``."""
    settings: Settings = request.app.state.settings
    if api_key and any(
        hmac.compare_digest(api_key.encode(), key.encode()) for key in settings.api_keys
    ):
        return api_key
    raise ApiError(
        401, "missing or invalid API key", "unauthorized", {"WWW-Authenticate": "ApiKey"}
    )


class RateLimiter:
    """Fixed-window counter per key. Bounded memory: keys are API keys (a configured set).

    TODO(Day 11): move to Redis so limits are shared across uvicorn workers/replicas.
    """

    def __init__(self, rate: str) -> None:
        self.limit, self.period = parse_rate_limit(rate)
        self._windows: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> float | None:
        """Record a hit; return ``None`` if allowed, else seconds until the window resets."""
        if self.limit <= 0:
            return None
        now = time.time() if now is None else now
        window = int(now // self.period)
        with self._lock:
            start, count = self._windows.get(key, (window, 0))
            if start != window:
                start, count = window, 0
            if count >= self.limit:
                return (window + 1) * self.period - now
            self._windows[key] = (start, count + 1)
        return None


def rate_limit(request: Request, api_key: str = Depends(require_api_key)) -> str:
    """Authenticated + rate-limited dependency for protected routers."""
    limiter: RateLimiter = request.app.state.rate_limiter
    retry_after = limiter.hit(api_key)
    if retry_after is not None:
        raise ApiError(
            429,
            "rate limit exceeded",
            "rate_limited",
            {"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    return api_key
