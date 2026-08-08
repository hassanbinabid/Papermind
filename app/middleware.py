"""
middleware.py — Request logging / timing middleware.

Tags every response with a request ID and processing time, and logs
method/path/status/duration. This is deliberately lightweight — not a
replacement for your LangSmith/Langfuse-style tracing on the other
projects, just a baseline observability layer at the HTTP boundary
that costs almost nothing to add.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("papermind.request")
logging.basicConfig(level=logging.INFO)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            "[%s] %s %s -> %s (%.2fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
