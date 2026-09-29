"""Production plumbing shared by the model servers: settings, logging, request IDs, errors, auth."""
import contextvars
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# Set per request by the middleware. Threads started with contextvars.copy_context()
# inherit it, so logs from worker threads carry the same request id.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


class _RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(threadName)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def add_production_middleware(app: FastAPI) -> None:
    """Request id + latency header + access log, and JSON errors that never leak stack traces."""
    log = logging.getLogger("serving.access")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
            log.info("%s %s -> %s in %.1fms", request.method, request.url.path,
                     response.status_code, elapsed_ms)
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logging.getLogger("serving.error").exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={
            "detail": "Internal server error",
            "request_id": request_id_var.get(),
        })


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Auth dependency. Disabled when SERVING_API_KEY is unset (local development)."""
    expected = os.getenv("SERVING_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
