"""
Structured JSON logging + request-ID middleware + CloudWatch shipping.

Every log line emitted anywhere in the app includes:
  - timestamp (ISO-8601)
  - level
  - logger name
  - message
  - request_id  (UUID injected per HTTP request, propagated via contextvars)
  - environment (from settings)

Two output channels:
  1. stdout (always) — JSON lines, consumed by Docker / container runtime logs

  2. CloudWatch Logs (optional) — direct shipping via watchtower when
     AWS_CLOUDWATCH_LOG_GROUP is set. Useful for ECS, EC2, Lambda, or local prod.
     Log stream per instance/task: {environment}/{hostname}
     Log group: whatever AWS_CLOUDWATCH_LOG_GROUP is set to (e.g. /nhl-draft/api)

CloudWatch Logs Insights queries:
  # Trace a full request end-to-end
  fields @timestamp, level, message, duration_ms
  | filter request_id = "abc-123"
  | sort @timestamp asc

  # Find all errors in the last hour
  fields @timestamp, message, logger
  | filter level = "ERROR"
  | sort @timestamp desc
  | limit 50

  # Simulation latency p99
  fields @timestamp, duration_ms
  | filter path = "/api/draft/simulate"
  | stats pct(duration_ms, 99) as p99 by bin(5m)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import Request
from pythonjsonlogger import jsonlogger

from app.config import settings

# ── Request-ID context ────────────────────────────────────────────────────────
# Each async task gets its own copy of this var; setting it in middleware
# propagates it to all log calls made during that request's lifetime.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id_var.get()


# ── JSON formatter ────────────────────────────────────────────────────────────

class _AppJsonFormatter(jsonlogger.JsonFormatter):
    """Adds request_id and environment to every log record."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["request_id"]  = _request_id_var.get()
        log_record["environment"] = settings.ENVIRONMENT
        log_record["logger"]      = record.name
        # Rename default fields to match CloudWatch / Datadog conventions
        log_record.setdefault("level",   record.levelname)
        log_record.setdefault("message", record.getMessage())


def configure_logging() -> None:
    """
    Configure structured JSON logging with optional CloudWatch shipping.
    Call once at app startup before any other logging occurs.

    Handlers added:
      - StreamHandler (stdout) — always present; consumed by Docker/container logs
      - WatchtowerHandler (CloudWatch) — only when AWS_CLOUDWATCH_LOG_GROUP is set
    """
    formatter = _AppJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # ── CloudWatch Logs (direct shipping) ─────────────────────────────────────
    # In ECS/EC2, logs can also be shipped by the CloudWatch agent or awslogs driver.
    # Enable watchtower when direct in-process shipping is preferred.
    if settings.AWS_CLOUDWATCH_LOG_GROUP:
        _add_cloudwatch_handler(root, formatter)


def _add_cloudwatch_handler(root: logging.Logger, formatter: logging.Formatter) -> None:
    """Attach a WatchtowerHandler for direct CloudWatch Logs shipping."""
    import socket
    try:
        import boto3
        import watchtower

        cw_client = boto3.client("logs", region_name=settings.AWS_REGION)
        stream_name = f"{settings.ENVIRONMENT}/{socket.gethostname()}"

        cw_handler = watchtower.CloudWatchLogHandler(
            boto3_client=cw_client,
            log_group=settings.AWS_CLOUDWATCH_LOG_GROUP,
            stream_name=stream_name,
            use_queues=True,          # async — never blocks the request path
            send_interval=10,         # flush every 10 seconds
            max_batch_size=10_000,    # CloudWatch max batch size
            max_batch_count=10_000,
        )
        cw_handler.setFormatter(formatter)
        root.addHandler(cw_handler)
        root.info(
            "cloudwatch.handler_attached",
            extra={
                "log_group":   settings.AWS_CLOUDWATCH_LOG_GROUP,
                "stream_name": stream_name,
            },
        )
    except ImportError:
        root.warning("watchtower not installed — CloudWatch direct shipping disabled")
    except Exception as exc:
        root.warning("cloudwatch.handler_failed", extra={"error": str(exc)})


# ── Request-ID + latency middleware ───────────────────────────────────────────

_logger = logging.getLogger("app.http")


async def request_id_middleware(request: Request, call_next):
    """
    FastAPI middleware that:
      1. Generates a UUID per request (or reads X-Request-ID if provided by a
         load balancer / API gateway).
      2. Injects it into the contextvars so every log call in this request's
         coroutine automatically carries it.
      3. Attaches it to the response as X-Request-ID for client-side tracing.
      4. Emits a structured access log with method, path, status, and latency.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = _request_id_var.set(request_id)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        _request_id_var.reset(token)

    duration_ms = (time.perf_counter() - start) * 1000
    _logger.info(
        "http.request",
        extra={
            "method":      request.method,
            "path":        request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )

    response.headers["X-Request-ID"] = request_id
    return response
