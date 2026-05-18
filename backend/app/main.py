from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.observability.logging import configure_logging, request_id_middleware
from app.api import teams
from app.api import lottery, prospects, teams_ext
from app.api.draft_sim import router as draft_sim_router
from app.api.counterfactual import router as counterfactual_router
from app.api.ml import router as ml_router
from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.standings import router as standings_router

# Configure structured JSON logging before anything else emits a log line
configure_logging()

import logging  # noqa: E402
logger = logging.getLogger(__name__)


# ── Sentry ────────────────────────────────────────────────────────────────────
def _init_sentry() -> None:
    if not settings.SENTRY_DSN:
        return
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    import logging as _logging

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            LoggingIntegration(level=_logging.INFO, event_level=_logging.ERROR),
        ],
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        environment=settings.APP_ENV,
        release=f"nhl-draft-api@{settings.APP_ENV}",
        # Don't send PII (user IPs, headers with keys)
        send_default_pii=False,
    )
    logger.info("sentry.initialized", extra={"environment": settings.APP_ENV})


# ── OpenTelemetry ─────────────────────────────────────────────────────────────
def _init_otel(app: FastAPI) -> None:
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        SQLAlchemyInstrumentor().instrument()
        logger.info(
            "otel.initialized",
            extra={"endpoint": settings.OTEL_EXPORTER_OTLP_ENDPOINT},
        )
    except Exception as exc:
        logger.warning("otel.init_failed", extra={"error": str(exc)})


# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
try:
    import importlib

    slowapi = importlib.import_module("slowapi")
    Limiter = slowapi.Limiter
    _rate_limit_exceeded_handler = slowapi._rate_limit_exceeded_handler
    get_remote_address = importlib.import_module("slowapi.util").get_remote_address
    RateLimitExceeded = importlib.import_module("slowapi.errors").RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
except ImportError:
    limiter = None

    def _rate_limit_exceeded_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )

    def get_remote_address(request: Request):
        return request.client.host if request.client else "127.0.0.1"

    class RateLimitExceeded(Exception):
        pass


def _check_config() -> None:
    """Fail fast at startup if critical env vars are missing in non-dev environments.

    Catching misconfiguration here (loud crash) is far better than silently
    returning 503 on every admin request and leaving operators confused.
    """
    if settings.APP_ENV != "development" and not settings.ADMIN_API_KEY:
        raise RuntimeError(
            "ADMIN_API_KEY is not set. All admin and ML endpoints will be "
            "unreachable. Set ADMIN_API_KEY in the runtime environment or "
            "Secrets Manager before deploying."
        )



@asynccontextmanager
async def lifespan(_app: FastAPI):
    _check_config()
    _init_sentry()
    _init_otel(_app)

    from app.ml.registry import registry
    registry.load()
    logger.info("app.startup", extra={"model_loaded": registry.is_loaded})
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="NHL Draft Simulator",
    version="0.1.0",
    description="Simulate NHL drafts using team tendencies, prospect needs, and lottery logic.",
    lifespan=lifespan,
)

# Rate limiter state + 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Key", "Cache-Control"],
    expose_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)

# Structured logging + request-ID middleware
app.middleware("http")(request_id_middleware)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Attach security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.APP_ENV != "development":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── API versioning ────────────────────────────────────────────────────────────
# All routes live under /api/v1/.  Each sub-router owns its own sub-path
# (e.g. agent_router has prefix="/agent", so its routes are /api/v1/agent/...).
#
# Backward-compat: routers are also mounted at /api for clients that haven't
# migrated.  /api/v1 is the canonical path.  In a future major release, the
# /api alias will be removed and clients must use /api/v1.
from fastapi import APIRouter as _APIRouter  # noqa: E402

_v1 = _APIRouter(prefix="/api/v1")
_v1.include_router(teams.router)       # /api/v1/teams/...
_v1.include_router(lottery.router)     # /api/v1/lottery/...
_v1.include_router(prospects.router)   # /api/v1/draft/...   (prospects)
_v1.include_router(teams_ext.router)   # /api/v1/teams/...   (extended)
_v1.include_router(draft_sim_router)   # /api/v1/draft/...   (simulation)
_v1.include_router(counterfactual_router)  # /api/v1/draft/counterfactual/...
_v1.include_router(ml_router)          # /api/v1/ml/...
_v1.include_router(admin_router)       # /api/v1/admin/...
_v1.include_router(agent_router)       # /api/v1/agent/...
_v1.include_router(standings_router)   # /api/v1/standings/...

app.include_router(_v1)

# Backward-compat aliases at /api (deprecated — use /api/v1)
_v0_compat = _APIRouter(prefix="/api")
_v0_compat.include_router(teams.router)
_v0_compat.include_router(lottery.router)
_v0_compat.include_router(prospects.router)
_v0_compat.include_router(teams_ext.router)
_v0_compat.include_router(draft_sim_router)
_v0_compat.include_router(counterfactual_router)
_v0_compat.include_router(ml_router)
_v0_compat.include_router(admin_router)
_v0_compat.include_router(agent_router)
_v0_compat.include_router(standings_router)

app.include_router(_v0_compat)


def _health_snapshot(db: Session) -> tuple[dict, bool]:
    from app.ml.registry import registry
    from app.models import Prospect, Team

    db_ok = True
    prospects_count = None
    teams_count = None
    try:
        db.execute(text("SELECT 1"))
        prospects_count = db.query(Prospect).count()
        teams_count = db.query(Team).count()
    except Exception:
        db_ok = False

    ready = db_ok and registry.is_loaded
    snapshot = {
        "status": "ok" if ready else "degraded",
        "db": "connected" if db_ok else "error",
        "prospects_loaded": prospects_count,
        "teams_loaded": teams_count,
        "model_loaded": registry.is_loaded,
        "model_trained_at": registry.meta.get("trained_at") if registry.is_loaded else None,
        "model_auc": registry.meta.get("validation_auc") if registry.is_loaded else None,
        "model_mode": registry.meta.get("mode", "evaluation") if registry.is_loaded else None,
    }
    return snapshot, ready


@app.get("/livez")
async def livez():
    """Lightweight liveness probe — proves the process is accepting requests."""
    return {"status": "alive", "service": settings.OTEL_SERVICE_NAME}


@app.get("/readyz")
async def readyz(db: Session = Depends(get_db)):
    """Readiness probe — returns 503 when core dependencies are unavailable."""
    snapshot, ready = _health_snapshot(db)
    payload = {
        "status": "ready" if ready else "not_ready",
        "db": snapshot["db"],
        "model_loaded": snapshot["model_loaded"],
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """Detailed dependency health endpoint for operators and smoke tests."""
    snapshot, ready = _health_snapshot(db)
    return JSONResponse(status_code=200 if ready else 503, content=snapshot)
