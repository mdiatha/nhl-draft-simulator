"""API key authentication for privileged endpoints (admin + ML training).

Usage:
    from app.middleware.auth import require_admin_key

    @router.post("/ingest")
    async def ingest(api_key: str = Depends(require_admin_key)):
        ...

The key is read from the X-Admin-Key header.

In local dev / CI: set ADMIN_API_KEY="" (empty) to bypass auth.
In production: ADMIN_API_KEY MUST be set. If it is not set in a production
environment (APP_ENV != "development"), the server will refuse all admin
requests with 503 to prevent accidental open access.
"""
import hmac
import logging

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


async def require_admin_key(x_admin_key: str = Header(default="")) -> None:
    """FastAPI dependency that enforces X-Admin-Key on protected endpoints.

    - Development (APP_ENV == "development"): no-op when ADMIN_API_KEY is empty.
    - All other environments: ADMIN_API_KEY must be set and the header must match.
      Returns 503 if ADMIN_API_KEY is missing in a non-dev environment (misconfiguration).
      Returns 401 if the header is absent.
      Returns 403 if the key is wrong.

    Uses hmac.compare_digest for constant-time comparison to prevent timing attacks.
    """
    is_dev = settings.APP_ENV == "development"

    if not settings.ADMIN_API_KEY:
        if is_dev:
            return  # auth disabled in local dev
        # Non-dev environment with no key set — server misconfiguration, refuse loudly
        logger.error(
            "auth.misconfigured ADMIN_API_KEY is not set in a non-development environment"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfiguration: admin key not configured. Contact the operator.",
        )

    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Admin-Key header is required for this endpoint.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison prevents timing-based key enumeration
    if not hmac.compare_digest(x_admin_key.encode(), settings.ADMIN_API_KEY.encode()):
        logger.warning("auth.invalid_key", extra={"env": settings.APP_ENV})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key.",
        )
