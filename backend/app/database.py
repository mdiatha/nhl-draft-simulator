import os
from typing import Optional

import redis as redis_lib
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# ── Primary database engine (read + write) ───────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    # 3 service instances × 4 Uvicorn workers = 12 persistent connections minimum.
    # pool_size=20 gives headroom for concurrent requests per worker.
    pool_size=20,
    max_overflow=10,   # hard cap: 30 total (20 + 10)
    # Recycle connections every hour so RDS Proxy's idle timeout (~900 s)
    # never kills a connection that SQLAlchemy thinks is still live.
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Read replica engine (read-only queries) ───────────────────────────────────
# Routes all Scout agent tool queries (get_gm_profile, search_prospects, etc.)
# to an RDS read replica. These are all SELECT-only — routing them to the
# replica keeps the primary free for writes (ingestion, training checkpoints,
# conversation persistence).
#
# Benefits:
#   - Primary write throughput is unaffected by analytical queries
#   - Read replicas are cheap (same instance class, async replication)
#   - Lag is typically <1 second — acceptable for draft analytics
#
# Fallback: if DATABASE_READ_REPLICA_URL is not set (dev or single-instance),
# agent tool queries use the primary engine. No code changes needed in tool
# executors — they receive a db session transparently.

if settings.DATABASE_READ_REPLICA_URL:
    _replica_engine = create_engine(
        settings.DATABASE_READ_REPLICA_URL,
        pool_pre_ping=True,
        pool_size=10,       # replica handles reads only — smaller pool
        max_overflow=5,
        pool_recycle=3600,
    )
    _ReplicaSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_replica_engine)
else:
    _replica_engine = engine
    _ReplicaSessionLocal = SessionLocal


class Base(DeclarativeBase):
    pass


def get_db():
    """Primary database session — used for writes and non-agent reads."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_read_db():
    """
    Read-replica database session — used for Scout agent tool queries.

    All 6 Scout tools (get_gm_profile, search_prospects, etc.) are read-only.
    Routing them here preserves primary capacity for ingestion + training writes.
    Falls back to the primary when no replica is configured.
    """
    db = _ReplicaSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Shared Redis client ───────────────────────────────────────────────────────
# Single connection pool reused across all API modules.

_redis_client: Optional[redis_lib.Redis] = None
REDIS_OK: bool = False

try:
    _redis_client = redis_lib.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    _redis_client.ping()
    REDIS_OK = True
except Exception:
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception:
            pass
    _redis_client = None
    REDIS_OK = False


def get_redis() -> tuple[Optional[redis_lib.Redis], bool]:
    """Return (redis_client, is_available). redis_client is None when Redis is down."""
    return _redis_client, REDIS_OK
