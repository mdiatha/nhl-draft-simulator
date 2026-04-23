"""Integration test fixtures using testcontainers.

These tests spin up a real PostgreSQL container and run against it.
They test the full data flow: ingestion → features → model predictions.
Much more valuable than mocked unit tests for catching schema/query bugs.

Run with: pytest tests/integration/ -v --tb=short
Requires Docker to be running.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PostgresContainer = pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers is not installed; skipping integration tests",
).PostgresContainer


@pytest.fixture(scope="session")
def pg_container():
    """Spin up a real PostgreSQL 16 + pgvector container for the test session.

    Must use the pgvector image — plain postgres:16-alpine lacks the vector
    extension required by migrations 009, 011, and 015. The image can be
    overridden via TESTCONTAINERS_POSTGRES_IMAGE but defaults to pgvector.
    """
    import os
    image = os.getenv("TESTCONTAINERS_POSTGRES_IMAGE", "pgvector/pgvector:pg16")
    with PostgresContainer(image) as pg:
        yield pg


@pytest.fixture(scope="session")
def db_engine(pg_container):
    """Create SQLAlchemy engine against the test container."""
    import os
    from alembic.config import Config
    from alembic import command

    url = pg_container.get_connection_url()
    engine = create_engine(url)

    # Run migrations instead of create_all so the schema matches the migration
    # chain exactly. create_all would create indexes from ORM models and then
    # migration 017 would try to create the same indexes again → DuplicateTable.
    # Set DATABASE_URL so alembic/env.py picks it up (set_main_option is
    # overridden by env.py reading os.environ).
    os.environ["DATABASE_URL"] = url
    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "../../alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Provide a transactional test session that rolls back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()
