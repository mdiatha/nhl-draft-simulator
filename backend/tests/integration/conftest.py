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
from app.database import Base

PostgresContainer = pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers is not installed; skipping integration tests",
).PostgresContainer


@pytest.fixture(scope="session")
def pg_container():
    """Spin up a real PostgreSQL 16 container for the test session."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_engine(pg_container):
    """Create SQLAlchemy engine against the test container."""
    url = pg_container.get_connection_url()
    # testcontainers returns postgresql+psycopg2://... format
    engine = create_engine(url)
    Base.metadata.create_all(engine)
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
