"""Add pipeline_steps column to ingestion_runs.

The ingestion_run model has a pipeline_steps JSON column that was added
to the ORM without a corresponding migration.

Revision ID: 016
Revises: 015
Create Date: 2026-04-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingestion_runs",
        sa.Column("pipeline_steps", JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_runs", "pipeline_steps")
