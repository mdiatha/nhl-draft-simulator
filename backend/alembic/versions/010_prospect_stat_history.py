"""Add prospect_stat_history table for real-time pre-draft stat tracking.

Stores a time-series of prospect stats fetched from the NHL API so we can:
  1. Track stat trends over the pre-draft season
  2. Feed live PPG updates into the feature store
  3. Detect stat anomalies (suspiciously high/low PPG) before they affect predictions

Revision ID: 010
Revises: 009
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prospect_stat_history",
        sa.Column("id",              sa.Integer(), nullable=False),
        sa.Column("prospect_id",     sa.Integer(), nullable=False),
        sa.Column("fetched_at",      sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("season_type",     sa.String(20),  nullable=True),   # pre_draft | regular | playoffs
        sa.Column("league",          sa.String(100), nullable=True),
        sa.Column("games_played",    sa.Integer(),   nullable=True),
        sa.Column("goals",           sa.Integer(),   nullable=True),
        sa.Column("assists",         sa.Integer(),   nullable=True),
        sa.Column("points",          sa.Integer(),   nullable=True),
        sa.Column("points_per_game", sa.Float(),     nullable=True),
        sa.Column("source_url",      sa.String(500), nullable=True),
        sa.Column("raw_payload",     postgresql.JSONB(), nullable=True),

        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["prospect_id"], ["prospects_2025.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_stat_history_prospect_fetched",
                    "prospect_stat_history", ["prospect_id", "fetched_at"])
    op.create_index("ix_stat_history_fetched",
                    "prospect_stat_history", ["fetched_at"])


def downgrade() -> None:
    op.drop_index("ix_stat_history_fetched",           table_name="prospect_stat_history")
    op.drop_index("ix_stat_history_prospect_fetched",  table_name="prospect_stat_history")
    op.drop_table("prospect_stat_history")
