"""Add prospect_features table — materialized feature store for ML inference.

Pre-materializes the feature vector for every 2025 prospect so scoring at
draft time is a single indexed DB read rather than recomputing 37 features
for all 224 prospects × 32 picks = 7,168 individual row constructions.

Also stores the commonly-queried scalar features denormalized for fast lookup.

Revision ID: 012
Revises: 011
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prospect_features",
        sa.Column("id",              sa.Integer(), nullable=False),
        sa.Column("prospect_id",     sa.Integer(), nullable=False),
        sa.Column("refreshed_at",    sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        # Full feature vector stored as JSONB for schema flexibility
        sa.Column("features_json",   postgresql.JSONB(), nullable=False),
        # Commonly-queried scalar fields denormalized for fast filtering
        sa.Column("css_rank_norm",   sa.Float(), nullable=True),
        sa.Column("ppg_league_norm", sa.Float(), nullable=True),
        sa.Column("pick_slot_norm",  sa.Float(), nullable=True),

        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["prospect_id"], ["prospects_2025.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("prospect_id", name="uq_prospect_features_prospect_id"),
    )
    op.create_index("ix_prospect_features_refreshed",
                    "prospect_features", ["refreshed_at"])


def downgrade() -> None:
    op.drop_index("ix_prospect_features_refreshed", table_name="prospect_features")
    op.drop_table("prospect_features")
