"""Add missing indexes for common filter/sort queries.

Prospects: position, css_ranking (used in list_prospects filters + draft sim ordering).
Draft picks: year, team_id (used in GM tendency engine + backtest + RAG indexing).
Scout embeddings: doc_type (used in retrieve_similar filter).
General managers: is_active (used in every GM lookup).

Revision ID: 013
Revises: 012
Create Date: 2026-04-01
"""
from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # prospects_2025 — most queried table at runtime
    op.create_index(
        "ix_prospects_2025_position",
        "prospects_2025",
        ["position"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_prospects_2025_css_ranking",
        "prospects_2025",
        ["css_ranking"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_prospects_2025_nationality",
        "prospects_2025",
        ["nationality"],
        if_not_exists=True,
    )

    # draft_picks_historical — used in backtest, tendency engine, RAG
    op.create_index(
        "ix_draft_picks_historical_year_team",
        "draft_picks_historical",
        ["year", "team_id"],
        if_not_exists=True,
    )

    # scout_embeddings — used in retrieve_similar WHERE doc_type IN (...)
    op.create_index(
        "ix_scout_embeddings_doc_type",
        "scout_embeddings",
        ["doc_type"],
        if_not_exists=True,
    )

    # general_managers — is_active filter in every GM lookup
    op.create_index(
        "ix_general_managers_is_active",
        "general_managers",
        ["is_active"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_general_managers_is_active", table_name="general_managers")
    op.drop_index("ix_scout_embeddings_doc_type", table_name="scout_embeddings")
    op.drop_index("ix_draft_picks_historical_year_team", table_name="draft_picks_historical")
    op.drop_index("ix_prospects_2025_nationality", table_name="prospects_2025")
    op.drop_index("ix_prospects_2025_css_ranking", table_name="prospects_2025")
    op.drop_index("ix_prospects_2025_position", table_name="prospects_2025")
