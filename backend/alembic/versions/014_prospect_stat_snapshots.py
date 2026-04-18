"""Add prospect_stat_snapshots table for versioned pre-draft vs end-of-season stats.

Revision ID: 014
Revises: 013
Create Date: 2026-04-02

Why this matters:
  The RAG index and training features are built from current prospect stats.
  If we re-run ingestion in July (post-draft), stats reflect end-of-season
  numbers, not what was available at draft time (typically late May).
  Training with post-draft stats leaks future information — the model "knows"
  how the season ended, which a real GM wouldn't know on draft day.

  This migration adds prospect_stat_snapshots: a point-in-time versioned table.
  Each snapshot has a snapshot_type ("pre_draft" | "end_of_season" | "mid_season")
  and a snapshot_date. Training data joins on snapshot_type = "pre_draft" for
  the relevant draft year, ensuring no data leakage.

  The scout_embeddings RAG index is built from "end_of_season" snapshots
  (most complete stats for semantic search), while ML training uses "pre_draft".

Schema:
  prospect_stat_snapshots
    id                   SERIAL PRIMARY KEY
    prospect_id          INTEGER FK → prospects_2025.id (nullable for historical)
    player_name          VARCHAR(100)  -- denormalized for lookup without JOIN
    draft_year           INTEGER       -- e.g. 2025
    draft_league         VARCHAR(100)
    snapshot_type        VARCHAR(30)   -- "pre_draft" | "end_of_season" | "mid_season"
    snapshot_date        DATE          -- actual date this snapshot was taken
    points_per_game      FLOAT
    goals_per_game       FLOAT
    assists_per_game     FLOAT
    games_played         INTEGER
    ppg_prev_season      FLOAT
    css_ranking          INTEGER
    created_at           TIMESTAMPTZ DEFAULT now()

Also adds a BM25 full-text search GIN index on scout_embeddings.content
required by the hybrid RRF search implementation.
"""
from alembic import op
import sqlalchemy as sa


revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── prospect_stat_snapshots ───────────────────────────────────────────────
    op.create_table(
        "prospect_stat_snapshots",
        sa.Column("id",               sa.Integer,     primary_key=True),
        sa.Column("prospect_id",      sa.Integer,     sa.ForeignKey("prospects_2025.id", ondelete="SET NULL"),
                  nullable=True, index=True),
        sa.Column("player_name",      sa.String(100), nullable=False),
        sa.Column("draft_year",       sa.Integer,     nullable=False),
        sa.Column("draft_league",     sa.String(100), nullable=True),
        sa.Column("snapshot_type",    sa.String(30),  nullable=False),
        sa.Column("snapshot_date",    sa.Date,        nullable=False),
        sa.Column("points_per_game",  sa.Float,       nullable=True),
        sa.Column("goals_per_game",   sa.Float,       nullable=True),
        sa.Column("assists_per_game", sa.Float,       nullable=True),
        sa.Column("games_played",     sa.Integer,     nullable=True),
        sa.Column("ppg_prev_season",  sa.Float,       nullable=True),
        sa.Column("css_ranking",      sa.Integer,     nullable=True),
        sa.Column("created_at",       sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )

    # Composite index for the primary access pattern: "give me pre_draft stats for year Y"
    op.create_index(
        "idx_snapshots_year_type",
        "prospect_stat_snapshots",
        ["draft_year", "snapshot_type"],
    )

    # Index for prospect lookup by name (used when prospect_id is null for historical)
    op.create_index(
        "idx_snapshots_player_name",
        "prospect_stat_snapshots",
        ["player_name"],
    )

    # ── BM25 full-text GIN index on scout_embeddings ──────────────────────────
    # Required by store.retrieve_bm25() for the hybrid RRF search.
    # to_tsvector() is expensive to compute at query time — the GIN index
    # pre-computes it so BM25 search is O(log N) rather than full-table scan.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_scout_embeddings_fts
        ON scout_embeddings
        USING GIN (to_tsvector('english', content))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_scout_embeddings_fts")
    op.drop_index("idx_snapshots_player_name", table_name="prospect_stat_snapshots")
    op.drop_index("idx_snapshots_year_type",   table_name="prospect_stat_snapshots")
    op.drop_table("prospect_stat_snapshots")
