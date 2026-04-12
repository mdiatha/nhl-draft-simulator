"""Add pgvector extension and scout_embeddings table

Enables the RAG "Ask the Scout" agent: GM tendency profiles and prospect
summaries are embedded and stored here for cosine-similarity retrieval.

Revision ID: 009
Revises: 008
Create Date: 2026-03-30
"""
from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension (requires PostgreSQL 11+ with pgvector installed)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "scout_embeddings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("doc_type", sa.String(50), nullable=False),   # "gm_profile", "prospect", "team_summary"
        sa.Column("ref_id", sa.Integer, nullable=True),          # FK to the source row (gm_id, prospect_id, etc.)
        sa.Column("ref_name", sa.String(200), nullable=True),    # human-readable label for retrieval display
        sa.Column("content", sa.Text, nullable=False),           # the raw text that was embedded
        # 1536-dim matches text-embedding-3-small; we use Claude's embed via a fixed dim
        # Store as text for portability, cast to vector in queries
        sa.Column("embedding", sa.Text, nullable=True),          # JSON-serialized float list
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_scout_embeddings_doc_type", "scout_embeddings", ["doc_type"])
    op.create_index("ix_scout_embeddings_ref_id", "scout_embeddings", ["ref_id"])

    # Conversation memory table for the agent
    op.create_table(
        "scout_conversations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),        # "user" or "assistant"
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_scout_conversations_session_id", "scout_conversations", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_scout_conversations_session_id", "scout_conversations")
    op.drop_table("scout_conversations")
    op.drop_index("ix_scout_embeddings_ref_id", "scout_embeddings")
    op.drop_index("ix_scout_embeddings_doc_type", "scout_embeddings")
    op.drop_table("scout_embeddings")
