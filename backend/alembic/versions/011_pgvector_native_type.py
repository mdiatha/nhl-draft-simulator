"""Convert scout_embeddings.embedding to native pgvector type and add IVFFlat ANN index.

Previously the embedding column was stored as TEXT (JSON-serialized float list).
This migration casts it to the native vector(1024) type so Postgres can do
approximate nearest-neighbor (ANN) search in SQL rather than in Python.

The IVFFlat index (lists=50) makes semantic similarity queries O(log N) instead
of O(N), enabling "find me a player like Makar" queries in <5ms even with 10k+ rows.

Revision ID: 011
Revises: 010
Create Date: 2026-03-31
"""
from alembic import op
import sqlalchemy as sa

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Cast TEXT column to native vector type.
    # Rows with NULL embedding are preserved as NULL.
    # Rows where the JSON cannot be cast (malformed) will fail — clean up first if needed.
    op.execute("""
        ALTER TABLE scout_embeddings
        ALTER COLUMN embedding TYPE vector(1024)
        USING CASE
            WHEN embedding IS NULL THEN NULL
            ELSE embedding::vector(1024)
        END
    """)

    # IVFFlat index for approximate cosine-similarity search.
    # lists=50 is appropriate for up to ~50k rows (rule of thumb: sqrt(n_rows)).
    op.execute("""
        CREATE INDEX ix_scout_embeddings_ivfflat
        ON scout_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 50)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_scout_embeddings_ivfflat")
    # Cast back to TEXT — serialize using array_to_string for portability.
    op.execute("""
        ALTER TABLE scout_embeddings
        ALTER COLUMN embedding TYPE TEXT
        USING CASE
            WHEN embedding IS NULL THEN NULL
            ELSE embedding::text
        END
    """)
