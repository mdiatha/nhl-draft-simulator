"""add points_per_game and age_at_draft to draft_picks_historical

Revision ID: 004
Revises: 003
Create Date: 2025-03-25 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('draft_picks_historical', sa.Column('points_per_game', sa.Float(), nullable=True))
    op.add_column('draft_picks_historical', sa.Column('age_at_draft', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('draft_picks_historical', 'age_at_draft')
    op.drop_column('draft_picks_historical', 'points_per_game')
