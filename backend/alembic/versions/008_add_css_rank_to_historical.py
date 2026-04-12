"""Add css_rank to draft_picks_historical

Stores the NHL Central Scouting final rank for each historical pick.
Used to replace the round-position proxy in training with the real
pre-draft consensus rank, eliminating the tautological training signal.

Revision ID: 008
Revises: 007
Create Date: 2026-03-29
"""
from alembic import op
import sqlalchemy as sa

revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'draft_picks_historical',
        sa.Column('css_rank', sa.Integer(), nullable=True),
    )
    op.create_index(
        'ix_draft_picks_historical_css_rank',
        'draft_picks_historical',
        ['css_rank'],
    )


def downgrade():
    op.drop_index('ix_draft_picks_historical_css_rank', table_name='draft_picks_historical')
    op.drop_column('draft_picks_historical', 'css_rank')
