"""add nhl_player_id to draft_picks_historical

Revision ID: 002
Revises: 001
Create Date: 2025-03-25 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'draft_picks_historical',
        sa.Column('nhl_player_id', sa.Integer(), nullable=True),
    )
    op.create_index(
        'ix_draft_picks_historical_nhl_player_id',
        'draft_picks_historical',
        ['nhl_player_id'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_draft_picks_historical_nhl_player_id',
        table_name='draft_picks_historical',
    )
    op.drop_column('draft_picks_historical', 'nhl_player_id')
