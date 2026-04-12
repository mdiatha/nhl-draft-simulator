"""add nhl_player_id to prospects_2025

Revision ID: 003
Revises: 002
Create Date: 2025-03-25 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'prospects_2025',
        sa.Column('nhl_player_id', sa.Integer(), nullable=True),
    )
    op.create_index(
        'ix_prospects_2025_nhl_player_id',
        'prospects_2025',
        ['nhl_player_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_prospects_2025_nhl_player_id', table_name='prospects_2025')
    op.drop_column('prospects_2025', 'nhl_player_id')
