"""Add season trend stats, remove games_played_nhl

Revision ID: 006
Revises: 005
Create Date: 2026-03-28
"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    # draft_picks_historical: drop games_played_nhl, add gp_pre_draft + ppg_prev_season
    op.drop_column('draft_picks_historical', 'games_played_nhl')
    op.add_column('draft_picks_historical', sa.Column('gp_pre_draft', sa.Integer(), nullable=True))
    op.add_column('draft_picks_historical', sa.Column('ppg_prev_season', sa.Float(), nullable=True))

    # prospects_2025: add ppg_prev_season
    op.add_column('prospects_2025', sa.Column('ppg_prev_season', sa.Float(), nullable=True))

    # drop unused tables (IF EXISTS — table may not exist on fresh databases)
    op.execute('DROP TABLE IF EXISTS roster_current')


def downgrade():
    op.add_column('draft_picks_historical', sa.Column('games_played_nhl', sa.Integer(), nullable=True, server_default='0'))
    op.drop_column('draft_picks_historical', 'gp_pre_draft')
    op.drop_column('draft_picks_historical', 'ppg_prev_season')
    op.drop_column('prospects_2025', 'ppg_prev_season')
    # roster_current not restored on downgrade — data is not recoverable
