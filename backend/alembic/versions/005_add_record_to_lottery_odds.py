"""add wins/losses/otl/points to lottery_odds

Revision ID: 005
Revises: 004
Create Date: 2025-03-26 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('lottery_odds', sa.Column('wins', sa.Integer(), nullable=True))
    op.add_column('lottery_odds', sa.Column('losses', sa.Integer(), nullable=True))
    op.add_column('lottery_odds', sa.Column('otl', sa.Integer(), nullable=True))
    op.add_column('lottery_odds', sa.Column('points', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('lottery_odds', 'points')
    op.drop_column('lottery_odds', 'otl')
    op.drop_column('lottery_odds', 'losses')
    op.drop_column('lottery_odds', 'wins')
