"""Add ingestion_runs table for data lineage tracking

Revision ID: 007
Revises: 006
Create Date: 2026-03-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'ingestion_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='running'),

        # What was fetched
        sa.Column('teams_upserted', sa.Integer(), nullable=True),
        sa.Column('gms_upserted', sa.Integer(), nullable=True),
        sa.Column('picks_upserted', sa.Integer(), nullable=True),
        sa.Column('prospects_upserted', sa.Integer(), nullable=True),
        sa.Column('prospect_stats_updated', sa.Integer(), nullable=True),

        # Source/provenance
        sa.Column('nhl_api_version', sa.String(length=20), nullable=True, server_default='v1'),
        sa.Column('triggered_by', sa.String(length=50), nullable=True),

        # Quality
        sa.Column('quality_passed', sa.Boolean(), nullable=True),
        sa.Column('quality_warnings', sa.JSON(), nullable=True),
        sa.Column('quality_failures', sa.JSON(), nullable=True),

        # Error info
        sa.Column('error_message', sa.String(length=500), nullable=True),

        # Link to model trained from this ingestion
        sa.Column('model_trained_at', sa.String(length=64), nullable=True),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id'),
    )
    op.create_index('ix_ingestion_runs_run_id', 'ingestion_runs', ['run_id'], unique=True)


def downgrade():
    op.drop_index('ix_ingestion_runs_run_id', table_name='ingestion_runs')
    op.drop_table('ingestion_runs')
