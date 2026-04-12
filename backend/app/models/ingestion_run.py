from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.sql import func
from app.database import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="running")  # running / success / failed

    # What was fetched
    teams_upserted = Column(Integer, nullable=True)
    gms_upserted = Column(Integer, nullable=True)
    picks_upserted = Column(Integer, nullable=True)
    prospects_upserted = Column(Integer, nullable=True)
    prospect_stats_updated = Column(Integer, nullable=True)

    # Source/provenance
    nhl_api_version = Column(String(20), nullable=True, default="v1")
    triggered_by = Column(String(50), nullable=True)  # "cron", "api", "lambda", "manual"

    # Quality
    quality_passed = Column(Boolean, nullable=True)
    quality_warnings = Column(JSON, nullable=True)   # list of warning strings
    quality_failures = Column(JSON, nullable=True)   # list of failure strings

    # Error info
    error_message = Column(String(500), nullable=True)

    # Link to model trained from this ingestion (if any)
    model_trained_at = Column(String(64), nullable=True)

    # ── Step-level checkpointing ──────────────────────────────────────────────
    # JSON dict: {step_name: {"status": "ok"|"failed"|"skipped", "completed_at": ISO, "detail": str}}
    # Populated by PipelineCheckpointer as each step completes.
    # On re-run, completed steps are skipped automatically — the pipeline resumes
    # from the first non-completed step instead of starting from scratch.
    #
    # Example:
    #   {
    #     "fetch_teams":     {"status": "ok", "completed_at": "2025-04-02T14:30:00Z"},
    #     "fetch_prospects": {"status": "ok", "completed_at": "2025-04-02T14:31:00Z"},
    #     "fetch_stats":     {"status": "failed", "completed_at": null, "detail": "429 rate limit"},
    #   }
    pipeline_steps = Column(JSON, nullable=True, default=dict)
