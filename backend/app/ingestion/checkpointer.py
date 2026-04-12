"""
Pipeline checkpointer for the ingestion pipeline.

Problem:
  If POST /api/admin/ingest fails midway through (network error on step 3 of 6,
  NHL API rate limit, etc.), there is no record of which steps completed.
  A re-run restarts from scratch — re-fetching and re-upserting data that was
  already written successfully, wasting time and API quota.

Solution:
  PipelineCheckpointer wraps each pipeline step. On success it writes the step's
  completion to IngestionRun.pipeline_steps (JSON column). On re-run with the
  same run_id, already-completed steps are skipped automatically and execution
  resumes from the first failed/unstarted step.

Usage:
    ck = PipelineCheckpointer(db, run_id)

    with ck.step("fetch_teams") as ctx:
        count = fetch_all_teams(db)
        ctx["teams_fetched"] = count     # stored in pipeline_steps detail

    with ck.step("fetch_prospects") as ctx:
        count = fetch_all_prospects(db)

    # If "fetch_teams" already succeeded, ck.step("fetch_teams") is a no-op
    # and the block body is skipped entirely.

Integration:
  Called from app/ingestion/nhl_api.py run_full_ingestion().
  IngestionRun row must already exist before calling PipelineCheckpointer.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class StepContext(dict):
    """Dict subclass returned by the step() context manager for capturing metadata."""


class PipelineCheckpointer:
    """
    Tracks step completion for a single IngestionRun and supports resumable execution.

    Parameters
    ----------
    db : Session
        Active SQLAlchemy session for reading/writing IngestionRun.pipeline_steps.
    run_id : str
        The run_id of the IngestionRun row to checkpoint against.
    force : bool
        If True, ignore existing checkpoints and re-run all steps.
        Useful when the data pipeline changes and cached step results are stale.
    """

    def __init__(self, db: Session, run_id: str, force: bool = False) -> None:
        self._db = db
        self._run_id = run_id
        self._force = force
        self._steps: dict = self._load_steps()

    def _load_steps(self) -> dict:
        from app.models.ingestion_run import IngestionRun
        run = self._db.query(IngestionRun).filter(IngestionRun.run_id == self._run_id).first()
        if run and run.pipeline_steps:
            return dict(run.pipeline_steps)
        return {}

    def _save_steps(self) -> None:
        from app.models.ingestion_run import IngestionRun
        self._db.query(IngestionRun).filter(
            IngestionRun.run_id == self._run_id
        ).update({"pipeline_steps": self._steps})
        self._db.commit()

    def is_done(self, step_name: str) -> bool:
        """Return True if this step already completed successfully."""
        if self._force:
            return False
        state = self._steps.get(step_name, {})
        return state.get("status") == "ok"

    @contextmanager
    def step(self, step_name: str) -> Iterator[StepContext]:
        """
        Context manager for a single pipeline step.

        If the step already succeeded (status="ok"), the block is skipped and
        a StepContext with {"skipped": True} is yielded immediately.

        On success: saves status="ok" with timestamp to pipeline_steps.
        On exception: saves status="failed" with error detail, then re-raises.

        Example:
            with ck.step("fetch_teams") as ctx:
                result = fetch_all_teams(db)
                ctx["rows"] = result
        """
        ctx = StepContext()

        if self.is_done(step_name):
            logger.info("checkpoint.skip step=%s (already completed)", step_name)
            ctx["skipped"] = True
            ctx["prior_detail"] = self._steps.get(step_name, {})
            yield ctx
            return

        logger.info("checkpoint.start step=%s", step_name)
        try:
            yield ctx
            # Step completed successfully
            self._steps[step_name] = {
                "status": "ok",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "detail": {k: v for k, v in ctx.items() if k != "skipped"},
            }
            self._save_steps()
            logger.info("checkpoint.done step=%s detail=%s", step_name, dict(ctx))
        except Exception as exc:
            self._steps[step_name] = {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "detail": str(exc)[:500],
            }
            self._save_steps()
            logger.error("checkpoint.failed step=%s error=%s", step_name, exc)
            raise

    def summary(self) -> dict:
        """Return a summary of all step statuses."""
        return {
            "run_id": self._run_id,
            "steps":  self._steps,
            "completed": [k for k, v in self._steps.items() if v.get("status") == "ok"],
            "failed":    [k for k, v in self._steps.items() if v.get("status") == "failed"],
        }
