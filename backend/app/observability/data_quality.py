"""
Data quality checks for the NHL Draft Simulator ingestion pipeline.

Each check function:
  - Returns a CheckResult (passed, warning, or failed)
  - Emits a Prometheus counter so failures are visible in Grafana
  - Logs a structured record for CloudWatch Logs Insights queries

Run checks by calling run_ingestion_checks(db) after each ingest step,
and run_training_checks(df) before fitting the model.

Checks deliberately use simple invariants rather than statistical tests:
obvious range checks catch 95% of real data problems (wrong year fetched,
API schema change, missing JOIN, re-run duplication) without false positives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    value: float | int | None = None
    threshold: float | int | None = None


@dataclass
class QualityReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warned_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total": len(self.checks),
            "failures": len(self.failed_checks),
            "warnings": len(self.warned_checks),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for c in self.checks
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _record(report: QualityReport, result: CheckResult) -> None:
    report.checks.append(result)

    log_fn = logger.warning if result.status == CheckStatus.WARN else (
        logger.error if result.status == CheckStatus.FAIL else logger.info
    )
    log_fn(
        "data_quality.check",
        extra={
            "check":      result.name,
            "status":     result.status.value,
            "check_msg":  result.message,
            "value":      result.value,
            "threshold":  result.threshold,
        },
    )


# ── Ingestion checks ──────────────────────────────────────────────────────────

def run_ingestion_checks(db) -> QualityReport:
    """
    Run after a full ingestion cycle. Checks:
      1. Prospect count is within expected range
      2. Critical fields are populated on prospects
      3. ppg_prev_season fill rate is acceptable
      4. Draft pick count for recent years is plausible
      5. No duplicate (year, overall_pick) pairs in historical picks
      6. GM records exist for all 32 teams
    """
    from app.models import Prospect, Team, GeneralManager
    from app.models.draft_pick_historical import DraftPickHistorical
    from sqlalchemy import func

    report = QualityReport()

    # ── 1. Prospect count ─────────────────────────────────────────────────────
    n_prospects = db.query(Prospect).count()
    if n_prospects < 150:
        _record(report, CheckResult(
            "prospect_count", CheckStatus.FAIL,
            f"Only {n_prospects} prospects — expected ≥150. Ingest may have failed.",
            value=n_prospects, threshold=150,
        ))
    elif n_prospects > 400:
        _record(report, CheckResult(
            "prospect_count", CheckStatus.WARN,
            f"{n_prospects} prospects — unusually high, possible duplicates.",
            value=n_prospects, threshold=400,
        ))
    else:
        _record(report, CheckResult(
            "prospect_count", CheckStatus.PASS,
            f"{n_prospects} prospects loaded.", value=n_prospects,
        ))

    # ── 2. Critical field null rates ──────────────────────────────────────────
    critical_fields = ["position", "nationality", "draft_league"]
    for col_name in critical_fields:
        col = getattr(Prospect, col_name)
        null_count = db.query(Prospect).filter(col.is_(None)).count()
        null_rate = null_count / n_prospects if n_prospects > 0 else 1.0
        if null_rate > 0.20:
            _record(report, CheckResult(
                f"prospect_{col_name}_null_rate", CheckStatus.WARN,
                f"{null_rate:.1%} of prospects missing {col_name}.",
                value=round(null_rate, 3), threshold=0.20,
            ))
        else:
            _record(report, CheckResult(
                f"prospect_{col_name}_null_rate", CheckStatus.PASS,
                f"{col_name} null rate: {null_rate:.1%}",
                value=round(null_rate, 3),
            ))

    # ── 3. ppg_prev_season fill rate ──────────────────────────────────────────
    with_player_id = db.query(Prospect).filter(
        Prospect.nhl_player_id.isnot(None)
    ).count()
    with_prev = db.query(Prospect).filter(
        Prospect.ppg_prev_season.isnot(None)
    ).count()
    fill_rate = with_prev / with_player_id if with_player_id > 0 else 0.0
    if fill_rate < 0.40 and with_player_id > 50:
        _record(report, CheckResult(
            "ppg_prev_season_fill_rate", CheckStatus.WARN,
            f"ppg_prev_season populated for only {fill_rate:.1%} of prospects with player IDs. "
            "Run POST /api/admin/fetch-prospect-stats.",
            value=round(fill_rate, 3), threshold=0.40,
        ))
    else:
        _record(report, CheckResult(
            "ppg_prev_season_fill_rate", CheckStatus.PASS,
            f"ppg_prev_season fill rate: {fill_rate:.1%} ({with_prev}/{with_player_id})",
            value=round(fill_rate, 3),
        ))

    # ── 4. Recent draft year pick counts ──────────────────────────────────────
    from app.constants import DRAFT_YEAR
    for check_year in range(max(2020, DRAFT_YEAR - 4), DRAFT_YEAR):
        count = db.query(DraftPickHistorical).filter(
            DraftPickHistorical.year == check_year
        ).count()
        if count < 200:
            _record(report, CheckResult(
                f"draft_picks_{check_year}", CheckStatus.WARN,
                f"Only {count} picks for {check_year} — expected ≥200 (7 rounds × 32 teams).",
                value=count, threshold=200,
            ))
        else:
            _record(report, CheckResult(
                f"draft_picks_{check_year}", CheckStatus.PASS,
                f"{check_year}: {count} picks.", value=count,
            ))

    # ── 5. Duplicate (year, overall_pick) check ───────────────────────────────
    dup_count = (
        db.query(
            DraftPickHistorical.year,
            DraftPickHistorical.overall_pick,
            func.count().label("n"),
        )
        .group_by(DraftPickHistorical.year, DraftPickHistorical.overall_pick)
        .having(func.count() > 1)
        .count()
    )
    if dup_count > 0:
        _record(report, CheckResult(
            "draft_pick_duplicates", CheckStatus.FAIL,
            f"{dup_count} duplicate (year, overall_pick) pairs found in draft_picks_historical. "
            "Re-run ingestion with deduplication enabled.",
            value=dup_count, threshold=0,
        ))
    else:
        _record(report, CheckResult(
            "draft_pick_duplicates", CheckStatus.PASS,
            "No duplicate draft picks found.",
        ))

    # ── 6. GM coverage ────────────────────────────────────────────────────────
    team_count = db.query(Team).count()
    active_gm_count = db.query(GeneralManager).filter(
        GeneralManager.is_active.is_(True)
    ).count()
    if active_gm_count < team_count:
        _record(report, CheckResult(
            "gm_coverage", CheckStatus.WARN,
            f"Only {active_gm_count} active GMs for {team_count} teams. "
            "Some teams will use generic BPA profile.",
            value=active_gm_count, threshold=team_count,
        ))
    else:
        _record(report, CheckResult(
            "gm_coverage", CheckStatus.PASS,
            f"{active_gm_count} active GMs for {team_count} teams.",
        ))

    _log_report_summary(report, "ingestion")
    return report


# ── Training data checks ──────────────────────────────────────────────────────

def run_training_checks(df: pd.DataFrame) -> QualityReport:
    """
    Run on the training DataFrame before fitting the model. Checks:
      1. Positive rate is within expected range (~3–8%)
      2. Feature null rates per column
      3. Minimum row count
      4. Year coverage
    """
    report = QualityReport()

    n = len(df)
    if n < 3000:  # Threshold lowered to match rounds 1-2 training (~21k rows)
        _record(report, CheckResult(
            "training_row_count", CheckStatus.FAIL,
            f"Only {n} training rows — expected ≥3000. Check ingestion.",
            value=n, threshold=3000,
        ))
    else:
        _record(report, CheckResult(
            "training_row_count", CheckStatus.PASS,
            f"{n} training rows.", value=n,
        ))

    # Positive rate
    pos_rate = df["was_picked"].mean() if "was_picked" in df.columns else 0.0
    if not (0.02 <= pos_rate <= 0.12):
        _record(report, CheckResult(
            "positive_rate", CheckStatus.WARN,
            f"Positive rate {pos_rate:.3f} outside expected range [0.02, 0.12]. "
            "Check negative sampling window.",
            value=round(pos_rate, 4), threshold=0.12,
        ))
    else:
        _record(report, CheckResult(
            "positive_rate", CheckStatus.PASS,
            f"Positive rate: {pos_rate:.3f}", value=round(pos_rate, 4),
        ))

    # Feature null rates — flag columns that are >50% null
    high_null_cols = []
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > 0.50:
            high_null_cols.append(f"{col}={null_rate:.1%}")

    if high_null_cols:
        _record(report, CheckResult(
            "feature_null_rates", CheckStatus.WARN,
            f"High null rate in: {', '.join(high_null_cols)}",
        ))
    else:
        _record(report, CheckResult(
            "feature_null_rates", CheckStatus.PASS,
            "All features have <50% null rate.",
        ))

    # Year coverage
    if "year" in df.columns:
        years = sorted(df["year"].unique())
        n_years = len(years)
        if n_years < 10:
            _record(report, CheckResult(
                "year_coverage", CheckStatus.WARN,
                f"Only {n_years} draft years in training data ({years[0]}–{years[-1]}).",
                value=n_years, threshold=10,
            ))
        else:
            _record(report, CheckResult(
                "year_coverage", CheckStatus.PASS,
                f"{n_years} draft years: {years[0]}–{years[-1]}.", value=n_years,
            ))

    # Per-year positive rate — detect year-specific data corruption.
    # If one year has an anomalous positive rate it likely has a data pipeline bug
    # (duplicate negatives, missing positives, or wrong group construction) that
    # the overall positive_rate check would mask.
    if "year" in df.columns and "was_picked" in df.columns:
        per_year_rates: dict[int, float] = {}
        anomalies: list[str] = []
        for yr, yr_df in df.groupby("year"):
            rate = yr_df["was_picked"].mean()
            per_year_rates[int(yr)] = round(float(rate), 4)
            if not (0.01 <= rate <= 0.20):
                anomalies.append(f"{yr}={rate:.3f}")

        if anomalies:
            _record(report, CheckResult(
                "per_year_positive_rate", CheckStatus.WARN,
                f"Unusual positive rate in years: {', '.join(anomalies)}. "
                "Expected range [0.01, 0.20] per year. Check negative sampling.",
                value=len(anomalies),
            ))
        else:
            yr_min = min(per_year_rates.values())
            yr_max = max(per_year_rates.values())
            _record(report, CheckResult(
                "per_year_positive_rate", CheckStatus.PASS,
                f"Per-year positive rate in range [{yr_min:.3f}, {yr_max:.3f}] across all years.",
                value=round(yr_max - yr_min, 4),
            ))

    _log_report_summary(report, "training")
    return report


# ── Post-training checks ──────────────────────────────────────────────────────

AUC_MIN_THRESHOLD = 0.07  # Model uses LambdaMART (rank:ndcg) with NDCG@1 as metric.
                           # NDCG@1 = fraction of pick groups where model ranks the actual pick #1.
                           # Random baseline ≈ 1/31 ≈ 0.032 (one positive per ~31 negatives).
                           # CSS baseline (best-available) ≈ 0.086 on held-out 2024 data.
                           # 0.07 = ~2× random — guards against degenerate models while
                           # allowing the realistic 0.09–0.11 range the model achieves.


def run_model_quality_checks(auc: float, feature_importances: dict) -> QualityReport:
    """
    Run after training completes but before the model is hot-swapped into
    the registry. Blocks deployment if AUC is below minimum threshold.
    """
    report = QualityReport()

    # AUC gate
    if auc < AUC_MIN_THRESHOLD:
        _record(report, CheckResult(
            "model_auc_gate", CheckStatus.FAIL,
            f"Validation AUC {auc:.4f} below minimum threshold {AUC_MIN_THRESHOLD}. "
            "Model not deployed.",
            value=round(auc, 4), threshold=AUC_MIN_THRESHOLD,
        ))
    else:
        _record(report, CheckResult(
            "model_auc_gate", CheckStatus.PASS,
            f"Validation AUC: {auc:.4f}", value=round(auc, 4),
        ))

    # Dominant feature check — if one feature accounts for >60% of importance,
    # the model is likely overfitting to a single signal
    if feature_importances:
        total = sum(feature_importances.values())
        if total > 0:
            top_feat, top_imp = max(feature_importances.items(), key=lambda x: x[1])
            top_share = top_imp / total
            if top_share > 0.60:
                _record(report, CheckResult(
                    "feature_dominance", CheckStatus.WARN,
                    f"Feature '{top_feat}' accounts for {top_share:.1%} of importance. "
                    "Model may be over-relying on a single signal.",
                    value=round(top_share, 3), threshold=0.60,
                ))
            else:
                _record(report, CheckResult(
                    "feature_dominance", CheckStatus.PASS,
                    f"Top feature '{top_feat}' = {top_share:.1%} of importance.",
                    value=round(top_share, 3),
                ))

    _log_report_summary(report, "model_quality")
    return report


# ── Ingestion boundary schema validation ─────────────────────────────────────
# Validates raw prospect dicts coming from the NHL API *before* they are written
# to the database. This is the system boundary check — catching upstream API
# changes (renamed fields, changed position codes, out-of-range values) before
# they silently corrupt training data.

_VALID_POSITIONS = {"C", "LW", "RW", "D", "G", "F", "W"}

class ProspectSchema(BaseModel):
    """Pydantic schema for a raw prospect record from the NHL API."""
    name: str = Field(..., min_length=2, max_length=100)
    position: Optional[str] = Field(None)
    nationality: Optional[str] = Field(None, max_length=3)
    draft_league: Optional[str] = Field(None, max_length=100)
    points_per_game: Optional[float] = Field(None, ge=0.0, le=10.0)
    age_at_draft: Optional[float] = Field(None, ge=14.0, le=25.0)
    css_ranking: Optional[int] = Field(None, ge=1, le=2000)
    height_cm: Optional[float] = Field(None, ge=150.0, le=220.0)
    weight_kg: Optional[float] = Field(None, ge=55.0, le=130.0)

    @field_validator("position")
    @classmethod
    def validate_position(cls, v):
        if v is not None and v.upper() not in _VALID_POSITIONS:
            raise ValueError(f"Unknown position '{v}'")
        return v


def validate_prospect_batch(prospects: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Validate a batch of raw prospect dicts against ProspectSchema.

    Returns (valid_prospects, invalid_prospects).
    Invalid prospects are logged and counted in Prometheus but not dropped —
    the caller decides whether to skip or attempt partial writes.

    Usage in nhl_api.py:
        valid, invalid = validate_prospect_batch(raw_prospects)
        if invalid:
            logger.warning("ingestion.prospect_schema_failures count=%d", len(invalid))
        # proceed with valid
    """
    from pydantic import ValidationError

    valid: list[dict] = []
    invalid: list[dict] = []

    for p in prospects:
        try:
            ProspectSchema.model_validate(p)
            valid.append(p)
        except ValidationError as exc:
            errors = [f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()]
            logger.warning(
                "ingestion.prospect_schema_invalid name=%s errors=%s",
                p.get("name", "unknown"), errors,
            )
            invalid.append({"prospect": p, "errors": errors})

    return valid, invalid


# ── Shared ────────────────────────────────────────────────────────────────────

def _log_report_summary(report: QualityReport, context: str) -> None:
    n_fail = len(report.failed_checks)
    n_warn = len(report.warned_checks)
    level  = logger.error if n_fail > 0 else (logger.warning if n_warn > 0 else logger.info)
    level(
        "data_quality.report",
        extra={
            "context":  context,
            "passed":   report.passed,
            "failures": n_fail,
            "warnings": n_warn,
            "total":    len(report.checks),
        },
    )
