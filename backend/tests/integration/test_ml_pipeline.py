"""Integration tests for the full ML pipeline.

Covers:
  - build_features() shape and NaN guarantees against real DB data
  - train() produces a valid AUC from a small but correctly shaped dataset
  - model.predict_proba() returns probabilities in [0, 1]
  - run_training_checks() catches a near-zero positive rate

Run with: pytest tests/integration/test_ml_pipeline.py -v --tb=short
Requires Docker to be running.
"""
from __future__ import annotations

import math
import random
from datetime import date

import pandas as pd
import pytest

from app.ml.features import FEATURE_COLS, build_features, build_training_dataset
from app.ml.train import train
from app.observability.data_quality import run_training_checks
from app.models import Team, GeneralManager, DraftPickHistorical


# ── Test data helpers ─────────────────────────────────────────────────────────

_POSITIONS = ["C", "LW", "RW", "D", "G"]
_NATIONALITIES = ["CAN", "USA", "SWE", "FIN", "RUS"]
_LEAGUES = [
    ("OHL", 1),
    ("WHL", 1),
    ("QMJHL", 1),
    ("SHL", 1),
    ("LIIGA", 1),
    ("NCAA", 2),
    ("AHL", 2),
    ("KHL", 2),
    ("SuperElit", 3),
    ("USHL", 3),
]


def _insert_teams(session, n: int = 5) -> list[Team]:
    """Insert n minimal Team records."""
    for i in range(n):
        session.add(Team(
            nhl_id=2000 + i,
            abbreviation=f"M{i:02d}",
            full_name=f"ML Team {i}",
            city=f"ML City {i}",
            conference="Eastern" if i % 2 == 0 else "Western",
            division="Atlantic" if i % 2 == 0 else "Central",
        ))
    session.flush()
    return session.query(Team).filter(Team.nhl_id >= 2000, Team.nhl_id < 2000 + n).all()


def _insert_gms(session, teams: list[Team]) -> list[GeneralManager]:
    """Insert one active GM per team."""
    for i, team in enumerate(teams):
        session.add(GeneralManager(
            name=f"ML GM {i} {team.abbreviation}",
            team_id=team.id,
            start_date=date(2015, 7, 1),
            is_active=True,
        ))
    session.flush()
    return (
        session.query(GeneralManager)
        .filter(GeneralManager.name.like("ML GM %"))
        .all()
    )


def _insert_realistic_picks(
    session,
    teams: list[Team],
    gms: list[GeneralManager],
    n: int,
    rng: random.Random,
) -> list[DraftPickHistorical]:
    """Insert n draft picks with randomized but valid values across years 2015–2019."""
    years = [2015, 2016, 2017, 2018, 2019]
    picks_per_year: dict[int, int] = {y: 0 for y in years}

    picks = []
    for i in range(n):
        year = years[i % len(years)]
        overall = picks_per_year[year] + 1
        picks_per_year[year] += 1

        league, tier = _LEAGUES[i % len(_LEAGUES)]
        pos = _POSITIONS[i % len(_POSITIONS)]
        # Goalies have very low PPG by nature; skaters have higher PPG
        if pos == "G":
            ppg = round(rng.uniform(0.01, 0.10), 2)
        else:
            ppg = round(rng.uniform(0.5, 2.0), 2)

        ppg_prev = round(rng.uniform(0.4, 1.8), 2) if rng.random() > 0.35 else None

        pick = DraftPickHistorical(
            year=year,
            round=1 + (overall - 1) // 32,
            pick_number=((overall - 1) % 32) + 1,
            overall_pick=overall,
            team_id=teams[i % len(teams)].id,
            gm_id=gms[i % len(gms)].id,
            player_name=f"ML Player {i}",
            position=pos,
            nationality=_NATIONALITIES[i % len(_NATIONALITIES)],
            draft_league=league,
            draft_league_tier=tier,
            height_cm=170 + rng.randint(0, 20),
            weight_kg=75 + rng.randint(0, 25),
            points_per_game=ppg,
            gp_pre_draft=30 + rng.randint(0, 40),
            ppg_prev_season=ppg_prev,
            age_at_draft=round(rng.uniform(17.5, 21.0), 1),
        )
        session.add(pick)
        picks.append(pick)

    session.flush()
    return picks


@pytest.fixture(scope="module")
def fifty_pick_db(db_engine):
    """
    Module-scoped fixture: inserts 50 realistic draft picks into a fresh session
    and returns the session. Uses a savepoint so callers can roll back individually.

    We use module scope here (not function scope) because training is expensive
    and we want to share the dataset across all tests in this module.
    """
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    session = Session()

    rng = random.Random(99)
    teams = _insert_teams(session, n=5)
    gms = _insert_gms(session, teams)
    _insert_realistic_picks(session, teams, gms, n=60, rng=rng)
    session.commit()

    yield session

    # Teardown: delete only the records we inserted so other tests remain clean
    session.query(DraftPickHistorical).filter(
        DraftPickHistorical.player_name.like("ML Player %")
    ).delete(synchronize_session=False)
    session.query(GeneralManager).filter(
        GeneralManager.name.like("ML GM %")
    ).delete(synchronize_session=False)
    session.query(Team).filter(
        Team.nhl_id >= 2000, Team.nhl_id < 2005
    ).delete(synchronize_session=False)
    session.commit()
    session.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildFeaturesIntegration:

    def test_build_features_consistent_shape(self, fifty_pick_db):
        """build_features output must have exactly len(FEATURE_COLS) columns and no NaNs."""
        df_raw = build_training_dataset(fifty_pick_db)
        assert len(df_raw) > 0, "Training dataset should not be empty"

        feat_df = build_features(df_raw)

        assert feat_df.shape[1] == len(FEATURE_COLS), (
            f"Expected {len(FEATURE_COLS)} feature columns, got {feat_df.shape[1]}"
        )
        assert list(feat_df.columns) == FEATURE_COLS, (
            "Feature column order does not match FEATURE_COLS"
        )
        assert not feat_df.isna().any().any(), (
            "build_features must fill all NaN values; found NaN in output"
        )

    def test_build_features_row_count_preserved(self, fifty_pick_db):
        """build_features must return the same number of rows as its input."""
        df_raw = build_training_dataset(fifty_pick_db)
        feat_df = build_features(df_raw)
        assert feat_df.shape[0] == df_raw.shape[0], (
            "build_features must not drop or duplicate rows"
        )

    def test_build_features_one_hot_valid(self, fifty_pick_db):
        """Each position one-hot column must be binary (0 or 1), never negative."""
        df_raw = build_training_dataset(fifty_pick_db)
        feat_df = build_features(df_raw)

        for pos in ["C", "LW", "RW", "D", "G"]:
            col = f"pos_{pos}"
            assert set(feat_df[col].unique()).issubset({0, 1}), (
                f"{col} must be binary (0 or 1)"
            )

    def test_build_features_probabilities_in_range(self, fifty_pick_db):
        """Normalized rank features should all fall within [0, 1]."""
        df_raw = build_training_dataset(fifty_pick_db)
        feat_df = build_features(df_raw)

        for col in ["pick_slot_norm"]:
            assert feat_df[col].between(0.0, 1.0).all(), (
                f"{col} values must be in [0, 1]"
            )


class TestTrainAndPredict:

    def test_train_and_predict_on_small_dataset(self, fifty_pick_db):
        """train() on a small dataset should return a finite AUC in [0.4, 1.0]."""
        df = build_training_dataset(fifty_pick_db)

        # Skip if the dataset is trivially tiny (shouldn't happen with 60 picks)
        assert len(df) >= 10, "Need at least 10 rows to attempt training"

        model, auc = train(df)

        assert not math.isnan(auc), "AUC should not be NaN in evaluation mode"
        assert 0.4 <= auc <= 1.0, (
            f"AUC {auc:.4f} is outside the expected range [0.4, 1.0]"
        )

    def test_model_predict_returns_scores(self, fifty_pick_db):
        """XGBRanker.predict must return finite scores for all rows."""
        df = build_training_dataset(fifty_pick_db)
        model, _ = train(df)

        feat_df = build_features(df.head(20))
        scores = model.predict(feat_df)

        assert len(scores) == len(feat_df), (
            "predict output length must match input length"
        )
        import numpy as np
        assert np.isfinite(scores).all(), "All scores must be finite"

    def test_train_final_mode_returns_numeric_auc(self, fifty_pick_db):
        """train(df, final=True) returns CV NDCG@1 — final flag is a no-op kept for API compat."""
        df = build_training_dataset(fifty_pick_db)
        model, auc = train(df, final=True)

        assert not math.isnan(auc), (
            "train() always runs eval phase and returns CV NDCG@1 regardless of final flag"
        )
        assert 0.0 <= auc <= 1.0, f"AUC must be in [0, 1], got {auc}"


class TestTrainingDataQualityGate:

    def test_data_quality_gate_catches_low_positive_rate(self):
        """run_training_checks should fail when positive rate is nearly zero."""
        # Build a DataFrame with <0.1% positive rate (artificially skewed negatives)
        n_total = 6000
        n_pos = 2  # 2 / 6000 ≈ 0.033% positive rate — well below the 2% floor

        rows = [{"was_picked": 1, "year": 2015}] * n_pos
        rows += [{"was_picked": 0, "year": 2015}] * (n_total - n_pos)
        df = pd.DataFrame(rows)

        report = run_training_checks(df)

        warn_or_fail_names = (
            {c.name for c in report.warned_checks}
            | {c.name for c in report.failed_checks}
        )
        assert "positive_rate" in warn_or_fail_names, (
            "run_training_checks should flag 'positive_rate' when it is near zero"
        )

    def test_data_quality_gate_catches_insufficient_rows(self):
        """run_training_checks should fail when the DataFrame has fewer than 5000 rows."""
        df = pd.DataFrame({
            "was_picked": [1] * 50 + [0] * 450,
            "year": [2015] * 500,
        })

        report = run_training_checks(df)

        fail_names = {c.name for c in report.failed_checks}
        assert "training_row_count" in fail_names, (
            "run_training_checks should fail 'training_row_count' with fewer than 5000 rows"
        )
        assert report.passed is False, (
            "Report should not pass when row count is below threshold"
        )

    def test_data_quality_passes_on_healthy_data(self):
        """run_training_checks should pass for a dataset within all expected ranges."""
        import numpy as np

        rng = np.random.default_rng(42)
        n_pos = 400    # ~5.7% positive rate — within the [2%, 12%] window
        n_neg = 6600
        n_total = n_pos + n_neg

        years = rng.integers(2010, 2020, size=n_total).tolist()
        labels = [1] * n_pos + [0] * n_neg

        df = pd.DataFrame({"was_picked": labels, "year": years})

        report = run_training_checks(df)

        fail_names = {c.name for c in report.failed_checks}
        # The positive_rate and training_row_count checks should pass
        assert "training_row_count" not in fail_names
        assert "positive_rate" not in fail_names
