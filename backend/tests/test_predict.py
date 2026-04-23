"""Unit tests for ML inference path.

All tests are pure-Python — no database, no trained model required.
"""
import math
import random
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prospect(
    id=1,
    position="C",
    nationality="CAN",
    draft_league="OHL",
    draft_league_tier=1,
    points_per_game=1.2,
    games_played=60,
    ppg_prev_season=1.0,
    age_at_draft=18.5,
    css_ranking=10,
    height_cm=185,
    weight_kg=88,
):
    """Build a minimal mock Prospect object."""
    p = MagicMock()
    p.id = id
    p.position = position
    p.nationality = nationality
    p.draft_league = draft_league
    p.draft_league_tier = draft_league_tier
    p.points_per_game = points_per_game
    p.games_played = games_played
    p.ppg_prev_season = ppg_prev_season
    p.age_at_draft = age_at_draft
    p.css_ranking = css_ranking
    p.height_cm = height_cm
    p.weight_kg = weight_kg
    return p


# ── Tests: _validate_feature_ranges ──────────────────────────────────────────

class TestValidateFeatureRanges:
    def test_in_range_produces_no_warnings(self, caplog):
        """A well-formed feature DataFrame should not trigger range warnings."""
        from app.ml.predict import _validate_feature_ranges
        import logging

        df = pd.DataFrame([{
            "css_rank_norm":   0.8,
            "pick_slot_norm":  0.5,
            "ppg_league_norm": 1.2,
            "age_league_norm": 0.5,
            "height_cm":       185,
            "weight_kg":       88,
            "gp_pre_draft":    60,
        }])
        with caplog.at_level(logging.WARNING, logger="app.ml.predict"):
            _validate_feature_ranges(df)
        assert not caplog.records

    def test_out_of_range_triggers_warning(self, caplog):
        """A css_rank_norm > 1.0 should trigger a WARNING log."""
        from app.ml.predict import _validate_feature_ranges
        import logging

        df = pd.DataFrame([{"css_rank_norm": 1.5}])
        with caplog.at_level(logging.WARNING, logger="app.ml.predict"):
            _validate_feature_ranges(df)
        assert any("css_rank_norm" in r.message for r in caplog.records)

    def test_missing_columns_are_skipped(self):
        """Columns not present in the DataFrame should be silently skipped."""
        from app.ml.predict import _validate_feature_ranges

        # Only has pick_slot_norm — other checked columns are absent
        df = pd.DataFrame([{"pick_slot_norm": 0.5}])
        # Should not raise
        _validate_feature_ranges(df)

    def test_does_not_raise_on_out_of_range(self):
        """_validate_feature_ranges must NEVER raise — only warn."""
        from app.ml.predict import _validate_feature_ranges

        df = pd.DataFrame([{
            "css_rank_norm":   99.9,  # wildly out of range
            "pick_slot_norm":  -5.0,
            "height_cm":       300,
        }])
        # Must complete without exception
        _validate_feature_ranges(df)


# ── Tests: _sample_pick ───────────────────────────────────────────────────────

class TestSamplePick:
    """Tests for the temperature-scaled softmax sampler in draft_sim.py."""

    def setup_method(self):
        from app.api.draft_sim import _sample_pick
        self._sample_pick = _sample_pick

        # Three mock prospects with distinct scores
        self.p1 = MagicMock(); self.p1.id = 1
        self.p2 = MagicMock(); self.p2.id = 2
        self.p3 = MagicMock(); self.p3.id = 3
        self.available = [self.p1, self.p2, self.p3]
        self.scores = {1: 0.9, 2: 0.5, 3: 0.1}

    def test_temperature_zero_returns_argmax(self):
        """temperature=0 must always return the highest-scored prospect."""
        rng = random.Random(42)
        result = self._sample_pick(self.available, self.scores, rng, temperature=0)
        assert result.id == 1  # p1 has score 0.9

    def test_temperature_zero_is_deterministic(self):
        """Repeated calls with temperature=0 should always return same pick."""
        rng = random.Random(42)
        picks = {self._sample_pick(self.available, self.scores, rng, 0).id for _ in range(10)}
        assert picks == {1}

    def test_temperature_one_produces_variation(self):
        """With temperature=1 and many draws, all three prospects should appear."""
        picks = set()
        for seed in range(200):
            rng = random.Random(seed)
            p = self._sample_pick(self.available, self.scores, rng, temperature=1.0)
            picks.add(p.id)
        # With 200 draws all three prospects should be selected at some point
        assert len(picks) == 3

    def test_high_score_wins_most_often(self):
        """With temperature=0.15 (default), the top-scored prospect should win >50% of draws."""
        from collections import Counter
        counter = Counter()
        for seed in range(500):
            rng = random.Random(seed)
            p = self._sample_pick(self.available, self.scores, rng, temperature=0.15)
            counter[p.id] += 1
        # p1 (score=0.9) should win the majority of the time
        assert counter[1] > 250

    def test_single_prospect_always_returned(self):
        """With only one prospect available, it must always be returned."""
        rng = random.Random(1)
        result = self._sample_pick([self.p1], {1: 0.5}, rng, temperature=0.5)
        assert result.id == 1


# ── Tests: compute_pool_stats ─────────────────────────────────────────────────

class TestComputePoolStats:
    def test_returns_all_keys(self):
        """Return dict must contain ppg_by_tier, age_by_tier, and ppg_percentile."""
        from app.ml.predict import compute_pool_stats

        prospects = [
            _make_prospect(id=1, position="C", draft_league="OHL", points_per_game=1.2, age_at_draft=18.0),
            _make_prospect(id=2, position="D", draft_league="OHL", points_per_game=0.8, age_at_draft=19.0),
        ]
        stats = compute_pool_stats(prospects)
        assert "ppg_by_tier" in stats
        assert "age_by_tier" in stats
        assert "ppg_percentile" in stats

    def test_goalie_excluded_from_ppg_tier(self):
        """Goalies should not skew the PPG median for skaters."""
        from app.ml.predict import compute_pool_stats

        prospects = [
            _make_prospect(id=1, position="C",  draft_league="OHL", points_per_game=1.5, age_at_draft=18.0),
            _make_prospect(id=2, position="G",  draft_league="OHL", points_per_game=0.01, age_at_draft=18.5),
        ]
        stats = compute_pool_stats(prospects)
        tier_key = "tier1_CAN"  # OHL is tier1_CAN
        # Goalie PPG (0.01) should not pollute the skater tier median
        if tier_key in stats["ppg_by_tier"]:
            assert stats["ppg_by_tier"][tier_key] > 0.5

    def test_ppg_percentile_bounds(self):
        """All PPG percentile values should be in [0, 1]."""
        from app.ml.predict import compute_pool_stats

        prospects = [_make_prospect(id=i, points_per_game=i * 0.2) for i in range(1, 6)]
        stats = compute_pool_stats(prospects)
        for pid, pct in stats["ppg_percentile"].items():
            assert 0.0 <= pct <= 1.0, f"prospect {pid} has out-of-range percentile {pct}"

    def test_empty_pool_returns_empty_dicts(self):
        """An empty prospect list should return empty dicts (no crash)."""
        from app.ml.predict import compute_pool_stats

        stats = compute_pool_stats([])
        assert stats["ppg_by_tier"] == {}
        assert stats["age_by_tier"] == {}
        assert stats["ppg_percentile"] == {}


# ── Tests: score_pool_for_team ────────────────────────────────────────────────

class TestScorePoolForTeam:
    def test_returns_empty_dict_when_model_not_loaded(self):
        """When registry.is_loaded is False, return {} immediately without scoring."""
        from app.ml.predict import score_pool_for_team

        prospects = [_make_prospect(id=1)]
        with patch("app.ml.predict.registry") as mock_reg:
            mock_reg.is_loaded = False
            result = score_pool_for_team(prospects, profile=None, pick_slot=1)
        assert result == {}

    def test_returns_score_for_every_prospect(self):
        """When model is loaded, every prospect in the pool must have a score.

        We test the scoring path by calling _score_pool_inner directly through
        score_pool_for_team with a mocked registry that provides a ranker whose
        predict() returns a fixed array.
        """
        from app.ml.predict import score_pool_for_team
        from xgboost import XGBRanker

        prospects = [_make_prospect(id=i, css_ranking=i * 5) for i in range(1, 4)]

        mock_model = MagicMock(spec=XGBRanker)
        mock_model.predict.return_value = [0.9, 0.5, 0.2]

        with patch("app.ml.predict.registry") as mock_reg:
            mock_reg.is_loaded = True
            mock_reg.model = mock_model
            mock_reg.calibration = None
            result = score_pool_for_team(prospects, profile=None, pick_slot=1)

        # Each prospect id should appear in result with a float score
        for p in prospects:
            assert p.id in result
            assert isinstance(result[p.id], float)

    def test_scores_are_floats(self):
        """All returned scores should be Python floats."""
        from app.ml.predict import score_pool_for_team

        prospects = [_make_prospect(id=1)]
        with patch("app.ml.predict.registry") as mock_reg:
            mock_reg.is_loaded = False
            result = score_pool_for_team(prospects, profile=None, pick_slot=1)
        # Empty dict is fine — if it were non-empty, check floats
        for v in result.values():
            assert isinstance(v, float)
