from __future__ import annotations

from types import SimpleNamespace

from app.agent.embeddings import prospect_to_text, cosine_similarity


def _prospect(**overrides):
    base = {
        "name": "Test Prospect",
        "position": "D",
        "nationality": "SWE",
        "draft_league": "SHL",
        "draft_league_tier": 1,
        "css_ranking": 7,
        "css_category": "EUR skater",
        "points_per_game": 0.82,
        "games_played": 42,
        "goals": 10,
        "assists": 24,
        "age_at_draft": 18.1,
        "height_cm": 191,
        "weight_kg": 88,
        "ppg_prev_season": 0.56,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _stat_row(**overrides):
    base = {
        "league": "SHL",
        "season_type": "pre_draft",
        "games_played": 42,
        "goals": 10,
        "assists": 24,
        "points": 34,
        "points_per_game": 0.82,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_prospect_to_text_contains_key_fields():
    text = prospect_to_text(_prospect())

    assert "Test Prospect" in text
    assert "D" in text  # position
    assert "SHL" in text  # league


def test_prospect_to_text_with_stat_rows():
    stat_rows = [
        _stat_row(points_per_game=0.82, points=34),
        _stat_row(league="J20 Nationell", season_type="previous",
                  games_played=38, goals=7, assists=14, points=21, points_per_game=0.55),
    ]
    text = prospect_to_text(_prospect(), stat_rows)

    assert "Test Prospect" in text
    assert len(text) > 50


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == -1.0
