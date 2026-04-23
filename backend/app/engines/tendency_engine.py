"""
GM tendency analysis engine.
Analyzes a GM's historical draft picks to compute their drafting fingerprint.

Two ML-backed techniques applied here:

  Recency weighting (temporal decay)
    Picks from recent years count more than older ones. A GM's behavior
    from 2010 is a weaker signal for 2025 than their 2023 picks.
    Weight = RECENCY_DECAY ^ (reference_year - pick_year)
    At RECENCY_DECAY=0.85: last year=1.0, 5yr ago=0.44, 10yr ago=0.20

  Bayesian shrinkage (empirical Bayes)
    With few picks, GM-specific estimates are noisy. Shrink toward the
    population prior (all GMs combined) proportional to sample size.
    alpha = n / (n + SHRINKAGE_K)
    At SHRINKAGE_K=30: 10 picks → 25% GM / 75% prior
                       30 picks → 50/50
                      100 picks → 77% GM / 23% prior
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.orm import Session

from app.models import GeneralManager, DraftPickHistorical, GMTendencyProfile
from app.constants import infer_league_key, nat_group

logger = logging.getLogger(__name__)

# ── Weighting parameters ──────────────────────────────────────────────────────

ROUND_WEIGHTS  = {1: 7, 2: 5, 3: 3}   # rounds 4+ default to 1
RECENCY_DECAY  = 0.85                  # exponential decay per year
SHRINKAGE_K    = 30                    # prior confidence in "equivalent picks"


def _get_round_weight(round_num: int) -> int:
    return ROUND_WEIGHTS.get(round_num, 1)


# ── Population prior computation ─────────────────────────────────────────────

def _compute_priors(all_picks: list) -> dict:
    """
    Compute population-level distributions from all historical picks.
    This is the prior used for Bayesian shrinkage: what does a generic GM
    look like when we have no information about them specifically?
    """
    pos_counts:    dict[str, float] = defaultdict(float)
    league_counts: dict[str, float] = defaultdict(float)
    nat_counts:    dict[str, float] = defaultdict(float)

    for p in all_picks:
        w = _get_round_weight(p.round or 4)
        if p.position:
            pos_counts[p.position] += w
        if p.draft_league or p.draft_league_tier:
            key = infer_league_key(p.draft_league or "", p.draft_league_tier)
            league_counts[key] += w
        if p.nationality:
            nat_counts[nat_group(p.nationality)] += w

    def _norm(d: dict) -> dict:
        total = sum(d.values())
        return {k: v / total for k, v in d.items()} if total > 0 else {}

    return {
        "position":    _norm(pos_counts),
        "league":      _norm(league_counts),
        "nationality": _norm(nat_counts),
    }


# ── Bayesian shrinkage ────────────────────────────────────────────────────────

def _shrink(gm_weights: dict, prior: dict, n_effective: float) -> dict:
    """
    Blend GM-specific weights toward the population prior.

    alpha = n / (n + k): rises from 0 (no data) toward 1 (lots of data).
    All keys from both dicts are preserved so the result covers the full
    vocabulary even if the GM never drafted a player of some type.
    """
    alpha = n_effective / (n_effective + SHRINKAGE_K)
    all_keys = set(gm_weights) | set(prior)
    return {
        k: round(alpha * gm_weights.get(k, 0.0) + (1 - alpha) * prior.get(k, 0.0), 6)
        for k in all_keys
    }


# ── Core tendency computation ─────────────────────────────────────────────────

def compute_gm_tendency(gm_id: int, db: Session, priors: dict | None = None) -> dict:
    """
    Compute tendency profile for a GM from their historical draft picks.

    Applies two ML-backed adjustments:
      - Recency weighting: recent picks weighted more than old ones
      - Bayesian shrinkage: blends toward population prior based on sample size

    priors: population-level distributions from _compute_priors(). If None,
    they are computed on-the-fly from all picks in the DB (slower — pass them
    in from compute_all_gm_tendencies for batch processing).

    Returns a dict with:
      position_weights, league_weights, nationality_weights,
      avg_ranking_deviation, total_picks
    """
    gm = db.query(GeneralManager).filter(GeneralManager.id == gm_id).first()
    if not gm:
        logger.warning(f"GM not found for gm_id={gm_id}")
        return _empty_tendency()

    # Gather picks across all career stints (career portability)
    all_gm_ids = [
        g.id for g in db.query(GeneralManager)
        .filter(GeneralManager.name == gm.name)
        .all()
    ]
    picks = db.query(DraftPickHistorical).filter(
        DraftPickHistorical.gm_id.in_(all_gm_ids)
    ).all()

    if not picks:
        logger.warning(f"No draft picks found for {gm.name} (gm_id={gm_id})")
        return _empty_tendency()

    # Compute priors on-the-fly if not provided
    if priors is None:
        all_picks = db.query(DraftPickHistorical).all()
        priors = _compute_priors(all_picks)

    reference_year = max(p.year for p in picks)

    df = pd.DataFrame([{
        "position":         p.position,
        "nationality":      p.nationality,
        "draft_league":     p.draft_league,
        "draft_league_tier": p.draft_league_tier,
        "round":            p.round,
        "overall_pick":     p.overall_pick,
        "year":             p.year,
    } for p in picks])

    # Combined weight = round importance × recency decay
    df["weight"] = df.apply(
        lambda row: _get_round_weight(row["round"]) * (RECENCY_DECAY ** (reference_year - row["year"])),
        axis=1,
    )

    # ── Raw weighted distributions ────────────────────────────────────────────

    pos_df = df[df["position"].notna()].copy()
    if not pos_df.empty:
        pos_weighted = pos_df.groupby("position")["weight"].sum()
        raw_pos = (pos_weighted / pos_weighted.sum()).to_dict()
    else:
        raw_pos = {}

    league_df = df[df["draft_league_tier"].notna()].copy()
    if not league_df.empty:
        league_df["league_label"] = league_df.apply(
            lambda row: infer_league_key(row["draft_league"] or "", row["draft_league_tier"]),
            axis=1,
        )
        league_weighted = league_df.groupby("league_label")["weight"].sum()
        raw_league = (league_weighted / league_weighted.sum()).to_dict()
    else:
        raw_league = {}

    nat_df = df[df["nationality"].notna()].copy()
    if not nat_df.empty:
        nat_df["nat_group"] = nat_df["nationality"].map(nat_group)
        nat_weighted = nat_df.groupby("nat_group")["weight"].sum()
        raw_nat = (nat_weighted / nat_weighted.sum()).to_dict()
    else:
        raw_nat = {}

    # ── Bayesian shrinkage toward population prior ────────────────────────────
    # n_effective = sum of weights (not raw count) to reflect quality of evidence
    n_effective = df["weight"].sum()

    position_weights    = _shrink(raw_pos,    priors["position"],    n_effective)
    league_weights      = _shrink(raw_league, priors["league"],      n_effective)
    nationality_weights = _shrink(raw_nat,    priors["nationality"], n_effective)

    # ── Ranking deviation ─────────────────────────────────────────────────────
    round_midpoints = {1: 16, 2: 48, 3: 80, 4: 112}
    deviations = [
        p.overall_pick - round_midpoints.get(p.round, 120)
        for p in picks if p.round in round_midpoints
    ]
    avg_ranking_deviation = round(sum(deviations) / len(deviations), 4) if deviations else 0.0

    return {
        "position_weights":    position_weights,
        "league_weights":      league_weights,
        "nationality_weights": nationality_weights,
        "avg_ranking_deviation": avg_ranking_deviation,
        "total_picks":         len(picks),
    }


# ── Archetype labels ──────────────────────────────────────────────────────────
# Five named archetypes used across the UI and agent responses.
ARCHETYPE_LABELS = ["BPA", "need-based", "safe", "euro-scout", "analytics"]

# Minimum GMs required to fit a GMM; below this fall back to rule-based.
_GMM_MIN_SAMPLES = 10

# Canonical archetype centroids in (eur_share, top2_concentration) space.
# Used to name unlabeled GMM clusters by finding the nearest centroid.
_ARCHETYPE_CENTROIDS: dict[str, tuple[float, float]] = {
    "BPA":        (0.33, 0.50),   # balanced — league-average EUR, moderate positional spread
    "need-based": (0.33, 0.72),   # high top-2 positional concentration
    "safe":       (0.12, 0.52),   # very low EUR share
    "euro-scout": (0.58, 0.50),   # high EUR share, balanced positions
    "analytics":  (0.38, 0.43),   # above-average EUR, low positional concentration
}


def _tendency_to_vec(tendency: dict) -> tuple[float, float]:
    """Extract the 2D feature vector used for archetype clustering."""
    pos_w = tendency.get("position_weights", {})
    nat_w = tendency.get("nationality_weights", {})
    eur_share = nat_w.get("NORDIC", 0.0) + nat_w.get("SLAVIC", 0.0) + nat_w.get("EUR_OTHER", 0.0)
    top2 = sum(sorted(pos_w.values(), reverse=True)[:2]) if pos_w else 0.0
    return eur_share, top2


def _label_cluster(centroid: tuple[float, float]) -> str:
    """Map a GMM cluster centroid to the nearest named archetype."""
    import math
    best_label = "BPA"
    best_dist  = float("inf")
    for label, ref in _ARCHETYPE_CENTROIDS.items():
        dist = math.sqrt((centroid[0] - ref[0]) ** 2 + (centroid[1] - ref[1]) ** 2)
        if dist < best_dist:
            best_dist  = dist
            best_label = label
    return best_label


def fit_archetype_clusters(tendencies: list[dict]) -> dict | None:
    """
    Fit a 5-component Gaussian Mixture Model on GM tendency vectors.

    Returns a mapping {cluster_id: archetype_label} if fitting succeeds,
    or None if there are too few samples for a reliable fit.

    Each GM is represented as (eur_share, top2_positional_concentration).
    GMs with < 15 picks are excluded — their shrunk vectors are too close to
    the population prior to carry useful cluster signal.

    The GMM is fit with covariance_type="full" and n_init=10 to avoid local
    minima. Cluster→label assignment uses nearest-centroid matching against
    the canonical _ARCHETYPE_CENTROIDS so labels stay interpretable.
    """
    try:
        from sklearn.mixture import GaussianMixture
        import numpy as np
    except ImportError:
        logger.warning("sklearn not available — archetype clustering requires scikit-learn")
        return None

    qualified = [t for t in tendencies if t.get("total_picks", 0) >= 15]
    if len(qualified) < _GMM_MIN_SAMPLES:
        logger.info(
            "fit_archetype_clusters: only %d qualified GMs (need %d) — skipping GMM",
            len(qualified), _GMM_MIN_SAMPLES,
        )
        return None

    X = np.array([_tendency_to_vec(t) for t in qualified])

    n_components = min(5, len(qualified))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        random_state=42,
        max_iter=300,
    )
    gmm.fit(X)

    # Map each cluster centroid to its nearest named archetype
    cluster_labels: dict[int, str] = {}
    used_labels: set[str] = set()
    # Sort clusters by eur_share so ties resolve deterministically
    centroid_order = sorted(range(n_components), key=lambda i: gmm.means_[i][0])
    for cluster_id in centroid_order:
        cx, cy = float(gmm.means_[cluster_id][0]), float(gmm.means_[cluster_id][1])
        label = _label_cluster((cx, cy))
        # If two clusters map to the same label, append index to distinguish
        if label in used_labels:
            label = f"{label}_{cluster_id}"
        used_labels.add(label)
        cluster_labels[cluster_id] = label

    logger.info(
        "fit_archetype_clusters: GMM fitted on %d GMs, %d components, cluster_labels=%s",
        len(qualified), n_components, cluster_labels,
    )
    return {"gmm": gmm, "cluster_labels": cluster_labels}


def classify_archetype(tendency: dict, gmm_result: dict | None = None) -> str:
    """
    Classify a GM's drafting archetype from their historical pick profile.

    When gmm_result is provided (fitted via fit_archetype_clusters), uses the
    GMM cluster assignment — boundaries are data-driven from the actual GM
    population rather than manually tuned thresholds.

    Falls back to rule-based classification when:
      - gmm_result is None (not enough GMs to fit, or sklearn unavailable)
      - GM has fewer than 15 picks (shrunk toward prior, unreliable cluster)

    Archetypes:
      BPA        — Best player available, balanced positional spread
      need-based — Concentrates picks at 1-2 positions
      safe       — Strong NA bias, avoids European prospects
      euro-scout — Actively targets European talent
      analytics  — Low positional concentration + above-average EUR targeting
    """
    total = tendency.get("total_picks", 0)

    if total < 15:
        return "BPA"

    # ── GMM path ──────────────────────────────────────────────────────────────
    if gmm_result is not None:
        try:
            import numpy as np
            vec = np.array([_tendency_to_vec(tendency)])
            cluster_id = int(gmm_result["gmm"].predict(vec)[0])
            label = gmm_result["cluster_labels"].get(cluster_id, "BPA")
            # Strip cluster-index suffix added to resolve label collisions
            return label.split("_")[0] if label.split("_")[0] in ARCHETYPE_LABELS else label
        except Exception as exc:
            logger.warning("classify_archetype.gmm_failed error=%s — falling back to rules", exc)

    # ── Rule-based fallback ───────────────────────────────────────────────────
    eur_share, top2 = _tendency_to_vec(tendency)

    if top2 > 0.65:
        return "need-based"
    if eur_share < 0.20:
        return "safe"
    if eur_share > 0.47:
        return "euro-scout"
    if top2 < 0.55 and eur_share > 0.30:
        return "analytics"
    return "BPA"


def compute_all_gm_tendencies(db: Session) -> list[GMTendencyProfile]:
    """
    Compute and upsert tendency profiles for all active GMs.

    Computes the population prior once from all historical picks, then passes
    it to each GM's tendency computation — avoids N redundant full-table scans.

    Archetype classification uses a GMM fitted on the full tendency population
    when enough GMs are available (>= _GMM_MIN_SAMPLES with 15+ picks). This
    makes cluster boundaries data-driven rather than manually tuned thresholds.
    Falls back to rule-based classification when sklearn is unavailable or the
    sample is too small.
    """
    # One query for priors — shared across all GMs
    all_picks = db.query(DraftPickHistorical).all()
    priors = _compute_priors(all_picks)
    logger.info(f"Population priors computed from {len(all_picks)} historical picks")

    active_gms = db.query(GeneralManager).filter(GeneralManager.is_active).all()

    # Compute all tendency dicts first so we can fit the GMM on the full population
    # before assigning archetypes — GMM needs the complete set of vectors.
    tendency_by_gm: dict[int, dict] = {}
    for gm in active_gms:
        logger.info(f"Computing tendency for GM: {gm.name} (id={gm.id})")
        tendency_by_gm[gm.id] = compute_gm_tendency(gm.id, db, priors=priors)

    # Fit GMM once on all tendency vectors; None = fall back to rules per GM
    gmm_result = fit_archetype_clusters(list(tendency_by_gm.values()))

    profiles = []

    for gm in active_gms:
        tendency = tendency_by_gm[gm.id]
        archetype = classify_archetype(tendency, gmm_result=gmm_result)

        existing = db.query(GMTendencyProfile).filter(
            GMTendencyProfile.gm_id == gm.id
        ).first()

        if existing:
            existing.position_weights    = tendency["position_weights"]
            existing.league_weights      = tendency["league_weights"]
            existing.nationality_weights = tendency["nationality_weights"]
            existing.avg_ranking_deviation = tendency["avg_ranking_deviation"]
            existing.tendency_archetype  = archetype
            existing.computed_at         = datetime.now(timezone.utc)
            profiles.append(existing)
        else:
            profile = GMTendencyProfile(
                gm_id=gm.id,
                computed_at=datetime.now(timezone.utc),
                position_weights=tendency["position_weights"],
                league_weights=tendency["league_weights"],
                nationality_weights=tendency["nationality_weights"],
                avg_ranking_deviation=tendency["avg_ranking_deviation"],
                tendency_archetype=archetype,
            )
            db.add(profile)
            profiles.append(profile)

    db.commit()
    logger.info(f"Computed tendencies for {len(profiles)} GMs")

    # Validate computed archetypes against known ground-truth for well-documented GMs.
    # Mismatches are logged as warnings — they don't block the computation but surface
    # threshold-tuning opportunities.
    validation = validate_archetypes(profiles)
    mismatches = [v for v in validation if not v["match"]]
    if validation:
        logger.info(
            "archetype_validation total=%d matched=%d mismatched=%d",
            len(validation), len(validation) - len(mismatches), len(mismatches),
        )
    for v in mismatches:
        logger.warning(
            "archetype_mismatch gm=%r expected=%r computed=%r note=%s",
            v["gm"], v["expected"], v["computed"], v.get("note", ""),
        )

    return profiles


def _empty_tendency() -> dict:
    return {
        "position_weights":    {},
        "league_weights":      {},
        "nationality_weights": {},
        "avg_ranking_deviation": 0.0,
        "total_picks":         0,
    }


# ── Archetype ground-truth validation ─────────────────────────────────────────
#
# Manually curated expected archetypes for well-known GMs, derived from publicly
# available draft analyses (The Athletic, EliteProspects, Hockey Reference, etc.).
# Used to sanity-check that the Bayesian tendency engine produces sensible outputs.
#
# Format: { "gm_name_substring": "expected_archetype" }
# Matching is case-insensitive substring search so "Dubas" matches "Kyle Dubas".
#
# Sources:
#   - Kyle Dubas: well-documented analytics-first approach (The Athletic, 2019-2024);
#     heavy European scouting (Auston Matthews, Timothy Liljegren, etc.)
#   - Lou Lamoriello: conservative NA-first, defence-heavy, traditional scouting
#     (decades of documented picks across NJD/TOR/NYI)
#   - Bill Zito: need-based drafting history in Columbus/Florida (2019–present)
#   - Don Sweeney: defenseman-heavy under Joe Nieuwendyk influence, safe NA bias
#   - Jim Nill: balanced BPA, slight Euro-scout lean (Modano influence, Stars)
#   - Chris Drury: early tenure, small sample → BPA default is correct
KNOWN_ARCHETYPES: dict[str, str] = {
    "Dubas":       "analytics",   # Euro-heavy, balanced positional spread
    "Lamoriello":  "safe",        # NA-first, conservative — EUR share historically < 15%
    "Zito":        "need-based",  # Columbus: D-heavy; Florida: forward-heavy
    "Sweeney":     "safe",        # BOS: strong NA bias, defenseman preference
    "Nill":        "BPA",         # DAL: balanced, slight Euro lean
    "Drury":       "BPA",         # NYR: small sample → shrinks to BPA
    "Benning":     "safe",        # VAN era: NA-heavy, physical profile
    "Yzerman":     "euro-scout",  # DET: long history of European first-rounders
}


def validate_archetypes(profiles: list) -> list[dict]:
    """
    Cross-check computed archetypes against KNOWN_ARCHETYPES ground-truth.

    Called after compute_all_gm_tendencies() to surface mismatches.
    Returns a list of validation results — one per GM in KNOWN_ARCHETYPES
    that appears in the computed profiles.

    A mismatch doesn't necessarily mean the model is wrong: GMs change
    philosophy over time, or have a short tenure with few picks. But
    systematic mismatches indicate the archetype thresholds need tuning.

    Example output:
        [
          {"gm": "Kyle Dubas", "expected": "analytics", "computed": "analytics", "match": True},
          {"gm": "Lou Lamoriello", "expected": "safe", "computed": "BPA", "match": False,
           "note": "total_picks=8 — short tenure, shrunk to prior"},
        ]
    """
    results = []
    for profile in profiles:
        gm_name = getattr(profile, "gm_name", None) or ""
        computed = getattr(profile, "tendency_archetype", "BPA") or "BPA"
        # total_picks tracked via tendency computation, not persisted directly

        for name_key, expected in KNOWN_ARCHETYPES.items():
            if name_key.lower() in gm_name.lower():
                match = computed == expected
                result = {
                    "gm":       gm_name,
                    "expected": expected,
                    "computed": computed,
                    "match":    match,
                }
                if not match:
                    result["note"] = (
                        "Mismatch — check EUR share and top-2 positional concentration. "
                        "May be explained by short tenure (picks shrink toward BPA prior)."
                    )
                results.append(result)

    return results


