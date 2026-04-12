"""Draft simulation endpoint."""
import json
import logging
import math
import random
from collections import Counter, defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, get_redis
from app.models import Team, GeneralManager, GMTendencyProfile, Prospect2025
from app.ml.predict import score_pool_for_team, compute_pool_stats
from app.ml.registry import registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/draft", tags=["draft"])

_redis, REDIS_OK = get_redis()


DEFAULT_TEMPERATURE = 0.15  # lower = more decisive; 1.0 = pure proportional sampling
                             # 0.15 strongly favors high-scored prospects while keeping
                             # realistic variation — at 0.4 with 224 prospects even
                             # CSS#188 had non-trivial selection probability

# Dynamic temperature: early picks are nearly deterministic BPA, late picks have
# more variance as teams factor in positional need and fit.
# Exponential curve keeps picks 1-10 very tight, then ramps quickly after pick 15.
# At pick 1:  T=0.15 * 0.10 = 0.015 (near-deterministic)
# At pick 10: T=0.15 * 0.35 = 0.053
# At pick 20: T=0.15 * 0.80 = 0.120
# At pick 32: T=0.15 * 2.00 = 0.300
def _pick_temperature(base_temperature: float, pick_num: int, total_picks: int = 32) -> float:
    """Scale temperature exponentially by pick position.

    Early picks (1-10): near-deterministic — teams take the best player available.
    Late picks (20-32): meaningful variance — need, fit, and upside gambles.
    """
    t = (pick_num - 1) / max(total_picks - 1, 1)   # 0.0 at pick 1, 1.0 at last pick
    scale = 0.10 * math.exp(math.log(20) * t)        # 0.10× → 2.0× exponential ramp
    return base_temperature * scale


class SimulateDraftRequest(BaseModel):
    lottery_result: list[int]   # ordered team_ids (picks 1-32)
    seed: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE  # controls pick randomness

    @field_validator("temperature")
    @classmethod
    def temperature_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be >= 0 (use 0 for deterministic argmax)")
        return v


def _sample_pick(
    available: list,
    scores: dict[int, float],
    rng: random.Random,
    temperature: float,
) -> object:
    """
    Sample a prospect from the available pool using temperature-scaled softmax.

    temperature < 1: sharpens the distribution — the model's top pick wins most
                     of the time, but plausible alternatives still occur.
    temperature = 1: sample exactly proportional to raw model scores.
    temperature → 0: collapses to argmax (deterministic).

    Steps:
      1. Divide each score by temperature (amplifies differences when T < 1).
      2. Exponentiate → unnormalized probabilities.
      3. Normalize to sum to 1.
      4. Sample one prospect using those weights.
    """
    if temperature <= 0:
        # Degenerate case: pure argmax
        return max(available, key=lambda p: scores.get(p.id, 0.0))

    scaled = [scores.get(p.id, 0.0) / temperature for p in available]
    # Subtract max for numerical stability before exp
    max_s = max(scaled)
    weights = [math.exp(s - max_s) for s in scaled]
    return rng.choices(available, weights=weights, k=1)[0]


@router.post("/simulate")
async def simulate_draft(body: SimulateDraftRequest, db: Session = Depends(get_db)):
    """
    Simulate the full 32-pick round 1 draft.

    Scores each pick using the trained XGBoost model via the in-memory registry.
    All DB queries happen once upfront — the pick loop is pure in-memory.
    Results are cached in Redis keyed by seed + lottery order.
    """
    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="No trained model loaded. Run POST /api/ml/train first.",
        )

    seed = body.seed if body.seed is not None else random.randint(0, 2**31)
    rng  = random.Random(seed)

    from app.observability.metrics import (
        SIMULATIONS_TOTAL, SIMULATION_DURATION, CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL,
    )
    import time as _time

    # Include model hash in cache key so stale simulations from the previous model
    # version are automatically bypassed after a hot-swap — no explicit cache flush.
    from app.ml.registry import registry as _registry
    _model_hash = _registry.model_hash
    cache_key = f"draft_v11:{_model_hash}:{seed}:{body.temperature}:{'-'.join(map(str, body.lottery_result))}"
    if REDIS_OK and _redis:
        cached = _redis.get(cache_key)
        if cached:
            SIMULATIONS_TOTAL.labels(status="cache_hit").inc()
            CACHE_HITS_TOTAL.inc()
            return json.loads(cached)

    CACHE_MISSES_TOTAL.inc()
    _sim_start = _time.perf_counter()

    # ── Load everything once ──────────────────────────────────────────────────
    prospects = (
        db.query(Prospect2025)
        .order_by(Prospect2025.css_ranking.nullslast())
        .all()
    )
    if not prospects:
        raise HTTPException(404, "No prospects loaded. Run ingestion first.")

    teams_map: dict[int, Team] = {t.id: t for t in db.query(Team).all()}
    gms_map: dict[int, GeneralManager] = {
        g.team_id: g
        for g in db.query(GeneralManager).filter(GeneralManager.is_active.is_(True)).all()
    }
    profiles_map: dict[int, GMTendencyProfile] = {
        p.gm_id: p for p in db.query(GMTendencyProfile).all()
    }

    # ── Precompute pool stats for league normalization (computed once) ────────
    pool_stats = compute_pool_stats(prospects)

    # ── Draft-state tracking ──────────────────────────────────────────────────
    pos_taken: Counter[str]                          = Counter()
    team_positions_drafted: defaultdict[int, Counter] = defaultdict(Counter)

    # ── Simulate ──────────────────────────────────────────────────────────────
    available = list(prospects)
    picks = []

    for pick_num, team_id in enumerate(body.lottery_result, 1):
        if not available:
            break
        team = teams_map.get(team_id)
        if not team:
            continue

        gm      = gms_map.get(team_id)
        profile = profiles_map.get(gm.id) if gm else None

        draft_state = {
            "pos_taken":      pos_taken,
            "team_positions": team_positions_drafted[team_id],
            "total_picked":   pick_num - 1,
        }

        scores = score_pool_for_team(available, profile, pick_num, draft_state, pool_stats)

        # Apply conformal calibration intervals if calibration data is available.
        # cal_intervals maps prospect_id → {in_prediction_set, nc_score, coverage}.
        cal_intervals = None
        if registry.calibration:
            from app.ml.calibration import apply_intervals
            cal_intervals = apply_intervals(scores, registry.calibration, alpha=0.10)

        effective_temp = _pick_temperature(body.temperature, pick_num, len(body.lottery_result))
        chosen = _sample_pick(available, scores, rng, effective_temp)

        # Update draft state before moving to next pick
        pos_taken[chosen.position or "F"] += 1
        team_positions_drafted[team_id][chosen.position or "F"] += 1

        available = [p for p in available if p.id != chosen.id]

        pick_data = {
            "pick":            pick_num,
            "team_id":         team_id,
            "team_name":       getattr(team, "full_name", team.abbreviation),
            "abbreviation":    team.abbreviation,
            "prospect_name":   chosen.name,
            "prospect_id":     chosen.id,
            "position":        chosen.position,
            "nationality":     chosen.nationality,
            "draft_league":    chosen.draft_league,
            "css_rank":        chosen.css_ranking,
            "points_per_game": chosen.points_per_game,
            "ml_score":        round(scores.get(chosen.id, 0.0), 4),
        }
        if cal_intervals:
            ci = cal_intervals.get(chosen.id, {})
            # in_prediction_set: True = model considered this a statistically
            # plausible pick at 90% confidence. False = a "surprise" selection.
            pick_data["in_prediction_set"] = ci.get("in_prediction_set")
            pick_data["nc_score"]          = ci.get("nc_score")
            pick_data["confidence"]        = ci.get("coverage")
        picks.append(pick_data)

    response = {"picks": picks, "seed": seed, "temperature": body.temperature, "total_picks": len(picks)}

    if REDIS_OK and _redis:
        _redis.setex(cache_key, 3600, json.dumps(response))

    SIMULATION_DURATION.observe(_time.perf_counter() - _sim_start)
    SIMULATIONS_TOTAL.labels(status="ok").inc()

    return response


# ── Draft summary endpoint ────────────────────────────────────────────────────

class DraftSummaryRequest(BaseModel):
    picks: list[dict]   # the picks array from /api/draft/simulate


@router.post("/summary")
async def draft_summary(body: DraftSummaryRequest):
    """
    Generate a natural-language summary of a completed draft simulation using Claude.

    Receives the picks array from /api/draft/simulate and returns a 2-3 paragraph
    analyst-style summary: biggest steals, notable reaches, position trends, and
    one team-specific spotlight.

    Returns {"summary": "..."} or {"summary": null, "error": "..."} if the LLM
    is unavailable.
    """
    from app.config import settings

    if not settings.ANTHROPIC_API_KEY:
        return {"summary": None, "error": "ANTHROPIC_API_KEY not configured"}

    if not body.picks:
        raise HTTPException(status_code=400, detail="picks array is empty")

    # Build a compact text representation of the draft board
    lines = ["2025 NHL Draft — Round 1 Results\n"]
    for p in body.picks:
        css = f"CSS #{p.get('css_rank')}" if p.get("css_rank") else "unranked"
        lines.append(
            f"Pick {p['pick']:>2}. {p['team_name']:<28} {p['prospect_name']:<22} "
            f"{p.get('position','?'):>2}  {css}  {p.get('draft_league','')}"
        )

    draft_text = "\n".join(lines)

    prompt = f"""{draft_text}

Please write a concise 2-3 paragraph analyst-style draft summary. Cover:
1. The biggest value picks (prospect drafted well below their CSS rank).
2. Any notable reaches (prospect drafted well above their CSS rank).
3. Overall positional trends across the draft (e.g. defense-heavy, center run early).
4. One team spotlight — a team that had a particularly strong or surprising draft.

Write in the style of a hockey analyst. Be specific — cite player names and pick numbers."""

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.content[0].text
        return {"summary": summary}
    except Exception as exc:
        logger.error("draft_summary.failed", extra={"error": str(exc)})
        return {"summary": None, "error": "Failed to generate summary"}


@router.post("/analysis")
async def draft_analysis(body: DraftSummaryRequest):
    """
    Generate a fully structured draft analysis using Claude's forced tool use.

    Unlike /summary (free-form text), this endpoint forces Claude to produce
    machine-readable JSON conforming to the DraftAnalysis schema. The frontend
    can render each field as a rich card:
      - steals / reaches  → pick highlight cards with deviation badges
      - position_trends   → bar chart or trend pills
      - team_spotlight    → team card with draft grade
      - headline / narrative → banner + closing text block

    Returns {"status": "ok", "analysis": DraftAnalysis} on success.
    Falls back to {"status": "error"} if LLM unavailable.
    """
    from app.config import settings
    from app.schemas.draft_analysis import DraftAnalysis

    if not settings.ANTHROPIC_API_KEY:
        return {"status": "error", "error": "ANTHROPIC_API_KEY not configured"}

    if not body.picks:
        raise HTTPException(status_code=400, detail="picks array is empty")

    # Build compact draft board text for context
    lines = ["2025 NHL Draft — Round 1 Results\n"]
    for p in body.picks:
        css   = f"CSS #{p.get('css_rank')}" if p.get("css_rank") else "unranked"
        dev   = ""
        if p.get("css_rank") and p.get("pick"):
            d = p["pick"] - p["css_rank"]
            dev = f" (dev={d:+d})" if d != 0 else ""
        lines.append(
            f"Pick {p['pick']:>2}. {p['team_name']:<28} {p['prospect_name']:<22} "
            f"{p.get('position','?'):>2}  {css}{dev}  {p.get('draft_league','')}"
        )
    draft_text = "\n".join(lines)

    # Force Claude to produce structured output via tool_choice
    analysis_tool = [{
        "name": "produce_draft_analysis",
        "description": "Produce a structured draft analysis with steals, reaches, position trends, and team spotlight.",
        "input_schema": DraftAnalysis.model_json_schema(),
    }]

    prompt = (
        f"{draft_text}\n\n"
        "Analyze this draft and call produce_draft_analysis with a complete structured analysis. "
        "Be specific — cite player names, pick numbers, and CSS ranks. "
        "deviation = pick_slot - css_rank (positive = steal, negative = reach)."
    )

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1200,
            tools=analysis_tool,
            tool_choice={"type": "tool", "name": "produce_draft_analysis"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the tool_use block input (already a dict from Claude)
        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise ValueError("Claude did not return a tool_use block")

        analysis = DraftAnalysis(**tool_block.input)
        return {"status": "ok", "analysis": analysis.model_dump()}

    except Exception as exc:
        logger.error("draft_analysis.failed", extra={"error": str(exc)})
        return {"status": "error", "error": "Failed to generate structured analysis"}


@router.post("/summary/stream")
async def draft_summary_stream(body: DraftSummaryRequest):
    """
    SSE streaming version of /summary. Emits tokens as they arrive from Claude.
    Frontend renders text progressively instead of waiting for the full response.
    """
    from app.config import settings

    if not settings.ANTHROPIC_API_KEY:
        async def _offline():
            yield f"data: {json.dumps({'token': 'AI Analysis unavailable — ANTHROPIC_API_KEY not configured.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(_offline(), media_type="text/event-stream")

    if not body.picks:
        raise HTTPException(status_code=400, detail="picks array is empty")

    lines = ["2025 NHL Draft — Round 1 Results\n"]
    for p in body.picks:
        css = f"CSS #{p.get('css_rank')}" if p.get("css_rank") else "unranked"
        lines.append(
            f"Pick {p['pick']:>2}. {p['team_name']:<28} {p['prospect_name']:<22} "
            f"{p.get('position','?'):>2}  {css}  {p.get('draft_league','')}"
        )
    draft_text = "\n".join(lines)
    prompt = f"""{draft_text}

Please write a concise 2-3 paragraph analyst-style draft summary. Cover:
1. The biggest value picks (prospect drafted well below their CSS rank).
2. Any notable reaches (prospect drafted well above their CSS rank).
3. Overall positional trends across the draft (e.g. defense-heavy, center run early).
4. One team spotlight — a team that had a particularly strong or surprising draft.

Write in the style of a hockey analyst. Be specific — cite player names and pick numbers."""

    import anthropic as _anthropic

    async def _stream():
        client = _anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        try:
            with client.messages.stream(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'token': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error("draft_summary_stream.failed", extra={"error": str(exc)})
            yield f"data: {json.dumps({'error': 'Stream failed.'})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
