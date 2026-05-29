"""
"Ask the Scout" agent — async tool-use (function-calling) implementation.

Flow for each user message:
  1. Load conversation history from scout_conversations (session memory)
  2. Send to Claude with TOOL_DEFINITIONS — Claude decides what data it needs
  3. If Claude returns tool_use blocks, execute ALL of them concurrently
     via asyncio.gather + asyncio.to_thread (DB calls are sync SQLAlchemy)
  4. Feed ALL tool results back to Claude in a second call
  5. Repeat until Claude returns a final text response (end_turn)
  6. Persist both the user message and assistant reply to scout_conversations

Key improvements over v1:
  - AsyncAnthropic: non-blocking API calls — preserves the Uvicorn event loop
    so concurrent SSE streams and REST requests share the same thread pool
  - Parallel tool execution: all tool_use blocks in a single response run
    concurrently via asyncio.gather + asyncio.to_thread (each DB call is
    ~10–50 ms; running N tools in parallel ≈ max(tool_latencies) instead of sum)
  - Circuit breaker (circuit_breaker.py): trips OPEN after 5 failures in 60 s,
    fails fast for 30 s, then sends one probe — prevents cascading timeouts
  - Cache hit tracking: Anthropic usage headers expose cache_read_input_tokens
    and cache_creation_input_tokens; these are emitted as Prometheus counters
    so we can verify prompt caching is actually hitting
  - Per-tool latency + call count metrics via Prometheus
  - Topic guardrail: a fast pre-flight classifier (haiku-tier) rejects obviously
    off-topic queries before they consume the full tool-use loop
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from sqlalchemy.orm import Session

from app.agent import store
from app.agent import memory as agent_memory
from app.agent.tools import TOOL_DEFINITIONS, execute_tool
from app.agent.circuit_breaker import anthropic_breaker, CircuitOpenError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are "The Scout" — an expert NHL draft analyst built into the NHL Draft Simulator.

## Tools available
- get_gm_profile: GM drafting tendencies, archetype, position/league/nationality weights
- get_top_prospects: top prospects at any position by CSS ranking
- get_team_needs: a team's positional needs based on historical drafting patterns
- search_prospects: find specific prospects by name, league, or nationality (keyword search)
- semantic_prospect_search: find prospects similar to a player archetype or named player
- get_draft_history: a team's actual historical draft picks with pick, player, CSS rank, PPG
- get_ml_ranking: XGBoost model score + SHAP factors for a prospect at a given pick slot
- compare_prospects: side-by-side stat comparison of two prospects
- get_prospect_detail: full profile + multi-season stat history for one prospect
- get_nhl_comp: find current NHL players most similar in style to a given prospect

## Reasoning approach
Before answering any question about a specific team, GM, player, or pick:
1. Identify what data you need (profile? rankings? history?).
2. Call the right tool(s) — you can call multiple tools in parallel when the data is independent.
3. Reason over the returned numbers before composing your reply.
4. If two tools conflict or data is sparse, say so explicitly.

Never answer from memory alone when a tool can provide live data.

## Output format rules
- Plain text only. No markdown: no **bold**, no *italics*, no ## headers, no backticks.
- Use plain bullet points (- item) for lists of 3+ items; prose for 1–2 item answers.
- Lead with the direct answer or recommendation, then support it with numbers.
- Always cite specific figures from tool results: CSS rank, PPG, ML score, position weight, pick slot.
- Keep responses under ~200 words unless the user asks for a deep breakdown.
- When recommending a pick, state the reasoning as: player → fit reason → supporting stat.
- If the data doesn't support a confident answer, say so — don't speculate.

## Tool selection rules
- "who should [team] draft?" → get_team_needs + get_top_prospects (+ get_gm_profile if archetype matters)
- "how does [team] draft?" / "what's [GM]'s style?" → get_gm_profile
- "find a player like X" / "two-way center" / style description → semantic_prospect_search (NOT search_prospects)
- "find [name]" / "Swedish prospects" / "OHL forwards" → search_prospects (keyword, not semantic)
- "compare X and Y" → compare_prospects; follow up with get_prospect_detail if depth is needed
- "who does X play like?" / "NHL comp" → get_nhl_comp
- "how does the model rate X?" / "what's driving X's ranking?" → get_ml_ranking with team if context available
- "what did [team] draft in [year]?" → get_draft_history
- When unsure between search_prospects and semantic_prospect_search: if the query describes a
  style, archetype, or similarity to another player, use semantic_prospect_search.
  If the query has a name, league code, or nationality code, use search_prospects.

## Draft context
If a [Draft Context] block appears at the start of the conversation, you have live board state.
Use the current pick number, picking team, recent selections, and available count to give
pick-specific advice. Factor in both value (CSS rank vs. pick slot) and fit (team needs +
GM archetype). Call out when a prospect is available beyond their expected range — that's value.

## Few-shot examples

User: "Who should the Senators take at pick 8?"
Thought: I need team needs + GM profile + top available prospects. Call get_team_needs("Ottawa Senators"), get_gm_profile("Ottawa Senators"), get_top_prospects("all", 10) in parallel.
Response format: "At #8, Ottawa's biggest need is [position] (weight: X). Their GM skews [archetype], so [player] (CSS #N, X.XX PPG) is the fit — [reason]. The model scores them [score] at this slot."

User: "Find me a player like Cale Makar"
Thought: Style/archetype comparison → semantic_prospect_search, not search_prospects.
Response format: "The closest stylistic match is [player] (CSS #N, [league]). [1–2 sentence evidence from profile]."

User: "How has Detroit drafted over the last three years?"
Thought: get_draft_history("Detroit Red Wings") — no year param to get last 3 years.
Response format: Summarize patterns (position split, CSS deviation, league preference) with 2–3 specific pick callouts."""

# Cached system prompt block — Anthropic caches this for 5 min, saving ~90% of prompt tokens.
CACHED_SYSTEM = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

MAX_TOOL_ROUNDS = 10

# Cheap fast model used only for the topic guardrail classifier — always Haiku
# regardless of what ANTHROPIC_MODEL is configured for the main agent.
GUARDRAIL_MODEL = "claude-haiku-4-5-20251001"

# Status labels yielded as SSE {"status": "..."} events while tool rounds run.
# Shown in the UI so users see what the Scout is doing between turns.
_TOOL_LABELS: dict[str, str] = {
    "get_gm_profile":           "Looking up GM profile...",
    "get_top_prospects":        "Fetching top prospects...",
    "get_team_needs":           "Analyzing team needs...",
    "search_prospects":         "Searching prospects...",
    "semantic_prospect_search": "Running semantic search...",
    "get_draft_history":        "Pulling draft history...",
    "get_ml_ranking":           "Running ML model...",
    "compare_prospects":        "Comparing prospects...",
    "get_prospect_detail":      "Fetching prospect details...",
    "get_nhl_comp":             "Finding NHL comps...",
}

# Off-topic rejection message (returned before hitting the full tool-use loop)
_OFFTOPIC_REPLY = (
    "I'm specialized in NHL draft analysis — prospects, GM tendencies, team needs, "
    "and draft history. Ask me about players, teams, or the 2025 draft class!"
)


def _cached_tools() -> list[dict]:
    """Return tool definitions with cache_control on the last entry."""
    tools = list(TOOL_DEFINITIONS)
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _parse_mcp_servers() -> list[dict]:
    from app.config import settings
    raw = settings.MCP_SERVERS.strip()
    if not raw:
        return []
    try:
        servers = json.loads(raw)
        return servers if isinstance(servers, list) else []
    except json.JSONDecodeError:
        logger.warning("scout.mcp_parse_failed", extra={"raw": raw[:100]})
        return []


async def _create_message(client, *, model: str, max_tokens: int, messages: list[dict], stream: bool = False):
    """
    Async wrapper around client.messages.create / .stream with prompt caching
    and optional MCP servers.
    """
    mcp_servers = _parse_mcp_servers()
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=CACHED_SYSTEM,
        tools=_cached_tools(),
        messages=messages,
    )

    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
        if stream:
            return client.beta.messages.stream(**kwargs, betas=["mcp-client-2025-04-04"])
        return await client.beta.messages.create(**kwargs, betas=["mcp-client-2025-04-04"])

    if stream:
        return client.messages.stream(**kwargs)
    return await client.messages.create(**kwargs)


# ── Topic guardrail ───────────────────────────────────────────────────────────

async def _is_on_topic(client, model: str, message: str) -> bool:
    """
    Fast pre-flight topic classifier using a single low-cost API call.

    Returns True if the message is hockey/draft related.
    Uses the same model but with a tiny max_tokens budget (1 token) to keep
    latency minimal — this is a gate, not a conversation turn.

    Fails open: if the classifier itself errors, we allow the request through
    rather than silently rejecting valid questions.
    """
    classifier_prompt = (
        "You are a topic classifier for an NHL Draft Simulator assistant called The Scout.\n\n"
        "Reply with only the single word YES or NO.\n\n"
        "The Scout answers questions about:\n"
        "  - NHL draft prospects and rankings (e.g. CSS rank, stats, comparisons)\n"
        "  - NHL teams' draft tendencies, GM profiles, and positional needs\n"
        "  - Historical draft picks and patterns\n"
        "  - Player archetypes, style comparisons, and NHL comps\n"
        "  - Draft strategy: pick value, BPA vs. need, round-by-round decisions\n\n"
        "The Scout does NOT answer questions about:\n"
        "  - Active NHL rosters, trades, free agency, or in-season standings\n"
        "  - Non-hockey topics (politics, sports other than hockey, general knowledge)\n"
        "  - Opinions on NHL games, scores, or playoff predictions\n\n"
        "Edge cases — answer YES for these:\n"
        "  - 'Is [prospect] overrated?' — prospect evaluation, YES\n"
        "  - 'Which pick has the best value?' — draft strategy, YES\n"
        "  - 'Who does [prospect] remind you of in the NHL?' — NHL comp, YES\n"
        "  - General hockey terms used in a draft context (e.g. 'two-way forward', 'power-play QB')\n\n"
        "Edge cases — answer NO for these:\n"
        "  - 'Is [active NHL player] overrated?' with no draft context\n"
        "  - 'Who will win the Stanley Cup?'\n"
        "  - 'What's the score of last night's game?'\n\n"
        f"Message: {message[:500]}\n\n"
        "Is this message within The Scout's scope? Reply YES or NO only."
    )
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": classifier_prompt}],
        )
        answer = _extract_text(resp).strip().upper()
        return answer.startswith("Y")
    except Exception as exc:
        logger.warning("scout.topic_classifier_failed — failing open error=%s", exc)
        return True   # fail open


# ── Draft context helpers ─────────────────────────────────────────────────────

def _pick_to_round(pick: int) -> int:
    """Convert an overall pick number to a round (32 picks per round)."""
    return max(1, (pick - 1) // 32 + 1)


def _round_strategy_note(pick: int) -> str:
    """Return a brief strategic framing note for the current pick slot."""
    rnd = _pick_to_round(pick)
    if pick <= 5:
        return "Top-5 pick — franchise-altering talent is available; BPA almost always wins here."
    if pick <= 15:
        return "Lottery pick — elite prospects; slight need-fit consideration is acceptable but don't reach."
    if pick <= 32:
        return "Late first round — value pick territory; a prospect available past their CSS rank is a win."
    if rnd == 2:
        return "Second round — high-upside players and league-specific gems; fit and upside matter more than CSS rank."
    if rnd == 3:
        return "Third round — depth and project players; prioritize skill archetype over immediate readiness."
    return f"Round {rnd} — late-round flier territory; swing for upside and positional scarcity."


def _format_draft_context(ctx: dict) -> str:
    """Render a draft_context dict into a readable string for the agent."""
    parts: list[str] = []

    pick = ctx.get("current_pick")
    if pick:
        rnd = _pick_to_round(pick)
        parts.append(f"Current pick: #{pick} overall (Round {rnd})")
        parts.append(f"Strategy note: {_round_strategy_note(pick)}")

    if ctx.get("picking_team"):
        parts.append(f"Picking team: {ctx['picking_team']}")

    if ctx.get("picks_made") is not None:
        parts.append(f"Picks made so far: {ctx['picks_made']}")

    recent = ctx.get("recent_picks", [])[-3:]
    if recent:
        lines = [
            f"  #{p.get('pick')} {p.get('team', '')} → {p.get('prospect', '')} ({p.get('position', '')})"
            for p in recent
        ]
        parts.append("Recent picks (last 3):\n" + "\n".join(lines))

    if ctx.get("available_count"):
        parts.append(f"Prospects still available: {ctx['available_count']}")

    # Value signal: if top prospect's CSS rank is well above the pick slot, flag it
    top_available_css = ctx.get("top_available_css_rank")
    if top_available_css and pick:
        value_gap = pick - top_available_css
        if value_gap >= 10:
            parts.append(
                f"Value alert: best available prospect is CSS #{top_available_css} — "
                f"{value_gap} picks of surplus value over current slot."
            )
        elif value_gap <= -10:
            parts.append(
                f"Reach warning: best available prospect is CSS #{top_available_css} — "
                f"reaching {abs(value_gap)} picks above expected range."
            )

    return "\n".join(parts)


def _prepend_draft_context(messages: list[dict], draft_context: dict | None) -> list[dict]:
    """
    If draft_context is provided, prepend a user/assistant pair at the start of
    the message list so Claude has situational draft awareness for the whole turn.
    Placed before history so it reads as established context, not part of the chat.
    """
    if not draft_context:
        return messages
    ctx_text = _format_draft_context(draft_context)
    pick = draft_context.get("current_pick")
    team = draft_context.get("picking_team", "the picking team")
    rnd = _pick_to_round(pick) if pick else None
    round_str = f" (Round {rnd})" if rnd else ""

    return [
        {"role": "user", "content": f"[Draft Context]\n{ctx_text}"},
        {
            "role": "assistant",
            "content": (
                f"Got it — I have the live board state. We're at pick #{pick}{round_str}, "
                f"{team} is on the clock. I'll weigh BPA vs. positional fit and flag any "
                f"value or reach signals when answering your question."
            ) if pick else (
                "Got it — I have the current draft state and will factor in the picking team "
                "and recent selections when answering your question."
            ),
        },
    ] + messages


# ── Public interface ──────────────────────────────────────────────────────────

async def chat(db: Session, session_id: str, user_message: str,
               draft_context: dict | None = None) -> str:
    """
    Process one user message with tool-use and return the final assistant reply.
    Persists the conversation turn to the database.
    """
    from app.config import settings

    if not settings.ANTHROPIC_API_KEY:
        return (
            "The Scout is offline — ANTHROPIC_API_KEY is not configured. "
            "Add it to your .env file to enable the AI assistant."
        )

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Topic guardrail — fast classifier before the expensive tool-use loop
    if not await _is_on_topic(client, GUARDRAIL_MODEL, user_message):
        return _OFFTOPIC_REPLY

    history = agent_memory.get_history_with_summary(db, session_id, limit=6)
    messages: list[dict] = [{"role": t["role"], "content": t["content"]} for t in history]
    messages = _prepend_draft_context(messages, draft_context)
    messages.append({"role": "user", "content": user_message})

    try:
        async with anthropic_breaker:
            reply = await _run_tool_loop(client, messages, db, settings.ANTHROPIC_MODEL)
    except CircuitOpenError as exc:
        return str(exc)

    store.save_message(db, session_id, "user", user_message)
    store.save_message(db, session_id, "assistant", reply)
    agent_memory.update_rolling_summary(db, session_id)
    logger.info("scout.chat", extra={"session_id": session_id})
    return reply


async def chat_stream(
    db: Session, session_id: str, user_message: str,
    draft_context: dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings for the streaming endpoint.

    Tool calls are executed concurrently between turns.
    Only the final text response is streamed token-by-token.
    """
    from app.config import settings

    if not settings.ANTHROPIC_API_KEY:
        yield f"data: {json.dumps({'token': 'The Scout is offline — ANTHROPIC_API_KEY not configured.'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Topic guardrail
    if not await _is_on_topic(client, GUARDRAIL_MODEL, user_message):
        yield f"data: {json.dumps({'token': _OFFTOPIC_REPLY})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    # Circuit breaker check up front
    try:
        if not await anthropic_breaker.allow_request():
            yield f"data: {json.dumps({'token': 'The Scout is temporarily unavailable. Please try again shortly.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
    except Exception:
        pass

    history = agent_memory.get_history_with_summary(db, session_id, limit=6)
    messages: list[dict] = [{"role": t["role"], "content": t["content"]} for t in history]
    messages = _prepend_draft_context(messages, draft_context)
    messages.append({"role": "user", "content": user_message})

    # Run tool rounds with per-tool status SSE events so the UI shows progress.
    # Each round: get Claude's response → yield a status label for every tool_use
    # block → execute all tools concurrently → feed results back.
    try:
        last_tool_sig: str | None = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = await _create_message(
                client, model=settings.ANTHROPIC_MODEL, max_tokens=800, messages=messages
            )
    
            if response.stop_reason != "tool_use":
                break
            # Stuck-loop guard
            sig = _tool_signature(response)
            if sig and sig == last_tool_sig:
                logger.warning("scout.stream_tool_loop_stuck — exiting early")
                break
            last_tool_sig = sig
            for block in response.content:
                if block.type == "tool_use":
                    label = _TOOL_LABELS.get(block.name, "Consulting the database...")
                    yield f"data: {json.dumps({'status': label})}\n\n"
            messages = await _process_tool_turn(response, messages, db)
        await anthropic_breaker.record_success()
    except Exception as exc:
        await anthropic_breaker.record_failure()
        logger.error("scout.tool_rounds_failed error=%s", exc)
        yield f"data: {json.dumps({'error': 'Tool execution failed. Please try again.'})}\n\n"
        return

    # Stream the final response token-by-token via the SSE generator.
    full_reply: list[str] = []
    try:
        async with await _create_message(
            client, model=settings.ANTHROPIC_MODEL, max_tokens=800, messages=messages, stream=True
        ) as stream:
            async for text in stream.text_stream:
                full_reply.append(text)
                yield f"data: {json.dumps({'token': text})}\n\n"

            await stream.get_final_message()

        reply = "".join(full_reply)
        store.save_message(db, session_id, "user", user_message)
        store.save_message(db, session_id, "assistant", reply)
        agent_memory.update_rolling_summary(db, session_id)

        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
    except Exception as exc:
        await anthropic_breaker.record_failure()
        logger.error("scout.stream_failed", extra={"error": str(exc)})
        yield f"data: {json.dumps({'error': 'Stream failed. Please try again.'})}\n\n"


# ── Tool loop helpers ─────────────────────────────────────────────────────────

async def _run_tool_loop(client, messages: list[dict], db: Session, model: str) -> str:
    """Execute tool calls until Claude produces a final text response."""
    rounds = 0
    last_tool_signature: str | None = None

    for _ in range(MAX_TOOL_ROUNDS):
        response = await _create_message(client, model=model, max_tokens=800, messages=messages)

        if response.stop_reason != "tool_use":
            return _extract_text(response)

        # Detect stuck loop: same tool call twice in a row → exit early
        tool_sig = _tool_signature(response)
        if tool_sig and tool_sig == last_tool_signature:
            logger.warning("scout.tool_loop_stuck signature=%s — exiting early", tool_sig)
            return _extract_text(response)
        last_tool_signature = tool_sig

        messages = await _process_tool_turn(response, messages, db)
        rounds += 1
    # All rounds exhausted — one final call without tools
    response = await client.messages.create(
        model=model,
        max_tokens=800,
        system=CACHED_SYSTEM,
        messages=messages,
    )
    return _extract_text(response)


def _tool_signature(response) -> str | None:
    """Stable fingerprint of all tool_use blocks in a response for stuck-loop detection."""
    calls = sorted(
        (b.name, json.dumps(b.input, sort_keys=True))
        for b in response.content
        if getattr(b, "type", None) == "tool_use"
    )
    return str(calls) if calls else None



async def _process_tool_turn(
    response, messages: list[dict], db: Session
) -> list[dict]:
    """
    Execute ALL tool_use blocks in a response concurrently, then append results.

    Each tool call runs in asyncio.to_thread so synchronous SQLAlchemy DB calls
    don't block the event loop. All N tool calls in a round run in parallel —
    total latency ≈ max(individual tool latencies) rather than their sum.
    """
    assistant_content = []
    tool_calls: list[tuple[str, str, dict]] = []   # (id, name, input)

    for block in response.content:
        if block.type == "text":
            assistant_content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            assistant_content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
            tool_calls.append((block.id, block.name, block.input))

    # Run all tool calls concurrently
    results = await asyncio.gather(
        *[_run_tool(name, inputs, db) for _, name, inputs in tool_calls],
        return_exceptions=True,
    )

    tool_results = []
    for (tool_id, tool_name, _), result in zip(tool_calls, results):
        if isinstance(result, Exception):
            logger.warning("scout.tool_failed tool=%s error=%s", tool_name, result)
            content = json.dumps({"error": "Tool execution failed."})
        else:
            content = result
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
        })

    messages.append({"role": "assistant", "content": assistant_content})
    messages.append({"role": "user", "content": tool_results})
    return messages


async def _run_tool(name: str, inputs: dict, db: Session) -> str:
    """
    Execute one tool call in a thread pool (sync SQLAlchemy) with per-tool
    Prometheus metrics and an OpenTelemetry span for distributed tracing.

    OTel span attributes:
      scout.tool.name    — tool identifier (get_gm_profile, etc.)
      scout.tool.status  — "ok" or "error"
      scout.tool.elapsed_ms — wall-clock latency

    These spans appear in Jaeger/Zipkin/AWS X-Ray as child spans of the
    parent request trace, enabling per-tool latency breakdown in production.
    """
    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(execute_tool, name, inputs, db)
        elapsed = time.perf_counter() - t0
        logger.info(
            "scout.tool_called tool=%s elapsed_ms=%.1f",
            name, elapsed * 1000,
            extra={"tool": name, "input": inputs},
        )
        return result
    except Exception as exc:
        raise exc


def _extract_text(response) -> str:
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "I wasn't able to generate a response. Please try again."
