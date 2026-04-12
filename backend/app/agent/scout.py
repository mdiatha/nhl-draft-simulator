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

You have access to tools that query the live draft database:
- get_gm_profile: GM drafting tendencies, archetype, position/league/nationality weights
- get_top_prospects: top prospects at any position by CSS ranking
- get_team_needs: a team's positional needs based on historical drafting patterns
- search_prospects: find specific prospects by name, league, or nationality (keyword search)
- semantic_prospect_search: find prospects similar to a player archetype or named player
  (e.g. "find me a player like Makar", "two-way offensive defenseman from Sweden")
  Use this when the query describes a style or player comparison rather than a name/league.
- get_draft_history: a team's actual historical draft picks for a given year or recent years,
  including player name, position, overall pick, CSS rank, and pre-draft stats.
  Use this when asked about what a team has done in past drafts or how picks panned out.
- get_ml_ranking: XGBoost model score for a prospect at a given pick slot, optionally for a
  specific team. When a team is provided, also returns SHAP factors explaining the score.
  Use when asked how the model rates a prospect, what's driving their ranking, or for a
  data-driven pick recommendation.
- compare_prospects: side-by-side stat comparison of two prospects (CSS rank, PPG, size, age,
  league). Use when asked to directly compare or choose between players.
- get_prospect_detail: full profile for one prospect including multi-season stat history.
  Use when the user wants deeper context on a specific player beyond basic stats.
- get_nhl_comp: find current NHL players who play most similarly to a given prospect, using
  semantic similarity of style profiles. Use for questions like "who does [prospect] play like?"
  or "what's their NHL comp?"

Always call a tool to get real data before answering questions about specific teams, GMs, or
prospects. Be concise, data-driven, and hockey-savvy. Cite specific numbers from tool results
(e.g. position weights, CSS ranks, PPG, ML scores). If the data doesn't support a confident
answer, say so.

If a [Draft Context] block appears at the start of the conversation, use it to give
pick-specific advice tailored to the current pick number, picking team, and board state."""

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


def _record_usage(usage) -> None:
    """Emit Anthropic token usage (including prompt cache) to Prometheus."""
    if not usage:
        return
    from app.observability.metrics import (
        SCOUT_PROMPT_CACHE_READ_TOKENS,
        SCOUT_PROMPT_CACHE_WRITE_TOKENS,
        SCOUT_INPUT_TOKENS,
        SCOUT_OUTPUT_TOKENS,
    )
    cache_read  = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    input_tok   = getattr(usage, "input_tokens", 0) or 0
    output_tok  = getattr(usage, "output_tokens", 0) or 0

    if cache_read:
        SCOUT_PROMPT_CACHE_READ_TOKENS.inc(cache_read)
    if cache_write:
        SCOUT_PROMPT_CACHE_WRITE_TOKENS.inc(cache_write)
    if input_tok:
        SCOUT_INPUT_TOKENS.inc(input_tok)
    if output_tok:
        SCOUT_OUTPUT_TOKENS.inc(output_tok)


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
        "You are a topic classifier. Reply with only the single word YES or NO.\n\n"
        "Is the following message related to NHL hockey, the NHL draft, hockey prospects, "
        "hockey teams, or general manager strategy?\n\n"
        f"Message: {message[:500]}"
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

def _format_draft_context(ctx: dict) -> str:
    """Render a draft_context dict into a readable string for the agent."""
    parts: list[str] = []
    if ctx.get("current_pick"):
        parts.append(f"Current pick: #{ctx['current_pick']} overall")
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
        parts.append("Recent picks:\n" + "\n".join(lines))
    if ctx.get("available_count"):
        parts.append(f"Prospects still available: {ctx['available_count']}")
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
    return [
        {"role": "user", "content": f"[Draft Context]\n{ctx_text}"},
        {
            "role": "assistant",
            "content": (
                "Got it — I have the current draft state and will factor in the pick number, "
                "picking team, and recent selections when answering your question."
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
    from app.observability.metrics import SCOUT_REQUESTS_TOTAL

    SCOUT_REQUESTS_TOTAL.labels(mode="sync").inc()

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
    from app.observability.metrics import SCOUT_REQUESTS_TOTAL

    SCOUT_REQUESTS_TOTAL.labels(mode="stream").inc()

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
            from app.observability.metrics import SCOUT_CIRCUIT_BREAKER_REJECTED
            SCOUT_CIRCUIT_BREAKER_REJECTED.inc()
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
        for _ in range(MAX_TOOL_ROUNDS):
            response = await _create_message(
                client, model=settings.ANTHROPIC_MODEL, max_tokens=800, messages=messages
            )
            _record_usage(getattr(response, "usage", None))
            if response.stop_reason != "tool_use":
                break
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

    # Stream the final response — publish to Redis Pub/Sub for multi-instance support.
    # Any instance can subscribe to the channel and fan out tokens to its SSE client.
    # Falls back to direct yield when Redis is not configured (dev mode).
    from app.agent.pubsub import publish_token, publish_done, new_msg_id

    msg_id = new_msg_id()
    use_pubsub = bool(settings.REDIS_URL)

    full_reply: list[str] = []
    try:
        async with await _create_message(
            client, model=settings.ANTHROPIC_MODEL, max_tokens=800, messages=messages, stream=True
        ) as stream:
            async for text in stream.text_stream:
                full_reply.append(text)
                # Direct yield (always) so the producing instance also serves the client
                yield f"data: {json.dumps({'token': text})}\n\n"
                # Also publish to Redis so other instances can serve reconnecting clients
                if use_pubsub:
                    await publish_token(settings.REDIS_URL, session_id, msg_id, text)

            final_message = await stream.get_final_message()
            _record_usage(getattr(final_message, "usage", None))

        reply = "".join(full_reply)
        store.save_message(db, session_id, "user", user_message)
        store.save_message(db, session_id, "assistant", reply)

        if use_pubsub:
            await publish_done(settings.REDIS_URL, session_id, msg_id, reply)

        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'msg_id': msg_id})}\n\n"
    except Exception as exc:
        await anthropic_breaker.record_failure()
        logger.error("scout.stream_failed", extra={"error": str(exc)})
        yield f"data: {json.dumps({'error': 'Stream failed. Please try again.'})}\n\n"


# ── Tool loop helpers ─────────────────────────────────────────────────────────

async def _run_tool_loop(client, messages: list[dict], db: Session, model: str) -> str:
    """Execute tool calls until Claude produces a final text response (end_turn)."""
    from app.observability.metrics import SCOUT_TOOL_ROUNDS
    rounds = 0

    for _ in range(MAX_TOOL_ROUNDS):
        response = await _create_message(client, model=model, max_tokens=800, messages=messages)
        _record_usage(getattr(response, "usage", None))

        if response.stop_reason == "end_turn":
            SCOUT_TOOL_ROUNDS.observe(rounds)
            return _extract_text(response)

        if response.stop_reason == "tool_use":
            messages = await _process_tool_turn(response, messages, db)
            rounds += 1
        else:
            SCOUT_TOOL_ROUNDS.observe(rounds)
            return _extract_text(response)

    SCOUT_TOOL_ROUNDS.observe(rounds)
    # Safety: all rounds exhausted — make one final call without tools
    response = await client.messages.create(
        model=model,
        max_tokens=800,
        system=CACHED_SYSTEM,
        messages=messages,
    )
    _record_usage(getattr(response, "usage", None))
    return _extract_text(response)



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
    from app.observability.metrics import SCOUT_TOOL_CALLS_TOTAL, SCOUT_TOOL_DURATION

    # Acquire OTel tracer — no-op tracer if OTel is not configured
    try:
        from opentelemetry import trace as _otel_trace
        _tracer = _otel_trace.get_tracer("scout.agent")
    except ImportError:
        _tracer = None

    t0 = time.perf_counter()

    if _tracer:
        span_ctx = _tracer.start_as_current_span(
            f"scout.tool.{name}",
            attributes={
                "scout.tool.name": name,
                "scout.tool.input_keys": ",".join(inputs.keys()),
            },
        )
    else:
        from contextlib import nullcontext
        span_ctx = nullcontext()

    try:
        with span_ctx as span:
            result = await asyncio.to_thread(execute_tool, name, inputs, db)
            elapsed = time.perf_counter() - t0
            SCOUT_TOOL_CALLS_TOTAL.labels(tool=name, status="ok").inc()
            SCOUT_TOOL_DURATION.labels(tool=name).observe(elapsed)
            if span and hasattr(span, "set_attribute"):
                span.set_attribute("scout.tool.status", "ok")
                span.set_attribute("scout.tool.elapsed_ms", round(elapsed * 1000, 1))
            logger.info(
                "scout.tool_called tool=%s elapsed_ms=%.1f",
                name, elapsed * 1000,
                extra={"tool": name, "input": inputs},
            )
            return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        SCOUT_TOOL_CALLS_TOTAL.labels(tool=name, status="error").inc()
        SCOUT_TOOL_DURATION.labels(tool=name).observe(elapsed)
        try:
            if span_ctx and hasattr(span_ctx, "set_attribute"):
                span_ctx.set_attribute("scout.tool.status", "error")
                span_ctx.record_exception(exc)
        except Exception:
            pass
        raise exc


def _extract_text(response) -> str:
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "I wasn't able to generate a response. Please try again."
