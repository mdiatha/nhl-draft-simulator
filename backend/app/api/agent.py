"""
Scout agent API endpoints.

POST /api/agent/chat        — send a message, get a reply (non-streaming)
POST /api/agent/chat/stream — SSE streaming version
POST /api/agent/chat/{msg_id} — retrieve a completed reply (SSE reconnection)
POST /api/agent/index       — enqueue RAG index rebuild via SQS (async)
POST /api/agent/index/sync  — synchronous RAG index rebuild (dev/admin)
GET  /api/agent/history     — fetch conversation history for a session
"""
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, get_read_db
from app.middleware.auth import require_admin_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=64)
    draft_context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, db: Session = Depends(get_read_db)):
    """
    Ask the Scout a question about NHL draft tendencies, prospects, or team needs.

    The agent is fully async (AsyncAnthropic, parallel tool execution).
    Tool queries run against the read replica (get_read_db) — all 6 tools are
    SELECT-only, so the primary is not touched during normal chat.
    Conversation persistence (save_message) still uses the read_db session here
    but store.py commits via the same session — acceptable for single-session
    workload. For strict separation, persistence could use a separate primary session.
    """
    from app.agent.scout import chat as scout_chat

    reply = await scout_chat(db, body.session_id, body.message, draft_context=body.draft_context)
    return ChatResponse(reply=reply, session_id=body.session_id)  # type: ignore[call-arg]


class IndexResponse(BaseModel):
    docs_indexed: int
    message: str


class IndexEnqueueResponse(BaseModel):
    queued: bool
    message: str


@router.post(
    "/index",
    response_model=IndexEnqueueResponse,
    dependencies=[Depends(require_admin_key)],
)
async def enqueue_index_rebuild(db: Session = Depends(get_db)):
    """
    Enqueue an async RAG index rebuild via SQS.

    Returns immediately with 202 — a Lambda/ECS worker picks up the message
    and calls build_index() in the background. This keeps the API responsive
    even though embedding 200+ prospects via Voyage AI takes several seconds.

    Falls back to synchronous rebuild when SQS is not configured (dev mode).
    """
    from app.config import settings
    from app.agent.sqs import enqueue_index_rebuild as _enqueue

    if settings.SQS_INDEX_QUEUE_URL:
        try:
            _enqueue(settings.SQS_INDEX_QUEUE_URL)
            return IndexEnqueueResponse(
                queued=True,
                message="Index rebuild queued — will complete in the background.",
            )
        except Exception as exc:
            logger.error("agent.sqs_enqueue_failed error=%s", exc)
            # Fall through to synchronous rebuild

    # Synchronous fallback (dev or SQS unavailable)
    from app.agent.embeddings import build_index
    try:
        count = build_index(db)
        return IndexEnqueueResponse(
            queued=False,
            message=f"Indexed {count} documents (synchronous — SQS not configured).",
        )
    except Exception as exc:
        logger.error("agent.index_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/index/sync",
    response_model=IndexResponse,
    dependencies=[Depends(require_admin_key)],
)
async def build_index_sync(db: Session = Depends(get_db)):
    """Synchronous index rebuild for admin/dev use. Blocks until complete."""
    from app.agent.embeddings import build_index
    try:
        count = build_index(db)
        return IndexResponse(docs_indexed=count, message=f"Indexed {count} documents.")
    except Exception as exc:
        logger.error("agent.index_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


class HistoryItem(BaseModel):
    role: str
    content: str


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, db: Session = Depends(get_read_db)):
    """
    SSE streaming version of /chat.

    Tool calls run concurrently between turns (parallel via asyncio.gather).
    Only the final text response is streamed token-by-token.

    SSE event format:
      data: {"token": "..."}        — partial response token
      data: {"done": true, "session_id": "..."} — stream complete
      data: {"error": "..."}        — error (stream terminates)

    SSE reconnection:
      On reconnect, the client should GET /api/agent/chat/reply/{session_id}
      to retrieve the last completed reply for a session.
    """
    from app.agent.scout import chat_stream as scout_stream

    async def _gen():
        async for chunk in scout_stream(db, body.session_id, body.message, draft_context=body.draft_context):
            yield chunk

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/reply/{session_id}")
async def get_last_reply(session_id: str, db: Session = Depends(get_db)):
    """
    SSE reconnection endpoint.

    If the client loses connection mid-stream, it can call this endpoint to
    retrieve the last completed assistant reply for the session. The frontend
    can then display the full answer without re-triggering a new generation.
    """
    from app.agent.store import get_history

    history = get_history(db, session_id, limit=2)
    # Find the last assistant message
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return {"session_id": session_id, "reply": msg["content"], "recovered": True}

    return {"session_id": session_id, "reply": None, "recovered": False}


@router.get("/history")
async def get_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Return conversation history for a session (for UI hydration on page reload)."""
    from app.agent.store import get_history as _get_history, count_messages

    total = count_messages(db, session_id)
    history = _get_history(db, session_id, limit=limit)
    return {"session_id": session_id, "messages": history, "total": total, "limit": limit, "offset": offset}
