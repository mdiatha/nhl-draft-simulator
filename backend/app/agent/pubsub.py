"""
In-memory pub/sub for single-instance SSE streaming.

Replaces the Redis pub/sub implementation. Since we run one instance,
the token generator and SSE consumer are always in the same process —
an asyncio.Queue per stream is sufficient.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()

# Active streams keyed by channel name → Queue
_streams: dict[str, asyncio.Queue] = {}


def _get_channel(session_id: str, msg_id: str) -> str:
    return f"scout:stream:{session_id}:{msg_id}"


async def publish_token(redis_url: str, session_id: str, msg_id: str, token: str) -> None:
    channel = _get_channel(session_id, msg_id)
    q = _streams.get(channel)
    if q:
        await q.put({"token": token})


async def publish_done(redis_url: str, session_id: str, msg_id: str, full_reply: str) -> None:
    channel = _get_channel(session_id, msg_id)
    q = _streams.get(channel)
    if q:
        await q.put({"done": True, "session_id": session_id})
        await q.put(_SENTINEL)


async def subscribe_stream(
    redis_url: str,
    session_id: str,
    msg_id: str,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    channel = _get_channel(session_id, msg_id)
    q: asyncio.Queue = asyncio.Queue()
    _streams[channel] = q
    try:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
                break
            try:
                payload = await asyncio.wait_for(q.get(), timeout=remaining)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
                break
            if payload is _SENTINEL:
                break
            if "token" in payload:
                yield f"data: {json.dumps({'token': payload['token']})}\n\n"
            elif payload.get("done"):
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                break
            elif "error" in payload:
                yield f"data: {json.dumps({'error': payload['error']})}\n\n"
                break
    finally:
        _streams.pop(channel, None)


async def get_buffered_reply(redis_url: str, session_id: str, msg_id: str) -> Optional[str]:
    return None


def new_msg_id() -> str:
    return str(uuid.uuid4())[:8]
