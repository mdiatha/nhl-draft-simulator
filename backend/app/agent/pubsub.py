"""
Redis Pub/Sub for horizontally-scalable SSE streaming.

Problem:
  When running multiple FastAPI instances, an SSE connection is sticky to the
  task that accepted it. If that task restarts, or if the initial HTTP request
  hits instance A but the client reconnects to instance B, the stream is lost.

  More critically: the token generation happens in the task that processes the
  Claude API call. If the client is connected to a different instance (load balancer
  round-robin), tokens never arrive.

Architecture:
  Token producer (any instance):
    1. Receives /chat/stream request
    2. Calls Claude, gets tokens
    3. Publishes each token to Redis channel "scout:stream:{session_id}:{msg_id}"
    4. Publishes {"done": true} when complete
    5. Buffers full reply in Redis key "scout:reply:{session_id}:{msg_id}" (TTL 5min)

  Token consumer (any instance):
    1. Subscribes to "scout:stream:{session_id}:{msg_id}"
    2. Forwards events to the SSE client
    3. If client reconnects, can check the buffer key for the completed reply

  This decouples stream generation from stream delivery — any instance can serve
  any client's SSE connection.

Usage:
    # Producer (in scout.py chat_stream):
    async for token in generate_response():
        await publish_token(session_id, msg_id, token)
    await publish_done(session_id, msg_id, full_reply)

    # Consumer (in api/agent.py):
    async for event in subscribe_stream(session_id, msg_id):
        yield event   # SSE to client
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# Redis channel prefix for SSE streams
_STREAM_CHANNEL = "scout:stream"
# Redis key prefix for completed reply buffer (reconnection recovery)
_REPLY_BUFFER    = "scout:reply"
# TTL for completed reply buffer (seconds)
_REPLY_TTL = 300   # 5 minutes — enough for client reconnect scenarios


def _get_channel(session_id: str, msg_id: str) -> str:
    return f"{_STREAM_CHANNEL}:{session_id}:{msg_id}"


def _get_buffer_key(session_id: str, msg_id: str) -> str:
    return f"{_REPLY_BUFFER}:{session_id}:{msg_id}"


async def publish_token(redis_url: str, session_id: str, msg_id: str, token: str) -> None:
    """Publish a single token to the Redis stream channel."""
    try:
        import redis.asyncio as aioredis
        async with aioredis.from_url(redis_url, decode_responses=True) as r:
            await r.publish(
                _get_channel(session_id, msg_id),
                json.dumps({"token": token}),
            )
    except Exception as exc:
        logger.debug("pubsub.publish_failed token_len=%d error=%s", len(token), exc)


async def publish_done(redis_url: str, session_id: str, msg_id: str, full_reply: str) -> None:
    """Publish stream completion and buffer the full reply for reconnection recovery."""
    try:
        import redis.asyncio as aioredis
        async with aioredis.from_url(redis_url, decode_responses=True) as r:
            # Buffer full reply so reconnecting clients can recover it
            await r.setex(_get_buffer_key(session_id, msg_id), _REPLY_TTL, full_reply)
            # Signal done on the stream channel
            await r.publish(
                _get_channel(session_id, msg_id),
                json.dumps({"done": True, "session_id": session_id}),
            )
    except Exception as exc:
        logger.warning("pubsub.publish_done_failed error=%s", exc)


async def subscribe_stream(
    redis_url: str,
    session_id: str,
    msg_id: str,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """
    Subscribe to a Redis stream channel and yield SSE-formatted events.

    Yields data lines suitable for SSE:
      'data: {"token": "..."}\n\n'
      'data: {"done": true}\n\n'

    Parameters
    ----------
    timeout : float
        Maximum seconds to wait for the stream to complete (default 120s).
        Prevents subscriber from hanging forever if the producer dies.
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning("pubsub.redis_not_available — falling back to empty stream")
        return

    channel = _get_channel(session_id, msg_id)

    async with aioredis.from_url(redis_url, decode_responses=True) as r:
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)

        try:
            deadline = asyncio.get_event_loop().time() + timeout
            async for message in pubsub.listen():
                if asyncio.get_event_loop().time() > deadline:
                    yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
                    break

                if message["type"] != "message":
                    continue

                try:
                    payload = json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if "token" in payload:
                    yield f"data: {json.dumps({'token': payload['token']})}\n\n"
                elif payload.get("done"):
                    yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                    break
                elif "error" in payload:
                    yield f"data: {json.dumps({'error': payload['error']})}\n\n"
                    break
        finally:
            await pubsub.unsubscribe(channel)


async def get_buffered_reply(redis_url: str, session_id: str, msg_id: str) -> Optional[str]:
    """
    Retrieve a completed reply from the buffer (for SSE reconnection).

    Returns None if the buffer has expired or the reply never completed.
    """
    try:
        import redis.asyncio as aioredis
        async with aioredis.from_url(redis_url, decode_responses=True) as r:
            return await r.get(_get_buffer_key(session_id, msg_id))
    except Exception as exc:
        logger.debug("pubsub.get_buffer_failed error=%s", exc)
        return None


def new_msg_id() -> str:
    """Generate a unique message ID for a new stream."""
    return str(uuid.uuid4())[:8]
