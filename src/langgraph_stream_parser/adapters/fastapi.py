"""
FastAPI adapter for streaming LangGraph events over WebSockets or SSE.

This adapter is stateless by design — conversation state lives in
LangGraph's checkpointer, keyed by ``session_id`` (used as ``thread_id``).

Requires: ``pip install langgraph-stream-parser[fastapi]``

Example:
    from fastapi import FastAPI, WebSocket
    from langgraph_stream_parser.adapters.fastapi import FastAPIAdapter

    app = FastAPI()
    adapter = FastAPIAdapter(graph=agent)

    @app.websocket("/chat/{session_id}")
    async def chat(ws: WebSocket, session_id: str):
        await adapter.handle_websocket(ws, session_id)
"""
import asyncio
import json
from typing import Any, AsyncIterator

from ..events import (
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
    event_to_dict,
)
from ..parser import StreamParser
from ..resume import create_resume_input, prepare_agent_input


class FastAPIAdapter:
    """Stream LangGraph events to a FastAPI client over WebSocket or SSE.

    State persistence is delegated entirely to LangGraph's checkpointer —
    the adapter keeps no conversation state. Each method uses
    ``session_id`` as the LangGraph ``thread_id``.

    Per-session concurrency is guarded by an internal lock map: two
    concurrent streams on the same ``session_id`` would corrupt
    LangGraph's checkpoint, so we serialize them.

    Attributes:
        graph: The LangGraph graph (compiled with a checkpointer).
        stream_mode: Stream mode for graph.astream() and the parser.
        parser_kwargs: Additional kwargs passed to each StreamParser.
    """

    def __init__(
        self,
        *,
        graph: Any,
        stream_mode: str | list[str] = "updates",
        **parser_kwargs: Any,
    ):
        """Initialize the adapter.

        Args:
            graph: A compiled LangGraph graph with a checkpointer.
            stream_mode: Stream mode for streaming and parsing.
            **parser_kwargs: Passed to each StreamParser instance.
        """
        self._graph = graph
        self._stream_mode = stream_mode
        self._parser_kwargs = parser_kwargs
        # Per-session locks + refcounts. Locks are created lazily inside the
        # running loop (not at construction) and removed when the last user
        # releases, so the dict does not grow unbounded across sessions.
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_refs: dict[str, int] = {}
        self._locks_mutex: asyncio.Lock | None = None

    # ── Public API ───────────────────────────────────────────────────

    async def handle_websocket(
        self,
        websocket: Any,
        session_id: str,
        *,
        accept: bool = True,
    ) -> None:
        """Handle a WebSocket connection for a chat session.

        Protocol (client → server):
            {"type": "message", "content": "..."}
            {"type": "decision", "decisions": [{"type": "approve"}, ...]}
            {"type": "cancel"}

        Protocol (server → client):
            Event dicts (via ``event_to_dict``) including "interrupt" events.
            {"type": "ack", "ref": "message|decision|cancel"}
            {"type": "error", "error": "..."}

        Args:
            websocket: A FastAPI WebSocket instance.
            session_id: Logical session identifier, used as LangGraph
                ``thread_id``.
            accept: If True, call ``websocket.accept()`` before handling.
        """
        if accept:
            await websocket.accept()

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Invalid JSON: {e}",
                    })
                    continue

                msg_type = message.get("type")

                if msg_type == "message":
                    content = message.get("content", "")
                    if not content:
                        await websocket.send_json({
                            "type": "error",
                            "error": "message requires 'content'",
                        })
                        continue
                    await websocket.send_json({"type": "ack", "ref": "message"})
                    input_data = prepare_agent_input(message=content)
                    await self._stream_to_websocket(
                        websocket, session_id, input_data
                    )

                elif msg_type == "decision":
                    decisions = message.get("decisions")
                    if not isinstance(decisions, list):
                        await websocket.send_json({
                            "type": "error",
                            "error": "decision requires 'decisions' list",
                        })
                        continue
                    await websocket.send_json({"type": "ack", "ref": "decision"})
                    input_data = create_resume_input(decisions=decisions)
                    await self._stream_to_websocket(
                        websocket, session_id, input_data
                    )

                elif msg_type == "cancel":
                    # Cancellation cannot interrupt an in-flight stream here
                    # (one message at a time over WS). This is a hook for
                    # clients that want an ack for UX parity.
                    await websocket.send_json({"type": "ack", "ref": "cancel"})

                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"unknown message type: {msg_type!r}",
                    })

        except asyncio.CancelledError:
            # Clean shutdown — propagate so the task can be cancelled.
            raise
        except Exception as e:
            # WebSocketDisconnect and any other failure mode end the loop.
            # Surface non-disconnect errors to the client if still open.
            if _is_disconnect(e):
                return
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })
            except Exception:
                pass

    async def sse_stream(
        self,
        session_id: str,
        input_data: Any,
    ) -> AsyncIterator[str]:
        """Stream parsed events as Server-Sent Events.

        Yields SSE-formatted strings ready to feed a
        ``StreamingResponse(media_type="text/event-stream")``.

        Args:
            session_id: Logical session identifier (LangGraph thread_id).
            input_data: Prepared agent input. Use ``prepare_agent_input()``
                or ``create_resume_input()`` to build this.

        Yields:
            SSE-formatted ``data: {...}\\n\\n`` lines.

        Example:
            from fastapi.responses import StreamingResponse

            @app.post("/chat/{session_id}")
            async def chat(session_id: str, body: dict):
                input_data = prepare_agent_input(message=body["message"])
                return StreamingResponse(
                    adapter.sse_stream(session_id, input_data),
                    media_type="text/event-stream",
                )
        """
        async for event in self._iter_events(session_id, input_data):
            payload = json.dumps(event_to_dict(event))
            yield f"data: {payload}\n\n"

    async def resume(
        self,
        session_id: str,
        decisions: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """SSE helper: resume a session from an interrupt with decisions.

        Args:
            session_id: Logical session identifier.
            decisions: Decision list, e.g. ``[{"type": "approve"}]``.

        Yields:
            SSE-formatted lines, same format as ``sse_stream``.
        """
        input_data = create_resume_input(decisions=decisions)
        async for line in self.sse_stream(session_id, input_data):
            yield line

    # ── Internals ────────────────────────────────────────────────────

    async def _stream_to_websocket(
        self,
        websocket: Any,
        session_id: str,
        input_data: Any,
    ) -> None:
        """Stream events from the graph to a WebSocket."""
        try:
            async for event in self._iter_events(session_id, input_data):
                await websocket.send_json(event_to_dict(event))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_disconnect(e):
                raise
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })

    async def _iter_events(
        self,
        session_id: str,
        input_data: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Run the graph for a single turn and yield parsed events.

        Serializes concurrent turns on the same session_id — most
        LangGraph checkpointers are not safe against concurrent writes
        to the same thread, so we run turns one at a time per session.
        """
        config = {"configurable": {"thread_id": session_id}}
        parser = StreamParser(
            stream_mode=self._stream_mode,
            **self._parser_kwargs,
        )

        lock = await self._acquire_session_lock(session_id)
        try:
            async with lock:
                stream = self._graph.astream(
                    input_data,
                    config=config,
                    stream_mode=self._stream_mode,
                )
                async for event in parser.aparse(stream):
                    yield event
                    # Stop after terminal events so subsequent client messages
                    # start a fresh turn.
                    if isinstance(event, (CompleteEvent, ErrorEvent)):
                        return
        finally:
            await self._release_session_lock(session_id)

    async def _acquire_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create the lock for a session and bump its refcount."""
        if self._locks_mutex is None:
            self._locks_mutex = asyncio.Lock()
        async with self._locks_mutex:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            self._lock_refs[session_id] = self._lock_refs.get(session_id, 0) + 1
            return lock

    async def _release_session_lock(self, session_id: str) -> None:
        """Drop the refcount for a session; delete the lock when it hits 0."""
        if self._locks_mutex is None:
            return
        async with self._locks_mutex:
            remaining = self._lock_refs.get(session_id, 0) - 1
            if remaining <= 0:
                self._locks.pop(session_id, None)
                self._lock_refs.pop(session_id, None)
            else:
                self._lock_refs[session_id] = remaining


def _is_disconnect(exc: Exception) -> bool:
    """Detect a WebSocket disconnect without importing fastapi at module load."""
    # Match by class name to avoid a hard import of starlette/fastapi.
    name = type(exc).__name__
    return name in {"WebSocketDisconnect", "ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"}
