"""Tests for FastAPIAdapter."""
import asyncio
import json
from typing import Any, AsyncIterator

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from langgraph_stream_parser.adapters.fastapi import FastAPIAdapter
from langgraph_stream_parser.resume import prepare_agent_input

from .fixtures.mocks import (
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    INTERRUPT_WITH_ACTIONS,
)


# ── Mock graph ───────────────────────────────────────────────────────


class MockGraph:
    """Minimal async graph that yields canned chunks per call.

    Tracks calls so tests can assert thread_id wiring and the input passed.
    """

    def __init__(self, chunks_per_call: list[list[Any]]):
        self._chunks_per_call = chunks_per_call
        self._call_idx = 0
        self.calls: list[dict[str, Any]] = []

    def astream(
        self,
        input_data: Any,
        config: dict[str, Any] | None = None,
        stream_mode: str | list[str] = "updates",
    ) -> AsyncIterator[Any]:
        self.calls.append({
            "input": input_data,
            "config": config,
            "stream_mode": stream_mode,
        })
        chunks = self._chunks_per_call[self._call_idx]
        self._call_idx += 1

        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()


# ── WebSocket: message flow ──────────────────────────────────────────


def _make_app(adapter: FastAPIAdapter) -> FastAPI:
    app = FastAPI()

    @app.websocket("/chat/{session_id}")
    async def ws_endpoint(ws: WebSocket, session_id: str):
        await adapter.handle_websocket(ws, session_id)

    return app


def _drain_until_complete(ws, max_events: int = 20) -> list[dict]:
    """Receive events until 'complete' or 'error' seen."""
    events = []
    for _ in range(max_events):
        msg = ws.receive_json()
        events.append(msg)
        if msg.get("type") in {"complete", "error"}:
            return events
    raise AssertionError(
        f"No terminal event after {max_events} messages: {events}"
    )


class TestWebSocketMessageFlow:
    def test_user_message_streams_content_and_complete(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-1") as ws:
                ws.send_text(json.dumps({
                    "type": "message",
                    "content": "Hello",
                }))

                ack = ws.receive_json()
                assert ack == {"type": "ack", "ref": "message"}

                content = ws.receive_json()
                assert content["type"] == "content"
                assert content["content"] == "Hello, how can I help?"

                complete = ws.receive_json()
                assert complete["type"] == "complete"

        # Verify thread_id wired through
        assert graph.calls[0]["config"] == {"configurable": {"thread_id": "sess-1"}}

    def test_tool_call_events_streamed(self):
        graph = MockGraph([[
            AI_MESSAGE_WITH_TOOL_CALLS,
            TOOL_MESSAGE_SUCCESS,
        ]])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-tools") as ws:
                ws.send_text(json.dumps({"type": "message", "content": "search"}))
                ack = ws.receive_json()
                events = [ack] + _drain_until_complete(ws)

        types = [e["type"] for e in events]
        assert "ack" in types
        assert "tool_start" in types
        assert "tool_end" in types
        assert "complete" in types

    def test_empty_content_rejected(self):
        graph = MockGraph([[]])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-empty") as ws:
                ws.send_text(json.dumps({"type": "message", "content": ""}))
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "content" in err["error"]

    def test_invalid_json_rejected(self):
        graph = MockGraph([])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-bad") as ws:
                ws.send_text("not json")
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "Invalid JSON" in err["error"]

    def test_unknown_message_type_rejected(self):
        graph = MockGraph([])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-unk") as ws:
                ws.send_text(json.dumps({"type": "nope"}))
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "unknown message type" in err["error"]

    def test_cancel_acks(self):
        graph = MockGraph([])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-cancel") as ws:
                ws.send_text(json.dumps({"type": "cancel"}))
                ack = ws.receive_json()
                assert ack == {"type": "ack", "ref": "cancel"}


# ── WebSocket: interrupt flow ────────────────────────────────────────


class TestWebSocketInterrupt:
    def test_interrupt_then_approve(self):
        # First call streams until interrupt; second call resumes.
        graph = MockGraph([
            [AI_MESSAGE_WITH_TOOL_CALLS, INTERRUPT_WITH_ACTIONS],
            [TOOL_MESSAGE_SUCCESS],
        ])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-int") as ws:
                ws.send_text(json.dumps({"type": "message", "content": "run it"}))
                ack = ws.receive_json()
                assert ack == {"type": "ack", "ref": "message"}

                turn1 = _drain_until_complete(ws)
                assert any(e["type"] == "interrupt" for e in turn1)
                interrupt_event = next(e for e in turn1 if e["type"] == "interrupt")
                assert "action_requests" in interrupt_event

                # Client sends decision to approve
                ws.send_text(json.dumps({
                    "type": "decision",
                    "decisions": [{"type": "approve"}],
                }))
                ack2 = ws.receive_json()
                assert ack2 == {"type": "ack", "ref": "decision"}

                turn2 = _drain_until_complete(ws)
                assert any(e["type"] == "tool_end" for e in turn2)

        # Verify 2 calls: original turn + resume
        assert len(graph.calls) == 2
        from langgraph.types import Command
        assert isinstance(graph.calls[1]["input"], Command)

    def test_decision_without_decisions_key_rejected(self):
        graph = MockGraph([])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/sess-d") as ws:
                ws.send_text(json.dumps({"type": "decision"}))
                err = ws.receive_json()
                assert err["type"] == "error"
                assert "decisions" in err["error"]


# ── Session isolation ────────────────────────────────────────────────


class TestSessionIsolation:
    def test_distinct_sessions_isolated(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE], [SIMPLE_AI_MESSAGE]])
        adapter = FastAPIAdapter(graph=graph)
        app = _make_app(adapter)

        with TestClient(app) as client:
            with client.websocket_connect("/chat/alice") as ws_a:
                ws_a.send_text(json.dumps({"type": "message", "content": "hi"}))
                ws_a.receive_json()  # ack
                _drain_until_complete(ws_a)

            with client.websocket_connect("/chat/bob") as ws_b:
                ws_b.send_text(json.dumps({"type": "message", "content": "hi"}))
                ws_b.receive_json()  # ack
                _drain_until_complete(ws_b)

        assert graph.calls[0]["config"]["configurable"]["thread_id"] == "alice"
        assert graph.calls[1]["config"]["configurable"]["thread_id"] == "bob"


# ── SSE ──────────────────────────────────────────────────────────────


class TestSSE:
    def test_sse_stream_emits_events(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = FastAPIAdapter(graph=graph)
        app = FastAPI()

        @app.post("/chat/{session_id}")
        async def chat(session_id: str, body: dict):
            input_data = prepare_agent_input(message=body["message"])
            return StreamingResponse(
                adapter.sse_stream(session_id, input_data),
                media_type="text/event-stream",
            )

        with TestClient(app) as client:
            resp = client.post("/chat/sess-sse", json={"message": "hello"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            body = resp.text
            # Each SSE frame is "data: {json}\n\n"
            frames = [line for line in body.split("\n\n") if line.startswith("data: ")]
            assert len(frames) >= 2  # at least content + complete

            payloads = [json.loads(f[len("data: "):]) for f in frames]
            types = [p["type"] for p in payloads]
            assert "content" in types
            assert "complete" in types

    def test_sse_resume_helper(self):
        graph = MockGraph([[SIMPLE_AI_MESSAGE]])
        adapter = FastAPIAdapter(graph=graph)
        app = FastAPI()

        @app.post("/chat/{session_id}/resume")
        async def resume(session_id: str, body: dict):
            return StreamingResponse(
                adapter.resume(session_id, body["decisions"]),
                media_type="text/event-stream",
            )

        with TestClient(app) as client:
            resp = client.post(
                "/chat/sess/resume",
                json={"decisions": [{"type": "approve"}]},
            )
            assert resp.status_code == 200
            frames = [
                line for line in resp.text.split("\n\n")
                if line.startswith("data: ")
            ]
            assert len(frames) >= 1

        # Verify it went through as a resume (Command input)
        from langgraph.types import Command
        assert isinstance(graph.calls[0]["input"], Command)


# ── Concurrency lock ─────────────────────────────────────────────────


class TestSessionLock:
    @pytest.mark.asyncio
    async def test_same_session_serialized(self):
        """Two concurrent _iter_events calls on same session must serialize."""
        class SlowGraph:
            def __init__(self):
                self.active = 0
                self.max_concurrent = 0
                self.lock = asyncio.Lock()

            def astream(self, input_data, config=None, stream_mode="updates"):
                async def gen():
                    async with self.lock:
                        self.active += 1
                        self.max_concurrent = max(self.max_concurrent, self.active)
                    try:
                        await asyncio.sleep(0.05)
                        yield SIMPLE_AI_MESSAGE
                    finally:
                        async with self.lock:
                            self.active -= 1

                return gen()

        slow = SlowGraph()
        adapter = FastAPIAdapter(graph=slow)

        async def one_turn():
            async for _ in adapter._iter_events("shared", {"msg": "x"}):
                pass

        await asyncio.gather(one_turn(), one_turn())
        assert slow.max_concurrent == 1

    @pytest.mark.asyncio
    async def test_lock_dict_cleaned_up(self):
        """After turns complete, session locks are removed — no leak."""
        graph = MockGraph([[SIMPLE_AI_MESSAGE], [SIMPLE_AI_MESSAGE]])
        adapter = FastAPIAdapter(graph=graph)

        async def one_turn(sid):
            async for _ in adapter._iter_events(sid, {"msg": "x"}):
                pass

        await one_turn("s1")
        await one_turn("s2")

        assert adapter._locks == {}
        assert adapter._lock_refs == {}

    @pytest.mark.asyncio
    async def test_different_sessions_concurrent(self):
        class SlowGraph:
            def __init__(self):
                self.active = 0
                self.max_concurrent = 0
                self.lock = asyncio.Lock()

            def astream(self, input_data, config=None, stream_mode="updates"):
                async def gen():
                    async with self.lock:
                        self.active += 1
                        self.max_concurrent = max(self.max_concurrent, self.active)
                    try:
                        await asyncio.sleep(0.05)
                        yield SIMPLE_AI_MESSAGE
                    finally:
                        async with self.lock:
                            self.active -= 1

                return gen()

        slow = SlowGraph()
        adapter = FastAPIAdapter(graph=slow)

        async def one_turn(sid):
            async for _ in adapter._iter_events(sid, {"msg": "x"}):
                pass

        await asyncio.gather(one_turn("a"), one_turn("b"))
        assert slow.max_concurrent == 2
