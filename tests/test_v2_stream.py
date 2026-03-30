"""Tests for LangGraph v2 StreamPart support."""
import pytest
from typing import AsyncIterator, Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    InterruptEvent,
    UsageEvent,
    CustomEvent,
    ValuesEvent,
    DebugEvent,
    CompleteEvent,
    ErrorEvent,
)
from langgraph_stream_parser.parser import _is_v2_stream_part, _unwrap_v2_chunk

from .fixtures.mocks import (
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    INTERRUPT_WITH_ACTIONS,
    AI_MESSAGE_WITH_USAGE,
    MESSAGES_CHUNK_TOKEN_1,
    MESSAGES_CHUNK_TOKEN_2,
    MESSAGES_CHUNK_WITH_AGENT_NAME,
    NAMESPACE_CHILD,
    V2_UPDATES_SIMPLE,
    V2_UPDATES_TOOL_CALL,
    V2_UPDATES_TOOL_RESULT,
    V2_UPDATES_INTERRUPT,
    V2_UPDATES_WITH_USAGE,
    V2_MESSAGES_TOKEN_1,
    V2_MESSAGES_TOKEN_2,
    V2_MESSAGES_WITH_AGENT,
    V2_CUSTOM,
    V2_VALUES,
    V2_DEBUG,
    V2_CHECKPOINT,
    V2_TASKS,
    V2_SUBGRAPH_UPDATES,
    V2_SUBGRAPH_MESSAGES,
    V2_SUBGRAPH_CUSTOM,
    V2_SUBGRAPH_VALUES,
    V2_SUBGRAPH_DEBUG,
    # v1 fixtures for backward compat tests
    DUAL_MESSAGES_TOKEN_1,
    DUAL_UPDATES_SIMPLE,
)


def make_stream(chunks: list) -> Iterator:
    """Create a stream iterator from a list of chunks."""
    return iter(chunks)


async def make_async_stream(chunks: list) -> AsyncIterator:
    """Create an async stream iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


# ─── Detection helpers ────────────────────────────────────────────────


class TestIsV2StreamPart:
    """Tests for _is_v2_stream_part detection."""

    def test_valid_v2_updates(self):
        assert _is_v2_stream_part(V2_UPDATES_SIMPLE) is True

    def test_valid_v2_messages(self):
        assert _is_v2_stream_part(V2_MESSAGES_TOKEN_1) is True

    def test_valid_v2_values(self):
        assert _is_v2_stream_part(V2_VALUES) is True

    def test_valid_v2_debug(self):
        assert _is_v2_stream_part(V2_DEBUG) is True

    def test_valid_v2_checkpoints(self):
        assert _is_v2_stream_part(V2_CHECKPOINT) is True

    def test_valid_v2_tasks(self):
        assert _is_v2_stream_part(V2_TASKS) is True

    def test_valid_v2_custom(self):
        assert _is_v2_stream_part(V2_CUSTOM) is True

    def test_valid_v2_subgraph(self):
        assert _is_v2_stream_part(V2_SUBGRAPH_UPDATES) is True

    def test_negative_plain_dict(self):
        """A v1 updates dict should not be detected as v2."""
        assert _is_v2_stream_part(SIMPLE_AI_MESSAGE) is False

    def test_negative_missing_type(self):
        assert _is_v2_stream_part({"ns": (), "data": {}}) is False

    def test_negative_missing_ns(self):
        assert _is_v2_stream_part({"type": "updates", "data": {}}) is False

    def test_negative_missing_data(self):
        assert _is_v2_stream_part({"type": "updates", "ns": ()}) is False

    def test_negative_tuple(self):
        """v1 multi-mode tuple should not be detected as v2."""
        assert _is_v2_stream_part(("updates", SIMPLE_AI_MESSAGE)) is False

    def test_negative_ns_not_tuple(self):
        assert _is_v2_stream_part({"type": "updates", "ns": "root", "data": {}}) is False

    def test_negative_type_not_string(self):
        assert _is_v2_stream_part({"type": 123, "ns": (), "data": {}}) is False

    def test_negative_none(self):
        assert _is_v2_stream_part(None) is False

    def test_negative_string(self):
        assert _is_v2_stream_part("not a chunk") is False


class TestUnwrapV2Chunk:
    """Tests for _unwrap_v2_chunk."""

    def test_root_namespace(self):
        stream_type, data, namespace = _unwrap_v2_chunk(V2_UPDATES_SIMPLE)
        assert stream_type == "updates"
        assert data == SIMPLE_AI_MESSAGE
        assert namespace is None  # () maps to None

    def test_subgraph_namespace(self):
        stream_type, data, namespace = _unwrap_v2_chunk(V2_SUBGRAPH_UPDATES)
        assert stream_type == "updates"
        assert namespace == NAMESPACE_CHILD

    def test_values_type(self):
        stream_type, data, namespace = _unwrap_v2_chunk(V2_VALUES)
        assert stream_type == "values"
        assert "step" in data


# ─── Stream mode validation ──────────────────────────────────────────


class TestV2StreamModeValidation:

    def test_v2_mode_accepted(self):
        parser = StreamParser(stream_mode="v2")
        assert parser._stream_mode == "v2"

    def test_v2_not_valid_in_list(self):
        with pytest.raises(ValueError):
            StreamParser(stream_mode=["v2", "updates"])


# ─── Parsing v2 updates ──────────────────────────────────────────────


class TestV2UpdatesParsing:

    def test_content_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_UPDATES_SIMPLE])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"
        assert content_events[0].role == "assistant"

    def test_tool_call_lifecycle(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([
            V2_UPDATES_TOOL_CALL,
            V2_UPDATES_TOOL_RESULT,
        ])))
        starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(starts) == 1
        assert starts[0].name == "search"
        assert len(ends) == 1
        assert ends[0].name == "search"
        assert ends[0].status == "success"

    def test_interrupt(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_UPDATES_INTERRUPT])))
        interrupts = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupts) == 1
        assert interrupts[0].needs_approval

    def test_usage_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_UPDATES_WITH_USAGE])))
        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 150
        assert usage_events[0].output_tokens == 42


# ─── Parsing v2 messages ─────────────────────────────────────────────


class TestV2MessagesParsing:

    def test_content_tokens(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([
            V2_MESSAGES_TOKEN_1,
            V2_MESSAGES_TOKEN_2,
        ])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"

    def test_agent_name_extraction(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_MESSAGES_WITH_AGENT])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].agent_name == "researcher"
        assert content_events[0].node == "agent"


# ─── Parsing v2 new types ────────────────────────────────────────────


class TestV2NewTypes:

    def test_custom_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_CUSTOM])))
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
        assert custom_events[0].data == {"progress": 0.5, "step": "analyzing"}

    def test_values_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_VALUES])))
        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        assert len(values_events) == 1
        assert values_events[0].data["step"] == 3

    def test_debug_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_DEBUG])))
        debug_events = [e for e in events if isinstance(e, DebugEvent)]
        assert len(debug_events) == 1
        assert debug_events[0].debug_type == "debug"
        assert debug_events[0].data["event"] == "node_start"

    def test_checkpoint_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_CHECKPOINT])))
        debug_events = [e for e in events if isinstance(e, DebugEvent)]
        assert len(debug_events) == 1
        assert debug_events[0].debug_type == "checkpoint"
        assert debug_events[0].data["checkpoint_id"] == "cp_abc123"

    def test_tasks_event(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_TASKS])))
        debug_events = [e for e in events if isinstance(e, DebugEvent)]
        assert len(debug_events) == 1
        assert debug_events[0].debug_type == "task"
        assert debug_events[0].data["task_id"] == "t1"

    def test_unknown_type_skipped(self):
        parser = StreamParser(stream_mode="v2")
        chunk = {"type": "future_type", "ns": (), "data": {"foo": "bar"}}
        events = list(parser.parse(make_stream([chunk])))
        # Only CompleteEvent should be emitted
        assert len(events) == 1
        assert isinstance(events[0], CompleteEvent)

    def test_non_v2_chunk_skipped(self):
        """Non-v2 chunks mixed in should be silently skipped."""
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([
            "garbage",
            V2_UPDATES_SIMPLE,
            42,
        ])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1


# ─── Stream lifecycle ─────────────────────────────────────────────────


class TestV2StreamLifecycle:

    def test_complete_event_at_end(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_UPDATES_SIMPLE])))
        assert isinstance(events[-1], CompleteEvent)

    def test_empty_stream(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([])))
        assert len(events) == 1
        assert isinstance(events[0], CompleteEvent)

    def test_error_handling(self):
        def bad_stream():
            yield V2_UPDATES_SIMPLE
            raise RuntimeError("connection lost")

        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(bad_stream()))
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "connection lost" in error_events[0].error


# ─── Namespace handling ───────────────────────────────────────────────


class TestV2Namespace:

    def test_root_namespace_is_none(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_UPDATES_SIMPLE])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].namespace is None

    def test_subgraph_namespace_on_content(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_SUBGRAPH_UPDATES])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_subgraph_namespace_on_messages(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_SUBGRAPH_MESSAGES])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_subgraph_namespace_on_custom(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_SUBGRAPH_CUSTOM])))
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert custom_events[0].namespace == NAMESPACE_CHILD

    def test_subgraph_namespace_on_values(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_SUBGRAPH_VALUES])))
        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        assert values_events[0].namespace == NAMESPACE_CHILD

    def test_subgraph_namespace_on_debug(self):
        parser = StreamParser(stream_mode="v2")
        events = list(parser.parse(make_stream([V2_SUBGRAPH_DEBUG])))
        debug_events = [e for e in events if isinstance(e, DebugEvent)]
        assert debug_events[0].namespace == NAMESPACE_CHILD

    def test_subgraph_tool_lifecycle_namespace(self):
        """Tool start/end events from subgraph carry namespace."""
        parser = StreamParser(stream_mode="v2")
        v2_tool_call = {"type": "updates", "ns": NAMESPACE_CHILD, "data": AI_MESSAGE_WITH_TOOL_CALLS}
        v2_tool_result = {"type": "updates", "ns": NAMESPACE_CHILD, "data": TOOL_MESSAGE_SUCCESS}
        events = list(parser.parse(make_stream([v2_tool_call, v2_tool_result])))
        starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert starts[0].namespace == NAMESPACE_CHILD
        assert ends[0].namespace == NAMESPACE_CHILD


# ─── Auto-detection ───────────────────────────────────────────────────


class TestV2AutoDetect:

    def test_auto_detect_v2(self):
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([V2_UPDATES_SIMPLE])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"

    def test_auto_detect_v2_first_chunk_preserved(self):
        """The first chunk used for detection should still be processed."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([V2_VALUES])))
        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        assert len(values_events) == 1

    def test_auto_detect_v2_multi_chunk(self):
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([
            V2_UPDATES_SIMPLE,
            V2_VALUES,
            V2_CUSTOM,
        ])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(content_events) == 1
        assert len(values_events) == 1
        assert len(custom_events) == 1

    def test_auto_detect_v1_updates_still_works(self):
        """v1 plain dict updates should still be detected as updates mode."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_auto_detect_v1_multi_still_works(self):
        """v1 multi-mode tuples should still be detected as dual mode."""
        parser = StreamParser(stream_mode="auto")
        events = list(parser.parse(make_stream([
            DUAL_MESSAGES_TOKEN_1,
            DUAL_UPDATES_SIMPLE,
        ])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) >= 1


# ─── parse_chunk ──────────────────────────────────────────────────────


class TestV2ParseChunk:

    def test_parse_chunk_v2_mode(self):
        parser = StreamParser(stream_mode="v2")
        events = parser.parse_chunk(V2_UPDATES_SIMPLE)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_parse_chunk_v2_values(self):
        parser = StreamParser(stream_mode="v2")
        events = parser.parse_chunk(V2_VALUES)
        assert len(events) == 1
        assert isinstance(events[0], ValuesEvent)

    def test_parse_chunk_v2_debug(self):
        parser = StreamParser(stream_mode="v2")
        events = parser.parse_chunk(V2_DEBUG)
        assert len(events) == 1
        assert isinstance(events[0], DebugEvent)

    def test_parse_chunk_non_v2_returns_empty(self):
        parser = StreamParser(stream_mode="v2")
        events = parser.parse_chunk("not a v2 chunk")
        assert events == []

    def test_parse_chunk_v2_subgraph(self):
        parser = StreamParser(stream_mode="v2")
        events = parser.parse_chunk(V2_SUBGRAPH_UPDATES)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD


# ─── Async parsing ────────────────────────────────────────────────────


class TestV2Async:

    @pytest.mark.asyncio
    async def test_aparse_v2_updates(self):
        parser = StreamParser(stream_mode="v2")
        events = []
        async for event in parser.aparse(make_async_stream([V2_UPDATES_SIMPLE])):
            events.append(event)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    @pytest.mark.asyncio
    async def test_aparse_v2_mixed_types(self):
        parser = StreamParser(stream_mode="v2")
        events = []
        async for event in parser.aparse(make_async_stream([
            V2_UPDATES_SIMPLE,
            V2_VALUES,
            V2_DEBUG,
            V2_CUSTOM,
        ])):
            events.append(event)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        debug_events = [e for e in events if isinstance(e, DebugEvent)]
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(content_events) == 1
        assert len(values_events) == 1
        assert len(debug_events) == 1
        assert len(custom_events) == 1

    @pytest.mark.asyncio
    async def test_aparse_v2_auto_detect(self):
        parser = StreamParser(stream_mode="auto")
        events = []
        async for event in parser.aparse(make_async_stream([V2_UPDATES_SIMPLE])):
            events.append(event)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    @pytest.mark.asyncio
    async def test_aparse_v2_subgraph(self):
        parser = StreamParser(stream_mode="v2")
        events = []
        async for event in parser.aparse(make_async_stream([V2_SUBGRAPH_UPDATES])):
            events.append(event)
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].namespace == NAMESPACE_CHILD


# ─── Event serialization ─────────────────────────────────────────────


class TestV2EventSerialization:

    def test_values_event_to_dict(self):
        event = ValuesEvent(data={"step": 3})
        d = event.to_dict()
        assert d["type"] == "values"
        assert d["data"] == {"step": 3}
        assert "namespace" not in d

    def test_values_event_to_dict_with_namespace(self):
        event = ValuesEvent(data={"step": 3}, namespace=NAMESPACE_CHILD)
        d = event.to_dict()
        assert d["namespace"] == list(NAMESPACE_CHILD)

    def test_debug_event_to_dict(self):
        event = DebugEvent(data={"event": "node_start"}, debug_type="debug")
        d = event.to_dict()
        assert d["type"] == "debug"
        assert d["debug_type"] == "debug"

    def test_checkpoint_event_to_dict(self):
        event = DebugEvent(data={"id": "cp1"}, debug_type="checkpoint")
        d = event.to_dict()
        assert d["debug_type"] == "checkpoint"

    def test_task_event_to_dict(self):
        event = DebugEvent(data={"id": "t1"}, debug_type="task")
        d = event.to_dict()
        assert d["debug_type"] == "task"


# ─── Mixed v2 stream with all types ──────────────────────────────────


class TestV2FullConversation:

    def test_mixed_stream(self):
        """Simulate a realistic v2 stream with multiple types interleaved."""
        parser = StreamParser(stream_mode="v2")
        chunks = [
            V2_DEBUG,                  # debug trace: node starting
            V2_MESSAGES_TOKEN_1,       # token "Hello"
            V2_MESSAGES_TOKEN_2,       # token " world"
            V2_UPDATES_SIMPLE,         # complete update with AI message
            V2_UPDATES_TOOL_CALL,      # AI calls a tool
            V2_UPDATES_TOOL_RESULT,    # tool returns result
            V2_VALUES,                 # full state snapshot
            V2_CHECKPOINT,             # checkpoint saved
            V2_CUSTOM,                 # custom progress data
        ]
        events = list(parser.parse(make_stream(chunks)))

        # Check we got the right types (excluding CompleteEvent)
        event_types = [type(e).__name__ for e in events[:-1]]
        assert "DebugEvent" in event_types
        assert "ContentEvent" in event_types
        assert "ToolCallStartEvent" in event_types
        assert "ToolCallEndEvent" in event_types
        assert "ValuesEvent" in event_types
        assert "CustomEvent" in event_types

        # Last event is always CompleteEvent
        assert isinstance(events[-1], CompleteEvent)

    def test_subgraph_mixed_stream(self):
        """v2 stream with both root and subgraph events."""
        parser = StreamParser(stream_mode="v2")
        chunks = [
            V2_UPDATES_SIMPLE,         # root graph
            V2_SUBGRAPH_UPDATES,       # subgraph
            V2_SUBGRAPH_VALUES,        # subgraph values
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 2

        root_content = [e for e in content_events if e.namespace is None]
        sub_content = [e for e in content_events if e.namespace == NAMESPACE_CHILD]
        assert len(root_content) == 1
        assert len(sub_content) == 1

        values_events = [e for e in events if isinstance(e, ValuesEvent)]
        assert len(values_events) == 1
        assert values_events[0].namespace == NAMESPACE_CHILD


# ─── Backward compatibility ──────────────────────────────────────────


class TestV2BackwardCompatibility:

    def test_v1_updates_unchanged(self):
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_v1_messages_unchanged(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_TOKEN_1])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1

    def test_v1_multi_mode_unchanged(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([
            DUAL_MESSAGES_TOKEN_1,
            DUAL_UPDATES_SIMPLE,
        ])))
        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) >= 1

    def test_v1_custom_unchanged(self):
        parser = StreamParser(stream_mode="custom")
        events = list(parser.parse(make_stream([{"progress": 0.5}])))
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
