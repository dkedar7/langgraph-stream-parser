"""Tests for subagent streaming: namespace preservation, custom mode, and agent_name."""
import pytest
from typing import AsyncIterator, Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    InterruptEvent,
    CustomEvent,
    CompleteEvent,
    ErrorEvent,
)

from .fixtures.mocks import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    TOOL_MESSAGE_SUCCESS,
    INTERRUPT_WITH_ACTIONS,
    MESSAGES_METADATA,
    MESSAGES_CHUNK_TOKEN_1,
    NAMESPACE_PARENT,
    NAMESPACE_CHILD,
    SUBGRAPH_SINGLE_PARENT,
    SUBGRAPH_SINGLE_CHILD,
    SUBGRAPH_MULTI_PARENT_MSG,
    SUBGRAPH_MULTI_CHILD_MSG,
    SUBGRAPH_MULTI_PARENT_UPD,
    SUBGRAPH_MULTI_CHILD_UPD,
    SUBGRAPH_MULTI_CHILD_TOOL_RESULT,
    SUBGRAPH_MULTI_CHILD_INTERRUPT,
    CUSTOM_CHUNK_SIMPLE,
    DUAL_CUSTOM_CHUNK,
    SUBGRAPH_CUSTOM_CHILD,
    MESSAGES_METADATA_WITH_AGENT,
    MESSAGES_CHUNK_WITH_AGENT_NAME,
    DUAL_MESSAGES_WITH_AGENT,
    SUBGRAPH_MULTI_CHILD_MSG_WITH_AGENT,
)


def make_stream(chunks: list) -> Iterator:
    return iter(chunks)


async def make_async_stream(chunks: list) -> AsyncIterator:
    for chunk in chunks:
        yield chunk


# ── Namespace preservation ────────────────────────────────────────────


class TestNamespacePreservation:
    """Namespace is preserved on events from subgraph chunks."""

    def test_parent_namespace_is_none(self):
        """Parent graph events have namespace=None."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_PARENT])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace is None

    def test_child_namespace_preserved(self):
        """Subgraph events carry the namespace tuple."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_CHILD])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_child_tool_events_have_namespace(self):
        """Tool call events from subgraphs carry namespace."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([
            SUBGRAPH_MULTI_CHILD_UPD,
            SUBGRAPH_MULTI_CHILD_TOOL_RESULT,
        ])))

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].namespace == NAMESPACE_CHILD
        assert len(tool_ends) == 1
        assert tool_ends[0].namespace == NAMESPACE_CHILD

    def test_child_interrupt_has_namespace(self):
        """Interrupt events from subgraphs carry namespace."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_CHILD_INTERRUPT])))

        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1
        assert interrupt_events[0].namespace == NAMESPACE_CHILD

    def test_multi_mode_parent_messages_no_namespace(self):
        """Parent messages in multi-mode have namespace=None."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_PARENT_MSG])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace is None

    def test_multi_mode_child_messages_have_namespace(self):
        """Child messages in multi-mode carry namespace."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([SUBGRAPH_MULTI_CHILD_MSG])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_parse_chunk_preserves_namespace(self):
        """parse_chunk also preserves namespace."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = parser.parse_chunk(SUBGRAPH_MULTI_CHILD_MSG)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_plain_dict_no_namespace(self):
        """Plain dict chunks (no subgraphs=True) have namespace=None."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].namespace is None

    def test_to_dict_includes_namespace(self):
        """to_dict() includes namespace when present."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SUBGRAPH_SINGLE_CHILD])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        d = content_events[0].to_dict()
        assert "namespace" in d
        assert d["namespace"] == list(NAMESPACE_CHILD)

    def test_to_dict_omits_namespace_when_none(self):
        """to_dict() omits namespace when it's None."""
        parser = StreamParser(stream_mode="updates")
        events = list(parser.parse(make_stream([SIMPLE_AI_MESSAGE])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        d = content_events[0].to_dict()
        assert "namespace" not in d


class TestNamespaceAsync:
    """Async parsing preserves namespace."""

    @pytest.mark.asyncio
    async def test_aparse_single_mode_namespace(self):
        parser = StreamParser(stream_mode="updates")
        events = []
        async for event in parser.aparse(make_async_stream([
            SUBGRAPH_SINGLE_PARENT,
            SUBGRAPH_SINGLE_CHILD,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].namespace is None
        assert content_events[1].namespace == NAMESPACE_CHILD

    @pytest.mark.asyncio
    async def test_aparse_multi_mode_namespace(self):
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = []
        async for event in parser.aparse(make_async_stream([
            SUBGRAPH_MULTI_PARENT_MSG,
            SUBGRAPH_MULTI_CHILD_MSG,
        ])):
            events.append(event)

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert content_events[0].namespace is None
        assert content_events[1].namespace == NAMESPACE_CHILD


# ── Custom mode ───────────────────────────────────────────────────────


class TestCustomMode:
    """stream_mode='custom' support."""

    def test_single_custom_mode(self):
        """Single custom mode yields CustomEvent."""
        parser = StreamParser(stream_mode="custom")
        events = list(parser.parse(make_stream([CUSTOM_CHUNK_SIMPLE])))

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
        assert custom_events[0].data == CUSTOM_CHUNK_SIMPLE
        assert custom_events[0].namespace is None
        assert isinstance(events[-1], CompleteEvent)

    def test_multi_mode_with_custom(self):
        """Custom chunks in multi-mode yield CustomEvent."""
        parser = StreamParser(stream_mode=["updates", "messages", "custom"])
        events = list(parser.parse(make_stream([DUAL_CUSTOM_CHUNK])))

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
        assert custom_events[0].data == {"progress": 0.75, "step": "writing"}

    def test_subgraph_custom_has_namespace(self):
        """Custom chunks from subgraphs carry namespace."""
        parser = StreamParser(stream_mode=["updates", "messages", "custom"])
        events = list(parser.parse(make_stream([SUBGRAPH_CUSTOM_CHILD])))

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
        assert custom_events[0].namespace == NAMESPACE_CHILD
        assert custom_events[0].data == {"progress": 1.0, "step": "done"}

    def test_custom_mode_validation(self):
        """'custom' is a valid stream_mode value."""
        parser = StreamParser(stream_mode="custom")
        assert parser._stream_mode == "custom"

    def test_custom_in_list_validation(self):
        """'custom' is valid in stream_mode list."""
        parser = StreamParser(stream_mode=["updates", "custom"])
        assert parser._stream_mode == ["updates", "custom"]

    def test_parse_chunk_custom_single(self):
        """parse_chunk works for single custom mode."""
        parser = StreamParser(stream_mode="custom")
        events = parser.parse_chunk(CUSTOM_CHUNK_SIMPLE)

        assert len(events) == 1
        assert isinstance(events[0], CustomEvent)
        assert events[0].data == CUSTOM_CHUNK_SIMPLE

    def test_parse_chunk_custom_multi(self):
        """parse_chunk works for custom in multi-mode."""
        parser = StreamParser(stream_mode=["updates", "messages", "custom"])
        events = parser.parse_chunk(DUAL_CUSTOM_CHUNK)

        assert len(events) == 1
        assert isinstance(events[0], CustomEvent)

    def test_custom_event_to_dict(self):
        """CustomEvent.to_dict() produces correct output."""
        event = CustomEvent(data={"key": "value"}, namespace=("sub:123",))
        d = event.to_dict()
        assert d["type"] == "custom"
        assert d["data"] == {"key": "value"}
        assert d["namespace"] == ["sub:123"]

    def test_custom_event_to_dict_no_namespace(self):
        """CustomEvent.to_dict() omits namespace when None."""
        event = CustomEvent(data="hello")
        d = event.to_dict()
        assert d["type"] == "custom"
        assert d["data"] == "hello"
        assert "namespace" not in d

    def test_mixed_custom_and_updates(self):
        """Custom and updates chunks coexist in multi-mode."""
        parser = StreamParser(stream_mode=["updates", "messages", "custom"])
        chunks = [
            ("messages", MESSAGES_CHUNK_TOKEN_1),
            DUAL_CUSTOM_CHUNK,
            ("updates", SIMPLE_AI_MESSAGE),  # content suppressed in dual mode
        ]
        events = list(parser.parse(make_stream(chunks)))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(content_events) == 1  # from messages only
        assert len(custom_events) == 1
        assert isinstance(events[-1], CompleteEvent)


class TestCustomModeAsync:
    @pytest.mark.asyncio
    async def test_aparse_custom_single(self):
        parser = StreamParser(stream_mode="custom")
        events = []
        async for event in parser.aparse(make_async_stream([CUSTOM_CHUNK_SIMPLE])):
            events.append(event)

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1
        assert custom_events[0].data == CUSTOM_CHUNK_SIMPLE
        assert isinstance(events[-1], CompleteEvent)

    @pytest.mark.asyncio
    async def test_aparse_custom_multi(self):
        parser = StreamParser(stream_mode=["updates", "messages", "custom"])
        events = []
        async for event in parser.aparse(make_async_stream([
            DUAL_CUSTOM_CHUNK,
            SUBGRAPH_CUSTOM_CHILD,
        ])):
            events.append(event)

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 2
        assert custom_events[0].namespace is None
        assert custom_events[1].namespace == NAMESPACE_CHILD


# ── Agent name extraction ─────────────────────────────────────────────


class TestAgentName:
    """lc_agent_name extraction from messages metadata."""

    def test_agent_name_extracted(self):
        """Agent name is extracted from metadata."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_WITH_AGENT_NAME])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].agent_name == "researcher"

    def test_agent_name_none_when_missing(self):
        """Agent name is None when not in metadata."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_TOKEN_1])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].agent_name is None

    def test_agent_name_in_dual_mode(self):
        """Agent name works in dual mode."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([DUAL_MESSAGES_WITH_AGENT])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].agent_name == "researcher"

    def test_agent_name_with_namespace(self):
        """Agent name and namespace work together."""
        parser = StreamParser(stream_mode=["updates", "messages"])
        events = list(parser.parse(make_stream([
            SUBGRAPH_MULTI_CHILD_MSG_WITH_AGENT,
        ])))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].agent_name == "researcher"
        assert content_events[0].namespace == NAMESPACE_CHILD

    def test_agent_name_to_dict(self):
        """to_dict() includes agent_name when present."""
        event = ContentEvent(
            content="test", agent_name="researcher", node="agent"
        )
        d = event.to_dict()
        assert d["agent_name"] == "researcher"

    def test_agent_name_to_dict_omitted_when_none(self):
        """to_dict() omits agent_name when None."""
        event = ContentEvent(content="test", node="agent")
        d = event.to_dict()
        assert "agent_name" not in d
