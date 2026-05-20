"""Tests for langchain-core 1.4 / langgraph 1.2 / deepagents 0.6 compatibility.

Covers:
  - Cache token fields on UsageEvent populated from
    ``usage_metadata.input_token_details``.
  - Server-tool content blocks (``server_tool_call``,
    ``server_tool_result``) excluded from ContentEvent text — they
    surface only via the tool lifecycle, not as bleeding text.
  - ``ls_agent_type == "subagent"`` metadata surfaces as
    ``is_subagent`` on ContentEvent and ReasoningEvent.
"""
from typing import Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ReasoningEvent,
    UsageEvent,
)

from .fixtures.mocks import (
    AI_MESSAGE_WITH_CACHED_USAGE,
    AI_MESSAGE_WITH_SERVER_TOOL_BLOCKS,
    MESSAGES_CHUNK_FROM_SUBAGENT,
    MESSAGES_CHUNK_SUBAGENT_REASONING,
)


def make_stream(chunks: list) -> Iterator:
    return iter(chunks)


class TestUsageCacheTokens:
    def test_cache_read_and_creation_extracted(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([AI_MESSAGE_WITH_CACHED_USAGE])))

        usage = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage) == 1
        assert usage[0].cache_read_tokens == 1100
        assert usage[0].cache_creation_tokens == 50
        # Base counts untouched.
        assert usage[0].input_tokens == 1200
        assert usage[0].output_tokens == 30

    def test_cache_to_dict_carries_fields(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([AI_MESSAGE_WITH_CACHED_USAGE])))
        usage = [e for e in events if isinstance(e, UsageEvent)][0]
        d = usage.to_dict()
        assert d["cache_read_tokens"] == 1100
        assert d["cache_creation_tokens"] == 50


class TestServerToolBlocksSkippedInText:
    def test_server_tool_blocks_dropped_from_content(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([AI_MESSAGE_WITH_SERVER_TOOL_BLOCKS])))

        content = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content) == 1
        # Only the two text blocks survive; the server_tool_call /
        # server_tool_result blocks are filtered out.
        assert "Looking up the answer..." in content[0].content
        assert "It is 75F." in content[0].content
        assert "server_tool_call" not in content[0].content
        assert "server_tool_result" not in content[0].content
        assert "web_search" not in content[0].content


class TestIsSubagentMetadata:
    def test_content_event_carries_is_subagent(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_FROM_SUBAGENT])))

        content = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content) == 1
        assert content[0].is_subagent is True
        # agent_name still carried alongside the flag.
        assert content[0].agent_name == "researcher"

    def test_reasoning_event_carries_is_subagent(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_SUBAGENT_REASONING])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        assert len(reasoning) == 1
        assert reasoning[0].is_subagent is True

    def test_is_subagent_false_by_default(self):
        event = ContentEvent(content="hi")
        assert event.is_subagent is False
        # to_dict omits the flag when False to keep payloads small.
        assert "is_subagent" not in event.to_dict()

    def test_is_subagent_in_to_dict_when_true(self):
        event = ContentEvent(content="hi", is_subagent=True)
        d = event.to_dict()
        assert d["is_subagent"] is True
