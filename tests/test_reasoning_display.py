"""Tests for ReasoningEvent and DisplayEvent."""
from typing import Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ReasoningEvent,
    DisplayEvent,
    ToolExtractedEvent,
    CompleteEvent,
    event_to_dict,
)

from .fixtures.mocks import (
    THINK_TOOL_MESSAGE,
    THINK_TOOL_STRING_CONTENT,
    DISPLAY_INLINE_ARTIFACT_MESSAGE,
    DISPLAY_INLINE_CONTENT_MESSAGE,
    WRITE_TODOS_MESSAGE,
    MESSAGES_CHUNK_REASONING_ONLY,
    MESSAGES_CHUNK_REASONING_AND_TEXT,
    MESSAGES_CHUNK_THINKING_BLOCK,
)


def make_stream(chunks: list) -> Iterator:
    return iter(chunks)


# ── ReasoningEvent from messages-mode content blocks ─────────────────


class TestReasoningFromContentBlocks:
    def test_reasoning_only_chunk(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_REASONING_ONLY])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        content = [e for e in events if isinstance(e, ContentEvent)]

        assert len(reasoning) == 1
        assert reasoning[0].source == "content_block"
        assert reasoning[0].content == "Let me think about this carefully..."
        assert reasoning[0].node == "agent"
        # No final ContentEvent — only the reasoning block was present
        assert len(content) == 0

    def test_reasoning_and_text_emits_both(self):
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_REASONING_AND_TEXT])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        content = [e for e in events if isinstance(e, ContentEvent)]

        assert len(reasoning) == 1
        assert reasoning[0].content == "First, I will analyze the input. "
        assert len(content) == 1
        assert content[0].content == "The answer is 42."

    def test_reasoning_before_content(self):
        """ReasoningEvent must be emitted before ContentEvent in one chunk."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_REASONING_AND_TEXT])))

        # Drop CompleteEvent at the end
        stream_events = [e for e in events if not isinstance(e, CompleteEvent)]
        assert isinstance(stream_events[0], ReasoningEvent)
        assert isinstance(stream_events[1], ContentEvent)

    def test_anthropic_thinking_block_recognized(self):
        """'thinking' type (Anthropic) is treated as reasoning."""
        parser = StreamParser(stream_mode="messages")
        events = list(parser.parse(make_stream([MESSAGES_CHUNK_THINKING_BLOCK])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        assert len(reasoning) == 1
        assert reasoning[0].content == "I should search for recent data."

    def test_to_dict_shape(self):
        event = ReasoningEvent(
            content="thinking...",
            source="content_block",
            node="agent",
        )
        d = event.to_dict()
        assert d["type"] == "reasoning"
        assert d["content"] == "thinking..."
        assert d["source"] == "content_block"
        assert d["node"] == "agent"


# ── ReasoningEvent from ThinkToolExtractor ──────────────────────────


class TestReasoningFromThinkTool:
    def test_think_tool_emits_reasoning(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([THINK_TOOL_MESSAGE])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        assert len(reasoning) == 1
        assert reasoning[0].source == "think_tool"
        assert reasoning[0].content == "I should search for more recent data."

    def test_think_tool_string_content_emits_reasoning(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([THINK_TOOL_STRING_CONTENT])))

        reasoning = [e for e in events if isinstance(e, ReasoningEvent)]
        assert len(reasoning) == 1
        assert reasoning[0].source == "think_tool"

    def test_think_tool_does_not_emit_tool_extracted(self):
        """Backward-compat guard: no generic ToolExtractedEvent for reflections."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([THINK_TOOL_MESSAGE])))

        generic = [
            e for e in events
            if isinstance(e, ToolExtractedEvent) and e.extracted_type == "reflection"
        ]
        assert len(generic) == 0


# ── DisplayEvent from DisplayInlineExtractor ────────────────────────


class TestDisplayEvent:
    def test_artifact_emits_display(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_ARTIFACT_MESSAGE])))

        displays = [e for e in events if isinstance(e, DisplayEvent)]
        assert len(displays) == 1
        d = displays[0]
        assert d.display_type == "dataframe"
        assert d.title == "Sales Data"
        assert d.status == "success"
        assert d.tool_name == "display_inline"
        assert d.tool_call_id == "call_display"

    def test_content_fallback_emits_display(self):
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_CONTENT_MESSAGE])))

        displays = [e for e in events if isinstance(e, DisplayEvent)]
        assert len(displays) == 1
        assert displays[0].display_type == "image"
        assert displays[0].title == "Chart"

    def test_display_does_not_emit_tool_extracted(self):
        """Backward-compat guard: no generic ToolExtractedEvent for display_inline."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_ARTIFACT_MESSAGE])))

        generic = [
            e for e in events
            if isinstance(e, ToolExtractedEvent)
            and e.extracted_type == "display_inline"
        ]
        assert len(generic) == 0

    def test_to_dict_shape(self):
        event = DisplayEvent(
            display_type="dataframe",
            data="<table>...</table>",
            title="Sales",
            tool_name="display_inline",
        )
        d = event.to_dict()
        assert d["type"] == "display"
        assert d["display_type"] == "dataframe"
        assert d["data"] == "<table>...</table>"
        assert d["title"] == "Sales"
        assert d["status"] == "success"
        assert d["tool_name"] == "display_inline"

    def test_to_dict_omits_none_fields(self):
        event = DisplayEvent(display_type="json", data={"k": 1})
        d = event.to_dict()
        assert "title" not in d
        assert "error" not in d
        assert "tool_name" not in d

    def test_event_to_dict_routes_correctly(self):
        """event_to_dict() dispatches DisplayEvent to its to_dict."""
        event = DisplayEvent(display_type="plotly", data='{"layout": {}}')
        d = event_to_dict(event)
        assert d["type"] == "display"
        assert d["display_type"] == "plotly"


# ── Backward compat: other extractors still use ToolExtractedEvent ──


class TestOtherExtractorsUnchanged:
    def test_todos_still_tool_extracted(self):
        """write_todos continues to emit ToolExtractedEvent."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([WRITE_TODOS_MESSAGE])))

        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        assert extracted[0].extracted_type == "todos"
        assert isinstance(extracted[0].data, list)
