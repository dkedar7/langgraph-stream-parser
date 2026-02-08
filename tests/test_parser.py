"""Tests for the main StreamParser class."""
import pytest
from typing import Iterator

from langgraph_stream_parser import (
    StreamParser,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    UsageEvent,
    CompleteEvent,
    ErrorEvent,
)

from .fixtures.mocks import (
    SIMPLE_AI_MESSAGE,
    AI_MESSAGE_WITH_TOOL_CALLS,
    AI_MESSAGE_WITH_CONTENT_AND_TOOLS,
    AI_MESSAGE_WITH_USAGE,
    TOOL_MESSAGE_SUCCESS,
    TOOL_MESSAGE_ERROR,
    TOOL_MESSAGE_ERROR_PREFIX,
    THINK_TOOL_MESSAGE,
    THINK_TOOL_STRING_CONTENT,
    WRITE_TODOS_MESSAGE,
    WRITE_TODOS_EMBEDDED,
    DISPLAY_INLINE_ARTIFACT_MESSAGE,
    DISPLAY_INLINE_CONTENT_MESSAGE,
    INTERRUPT_SIMPLE,
    INTERRUPT_WITH_ACTIONS,
    INTERRUPT_MULTIPLE_ACTIONS,
    STATE_UPDATE_WITH_EXTRA_KEYS,
    MULTI_MESSAGE_CONTENT,
)


def make_stream(chunks: list) -> Iterator:
    """Create a stream iterator from a list of chunks."""
    return iter(chunks)


class TestStreamParserInit:
    def test_default_options(self):
        parser = StreamParser()
        assert parser._track_tool_lifecycle is True
        assert parser._skip_tools == set()
        assert parser._include_state_updates is False

    def test_custom_options(self):
        parser = StreamParser(
            track_tool_lifecycle=False,
            skip_tools=["tool1", "tool2"],
            include_state_updates=True,
        )
        assert parser._track_tool_lifecycle is False
        assert parser._skip_tools == {"tool1", "tool2"}
        assert parser._include_state_updates is True

    def test_builtin_extractors_registered(self):
        parser = StreamParser()
        assert "think_tool" in parser._extractors
        assert "write_todos" in parser._extractors


class TestStreamParserParse:
    def test_simple_ai_message(self):
        parser = StreamParser()
        stream = make_stream([SIMPLE_AI_MESSAGE])

        events = list(parser.parse(stream))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello, how can I help?"
        assert content_events[0].node == "agent"

        # Should end with CompleteEvent
        assert isinstance(events[-1], CompleteEvent)

    def test_ai_message_with_tool_calls(self):
        parser = StreamParser()
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS])

        events = list(parser.parse(stream))

        tool_start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tool_start_events) == 1
        assert tool_start_events[0].name == "search"
        assert tool_start_events[0].args == {"query": "weather"}
        assert tool_start_events[0].node == "agent"

    def test_ai_message_with_content_and_tools(self):
        parser = StreamParser()
        stream = make_stream([AI_MESSAGE_WITH_CONTENT_AND_TOOLS])

        events = list(parser.parse(stream))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        tool_events = [e for e in events if isinstance(e, ToolCallStartEvent)]

        assert len(content_events) == 1
        assert len(tool_events) == 1
        assert "search" in content_events[0].content or tool_events[0].name == "search"

    def test_tool_message_success(self):
        parser = StreamParser()
        # First send tool call, then result
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS, TOOL_MESSAGE_SUCCESS])

        events = list(parser.parse(stream))

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].status == "success"
        assert end_events[0].name == "search"
        assert "sunny" in end_events[0].result

    def test_tool_message_error(self):
        parser = StreamParser()
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS, TOOL_MESSAGE_ERROR])

        events = list(parser.parse(stream))

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].status == "error"

    def test_tool_message_error_from_prefix(self):
        parser = StreamParser()
        stream = make_stream([TOOL_MESSAGE_ERROR_PREFIX])

        events = list(parser.parse(stream))

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].status == "error"

    def test_think_tool_extraction(self):
        parser = StreamParser()
        stream = make_stream([THINK_TOOL_MESSAGE])

        events = list(parser.parse(stream))

        extracted_events = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted_events) == 1
        assert extracted_events[0].tool_name == "think_tool"
        assert extracted_events[0].extracted_type == "reflection"
        assert extracted_events[0].data == "I should search for more recent data."

    def test_think_tool_string_content(self):
        parser = StreamParser()
        stream = make_stream([THINK_TOOL_STRING_CONTENT])

        events = list(parser.parse(stream))

        extracted_events = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted_events) == 1
        assert "reflection" in extracted_events[0].data.lower() or "problem" in extracted_events[0].data.lower()

    def test_write_todos_extraction(self):
        parser = StreamParser()
        stream = make_stream([WRITE_TODOS_MESSAGE])

        events = list(parser.parse(stream))

        extracted_events = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted_events) == 1
        assert extracted_events[0].extracted_type == "todos"
        assert len(extracted_events[0].data) == 2

    def test_write_todos_embedded(self):
        parser = StreamParser()
        stream = make_stream([WRITE_TODOS_EMBEDDED])

        events = list(parser.parse(stream))

        extracted_events = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted_events) == 1
        assert len(extracted_events[0].data) == 2

    def test_interrupt_with_actions(self):
        parser = StreamParser()
        stream = make_stream([INTERRUPT_WITH_ACTIONS])

        events = list(parser.parse(stream))

        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1

        interrupt = interrupt_events[0]
        assert interrupt.needs_approval is True
        assert len(interrupt.action_requests) == 1
        assert interrupt.action_requests[0]["tool"] == "bash"
        assert len(interrupt.review_configs) == 1
        assert "approve" in interrupt.review_configs[0]["allowed_decisions"]

    def test_interrupt_multiple_actions(self):
        parser = StreamParser()
        stream = make_stream([INTERRUPT_MULTIPLE_ACTIONS])

        events = list(parser.parse(stream))

        interrupt_events = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupt_events) == 1
        assert len(interrupt_events[0].action_requests) == 2

    def test_state_updates_disabled_by_default(self):
        parser = StreamParser()
        stream = make_stream([STATE_UPDATE_WITH_EXTRA_KEYS])

        events = list(parser.parse(stream))

        state_events = [e for e in events if isinstance(e, StateUpdateEvent)]
        assert len(state_events) == 0

    def test_state_updates_enabled(self):
        parser = StreamParser(include_state_updates=True)
        stream = make_stream([STATE_UPDATE_WITH_EXTRA_KEYS])

        events = list(parser.parse(stream))

        state_events = [e for e in events if isinstance(e, StateUpdateEvent)]
        assert len(state_events) == 2  # current_step and total_steps
        keys = {e.key for e in state_events}
        assert "current_step" in keys
        assert "total_steps" in keys

    def test_multi_content_message(self):
        parser = StreamParser()
        stream = make_stream([MULTI_MESSAGE_CONTENT])

        events = list(parser.parse(stream))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        assert len(content_events) == 1
        assert "42" in content_events[0].content


class TestStreamParserSkipTools:
    def test_skip_tools(self):
        parser = StreamParser(skip_tools=["search"])
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS])

        events = list(parser.parse(stream))

        tool_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(tool_events) == 0

    def test_skip_tools_no_end_event(self):
        parser = StreamParser(skip_tools=["search"])
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS, TOOL_MESSAGE_SUCCESS])

        events = list(parser.parse(stream))

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 0


class TestStreamParserLifecycleTracking:
    def test_disable_lifecycle_tracking(self):
        parser = StreamParser(track_tool_lifecycle=False)
        stream = make_stream([AI_MESSAGE_WITH_TOOL_CALLS, TOOL_MESSAGE_SUCCESS])

        events = list(parser.parse(stream))

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]

        assert len(start_events) == 0
        assert len(end_events) == 0


class TestStreamParserCustomExtractor:
    def test_register_custom_extractor(self):
        class CanvasExtractor:
            tool_name = "add_to_canvas"
            extracted_type = "canvas_item"

            def extract(self, content):
                if isinstance(content, dict):
                    return content
                return {"type": "text", "data": str(content)}

        parser = StreamParser()
        parser.register_extractor(CanvasExtractor())

        assert "add_to_canvas" in parser._extractors

    def test_unregister_extractor(self):
        parser = StreamParser()
        assert "think_tool" in parser._extractors

        parser.unregister_extractor("think_tool")
        assert "think_tool" not in parser._extractors


class TestStreamParserParseChunk:
    def test_parse_single_chunk(self):
        parser = StreamParser()
        events = parser.parse_chunk(SIMPLE_AI_MESSAGE)

        assert len(events) == 1
        assert isinstance(events[0], ContentEvent)

    def test_parse_interrupt_chunk(self):
        parser = StreamParser()
        events = parser.parse_chunk(INTERRUPT_WITH_ACTIONS)

        assert len(events) == 1
        assert isinstance(events[0], InterruptEvent)


class TestStreamParserErrors:
    def test_exception_yields_error_event(self):
        parser = StreamParser()

        def bad_stream():
            yield SIMPLE_AI_MESSAGE
            raise ValueError("Stream error!")

        events = list(parser.parse(bad_stream()))

        # Should have content event, then error event
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "Stream error" in error_events[0].error

    def test_unsupported_stream_mode(self):
        # Unsupported stream mode raises ValueError at construction time
        with pytest.raises(ValueError, match="Unsupported stream_mode"):
            StreamParser(stream_mode="values")


class TestStreamParserUsage:
    def test_usage_event_emitted(self):
        parser = StreamParser()
        stream = make_stream([AI_MESSAGE_WITH_USAGE])

        events = list(parser.parse(stream))

        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 150
        assert usage_events[0].output_tokens == 42
        assert usage_events[0].total_tokens == 192
        assert usage_events[0].node == "agent"

    def test_no_usage_event_without_metadata(self):
        parser = StreamParser()
        stream = make_stream([SIMPLE_AI_MESSAGE])

        events = list(parser.parse(stream))

        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(usage_events) == 0

    def test_usage_with_content(self):
        """Usage event should be emitted alongside content."""
        parser = StreamParser()
        stream = make_stream([AI_MESSAGE_WITH_USAGE])

        events = list(parser.parse(stream))

        content_events = [e for e in events if isinstance(e, ContentEvent)]
        usage_events = [e for e in events if isinstance(e, UsageEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Done."
        assert len(usage_events) == 1


class TestStreamParserDisplayInline:
    def test_artifact_extraction(self):
        """When ToolMessage has artifact, extractor receives the artifact dict."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_ARTIFACT_MESSAGE])))

        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        assert extracted[0].tool_name == "display_inline"
        assert extracted[0].extracted_type == "display_inline"
        assert extracted[0].data["display_type"] == "dataframe"
        assert extracted[0].data["title"] == "Sales Data"

    def test_artifact_extraction_tool_end_has_stub(self):
        """ToolCallEndEvent.result should be the stub content, not the artifact."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_ARTIFACT_MESSAGE])))

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        # The result is the ToolMessage.content (the stub), not the artifact
        assert end_events[0].result == "Displayed dataframe inline: Sales Data"

    def test_content_fallback_extraction(self):
        """When no artifact, extractor falls back to parsing content as JSON."""
        parser = StreamParser()
        events = list(parser.parse(make_stream([DISPLAY_INLINE_CONTENT_MESSAGE])))

        extracted = [e for e in events if isinstance(e, ToolExtractedEvent)]
        assert len(extracted) == 1
        assert extracted[0].data["display_type"] == "image"
        assert extracted[0].data["title"] == "Chart"


class TestStreamParserReset:
    def test_reset_clears_pending(self):
        parser = StreamParser()

        # Process a tool call start
        parser.parse_chunk(AI_MESSAGE_WITH_TOOL_CALLS)
        assert len(parser._pending_tool_calls) > 0

        # Reset should clear it
        parser.reset()
        assert len(parser._pending_tool_calls) == 0
