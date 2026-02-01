"""Tests for event dataclasses."""
import pytest
from datetime import datetime

from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
)


class TestContentEvent:
    def test_basic_creation(self):
        event = ContentEvent(content="Hello world")
        assert event.content == "Hello world"
        assert event.node is None
        assert isinstance(event.timestamp, datetime)

    def test_with_node(self):
        event = ContentEvent(content="Hello", node="agent")
        assert event.node == "agent"


class TestToolCallStartEvent:
    def test_basic_creation(self):
        event = ToolCallStartEvent(
            id="call_123",
            name="search",
            args={"query": "test"}
        )
        assert event.id == "call_123"
        assert event.name == "search"
        assert event.args == {"query": "test"}
        assert event.node is None

    def test_with_node(self):
        event = ToolCallStartEvent(
            id="call_123",
            name="search",
            args={},
            node="agent"
        )
        assert event.node == "agent"


class TestToolCallEndEvent:
    def test_success(self):
        event = ToolCallEndEvent(
            id="call_123",
            name="search",
            result="Found 5 results",
            status="success"
        )
        assert event.status == "success"
        assert event.error_message is None

    def test_error(self):
        event = ToolCallEndEvent(
            id="call_123",
            name="search",
            result=None,
            status="error",
            error_message="API error"
        )
        assert event.status == "error"
        assert event.error_message == "API error"

    def test_with_duration(self):
        event = ToolCallEndEvent(
            id="call_123",
            name="search",
            result="result",
            status="success",
            duration_ms=150.5
        )
        assert event.duration_ms == 150.5


class TestToolExtractedEvent:
    def test_basic_creation(self):
        event = ToolExtractedEvent(
            tool_name="think_tool",
            extracted_type="reflection",
            data="I should search more"
        )
        assert event.tool_name == "think_tool"
        assert event.extracted_type == "reflection"
        assert event.data == "I should search more"

    def test_with_complex_data(self):
        todos = [{"task": "Do A", "done": False}]
        event = ToolExtractedEvent(
            tool_name="write_todos",
            extracted_type="todos",
            data=todos
        )
        assert event.data == todos


class TestInterruptEvent:
    def test_empty_interrupt(self):
        event = InterruptEvent(
            action_requests=[],
            review_configs=[]
        )
        assert event.needs_approval is False

    def test_with_action_requests(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve"]}]
        )
        assert event.needs_approval is True

    def test_raw_value_preserved(self):
        raw = {"custom": "data"}
        event = InterruptEvent(
            action_requests=[],
            review_configs=[],
            raw_value=raw
        )
        assert event.raw_value == raw


class TestStateUpdateEvent:
    def test_basic_creation(self):
        event = StateUpdateEvent(
            node="agent",
            key="current_step",
            value=3
        )
        assert event.node == "agent"
        assert event.key == "current_step"
        assert event.value == 3


class TestCompleteEvent:
    def test_basic_creation(self):
        event = CompleteEvent()
        assert isinstance(event.timestamp, datetime)


class TestErrorEvent:
    def test_basic_creation(self):
        event = ErrorEvent(error="Something went wrong")
        assert event.error == "Something went wrong"
        assert event.exception is None

    def test_with_exception(self):
        exc = ValueError("Bad value")
        event = ErrorEvent(error="Bad value", exception=exc)
        assert event.exception is exc
