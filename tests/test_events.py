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
    UsageEvent,
    CompleteEvent,
    ErrorEvent,
    event_to_dict,
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


class TestUsageEvent:
    def test_basic_creation(self):
        event = UsageEvent(input_tokens=100, output_tokens=50, total_tokens=150)
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.node is None
        assert isinstance(event.timestamp, datetime)

    def test_with_node(self):
        event = UsageEvent(input_tokens=10, output_tokens=5, total_tokens=15, node="agent")
        assert event.node == "agent"


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


class TestToDict:
    """Tests for to_dict() methods and event_to_dict() function."""

    def test_content_event_to_dict(self):
        event = ContentEvent(content="Hello", role="assistant", node="agent")
        d = event.to_dict()
        assert d["type"] == "content"
        assert d["content"] == "Hello"
        assert d["role"] == "assistant"
        assert d["node"] == "agent"

    def test_tool_call_start_to_dict(self):
        event = ToolCallStartEvent(
            id="call_1", name="search", args={"query": "test"}, node="agent"
        )
        d = event.to_dict()
        assert d["type"] == "tool_start"
        assert d["id"] == "call_1"
        assert d["name"] == "search"
        assert d["args"] == {"query": "test"}

    def test_tool_call_end_to_dict(self):
        event = ToolCallEndEvent(
            id="call_1",
            name="search",
            result="Found results",
            status="success",
            duration_ms=150.0,
        )
        d = event.to_dict()
        assert d["type"] == "tool_end"
        assert d["status"] == "success"
        assert d["duration_ms"] == 150.0

    def test_tool_call_end_truncates_long_result(self):
        long_result = "x" * 1000
        event = ToolCallEndEvent(
            id="call_1", name="search", result=long_result, status="success"
        )
        d = event.to_dict(max_result_len=100)
        assert len(d["result"]) == 103  # 100 + "..."
        assert d["result"].endswith("...")

    def test_tool_extracted_to_dict(self):
        event = ToolExtractedEvent(
            tool_name="think_tool", extracted_type="reflection", data="My thoughts"
        )
        d = event.to_dict()
        assert d["type"] == "extraction"
        assert d["tool_name"] == "think_tool"
        assert d["extracted_type"] == "reflection"
        assert d["data"] == "My thoughts"

    def test_interrupt_to_dict(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve", "reject"]}],
        )
        d = event.to_dict()
        assert d["type"] == "interrupt"
        assert d["action_requests"] == [{"tool": "bash", "args": {"cmd": "ls"}}]
        assert set(d["allowed_decisions"]) == {"approve", "reject"}

    def test_interrupt_allowed_decisions_property(self):
        event = InterruptEvent(
            action_requests=[],
            review_configs=[{"allowed_decisions": ["approve", "edit"]}],
        )
        assert event.allowed_decisions == {"approve", "edit"}

    def test_interrupt_allowed_decisions_default(self):
        event = InterruptEvent(action_requests=[], review_configs=[])
        assert event.allowed_decisions == {"approve", "reject"}

    def test_interrupt_build_decisions(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash"}, {"tool": "write"}],
            review_configs=[],
        )
        decisions = event.build_decisions("approve")
        assert len(decisions) == 2
        assert all(d["type"] == "approve" for d in decisions)

    def test_interrupt_build_decisions_with_modifier(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "rm -rf /"}}],
            review_configs=[],
        )
        decisions = event.build_decisions(
            "edit", args_modifier=lambda args: {"cmd": "ls", "safe": True}
        )
        assert decisions[0]["type"] == "edit"
        assert decisions[0]["args"] == {"cmd": "ls", "safe": True}

    def test_state_update_to_dict(self):
        event = StateUpdateEvent(node="agent", key="step", value=3)
        d = event.to_dict()
        assert d["type"] == "state_update"
        assert d["node"] == "agent"
        assert d["key"] == "step"
        assert d["value"] == 3

    def test_complete_to_dict(self):
        event = CompleteEvent()
        d = event.to_dict()
        assert d == {"type": "complete"}

    def test_usage_event_to_dict(self):
        event = UsageEvent(
            input_tokens=100, output_tokens=50, total_tokens=150, node="agent"
        )
        d = event.to_dict()
        assert d["type"] == "usage"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["node"] == "agent"

    def test_error_to_dict(self):
        event = ErrorEvent(error="Something went wrong")
        d = event.to_dict()
        assert d["type"] == "error"
        assert d["error"] == "Something went wrong"

    def test_event_to_dict_function(self):
        """Test the convenience function works for all event types."""
        events = [
            ContentEvent(content="Hi"),
            ToolCallStartEvent(id="1", name="test", args={}),
            ToolCallEndEvent(id="1", name="test", result="ok", status="success"),
            ToolExtractedEvent(tool_name="t", extracted_type="x", data="d"),
            InterruptEvent(action_requests=[], review_configs=[]),
            StateUpdateEvent(node="n", key="k", value="v"),
            UsageEvent(input_tokens=10, output_tokens=5, total_tokens=15),
            CompleteEvent(),
            ErrorEvent(error="err"),
        ]
        for event in events:
            d = event_to_dict(event)
            assert "type" in d
