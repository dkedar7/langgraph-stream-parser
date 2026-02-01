"""Tests for Jupyter adapter."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from langgraph_stream_parser.adapters.jupyter import (
    JupyterDisplay,
    ToolStatus,
    ToolState,
)
from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    CompleteEvent,
    ErrorEvent,
)


class TestToolState:
    def test_duration_ms_not_ended(self):
        state = ToolState(id="1", name="test", args={})
        assert state.duration_ms is None

    def test_duration_ms_calculated(self):
        start = datetime.now()
        end = start + timedelta(milliseconds=500)
        state = ToolState(
            id="1", name="test", args={},
            start_time=start, end_time=end
        )
        assert state.duration_ms is not None
        assert 490 <= state.duration_ms <= 510  # Allow small variance


class TestJupyterDisplayInit:
    def test_default_options(self):
        display = JupyterDisplay()
        assert display._show_timestamps is False
        assert display._show_tool_args is True
        assert display._max_content_preview == 200

    def test_custom_options(self):
        display = JupyterDisplay(
            show_timestamps=True,
            show_tool_args=False,
            max_content_preview=100,
        )
        assert display._show_timestamps is True
        assert display._show_tool_args is False
        assert display._max_content_preview == 100


class TestJupyterDisplayHelpers:
    def setup_method(self):
        self.display = JupyterDisplay()

    def test_get_status_icon_pending(self):
        icon = self.display._get_status_icon(ToolStatus.PENDING)
        assert "..." in icon

    def test_get_status_icon_running(self):
        icon = self.display._get_status_icon(ToolStatus.RUNNING)
        assert "..." in icon
        assert "yellow" in icon

    def test_get_status_icon_success(self):
        icon = self.display._get_status_icon(ToolStatus.SUCCESS)
        assert "OK" in icon
        assert "green" in icon

    def test_get_status_icon_error(self):
        icon = self.display._get_status_icon(ToolStatus.ERROR)
        assert "ERR" in icon
        assert "red" in icon

    def test_format_duration_none(self):
        assert self.display._format_duration(None) == ""

    def test_format_duration_ms(self):
        assert self.display._format_duration(500) == "500ms"

    def test_format_duration_seconds(self):
        assert self.display._format_duration(1500) == "1.5s"

    def test_format_args_empty(self):
        assert self.display._format_args({}) == ""

    def test_format_args_simple(self):
        result = self.display._format_args({"key": "value"})
        assert "key=value" in result

    def test_format_args_truncates_long_values(self):
        long_value = "x" * 50
        result = self.display._format_args({"key": long_value})
        assert "..." in result
        assert len(result) <= 45  # key= plus truncated value


class TestJupyterDisplayEventProcessing:
    def setup_method(self):
        self.display = JupyterDisplay()

    def test_process_content_event(self):
        event = ContentEvent(content="Hello ", role="assistant")
        self.display._process_event(event)
        assert self.display._current_content == "Hello "
        assert self.display._current_role == "assistant"

        event2 = ContentEvent(content="World", role="assistant")
        self.display._process_event(event2)
        assert self.display._current_content == "Hello World"

    def test_process_content_event_role_change(self):
        # Assistant message
        self.display._process_event(ContentEvent(content="Hi!", role="assistant"))
        assert self.display._current_content == "Hi!"

        # Human message - should flush previous
        self.display._process_event(ContentEvent(content="Hello", role="human"))
        # Check display_items instead of _messages
        message_items = [item for item in self.display._display_items if item[0] == "message"]
        assert len(message_items) == 1
        assert message_items[0][1] == ("assistant", "Hi!")
        assert self.display._current_content == "Hello"
        assert self.display._current_role == "human"

    def test_process_tool_start_event(self):
        event = ToolCallStartEvent(
            id="call_1",
            name="search",
            args={"query": "test"},
        )
        self.display._process_event(event)

        assert "call_1" in self.display._tools
        tool = self.display._tools["call_1"]
        assert tool.name == "search"
        assert tool.status == ToolStatus.RUNNING

    def test_process_tool_end_event_success(self):
        # First start the tool
        start_event = ToolCallStartEvent(
            id="call_1", name="search", args={}
        )
        self.display._process_event(start_event)

        # Then end it
        end_event = ToolCallEndEvent(
            id="call_1",
            name="search",
            result="Found results",
            status="success",
        )
        self.display._process_event(end_event)

        tool = self.display._tools["call_1"]
        assert tool.status == ToolStatus.SUCCESS
        assert tool.end_time is not None

    def test_process_tool_end_event_error(self):
        start_event = ToolCallStartEvent(
            id="call_1", name="search", args={}
        )
        self.display._process_event(start_event)

        end_event = ToolCallEndEvent(
            id="call_1",
            name="search",
            result=None,
            status="error",
            error_message="Connection failed",
        )
        self.display._process_event(end_event)

        tool = self.display._tools["call_1"]
        assert tool.status == ToolStatus.ERROR
        assert tool.error_message == "Connection failed"

    def test_process_extraction_event(self):
        event = ToolExtractedEvent(
            tool_name="think_tool",
            extracted_type="reflection",
            data="My thoughts",
        )
        self.display._process_event(event)

        extraction_items = [item for item in self.display._display_items if item[0] == "extraction"]
        assert len(extraction_items) == 1
        assert extraction_items[0][1].data == "My thoughts"

    def test_process_interrupt_event(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve", "reject"]}],
        )
        self.display._process_event(event)

        assert self.display._interrupt is not None
        assert len(self.display._interrupt.action_requests) == 1

    def test_process_error_event(self):
        event = ErrorEvent(error="Something went wrong")
        self.display._process_event(event)

        assert self.display._error is not None
        assert self.display._error.error == "Something went wrong"

    def test_process_complete_event(self):
        event = CompleteEvent()
        self.display._process_event(event)

        assert self.display._complete is True


class TestJupyterDisplayReset:
    def test_reset_clears_state(self):
        display = JupyterDisplay()

        # Populate some state
        display._display_items.append(("message", ("assistant", "Some content")))
        display._current_content = "In progress"
        display._current_role = "assistant"
        display._tools["call_1"] = ToolState(id="1", name="test", args={})
        display._tools_item_index = 1
        display._interrupt = InterruptEvent(action_requests=[], review_configs=[])
        display._error = ErrorEvent(error="err")
        display._complete = True

        # Reset
        display.reset()

        assert len(display._display_items) == 0
        assert display._current_content == ""
        assert display._current_role is None
        assert len(display._tools) == 0
        assert display._tools_item_index is None
        assert display._interrupt is None
        assert display._error is None
        assert display._complete is False


class TestJupyterDisplayDependencyCheck:
    def test_raises_without_rich(self):
        display = JupyterDisplay()
        display._rich_available = False
        display._ipython_available = True

        with pytest.raises(ImportError) as exc_info:
            display._check_dependencies()

        assert "rich is required" in str(exc_info.value)

    def test_raises_without_ipython(self):
        display = JupyterDisplay()
        display._rich_available = True
        display._ipython_available = False

        with pytest.raises(ImportError) as exc_info:
            display._check_dependencies()

        assert "IPython is required" in str(exc_info.value)


class TestJupyterDisplayRendering:
    """Test rendering with mocked IPython."""

    def setup_method(self):
        self.display = JupyterDisplay()

    @patch('langgraph_stream_parser.adapters.jupyter.JupyterDisplay._check_dependencies')
    @patch('langgraph_stream_parser.adapters.jupyter.JupyterDisplay._render')
    def test_update_calls_render(self, mock_render, mock_check):
        event = ContentEvent(content="test")
        self.display.update(event)

        mock_check.assert_called_once()
        mock_render.assert_called_once()

    @patch('langgraph_stream_parser.adapters.jupyter.JupyterDisplay._check_dependencies')
    @patch('langgraph_stream_parser.adapters.jupyter.JupyterDisplay._render')
    def test_stream_processes_all_events(self, mock_render, mock_check):
        mock_stream = iter([
            {"agent": {"messages": [MagicMock(content="Hello", tool_calls=[])]}},
        ])
        mock_parser = MagicMock()
        mock_parser.parse.return_value = iter([
            ContentEvent(content="Hello"),
            CompleteEvent(),
        ])

        self.display.stream(mock_stream, parser=mock_parser)

        assert mock_render.call_count == 2  # Once for each event
