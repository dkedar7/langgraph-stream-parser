"""Tests for CLIAdapter."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import timedelta

from langgraph_stream_parser.adapters.base import ToolStatus, ToolState
from langgraph_stream_parser.adapters.cli import CLIAdapter, Spinner
from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    CompleteEvent,
    ErrorEvent,
)


class TestCLIAdapterInit:
    def test_default_options(self):
        adapter = CLIAdapter()
        assert adapter._show_timestamps is False
        assert adapter._show_tool_args is True
        assert adapter._max_content_preview == 200
        assert adapter._use_spinner is True
        assert adapter._use_colors is True
        assert adapter._reflection_types == {"reflection"}
        assert adapter._todo_types == {"todos"}

    def test_custom_options(self):
        adapter = CLIAdapter(
            show_timestamps=True,
            show_tool_args=False,
            max_content_preview=100,
            use_spinner=False,
            use_colors=False,
        )
        assert adapter._show_timestamps is True
        assert adapter._show_tool_args is False
        assert adapter._max_content_preview == 100
        assert adapter._use_spinner is False
        assert adapter._use_colors is False

    def test_custom_extraction_types(self):
        adapter = CLIAdapter(
            reflection_types=["thinking", "reasoning"],
            todo_types=["tasks", "checklist"],
        )
        assert adapter._reflection_types == {"thinking", "reasoning"}
        assert adapter._todo_types == {"tasks", "checklist"}


class TestCLIAdapterColorHelper:
    def test_color_enabled(self):
        adapter = CLIAdapter(use_colors=True)
        assert adapter._c("\033[31m") == "\033[31m"

    def test_color_disabled(self):
        adapter = CLIAdapter(use_colors=False)
        assert adapter._c("\033[31m") == ""


class TestCLIAdapterEventProcessing:
    def setup_method(self):
        self.adapter = CLIAdapter(use_spinner=False, use_colors=False)

    def test_process_content_event(self):
        event = ContentEvent(content="Hello ", role="assistant")
        self.adapter._process_event(event)
        assert self.adapter._current_content == "Hello "
        assert self.adapter._current_role == "assistant"

    def test_process_tool_start_event(self):
        event = ToolCallStartEvent(
            id="call_1",
            name="search",
            args={"query": "test"},
        )
        self.adapter._process_event(event)

        assert "call_1" in self.adapter._tool_indices
        idx = self.adapter._tool_indices["call_1"]
        _, tool = self.adapter._display_items[idx]
        assert tool.name == "search"
        assert tool.status == ToolStatus.RUNNING

    def test_process_tool_end_event_success(self):
        start_event = ToolCallStartEvent(
            id="call_1", name="search", args={}
        )
        self.adapter._process_event(start_event)

        end_event = ToolCallEndEvent(
            id="call_1",
            name="search",
            result="Found results",
            status="success",
        )
        self.adapter._process_event(end_event)

        idx = self.adapter._tool_indices["call_1"]
        _, tool = self.adapter._display_items[idx]
        assert tool.status == ToolStatus.SUCCESS

    def test_process_interrupt_event(self):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve", "reject"]}],
        )
        self.adapter._process_event(event)

        assert self.adapter._interrupt is not None
        assert len(self.adapter._interrupt.action_requests) == 1

    def test_process_error_event(self):
        event = ErrorEvent(error="Something went wrong")
        self.adapter._process_event(event)

        assert self.adapter._error is not None
        assert self.adapter._error.error == "Something went wrong"

    def test_process_complete_event(self):
        event = CompleteEvent()
        self.adapter._process_event(event)

        assert self.adapter._complete is True


class TestCLIAdapterReset:
    def test_reset_clears_state(self):
        adapter = CLIAdapter(use_spinner=False)

        # Populate some state
        adapter._display_items.append(("message", ("assistant", "Some content")))
        adapter._current_content = "In progress"
        adapter._current_role = "assistant"
        adapter._tool_indices["call_1"] = 0
        adapter._last_rendered_count = 5
        adapter._active_tools.add("call_1")

        # Reset
        adapter.reset()

        assert len(adapter._display_items) == 0
        assert adapter._current_content == ""
        assert adapter._current_role is None
        assert len(adapter._tool_indices) == 0
        assert adapter._last_rendered_count == 0
        assert len(adapter._active_tools) == 0


class TestCLIAdapterRendering:
    def setup_method(self):
        self.adapter = CLIAdapter(use_spinner=False, use_colors=False)

    def test_print_message_user(self, capsys):
        self.adapter._print_message("human", "Hello!")
        captured = capsys.readouterr()
        assert "You" in captured.out
        assert "Hello!" in captured.out

    def test_print_message_assistant(self, capsys):
        self.adapter._print_message("assistant", "Hi there!")
        captured = capsys.readouterr()
        assert "Hi there!" in captured.out

    def test_print_tool_start(self, capsys):
        tool = ToolState(
            id="1",
            name="search",
            args={"query": "test"},
            status=ToolStatus.RUNNING,
        )
        self.adapter._print_tool_start(tool)
        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "test" in captured.out

    def test_print_tool_result_success(self, capsys):
        tool = ToolState(
            id="1",
            name="search",
            args={},
            status=ToolStatus.SUCCESS,
        )
        tool.end_time = tool.start_time + timedelta(milliseconds=500)
        self.adapter._print_tool_result(tool)
        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "completed" in captured.out

    def test_print_tool_result_error(self, capsys):
        tool = ToolState(
            id="1",
            name="search",
            args={},
            status=ToolStatus.ERROR,
            error_message="Connection failed",
        )
        self.adapter._print_tool_result(tool)
        captured = capsys.readouterr()
        assert "failed" in captured.out
        assert "Connection failed" in captured.out

    def test_print_extraction(self, capsys):
        event = ToolExtractedEvent(
            tool_name="think_tool",
            extracted_type="reflection",
            data="My thoughts",
        )
        self.adapter._print_extraction(event)
        captured = capsys.readouterr()
        assert "reflection" in captured.out
        assert "My thoughts" in captured.out

    def test_print_extraction_todos(self, capsys):
        event = ToolExtractedEvent(
            tool_name="todo_tool",
            extracted_type="todos",
            data=[
                {"status": "completed", "content": "Task 1"},
                {"status": "in_progress", "content": "Task 2"},
                {"status": "pending", "content": "Task 3"},
            ],
        )
        self.adapter._print_extraction(event)
        captured = capsys.readouterr()
        assert "Task 1" in captured.out
        assert "Task 2" in captured.out
        assert "Task 3" in captured.out

    def test_print_interrupt(self, capsys):
        event = InterruptEvent(
            action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
            review_configs=[{"allowed_decisions": ["approve", "reject"]}],
        )
        self.adapter._print_interrupt(event)
        captured = capsys.readouterr()
        assert "Action Required" in captured.out
        assert "bash" in captured.out

    def test_render_incremental(self, capsys):
        # Process and render first message
        self.adapter._process_event(ContentEvent(content="Hello", role="assistant"))
        self.adapter._flush_current_message()
        self.adapter.render()

        first_output = capsys.readouterr()
        assert "Hello" in first_output.out

        # Process and render second message
        self.adapter._process_event(ContentEvent(content="World", role="human"))
        self.adapter._flush_current_message()
        self.adapter.render()

        second_output = capsys.readouterr()
        assert "World" in second_output.out
        # Should not repeat first message
        assert second_output.out.count("Hello") == 0


class TestCLIAdapterArgPreview:
    def setup_method(self):
        self.adapter = CLIAdapter()

    def test_arg_preview_empty(self):
        assert self.adapter._get_arg_preview({}) == ""

    def test_arg_preview_short(self):
        preview = self.adapter._get_arg_preview({"query": "test"})
        assert preview == "test"

    def test_arg_preview_truncates(self):
        long_value = "x" * 100
        preview = self.adapter._get_arg_preview({"query": long_value})
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")


class TestCLIAdapterRun:
    def setup_method(self):
        self.adapter = CLIAdapter(use_spinner=False, use_colors=False)

    def test_run_processes_all_events(self, capsys):
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([
            {"agent": {"messages": [MagicMock(content="Hello", tool_calls=[])]}},
        ])
        mock_parser = MagicMock()
        mock_parser.parse.return_value = iter([
            ContentEvent(content="Hello", role="assistant"),
            CompleteEvent(),
        ])

        self.adapter.run(
            graph=mock_graph,
            input_data={"messages": [("user", "test")]},
            parser=mock_parser,
        )

        captured = capsys.readouterr()
        assert "Hello" in captured.out


class TestCLIAdapterPromptInterrupt:
    def setup_method(self):
        self.adapter = CLIAdapter(use_spinner=False, use_colors=False)

    @patch('builtins.input', return_value="1")
    def test_prompt_interrupt_approve_fallback(self, mock_input):
        """Test fallback mode with numbered input."""
        # Mock sys.stdin.isatty to return False for fallback mode
        with patch('sys.stdin.isatty', return_value=False):
            event = InterruptEvent(
                action_requests=[{"tool": "bash", "args": {"cmd": "ls"}}],
                review_configs=[{"allowed_decisions": ["approve", "reject"]}],
            )
            decisions = self.adapter.prompt_interrupt(event)

            assert decisions is not None
            assert len(decisions) == 1
            assert decisions[0]["type"] == "approve"

    @patch('builtins.input', return_value="2")
    def test_prompt_interrupt_reject_fallback(self, mock_input):
        """Test fallback mode with numbered input."""
        with patch('sys.stdin.isatty', return_value=False):
            event = InterruptEvent(
                action_requests=[{"tool": "bash", "args": {"cmd": "rm -rf /"}}],
                review_configs=[{"allowed_decisions": ["approve", "reject"]}],
            )
            decisions = self.adapter.prompt_interrupt(event)

            assert decisions is not None
            assert len(decisions) == 1
            assert decisions[0]["type"] == "reject"


class TestSpinner:
    def test_spinner_start_stop(self):
        spinner = Spinner("Testing")
        spinner.start()
        assert spinner.running is True
        spinner.stop()
        assert spinner.running is False

    def test_spinner_message(self):
        spinner = Spinner("Custom message")
        assert spinner.message == "Custom message"
