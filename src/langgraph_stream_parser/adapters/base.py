"""
Base adapter class for rendering LangGraph stream events.

Provides shared state tracking, event processing, and interrupt handling
that can be reused by adapter implementations for different environments.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..events import (
    StreamEvent,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
)
from ..parser import StreamParser
from ..resume import create_resume_input


class ToolStatus(Enum):
    """Status of a tool in its lifecycle."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ToolState:
    """Tracks the state of a single tool call."""
    id: str
    name: str
    args: dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    result: Any = None
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


@dataclass
class MessageState:
    """Tracks a message in the conversation."""
    role: str  # "assistant" or "human"
    content: str


class BaseAdapter(ABC):
    """Base class for LangGraph stream display adapters.

    Provides shared functionality for:
    - State tracking (messages, tools, extractions)
    - Event processing
    - Interrupt handling with user prompts
    - Run loop with automatic resumption

    Subclasses must implement:
    - render(): Display the current state
    - prompt_interrupt(): Get user decision for interrupts

    Example:
        from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

        display = JupyterDisplay()
        display.run(
            graph=agent,
            input_data={"messages": [("user", "Hello")]},
            config={"configurable": {"thread_id": "my-session"}}
        )
    """

    # Default extracted types for special rendering
    DEFAULT_REFLECTION_TYPES: set[str] = {"reflection"}
    DEFAULT_TODO_TYPES: set[str] = {"todos"}

    def __init__(
        self,
        *,
        show_timestamps: bool = False,
        show_tool_args: bool = True,
        max_content_preview: int = 200,
        reflection_types: set[str] | list[str] | None = None,
        todo_types: set[str] | list[str] | None = None,
    ):
        """Initialize the adapter.

        Args:
            show_timestamps: Show timestamps on events.
            show_tool_args: Show tool arguments in tool displays.
            max_content_preview: Max characters for extracted content preview.
            reflection_types: Set of extracted_type values to render as reflections
                (italic/special formatting). Defaults to {"reflection"}.
            todo_types: Set of extracted_type values to render as todo lists
                (checkbox formatting). Defaults to {"todos"}.
        """
        self._show_timestamps = show_timestamps
        self._show_tool_args = show_tool_args
        self._max_content_preview = max_content_preview

        # Configurable extraction types for special rendering
        self._reflection_types: set[str] = (
            set(reflection_types) if reflection_types is not None
            else self.DEFAULT_REFLECTION_TYPES.copy()
        )
        self._todo_types: set[str] = (
            set(todo_types) if todo_types is not None
            else self.DEFAULT_TODO_TYPES.copy()
        )

        # State tracking - chronological list of display items
        # Each item is a tuple: (type, data) where type is "message", "tool", "extraction"
        self._display_items: list[tuple[str, Any]] = []

        # Current message being accumulated
        self._current_role: str | None = None
        self._current_content: str = ""

        # Tool state - maps tool_id to index in _display_items
        self._tool_indices: dict[str, int] = {}

        # Final state
        self._interrupt: InterruptEvent | None = None
        self._error: ErrorEvent | None = None
        self._complete: bool = False

    def reset(self) -> None:
        """Reset state for a new stream."""
        self._display_items.clear()
        self._current_role = None
        self._current_content = ""
        self._tool_indices.clear()
        self._interrupt = None
        self._error = None
        self._complete = False

    def run(
        self,
        graph: Any,
        input_data: Any,
        *,
        config: dict[str, Any] | None = None,
        parser: StreamParser | None = None,
        stream_mode: str | list[str] = "updates",
        **stream_kwargs: Any,
    ) -> None:
        """Run a LangGraph agent with live display and interrupt handling.

        This is the main entry point for displaying agent streams. It handles:
        - Streaming events with live display updates
        - Automatic interrupt prompts and agent resumption
        - Completion and error states

        For custom event processing (filtering, transformation, custom display),
        use StreamParser.parse() directly with update() instead.

        Args:
            graph: The LangGraph graph/agent to run.
            input_data: Initial input for the agent.
            config: LangGraph config dict (must include thread_id for resumption).
            parser: Optional pre-configured StreamParser with stream_mode set.
            stream_mode: The stream mode to use. Passed to both graph.stream()
                and StreamParser (when parser is auto-created).
            **stream_kwargs: Additional arguments passed to graph.stream()
                (e.g., interrupt_before, interrupt_after, debug).

        Example:
            display = JupyterDisplay()
            display.run(
                graph=agent,
                input_data={"messages": [("user", "Hello")]},
                config={"configurable": {"thread_id": "my-session"}},
                interrupt_before=["tools"],
            )
        """
        self.reset()

        if parser is None:
            parser = StreamParser(stream_mode=stream_mode)

        current_input = input_data

        while True:
            # Stream until completion or interrupt
            stream = graph.stream(
                current_input,
                config=config,
                stream_mode=stream_mode,
                **stream_kwargs,
            )

            for event in parser.parse(stream):
                self.update(event)

                # Check for interrupt
                if isinstance(event, InterruptEvent):
                    # Get user decision
                    decision = self.prompt_interrupt(event)

                    if decision is None:
                        # User cancelled
                        return

                    # Clear interrupt state and prepare to resume
                    self._interrupt = None
                    current_input = create_resume_input(decisions=decision)
                    break

                # Check for completion or error
                if isinstance(event, (CompleteEvent, ErrorEvent)):
                    return

    def update(self, event: StreamEvent) -> None:
        """Update display with a single event.

        Args:
            event: A StreamEvent to display.
        """
        self._process_event(event)
        self.render()

    def _flush_current_message(self) -> None:
        """Flush current message buffer to display items list.

        Merges with the previous message if it has the same role.
        """
        if self._current_content and self._current_role:
            # Check if we can merge with the last message of same role
            if self._display_items:
                last_type, last_data = self._display_items[-1]
                if last_type == "message":
                    last_role, last_content = last_data
                    if last_role == self._current_role:
                        # Merge with previous message
                        self._display_items[-1] = ("message", (self._current_role, last_content + "\n" + self._current_content))
                        self._current_content = ""
                        return

            # Otherwise add as new message
            self._display_items.append(("message", (self._current_role, self._current_content)))
            self._current_content = ""

    def _process_event(self, event: StreamEvent) -> None:
        """Process an event and update internal state."""
        match event:
            case ContentEvent(content=text, role=role):
                # If role changes, flush the previous message
                if self._current_role is not None and self._current_role != role:
                    self._flush_current_message()
                self._current_role = role
                self._current_content += text

            case ToolCallStartEvent(id=tool_id, name=name, args=args):
                # Flush any pending content before tool
                self._flush_current_message()

                # Create tool state and add to display items
                tool_state = ToolState(
                    id=tool_id,
                    name=name,
                    args=args,
                    status=ToolStatus.RUNNING,
                )
                self._tool_indices[tool_id] = len(self._display_items)
                self._display_items.append(("tool", tool_state))

            case ToolCallEndEvent(
                id=tool_id,
                result=result,
                status=status,
                error_message=error_msg,
            ):
                if tool_id in self._tool_indices:
                    idx = self._tool_indices[tool_id]
                    _, tool = self._display_items[idx]
                    tool.end_time = datetime.now()
                    tool.result = result
                    if status == "success":
                        tool.status = ToolStatus.SUCCESS
                    else:
                        tool.status = ToolStatus.ERROR
                        tool.error_message = error_msg

            case ToolExtractedEvent():
                self._display_items.append(("extraction", event))

            case InterruptEvent():
                self._flush_current_message()
                self._interrupt = event

            case ErrorEvent():
                self._error = event

            case CompleteEvent():
                self._flush_current_message()
                self._complete = True

            case StateUpdateEvent():
                pass  # Ignored in display

    # Helper methods for subclasses

    def get_allowed_decisions(self, event: InterruptEvent) -> set[str]:
        """Extract allowed decisions from an interrupt event."""
        allowed = set()
        for config in event.review_configs:
            allowed.update(config.get("allowed_decisions", []))
        if not allowed:
            allowed = {"approve", "reject"}
        return allowed

    def build_decisions(
        self,
        event: InterruptEvent,
        decision_type: str,
        args_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build decision list for resuming from interrupt.

        Args:
            event: The InterruptEvent.
            decision_type: The decision type (e.g., "approve", "reject", "edit").
            args_modifier: Optional function to modify args for "edit" decisions.

        Returns:
            List of decision dicts ready for create_resume_input().
        """
        decisions = []
        for action in event.action_requests:
            decision: dict[str, Any] = {"type": decision_type}

            if decision_type == "edit" and args_modifier:
                original_args = action.get("args", {})
                decision["args"] = args_modifier(original_args)

            decisions.append(decision)

        return decisions

    @staticmethod
    def format_duration(duration_ms: float | None) -> str:
        """Format duration for display."""
        if duration_ms is None:
            return ""
        if duration_ms < 1000:
            return f"{duration_ms:.0f}ms"
        return f"{duration_ms / 1000:.1f}s"

    @staticmethod
    def format_args(args: dict[str, Any], max_value_len: int = 30, max_total_len: int = 40) -> str:
        """Format tool arguments for display."""
        if not args:
            return ""
        parts = []
        for key, value in args.items():
            value_str = str(value)
            if len(value_str) > max_value_len:
                value_str = value_str[:max_value_len - 3] + "..."
            parts.append(f"{key}={value_str}")
        result = ", ".join(parts)
        if len(result) > max_total_len:
            result = result[:max_total_len - 3] + "..."
        return result

    @staticmethod
    def format_todos(todos: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Format todo items for display.

        Args:
            todos: List of todo dicts with status and content/task keys.

        Returns:
            List of (status, content) tuples where status is
            "completed", "in_progress", or "pending".
        """
        result = []
        for item in todos:
            if isinstance(item, dict):
                # Support both formats: {status, content} and {done, task}
                status = item.get("status", "pending")
                content = item.get("content") or item.get("task", str(item))

                # Handle done field as fallback
                if "done" in item:
                    status = "completed" if item["done"] else "pending"

                result.append((status, content))
        return result

    # Abstract methods for subclasses to implement

    @abstractmethod
    def render(self) -> None:
        """Render the current state to the output.

        Subclasses must implement this to display:
        - Messages (from self._display_items where type == "message")
        - Tools (from self._display_items where type == "tool")
        - Extractions (from self._display_items where type == "extraction")
        - Current in-progress message (self._current_content)
        - Interrupt (self._interrupt)
        - Error (self._error)
        - Completion (self._complete)
        """
        pass

    @abstractmethod
    def prompt_interrupt(self, event: InterruptEvent) -> list[dict[str, Any]] | None:
        """Prompt user for interrupt decision.

        Args:
            event: The InterruptEvent requiring a decision.

        Returns:
            List of decision dicts for create_resume_input(),
            or None if the user cancelled.
        """
        pass
