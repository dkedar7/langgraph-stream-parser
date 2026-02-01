"""
Event dataclasses for LangGraph stream parsing.

These typed event objects provide a consistent interface for consuming
LangGraph streaming outputs, regardless of the underlying stream mode
or message types.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union


@dataclass
class ContentEvent:
    """Text content from a message.

    Attributes:
        content: The text content from the message.
        role: The role of the message sender ("assistant" or "human").
        node: The name of the graph node that produced this content.
        timestamp: When the event was created.
    """
    content: str
    role: Literal["assistant", "human"] = "assistant"
    node: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCallStartEvent:
    """Tool call initiated by AI.

    Emitted when an AI message contains tool calls. This indicates
    the tool is about to be executed.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool being called.
        args: Arguments passed to the tool.
        node: The name of the graph node that initiated the call.
        timestamp: When the event was created.
    """
    id: str
    name: str
    args: dict[str, Any]
    node: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCallEndEvent:
    """Tool call completed with result.

    Emitted when a ToolMessage is received, indicating the tool
    has finished executing.

    Attributes:
        id: Unique identifier matching the ToolCallStartEvent.
        name: Name of the tool that was called.
        result: The result returned by the tool.
        status: Whether the tool succeeded or errored.
        error_message: Error details if status is "error".
        duration_ms: Execution time in milliseconds if available.
        timestamp: When the event was created.
    """
    id: str
    name: str
    result: Any
    status: Literal["success", "error"]
    error_message: str | None = None
    duration_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolExtractedEvent:
    """Special content extracted from a tool result.

    Emitted when a registered ToolExtractor successfully extracts
    meaningful data from a tool's output. For example, extracting
    a reflection from think_tool or a todo list from write_todos.

    Attributes:
        tool_name: Name of the tool the content was extracted from.
        extracted_type: Type identifier for the extracted content
            (e.g., "reflection", "todos", "canvas_item").
        data: The extracted data.
        timestamp: When the event was created.
    """
    tool_name: str
    extracted_type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InterruptEvent:
    """Human-in-the-loop interrupt requiring user decision.

    Emitted when the graph hits an interrupt point and requires
    user input to continue. Use create_resume_input() to create
    the appropriate input to resume execution.

    Attributes:
        action_requests: List of actions requiring user approval.
            Each item contains 'tool', 'tool_call_id', 'args', etc.
        review_configs: Configuration for how actions should be reviewed.
            Each item may contain 'allowed_decisions' list.
        raw_value: The original interrupt value for custom handling.
        timestamp: When the event was created.
    """
    action_requests: list[dict[str, Any]]
    review_configs: list[dict[str, Any]]
    raw_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def needs_approval(self) -> bool:
        """Check if this interrupt has action requests needing approval."""
        return len(self.action_requests) > 0


@dataclass
class StateUpdateEvent:
    """Raw state update for non-message state keys.

    Emitted when include_state_updates=True and the update contains
    state keys other than "messages".

    Attributes:
        node: The name of the graph node that produced this update.
        key: The state key that was updated.
        value: The new value for the state key.
        timestamp: When the event was created.
    """
    node: str
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompleteEvent:
    """Stream completed successfully.

    Emitted when the graph stream finishes without error.

    Attributes:
        timestamp: When the stream completed.
    """
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorEvent:
    """Error occurred during streaming.

    Emitted when an error occurs during stream processing.
    The parser catches exceptions and yields ErrorEvent instead
    of raising, allowing graceful error handling.

    Attributes:
        error: Human-readable error message.
        exception: The original exception if available.
        timestamp: When the error occurred.
    """
    error: str
    exception: Exception | None = None
    timestamp: datetime = field(default_factory=datetime.now)


# Union type for all events - useful for type hints
StreamEvent = Union[
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
]
