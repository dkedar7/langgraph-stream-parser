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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "content",
            "content": self.content,
            "role": self.role,
            "node": self.node,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "tool_start",
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "node": self.node,
        }


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

    def to_dict(self, max_result_len: int = 500) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs.

        Args:
            max_result_len: Maximum length for result string (truncated if longer).
        """
        result_str = str(self.result)
        if len(result_str) > max_result_len:
            result_str = result_str[:max_result_len] + "..."
        return {
            "type": "tool_end",
            "id": self.id,
            "name": self.name,
            "result": result_str,
            "status": self.status,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "extraction",
            "tool_name": self.tool_name,
            "extracted_type": self.extracted_type,
            "data": self.data,
        }


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

    @property
    def allowed_decisions(self) -> set[str]:
        """Get the set of allowed decision types from review configs."""
        allowed = set()
        for config in self.review_configs:
            allowed.update(config.get("allowed_decisions", []))
        if not allowed:
            allowed = {"approve", "reject"}
        return allowed

    def build_decisions(
        self,
        decision_type: str,
        args_modifier: Any = None,
    ) -> list[dict[str, Any]]:
        """Build decision list for resuming from this interrupt.

        Args:
            decision_type: The decision type (e.g., "approve", "reject", "edit").
            args_modifier: Optional function to modify args for "edit" decisions.
                Takes original args dict and returns modified args dict.

        Returns:
            List of decision dicts ready for create_resume_input().

        Example:
            # Approve all actions
            decisions = interrupt.build_decisions("approve")
            resume_input = create_resume_input(decisions=decisions)

            # Edit args before approval
            def modify(args):
                args["safe_mode"] = True
                return args
            decisions = interrupt.build_decisions("edit", args_modifier=modify)
        """
        decisions = []
        for action in self.action_requests:
            decision: dict[str, Any] = {"type": decision_type}

            if decision_type == "edit" and args_modifier is not None:
                original_args = action.get("args", {})
                decision["args"] = args_modifier(original_args)

            decisions.append(decision)

        return decisions

    def create_resume(
        self,
        decision_type: str,
        args_modifier: Any = None,
    ) -> Any:
        """Create resume input to continue from this interrupt.

        This is a convenience method that combines build_decisions() and
        create_resume_input() into a single call.

        Args:
            decision_type: The decision type (e.g., "approve", "reject", "edit").
            args_modifier: Optional function to modify args for "edit" decisions.

        Returns:
            A LangGraph Command object ready to pass to graph.stream().

        Example:
            # Approve and resume in one call
            resume_input = interrupt.create_resume("approve")
            for event in parser.parse(graph.stream(resume_input, config=config)):
                handle_event(event)
        """
        # Import here to avoid circular dependency
        from .resume import create_resume_input

        decisions = self.build_decisions(decision_type, args_modifier)
        return create_resume_input(decisions=decisions)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "interrupt",
            "action_requests": self.action_requests,
            "review_configs": self.review_configs,
            "allowed_decisions": list(self.allowed_decisions),
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "state_update",
            "node": self.node,
            "key": self.key,
            "value": self.value,
        }


@dataclass
class CompleteEvent:
    """Stream completed successfully.

    Emitted when the graph stream finishes without error.

    Attributes:
        timestamp: When the stream completed.
    """
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {"type": "complete"}


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for web APIs."""
        return {
            "type": "error",
            "error": self.error,
        }


def event_to_dict(event: "StreamEvent") -> dict[str, Any]:
    """Convert any StreamEvent to a JSON-serializable dict.

    This is a convenience function for web APIs that need to serialize
    events to JSON for transmission over WebSockets, HTTP responses, etc.

    Args:
        event: Any StreamEvent instance.

    Returns:
        A dict with a "type" key and event-specific fields.

    Example:
        for event in parser.parse(stream):
            await websocket.send_json(event_to_dict(event))
    """
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return {"type": "unknown", "event": str(event)}


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
