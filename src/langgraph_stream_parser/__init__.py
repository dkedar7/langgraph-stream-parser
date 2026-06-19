"""
langgraph-stream-parser: Universal parser for LangGraph streaming outputs.

This package provides a typed interface for parsing streaming outputs from
LangGraph agents, normalizing the complex output shapes into consistent,
typed event objects.

Basic Usage:
    from langgraph_stream_parser import StreamParser
    from langgraph_stream_parser.events import ContentEvent, InterruptEvent

    parser = StreamParser()

    for event in parser.parse(graph.stream(input, stream_mode="updates")):
        match event:
            case ContentEvent(content=text):
                print(text, end="")
            case InterruptEvent(action_requests=actions):
                # Handle human-in-the-loop
                ...

Legacy Dict-Based API:
    from langgraph_stream_parser import stream_graph_updates

    for update in stream_graph_updates(agent, input_data, config=config):
        if "chunk" in update:
            print(update["chunk"], end="")
        elif update.get("status") == "interrupt":
            # Handle interrupt
            ...
"""
from importlib.metadata import PackageNotFoundError, version

from .parser import StreamParser
from .events import (
    ContentEvent,
    ReasoningEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    DisplayEvent,
    InterruptEvent,
    StateUpdateEvent,
    UsageEvent,
    CustomEvent,
    ValuesEvent,
    DebugEvent,
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
    event_to_dict,
)
from .extractors.base import ToolExtractor
from .extractors.builtins import (
    DisplayInlineExtractor,
    GenericToolExtractor,
    ThinkToolExtractor,
    TodoExtractor,
)
from .resume import create_resume_input, prepare_agent_input
from .host import load_agent_spec, HostConfig, Workspace
from .tasks import (
    TaskRunner,
    TaskStore,
    InMemoryTaskStore,
    Task,
    TaskState,
    TASK_TOOLS,
    set_runner,
    get_runner,
    current_task_id,
    outcome_to_state,
)
from .compat import (
    stream_graph_updates,
    astream_graph_updates,
    resume_graph_from_interrupt,
    aresume_graph_from_interrupt,
)

# Single source of truth is the installed distribution metadata (driven by
# pyproject's version), so a hard-coded constant can never drift out of sync.
try:
    __version__ = version("langgraph-stream-parser")
except PackageNotFoundError:  # pragma: no cover - editable/source checkout
    __version__ = "0.0.0+local"

__all__ = [
    # Main parser
    "StreamParser",
    # Event types
    "ContentEvent",
    "ReasoningEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
    "ToolExtractedEvent",
    "DisplayEvent",
    "InterruptEvent",
    "StateUpdateEvent",
    "UsageEvent",
    "CustomEvent",
    "ValuesEvent",
    "DebugEvent",
    "CompleteEvent",
    "ErrorEvent",
    "StreamEvent",
    # Extractors
    "ToolExtractor",
    "ThinkToolExtractor",
    "TodoExtractor",
    "DisplayInlineExtractor",
    "GenericToolExtractor",
    # Resume utilities
    "create_resume_input",
    "prepare_agent_input",
    # Host conventions
    "load_agent_spec",
    "HostConfig",
    "Workspace",
    # Task-delegation engine
    "TaskRunner",
    "TaskStore",
    "InMemoryTaskStore",
    "Task",
    "TaskState",
    "TASK_TOOLS",
    "set_runner",
    "get_runner",
    "current_task_id",
    "outcome_to_state",
    # Serialization
    "event_to_dict",
    # Legacy/compat functions
    "stream_graph_updates",
    "astream_graph_updates",
    "resume_graph_from_interrupt",
    "aresume_graph_from_interrupt",
]
