"""
Backward-compatible convenience functions.

These functions provide a dict-based API matching the reference
implementation for users who prefer that style or need to migrate
gradually.
"""
from typing import Any, AsyncIterator, Iterator

from .events import (
    CompleteEvent,
    ContentEvent,
    ErrorEvent,
    InterruptEvent,
    StreamEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolExtractedEvent,
)
from .parser import StreamParser
from .resume import prepare_agent_input


def _event_to_dict(event: StreamEvent) -> dict[str, Any] | None:
    """Convert a StreamEvent to the legacy dict format.

    Args:
        event: A StreamEvent object.

    Returns:
        Dict in the legacy format, or None for events that
        shouldn't produce output.
    """
    match event:
        case ContentEvent(content=content, node=node):
            result: dict[str, Any] = {
                "chunk": content,
                "status": "streaming",
            }
            if node:
                result["node"] = node
            return result

        case ToolCallStartEvent(id=id, name=name, args=args, node=node):
            result = {
                "tool_calls": [{
                    "id": id,
                    "name": name,
                    "args": args,
                }],
                "status": "streaming",
            }
            if node:
                result["node"] = node
            return result

        case ToolCallEndEvent():
            # Legacy format doesn't emit separate tool end events
            return None

        case ToolExtractedEvent(tool_name=tool_name, extracted_type=ext_type, data=data):
            if ext_type == "reflection":
                return {
                    "chunk": data,
                    "status": "streaming",
                }
            elif ext_type == "todos":
                return {
                    "todo_list": data,
                    "status": "streaming",
                }
            else:
                # Generic extracted content
                return {
                    "extracted": {
                        "tool": tool_name,
                        "type": ext_type,
                        "data": data,
                    },
                    "status": "streaming",
                }

        case InterruptEvent(action_requests=actions, review_configs=configs):
            return {
                "interrupt": {
                    "action_requests": actions,
                    "review_configs": configs,
                },
                "status": "interrupt",
            }

        case CompleteEvent():
            return {"status": "complete"}

        case ErrorEvent(error=error):
            return {
                "error": error,
                "status": "error",
            }

        case _:
            return None


def stream_graph_updates(
    agent: Any,
    input_data: Any,
    config: dict[str, Any] | None = None,
    stream_mode: str | list[str] = "updates",
) -> Iterator[dict[str, Any]]:
    """Stream updates from a LangGraph agent.

    This is a convenience function that provides the same dict-based
    output format as the reference implementation.

    Args:
        agent: The LangGraph agent/graph instance.
        input_data: Input data for the agent (can be dict, Command, or any agent input).
        config: Optional configuration for the agent.
        stream_mode: Stream mode for LangGraph (default: "updates").

    Yields:
        Dictionaries containing:
        - {"chunk": str, "status": "streaming"} for text content
        - {"tool_calls": list, "status": "streaming"} for tool calls
        - {"todo_list": list, "status": "streaming"} for todos
        - {"interrupt": dict, "status": "interrupt"} for interrupts
        - {"status": "complete"} when finished
        - {"error": str, "status": "error"} on errors

    Example:
        for update in stream_graph_updates(agent, input_data, config=config):
            if update.get("status") == "interrupt":
                interrupt = update["interrupt"]
                # Handle interrupt...
            elif "chunk" in update:
                print(update["chunk"], end="")
            elif update.get("status") == "complete":
                break
    """
    parser = StreamParser(
        stream_mode=stream_mode,
        track_tool_lifecycle=True,
        skip_tools=["think_tool", "write_todos"],
    )

    try:
        stream = agent.stream(input_data, config=config, stream_mode=stream_mode)

        for event in parser.parse(stream):
            result = _event_to_dict(event)
            if result is not None:
                yield result

    except Exception as e:
        yield {
            "error": f"Error streaming from agent: {str(e)}",
            "status": "error",
        }


async def astream_graph_updates(
    agent: Any,
    input_data: Any,
    config: dict[str, Any] | None = None,
    stream_mode: str | list[str] = "updates",
) -> AsyncIterator[dict[str, Any]]:
    """Async version of stream_graph_updates.

    Args:
        agent: The LangGraph agent/graph instance.
        input_data: Input data for the agent.
        config: Optional configuration for the agent.
        stream_mode: Stream mode for LangGraph (default: "updates").

    Yields:
        Same format as stream_graph_updates.
    """
    parser = StreamParser(
        stream_mode=stream_mode,
        track_tool_lifecycle=True,
        skip_tools=["think_tool", "write_todos"],
    )

    try:
        stream = agent.astream(input_data, config=config, stream_mode=stream_mode)

        async for event in parser.aparse(stream):
            result = _event_to_dict(event)
            if result is not None:
                yield result

    except Exception as e:
        yield {
            "error": f"Error streaming from agent: {str(e)}",
            "status": "error",
        }


def resume_graph_from_interrupt(
    agent: Any,
    decisions: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    stream_mode: str = "updates",
) -> Iterator[dict[str, Any]]:
    """Resume a LangGraph agent from an interrupt.

    This is a convenience wrapper around stream_graph_updates that
    prepares the resume input automatically.

    Args:
        agent: The LangGraph agent/graph instance.
        decisions: List of decision objects with 'type' and optional fields.
        config: Optional configuration for the agent.
        stream_mode: Stream mode for LangGraph (default: "updates").

    Yields:
        Same format as stream_graph_updates.

    Example:
        for update in resume_graph_from_interrupt(
            agent,
            decisions=[{"type": "approve"}],
            config=config
        ):
            if "chunk" in update:
                print(update["chunk"], end="")
    """
    try:
        # Prepare resume input
        resume_input = prepare_agent_input(decisions=decisions)

        # Use the same streaming logic as regular streaming
        for chunk in stream_graph_updates(
            agent, resume_input, config=config, stream_mode=stream_mode
        ):
            yield chunk

    except Exception as e:
        yield {
            "error": f"Error resuming from interrupt: {str(e)}",
            "status": "error",
        }


async def aresume_graph_from_interrupt(
    agent: Any,
    decisions: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    stream_mode: str = "updates",
) -> AsyncIterator[dict[str, Any]]:
    """Async version of resume_graph_from_interrupt.

    Args:
        agent: The LangGraph agent/graph instance.
        decisions: List of decision objects.
        config: Optional configuration for the agent.
        stream_mode: Stream mode for LangGraph.

    Yields:
        Same format as stream_graph_updates.
    """
    try:
        resume_input = prepare_agent_input(decisions=decisions)

        async for chunk in astream_graph_updates(
            agent, resume_input, config=config, stream_mode=stream_mode
        ):
            yield chunk

    except Exception as e:
        yield {
            "error": f"Error resuming from interrupt: {str(e)}",
            "status": "error",
        }
