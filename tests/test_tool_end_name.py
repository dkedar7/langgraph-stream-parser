"""tool_end must carry the real tool name, not "unknown" (gh #-dogfood).

A ToolMessage often lacks a ``name``; tool_end then reported name="unknown" even
though the correlated tool_start (same id) carried the name. Found via the
langstage-vscode sidecar. The handler now backfills the name from the tracked
start event.
"""
from langchain_core.messages import AIMessage, ToolMessage

from langgraph_stream_parser import (
    StreamParser,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


def test_tool_end_backfills_name_from_start_event():
    stream = [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "c1", "name": "add", "args": {"a": 2, "b": 3}}],
                    )
                ]
            }
        },
        # ToolMessage with NO name (the case that produced "unknown").
        {"tools": {"messages": [ToolMessage(content="5", tool_call_id="c1")]}},
    ]
    events = list(StreamParser(stream_mode="updates").parse(iter(stream)))
    starts = [e.name for e in events if isinstance(e, ToolCallStartEvent)]
    ends = [e.name for e in events if isinstance(e, ToolCallEndEvent)]
    assert starts == ["add"]
    assert ends == ["add"], f"tool_end name not backfilled: {ends}"


def test_tool_end_uses_message_name_when_present():
    """When the ToolMessage carries a name, it's still used (no regression)."""
    stream = [
        {
            "agent": {
                "messages": [
                    AIMessage(content="", tool_calls=[{"id": "c1", "name": "add", "args": {}}])
                ]
            }
        },
        {"tools": {"messages": [ToolMessage(content="5", name="add", tool_call_id="c1")]}},
    ]
    events = list(StreamParser(stream_mode="updates").parse(iter(stream)))
    ends = [e.name for e in events if isinstance(e, ToolCallEndEvent)]
    assert ends == ["add"]
