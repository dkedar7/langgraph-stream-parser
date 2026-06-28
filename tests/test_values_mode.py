"""stream_mode="values" support (gh #43).

A real `graph.stream(..., stream_mode="values")` used to be silently dropped:
the parser rejected the constructor arg, and feeding a values stream (full-state
dicts) fell through to the updates handler, which saw no node-keyed update and
yielded a lone CompleteEvent — an empty turn. ValuesEvent was also unreachable
from any real graph.stream() call. These tests lock in: values mode is accepted,
emits a ValuesEvent per snapshot plus the run's content (deduped, no history
replay), and auto-detect routes a values stream correctly.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import (
    CompleteEvent,
    ContentEvent,
    ValuesEvent,
)


def _graph():
    b = StateGraph(MessagesState)
    b.add_node("respond", lambda s: {"messages": [AIMessage(content="hello from values mode")]})
    b.add_edge(START, "respond")
    b.add_edge("respond", END)
    return b.compile()


def _names(evs):
    return [type(e).__name__ for e in evs]


def _contents(evs):
    return [e.content for e in evs if isinstance(e, ContentEvent)]


def test_values_mode_is_accepted():
    # Previously raised ValueError: "Unsupported stream_mode: 'values'".
    StreamParser(stream_mode="values")


def test_explicit_values_emits_valueevent_and_content():
    g = _graph()
    evs = list(StreamParser(stream_mode="values").parse(
        g.stream({"messages": [("user", "hi")]}, stream_mode="values")))
    # ValuesEvent is now reachable from a real graph.stream() call...
    assert any(isinstance(e, ValuesEvent) for e in evs)
    # ...and the assistant content renders (no more silent empty turn).
    assert _contents(evs) == ["hello from values mode"]
    assert isinstance(evs[-1], CompleteEvent)


def test_values_snapshots_carry_full_state():
    g = _graph()
    evs = list(StreamParser(stream_mode="values").parse(
        g.stream({"messages": [("user", "hi")]}, stream_mode="values")))
    values = [e for e in evs if isinstance(e, ValuesEvent)]
    assert values, "expected a ValuesEvent per snapshot"
    # The final snapshot holds the whole accumulated state.
    assert "messages" in values[-1].data
    assert len(values[-1].data["messages"]) == 2  # human + ai


def test_no_human_echo_and_no_history_replay():
    g = _graph()
    # A multi-turn input: the first values snapshot is the full prior history.
    multi = {"messages": [("user", "earlier q"), ("ai", "earlier a"), ("user", "hi")]}
    evs = list(StreamParser(stream_mode="values").parse(
        g.stream(multi, stream_mode="values")))
    # Only THIS run's output renders — not the user echo, not the prior turns.
    assert _contents(evs) == ["hello from values mode"]


def test_auto_detect_routes_values_stream():
    g = _graph()
    evs = list(StreamParser(stream_mode="auto").parse(
        g.stream({"messages": [("user", "hi")]}, stream_mode="values")))
    assert any(isinstance(e, ValuesEvent) for e in evs)
    assert _contents(evs) == ["hello from values mode"]


def test_auto_detect_still_routes_updates_stream():
    # Regression: a real updates stream must NOT be mistaken for values.
    g = _graph()
    evs = list(StreamParser(stream_mode="auto").parse(
        g.stream({"messages": [("user", "hi")]}, stream_mode="updates")))
    assert not any(isinstance(e, ValuesEvent) for e in evs)
    assert _contents(evs) == ["hello from values mode"]


def test_parse_chunk_values_snapshot():
    parser = StreamParser(stream_mode="values")
    snapshot = {"messages": [AIMessage(content="x", id="m1")]}
    evs = parser.parse_chunk(snapshot)
    assert any(isinstance(e, ValuesEvent) for e in evs)
    assert _contents(evs) == ["x"]


def test_async_values_parity():
    g = _graph()

    async def go():
        return [ev async for ev in StreamParser(stream_mode="values").aparse(
            g.astream({"messages": [("user", "hi")]}, stream_mode="values"))]

    evs = asyncio.run(go())
    assert any(isinstance(e, ValuesEvent) for e in evs)
    assert _contents(evs) == ["hello from values mode"]


def test_invalid_mode_still_rejected():
    with pytest.raises(ValueError):
        StreamParser(stream_mode="bogus")
