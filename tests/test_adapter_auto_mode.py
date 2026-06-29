"""Adapters must not forward parser-only stream modes to graph.stream() (gh #45).

`stream_mode="auto"` is advertised, but it's a parser-side concept: LangGraph's
graph.stream() doesn't understand it and yields zero chunks, so an adapter that
forwarded it verbatim rendered a silent blank turn. The adapter now drives the
graph with a real multi-mode stream and lets the parser auto-detect; "v2" (which
graph.stream cannot produce) fails loudly instead of rendering nothing.
"""

import io
from contextlib import redirect_stdout

import pytest

from langgraph_stream_parser.adapters.base import graph_stream_mode
from langgraph_stream_parser.adapters.print import PrintAdapter
from langgraph_stream_parser.demo import create_stub_agent


def test_graph_stream_mode_maps_auto_and_passthrough():
    # "auto" becomes a real multi-mode the graph can stream + the parser detects.
    assert graph_stream_mode("auto") == ["updates", "messages"]
    # Real modes pass through untouched.
    for m in ("updates", "messages", "custom", "values"):
        assert graph_stream_mode(m) == m
    assert graph_stream_mode(["updates", "messages"]) == ["updates", "messages"]


def test_graph_stream_mode_rejects_v2():
    with pytest.raises(ValueError, match="parser-only"):
        graph_stream_mode("v2")


def _run(stream_mode: str) -> str:
    g = create_stub_agent()
    buf = io.StringIO()
    with redirect_stdout(buf):
        PrintAdapter().run(graph=g, input_data={"messages": [("user", "Hi")]}, stream_mode=stream_mode)
    return buf.getvalue()


def test_auto_renders_the_turn_like_updates():
    auto = _run("auto")
    updates = _run("updates")
    # Previously `auto` rendered nothing; now it renders the reply, same as updates.
    assert "You said: Hi" in auto, f"auto rendered nothing: {auto!r}"
    assert "You said: Hi" in updates


def test_v2_via_adapter_raises_not_blank():
    g = create_stub_agent()
    with pytest.raises(ValueError, match="parser-only"):
        PrintAdapter().run(graph=g, input_data={"messages": [("user", "Hi")]}, stream_mode="v2")
