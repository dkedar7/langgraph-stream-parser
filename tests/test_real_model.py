"""Opt-in real-model test: a live model streamed through the real parser.

Every other test in this suite feeds the parser synthetic / mocked events. This
one runs an actual LLM (via OpenRouter, a cheap model) through a real LangGraph
agent and asserts the parser turns its streaming output into the wire events
downstreams render: token content, a real tool call, and completion.

It is **opt-in**: it skips unless ``OPENROUTER_API_KEY`` is set and
``langchain-openai`` is installed (the ``real`` extra), so default CI stays
free, fast, and deterministic. Run it with::

    uv pip install -e ".[dev,real]"
    OPENROUTER_API_KEY=... pytest tests/test_real_model.py -v
"""
import os

import pytest

pytestmark = pytest.mark.real_model

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"  # cheap, reliable tool-calling


def _require_real_model():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — skipping real-model test")
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langgraph.prebuilt")


def _build_agent():
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    model = ChatOpenAI(
        model=MODEL,
        base_url=OPENROUTER_BASE,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )
    try:
        from langchain.agents import create_agent as make_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent as make_agent
    return make_agent(model, tools=[multiply])


def test_real_model_streams_content_tool_call_and_completes():
    _require_real_model()
    from langgraph_stream_parser import prepare_agent_input, stream_graph_updates

    agent = _build_agent()
    inp = prepare_agent_input(
        message="What is 17 times 23? Use the multiply tool, then state the result."
    )

    saw_tool_call = False
    saw_content = False
    saw_complete = False
    tool_names: list[str] = []

    for ev in stream_graph_updates(agent, inp, stream_mode=["updates", "messages"]):
        status = ev.get("status")
        if ev.get("tool_calls"):
            saw_tool_call = True
            tool_names += [tc.get("name") for tc in ev["tool_calls"]]
        if ev.get("chunk"):
            saw_content = True
        if status == "complete":
            saw_complete = True

    assert saw_tool_call, "expected at least one tool call from the real model"
    assert "multiply" in tool_names, f"expected the multiply tool, got {tool_names}"
    assert saw_content, "expected streamed text content"
    assert saw_complete, "expected a completion event"
