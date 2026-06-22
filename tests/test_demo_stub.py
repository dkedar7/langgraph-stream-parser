"""Tests for the keyless stub agent behind every surface's --demo mode.

Unlike the default agent (which needs deepagents + an API key), the stub only
needs langgraph + langchain-core — both in the dev extras — so these tests
exercise it end to end through the parser.
"""
from langgraph_stream_parser import StreamParser, load_agent_spec, prepare_agent_input
from langgraph_stream_parser.demo import create_stub_agent
from langgraph_stream_parser.events import CompleteEvent, ContentEvent, ErrorEvent

STREAM_MODE = ["updates", "messages"]


def _run_turn(graph, message: str, thread_id: str = "t1"):
    parser = StreamParser(stream_mode=STREAM_MODE)
    stream = graph.stream(
        prepare_agent_input(message=message),
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=STREAM_MODE,
    )
    return list(parser.parse(stream))


def _content(events) -> str:
    return "".join(e.content for e in events if isinstance(e, ContentEvent))


def test_streams_echo_through_the_parser():
    graph = create_stub_agent()
    events = _run_turn(graph, "hello demo")

    assert "(demo agent) You said: hello demo" in _content(events)
    assert isinstance(events[-1], CompleteEvent)


def test_streams_token_by_token():
    graph = create_stub_agent()
    events = _run_turn(graph, "one two three four")
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    # The echo splits on spaces — a multi-word message must arrive in pieces.
    assert len(content_events) > 1


def test_streams_config_free():
    """The documented keyless Quick Start streams with NO config/thread_id.

    Regression (gh #-dogfood): the stub used to compile with a *default*
    MemorySaver, so a config-free ``.stream()`` raised a "checkpointer requires
    thread_id" error — which the parser turned into a lone ErrorEvent and a
    silent, blank reply for anyone copy-pasting the README Quick Start. The stub
    now compiles without a checkpointer by default, so it just works.
    """
    graph = create_stub_agent()
    parser = StreamParser(stream_mode=STREAM_MODE)
    events = list(
        parser.parse(
            graph.stream(prepare_agent_input(message="no config"), stream_mode=STREAM_MODE)
        )
    )
    assert "(demo agent) You said: no config" in _content(events)
    assert isinstance(events[-1], CompleteEvent)
    assert not any(isinstance(e, ErrorEvent) for e in events)


def test_multi_turn_thread_persists():
    # Multi-turn persistence is now opt-in: pass an explicit checkpointer.
    from langgraph.checkpoint.memory import MemorySaver

    graph = create_stub_agent(checkpointer=MemorySaver())
    _run_turn(graph, "first", thread_id="conv")
    state = graph.get_state({"configurable": {"thread_id": "conv"}})
    _run_turn(graph, "second", thread_id="conv")
    state2 = graph.get_state({"configurable": {"thread_id": "conv"}})
    assert len(state2.values["messages"]) > len(state.values["messages"])


def test_custom_name_and_prefix():
    graph = create_stub_agent(name="My Demo", reply_prefix="echo: ")
    assert graph.name == "My Demo"
    events = _run_turn(graph, "hi")
    assert "echo: hi" in _content(events)


def test_loadable_via_standard_spec_string():
    graph = load_agent_spec("langgraph_stream_parser.demo.stub:graph")
    events = _run_turn(graph, "spec works", thread_id="spec")
    assert "spec works" in _content(events)
    # The module-level graph is cached — same object on a second load.
    assert load_agent_spec("langgraph_stream_parser.demo.stub:graph") is graph
