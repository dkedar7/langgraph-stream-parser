"""Dict-form messages must render — LangGraph's add_messages accepts dicts.

Regression (gh #-dogfood): a CompiledGraph whose node returns a dict message
(`{"role": "assistant", "content": ...}` or `{"type": "ai", ...}`) produced NO
content — the updates handler dispatched on the message's class name ("dict"),
which matched no branch. This broke the family's "runs any CompiledGraph" promise
(found via langstage-cli). The handler now coerces dict messages to LangChain
Message objects first.
"""
from langgraph_stream_parser import (
    ContentEvent,
    StreamParser,
    ToolCallStartEvent,
)


def _content(events):
    return [e.content for e in events if isinstance(e, ContentEvent)]


def test_dict_message_role_form_renders_single_updates():
    parser = StreamParser(stream_mode="updates")
    stream = [{"respond": {"messages": [{"role": "assistant", "content": "hello"}]}}]
    assert _content(parser.parse(iter(stream))) == ["hello"]


def test_dict_message_type_form_renders_single_updates():
    parser = StreamParser(stream_mode="updates")
    stream = [{"respond": {"messages": [{"type": "ai", "content": "hi there"}]}}]
    assert _content(parser.parse(iter(stream))) == ["hi there"]


def test_dict_message_renders_in_dual_mode():
    """The surfaces (cli/web/vscode) run dual ["updates","messages"]; a finished
    dict message with no token stream must still render."""
    parser = StreamParser(stream_mode=["updates", "messages"])
    stream = [("updates", {"respond": {"messages": [{"role": "assistant", "content": "dual ok"}]}})]
    assert _content(parser.parse(iter(stream))) == ["dual ok"]


def test_dict_message_tool_call_still_emits_tool_event():
    parser = StreamParser(stream_mode="updates")
    stream = [
        {
            "agent": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "name": "search", "args": {"q": "x"}}
                        ],
                    }
                ]
            }
        }
    ]
    events = list(parser.parse(iter(stream)))
    tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(tool_starts) == 1 and tool_starts[0].name == "search"


def test_object_messages_still_work():
    """Coercion must not disturb existing LangChain Message objects."""
    from langchain_core.messages import AIMessage

    parser = StreamParser(stream_mode="updates")
    stream = [{"respond": {"messages": [AIMessage(content="object ok")]}}]
    assert _content(parser.parse(iter(stream))) == ["object ok"]
