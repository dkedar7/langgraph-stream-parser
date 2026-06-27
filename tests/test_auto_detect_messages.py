"""stream_mode="auto" detects a pure messages-mode stream (gh #41).

Auto-detect used to fall through to "updates" for a messages-mode first chunk
``(message, metadata)`` and emit zero ContentEvents — a silent empty turn.
"""

from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.demo import create_stub_agent
from langgraph_stream_parser.events import ContentEvent
from langgraph_stream_parser.parser import _is_messages_mode_chunk


def _content(parser_mode):
    graph = create_stub_agent()
    parser = StreamParser(stream_mode=parser_mode)
    stream = graph.stream({"messages": [("user", "Hello")]}, stream_mode="messages")
    return "".join(e.content for e in parser.parse(stream) if isinstance(e, ContentEvent))


def test_auto_matches_explicit_messages_mode():
    explicit = _content("messages")
    auto = _content("auto")
    assert "Hello" in explicit          # sanity: messages mode renders content
    assert auto == explicit             # auto-detect now has parity (was "")


def test_detector_recognizes_messages_chunk():
    from langchain_core.messages import AIMessageChunk

    assert _is_messages_mode_chunk((AIMessageChunk(content="hi"), {"langgraph_node": "x"}))
    # NOT a messages chunk: multi-mode (str first), subgraph (tuple first), plain dict.
    assert not _is_messages_mode_chunk(("updates", {}))
    assert not _is_messages_mode_chunk(((), {}))
    assert not _is_messages_mode_chunk({"foo": "bar"})
