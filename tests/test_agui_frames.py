"""The two shared AG-UI mapping helpers in the agui module (ADR 0002 dedupe).

``iter_event_frames`` -> event_to_dict wire (vscode + web).
``iter_chunk_frames`` -> stream_graph_updates chunk-dict wire (cli + jupyter).
Skipped unless the agui extra is installed (dev pulls it, so CI runs it).
"""
import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langgraph_stream_parser import load_agent_spec
from langgraph_stream_parser.agui import build_agent, iter_chunk_frames, iter_event_frames

pytestmark = pytest.mark.asyncio


async def _collect(aiter):
    return [frame async for frame in aiter]


async def test_iter_chunk_frames_shape():
    agent = build_agent(load_agent_spec("langgraph_stream_parser.demo.stub:graph"))
    frames = await _collect(iter_chunk_frames(agent, "chunk wire", "t1"))
    assert frames[-1] == {"status": "complete"}
    content = [f for f in frames if f.get("status") == "streaming" and "chunk" in f]
    assert content and all(set(f) == {"status", "chunk", "node"} for f in content)
    assert "chunk wire" in "".join(f["chunk"] for f in content)


async def test_iter_event_frames_shape():
    agent = build_agent(load_agent_spec("langgraph_stream_parser.demo.stub:graph"))
    frames = await _collect(iter_event_frames(agent, "event wire", "t2"))
    assert frames[-1] == {"type": "complete"}
    content = [f for f in frames if f.get("type") == "content"]
    assert content and all(set(f) == {"type", "content", "role", "node"} for f in content)
    assert "event wire" in "".join(f["content"] for f in content)
