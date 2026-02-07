# langgraph-stream-parser — Dual Stream Mode Support

**Repository**: https://github.com/dkedar7/langgraph-stream-parser
**Prerequisite for**: cowork-dash v2 (Phase 2: WebSocket streaming)

---

## 1. Problem

`StreamParser.parse()` and `StreamParser.aparse()` currently accept the output of `graph.stream()` / `graph.astream()` when called with a single `stream_mode` string (e.g., `stream_mode="updates"`). Each chunk is a plain dict.

LangGraph also supports passing a **list** of stream modes:

```python
stream = agent.astream(
    input_data,
    config=config,
    stream_mode=["updates", "messages"],
)
```

When `stream_mode` is a list, each yielded chunk is a **tuple** `(mode_name, data)` instead of a plain dict:

```python
# Interleaved output from dual stream mode:
("updates", {"agent": {"messages": [AIMessage(content="Hello world", ...)]}})
("messages", (AIMessageChunk(content="Hello"), {"langgraph_node": "agent"}))
("messages", (AIMessageChunk(content=" world"), {"langgraph_node": "agent"}))
("updates", {"tools": {"messages": [ToolMessage(...)]}})
```

The parser does not handle this tuple format today. Cowork-dash v2 needs dual mode (`["updates", "messages"]`) to get both:
- **`"updates"`**: Node-level state diffs — used for tool call lifecycle, interrupts, state updates
- **`"messages"`**: Token-level LLM chunks — used for real-time text streaming in the UI

---

## 2. Goal

Extend `StreamParser` so that `parse()` and `aparse()` accept streams from **any** `stream_mode` configuration — a single string or a list of strings — and yield the same typed event dataclasses either way. The caller should not need to know which stream mode was used.

---

## 3. LangGraph Stream Mode Reference

| Mode | Chunk format | What it contains |
|---|---|---|
| `"updates"` | `dict` — `{node_name: state_diff}` | Full state diff after each node completes. Contains complete `AIMessage`s (with all content + tool calls), `ToolMessage`s, interrupt signals |
| `"messages"` | `tuple` — `(BaseMessageChunk, metadata_dict)` | Token-level LLM output. `metadata` includes `langgraph_node`, `langgraph_tags`, etc. Yields `AIMessageChunk` for content tokens AND for streamed tool call fragments |
| `"values"` | `dict` — full state snapshot | Entire state after each node. Not needed for cowork-dash |
| `"custom"` | `any` | Custom data from `get_stream_writer()`. Not needed for cowork-dash |
| `"debug"` | `dict` | Debug traces. Not needed for cowork-dash |

When `stream_mode` is a list, output is `(mode_name: str, chunk)` tuples with chunks from each mode interleaved.

---

## 4. Design

### 4.1 Detection: tuple vs dict

The parser should auto-detect the stream format on the first chunk:

```python
def _is_multi_mode(chunk) -> bool:
    """Check if this chunk is from multi-mode streaming (a tuple of (str, data))."""
    return (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[0], str)
    )
```

### 4.2 Routing

When multi-mode is detected, the parser demultiplexes by mode name:

```
incoming chunk (mode, data)
         │
         ├── mode == "updates"  →  existing _process_update(data)  →  yields events
         ├── mode == "messages" →  new _process_message_chunk(data) →  yields events
         └── mode == other      →  ignored (or passed through as raw event)
```

### 4.3 Processing `"messages"` chunks

Each `"messages"` chunk is a tuple `(message_chunk, metadata)`:

```python
from langchain_core.messages import AIMessageChunk

def _process_message_chunk(
    self,
    message_chunk: AIMessageChunk,
    metadata: dict,
) -> Iterator[StreamEvent]:
    node = metadata.get("langgraph_node", "")

    # Token content
    if message_chunk.content:
        yield ContentEvent(
            content=message_chunk.content,
            node=node,
        )

    # Streamed tool call fragments (partial tool_calls on the chunk)
    # These arrive incrementally — the parser should accumulate them
    # and yield ToolCallStartEvent once the tool name + args are available.
    if message_chunk.tool_call_chunks:
        for tc_chunk in message_chunk.tool_call_chunks:
            self._accumulate_tool_call_chunk(tc_chunk, node)
```

#### Tool call accumulation from `"messages"` mode

In `"messages"` mode, tool calls arrive as fragments across multiple `AIMessageChunk`s:

```python
# Fragment 1: tool name arrives
AIMessageChunk(content="", tool_call_chunks=[{"name": "write_file", "args": "", "id": "call_abc", "index": 0}])

# Fragment 2-N: args stream in
AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": '{"file', "id": None, "index": 0}])
AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": '_path":', "id": None, "index": 0}])
# ...
```

The parser must buffer these fragments per `index` and emit `ToolCallStartEvent` once accumulated. However, this is **only needed if `"updates"` mode is not also present** — see deduplication below.

### 4.4 Deduplication

When both `"updates"` and `"messages"` are active, the same content appears in both streams:
- **Text content**: `"messages"` yields it token-by-token, `"updates"` yields it as complete messages after the node finishes
- **Tool calls**: `"messages"` yields them as fragments, `"updates"` yields them as complete `tool_calls` on the `AIMessage`

The deduplication strategy:

| Event type | Source when dual mode | Rationale |
|---|---|---|
| `ContentEvent` | `"messages"` only | Token-level granularity for streaming UI |
| `ToolCallStartEvent` | `"updates"` only | Complete args, no accumulation needed |
| `ToolCallEndEvent` | `"updates"` only | Tool results only appear in updates |
| `ToolExtractedEvent` | `"updates"` only | Derived from tool results |
| `InterruptEvent` | `"updates"` only | Interrupts are state-level |
| `StateUpdateEvent` | `"updates"` only | State diffs |
| `CompleteEvent` | `"updates"` only | End of stream |
| `ErrorEvent` | either | Whichever reports it |

In practice, this means: **when both modes are present, suppress `ContentEvent` generation from `"updates"` chunks** (since `"messages"` provides the token-level version), and **ignore tool call fragments from `"messages"` chunks** (since `"updates"` provides complete tool calls).

### 4.5 Configuration

Add a new parameter to `StreamParser`:

```python
class StreamParser:
    def __init__(
        self,
        *,
        track_tool_lifecycle: bool = True,
        skip_tools: list[str] | None = None,
        include_state_updates: bool = False,
        # New parameter:
        stream_mode: str | list[str] = "updates",
    ):
        """
        stream_mode: Tells the parser what stream format to expect.
          - "updates" (default): chunks are plain dicts (current behavior)
          - "messages": chunks are (AIMessageChunk, metadata) tuples
          - ["updates", "messages"]: chunks are (mode_name, data) tuples
          - "auto": auto-detect from the first chunk (inspect for tuple format)

        When set to a list, the parser uses the deduplication strategy
        described above. When set to "auto", it detects on first chunk.
        """
```

For backward compatibility, the default remains `"updates"` and all existing behavior is unchanged. Callers opt in via `stream_mode=["updates", "messages"]` or `stream_mode="auto"`.

---

## 5. Implementation

### 5.1 Updated `aparse()`

```python
async def aparse(self, stream: AsyncIterator) -> AsyncIterator[StreamEvent]:
    """Parse an async stream of LangGraph chunks into typed events.

    Handles both single-mode (dict chunks) and multi-mode (tuple chunks).
    """
    detected_mode = None

    async for chunk in stream:
        # Auto-detect on first chunk if needed
        if detected_mode is None:
            if self._stream_mode == "auto":
                detected_mode = "multi" if _is_multi_mode(chunk) else "single"
            elif isinstance(self._stream_mode, list):
                detected_mode = "multi"
            else:
                detected_mode = "single"

        if detected_mode == "multi":
            mode_name, data = chunk
            async for event in self._process_mode(mode_name, data):
                yield event
        else:
            # Single mode — existing behavior unchanged
            async for event in self._process_update(chunk):
                yield event

    yield CompleteEvent()


async def _process_mode(
    self,
    mode_name: str,
    data: Any,
) -> AsyncIterator[StreamEvent]:
    """Route a chunk from multi-mode streaming to the appropriate handler."""
    if mode_name == "updates":
        # Use existing update processing, but suppress ContentEvent
        # generation (messages mode handles content at token level)
        async for event in self._process_update(data, suppress_content=True):
            yield event

    elif mode_name == "messages":
        message_chunk, metadata = data
        async for event in self._process_message_chunk(message_chunk, metadata):
            yield event

    # Other modes ("values", "custom", "debug") are silently ignored


async def _process_message_chunk(
    self,
    message_chunk,
    metadata: dict,
) -> AsyncIterator[StreamEvent]:
    """Extract ContentEvents from a messages-mode chunk."""
    node = metadata.get("langgraph_node", "")

    # Only yield text content — tool call fragments are ignored
    # because "updates" mode provides complete tool calls
    if hasattr(message_chunk, "content") and message_chunk.content:
        yield ContentEvent(content=message_chunk.content, node=node)
```

### 5.2 Updated `_process_update()` — suppress_content flag

The existing `_process_update()` method needs a `suppress_content` parameter. When `True`, it skips yielding `ContentEvent` for AI message text content (since `"messages"` mode already provides this at token granularity). All other events (tool calls, interrupts, state updates) are yielded normally.

```python
async def _process_update(
    self,
    chunk: dict,
    suppress_content: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Process a single 'updates'-mode chunk.

    Args:
        chunk: Node-level state diff, e.g. {"agent": {"messages": [...]}}
        suppress_content: If True, skip ContentEvent for AI message text.
            Used when "messages" mode is also active (dual mode).
    """
    # ... existing logic ...
    # When yielding ContentEvent from AI messages:
    if not suppress_content:
        yield ContentEvent(content=..., node=...)
    # Tool calls, interrupts, state updates — always yielded
```

### 5.3 Sync `parse()` — mirror changes

Apply the same detection/routing/deduplication logic to the sync `parse()` generator. The implementation should share the core logic with `aparse()` where possible.

---

## 6. Public API Changes

### New parameter

```python
# Before
parser = StreamParser(track_tool_lifecycle=True)

# After — explicit dual mode
parser = StreamParser(
    track_tool_lifecycle=True,
    stream_mode=["updates", "messages"],
)

# After — auto-detect
parser = StreamParser(
    track_tool_lifecycle=True,
    stream_mode="auto",
)
```

### No changes to event types

The output event types (`ContentEvent`, `ToolCallStartEvent`, etc.) are unchanged. The only difference is that `ContentEvent`s arrive token-by-token instead of as complete message blocks.

### No changes to `create_resume_input()`

Interrupt handling is unaffected.

---

## 7. Messages-Only Mode

For completeness, `stream_mode="messages"` (without `"updates"`) should also work. In this mode:
- `ContentEvent` is yielded per token chunk (same as dual mode)
- Tool calls must be accumulated from `AIMessageChunk.tool_call_chunks` and emitted as `ToolCallStartEvent` once complete
- `ToolCallEndEvent` cannot be emitted (tool results don't appear in `"messages"` mode)
- `InterruptEvent` cannot be emitted (interrupts don't appear in `"messages"` mode)

This mode is incomplete for cowork-dash (no tool lifecycle or interrupts), but supporting it makes the library more general. It can be implemented later if not needed immediately.

---

## 8. Testing

### 8.1 Unit tests

```python
# Test: dual mode produces token-level ContentEvents
async def test_dual_mode_content_streaming():
    """Dual mode yields ContentEvent per token from 'messages', not from 'updates'."""
    chunks = [
        ("messages", (AIMessageChunk(content="Hello"), {"langgraph_node": "agent"})),
        ("messages", (AIMessageChunk(content=" world"), {"langgraph_node": "agent"})),
        ("updates", {"agent": {"messages": [AIMessage(content="Hello world")]}}),
    ]
    parser = StreamParser(stream_mode=["updates", "messages"])
    events = list(parser.parse(iter(chunks)))

    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 2  # from "messages", not "updates"
    assert content_events[0].content == "Hello"
    assert content_events[1].content == " world"


# Test: dual mode still yields tool call events from "updates"
async def test_dual_mode_tool_calls():
    """Tool calls come from 'updates' mode, not accumulated from 'messages' fragments."""
    chunks = [
        ("messages", (AIMessageChunk(content="", tool_call_chunks=[...]), {"langgraph_node": "agent"})),
        ("updates", {"agent": {"messages": [AIMessage(content="", tool_calls=[{
            "name": "write_file",
            "args": {"file_path": "/test.md", "content": "hi"},
            "id": "call_abc",
        }])]}}),
        ("updates", {"tools": {"messages": [ToolMessage(content="File written.", tool_call_id="call_abc")]}}),
    ]
    parser = StreamParser(stream_mode=["updates", "messages"])
    events = list(parser.parse(iter(chunks)))

    tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
    assert len(tool_starts) == 1
    assert tool_starts[0].name == "write_file"
    assert len(tool_ends) == 1


# Test: auto-detect mode
async def test_auto_detect_single_mode():
    """Auto mode detects plain dict chunks as single-mode."""
    chunks = [{"agent": {"messages": [AIMessage(content="Hello")]}}]
    parser = StreamParser(stream_mode="auto")
    events = list(parser.parse(iter(chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1


# Test: backward compatibility
async def test_default_single_mode_unchanged():
    """Default stream_mode='updates' works exactly as before."""
    # ... existing test cases should pass without modification
```

### 8.2 Integration test

```python
async def test_dual_mode_with_real_agent():
    """End-to-end: real deepagent with dual stream mode."""
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_deep_agent(checkpointer=InMemorySaver())
    parser = StreamParser(stream_mode=["updates", "messages"])

    stream = agent.astream(
        {"messages": [{"role": "user", "content": "Say hello"}]},
        config={"configurable": {"thread_id": "test"}},
        stream_mode=["updates", "messages"],
    )

    events = []
    async for event in parser.aparse(stream):
        events.append(event)

    # Should have token-level content events
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) > 1  # multiple tokens, not one big block
    assert isinstance(events[-1], CompleteEvent)
```

---

## 9. Scope & Non-Goals

**In scope**:
- Dual mode: `stream_mode=["updates", "messages"]`
- Auto-detection: `stream_mode="auto"`
- Deduplication strategy for dual mode
- Backward compatibility: default `stream_mode="updates"` unchanged

**Out of scope (can be added later)**:
- Messages-only mode with tool call accumulation (Section 7)
- Support for `"values"`, `"custom"`, `"debug"` modes
- Subgraph token streaming (nested `namespace` tuples)

---

## 10. Migration

This is a **backward-compatible** addition. No existing code breaks:
- Default `stream_mode="updates"` behavior is unchanged
- All existing event types are unchanged
- All existing `StreamParser` constructor parameters are unchanged
- `create_resume_input()` is unchanged

Consumers opt in by passing `stream_mode=["updates", "messages"]` or `stream_mode="auto"` to `StreamParser()`.