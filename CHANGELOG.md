# Changelog

## [0.1.7] - 2026-04-18

### Added
- `FastAPIAdapter` for streaming LangGraph events over WebSocket and Server-Sent Events; stateless by design — conversation state is keyed by `session_id` used as LangGraph `thread_id`
- Per-session asyncio lock with refcounted cleanup to serialize concurrent turns on the same thread
- `BaseAdapter._text_prompt_interrupt()` helper, shared by `PrintAdapter` and `JupyterDisplay`
- `BaseAdapter._truncate()` helper for preview-length capping
- `fastapi` optional dependency group (`pip install langgraph-stream-parser[fastapi]`)

### Changed
- Hoisted `_last_rendered_count` incremental-render cursor from Print/CLI into `BaseAdapter`
- Slimmed `examples/fastapi_websocket.py` from ~455 to ~234 lines by using the new adapter

### Fixed
- `UsageEvent` now has an explicit case in `BaseAdapter._process_event` instead of silently falling through

## [0.1.6] - 2026-03-28

### Added
- v2 StreamPart parsing (`stream_mode="v2"`) with auto-detection of `{"type", "ns", "data"}` dict format
- `ValuesEvent` for full state snapshots from `stream_mode="values"` (v2)
- `DebugEvent` for debug, checkpoint, and task trace data from v2 streaming
- Routing for v2 stream types: updates, messages, custom, values, debug, checkpoints, tasks

## [0.1.5] - 2026-02-06

### Added
- Subgraph namespace preservation on events (`namespace` field on `ContentEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolExtractedEvent`, `InterruptEvent`, `StateUpdateEvent`, `UsageEvent`)
- `agent_name` field on `ContentEvent`, extracted from `lc_agent_name` metadata in messages mode (for deep agent subagents)
- `CustomEvent` for data emitted via `get_stream_writer()` (`stream_mode="custom"`)
- `stream_mode="custom"` support in single and multi-mode parsing

## [0.1.4] - 2026-02-06

### Added
- `context_parts` parameter on `prepare_agent_input()` for prepending context lines (e.g., timestamp, working directory) to user messages

## [0.1.3] - 2026-02-09

### Fixed
- Handle multi-element interrupt tuples from LangGraph subgraphs
- Aggregate `action_requests` and `review_configs` across all Interrupt objects in a tuple

## [0.1.2] - 2026-02-08

### Added
- Subgraph namespace stripping for `subgraphs=True` streams
- Automatic handling of single-mode `(namespace, data)` and multi-mode `(namespace, mode, data)` chunk formats
- All parent and subgraph chunks processed uniformly with namespace stripped

## [0.1.1] - 2026-02-07

### Added
- Dual stream mode support (`stream_mode=["updates", "messages"]`) with automatic deduplication
- Auto-detection mode (`stream_mode="auto"`) that inspects the first chunk
- `MessagesHandler` for token-level content streaming from `stream_mode="messages"`
- `UsageEvent` for token usage metadata from AIMessage `usage_metadata`
- `DisplayInlineExtractor` for extracting inline display artifacts
- Event serialization helpers (`InterruptEvent.build_decisions()`, `InterruptEvent.create_resume()`)

### Changed
- `stream_mode` is now a constructor parameter on `StreamParser` (moved from `parse()`/`aparse()`)
- `UpdatesHandler` accepts `suppress_content` flag for dual-mode deduplication

## [0.1.0] - 2026-02-01

### Initial Release

- Add `StreamParser` for parsing LangGraph stream outputs into typed events
- Add typed event classes: `ContentEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `ToolExtractedEvent`, `InterruptEvent`, `StateUpdateEvent`, `CompleteEvent`, `ErrorEvent`
- Add tool lifecycle tracking (start → end)
- Add extensible extractor system with built-in `ThinkToolExtractor` and `TodoExtractor`
- Add interrupt handling with `create_resume_input()` and `prepare_agent_input()`
- Add async support via `aparse()`
- Add legacy dict-based API for backward compatibility (`stream_graph_updates`, `resume_graph_from_interrupt`)

### Display Adapters

- Add `BaseAdapter` abstract class for building custom display adapters
- Add `PrintAdapter` for plain text output in any Python environment
- Add `CLIAdapter` for styled terminal output with ANSI colors and spinner animation
- Add `JupyterDisplay` for rich notebook display with live updates
- Add configurable `reflection_types` and `todo_types` for custom tool rendering
- Add `**stream_kwargs` pass-through to `graph.stream()`
