# Changelog

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
