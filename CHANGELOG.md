# Changelog

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
