"""
Main StreamParser class for parsing LangGraph streaming outputs.

This is the primary interface for the langgraph-stream-parser package.
"""
from typing import Any, AsyncIterator, Iterator

from .events import (
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
    ToolCallStartEvent,
)
from .extractors.base import ToolExtractor
from .extractors.builtins import ThinkToolExtractor, TodoExtractor
from .handlers.updates import UpdatesHandler


class StreamParser:
    """Universal parser for LangGraph streaming outputs.

    Normalizes various output formats into typed StreamEvent objects
    that are easy to consume in application code.

    Example:
        parser = StreamParser()

        # Register custom tool extractor
        parser.register_extractor(MyCanvasExtractor())

        # Parse stream
        for event in parser.parse(graph.stream(input, stream_mode="updates")):
            match event:
                case ContentEvent(content=text):
                    print(text, end="")
                case ToolCallStartEvent(name=name):
                    print(f"Calling {name}...")
                case InterruptEvent(action_requests=actions):
                    # Handle HITL
                    ...
    """

    def __init__(
        self,
        *,
        track_tool_lifecycle: bool = True,
        skip_tools: list[str] | None = None,
        include_state_updates: bool = False,
    ):
        """Initialize the parser.

        Args:
            track_tool_lifecycle: If True, emit ToolCallStartEvent when tools
                are called and ToolCallEndEvent when results arrive.
                If False, only emit ToolExtractedEvent for registered extractors.
            skip_tools: Tool names to skip entirely (no events emitted).
                Useful for internal tools you don't want to expose in UI.
            include_state_updates: If True, emit StateUpdateEvent for non-message
                state keys in updates mode.
        """
        self._track_tool_lifecycle = track_tool_lifecycle
        self._skip_tools = set(skip_tools or [])
        self._include_state_updates = include_state_updates
        self._extractors: dict[str, ToolExtractor] = {}
        self._pending_tool_calls: dict[str, ToolCallStartEvent] = {}

        # Register built-in extractors
        self._register_builtin_extractors()

    def _register_builtin_extractors(self) -> None:
        """Register the built-in tool extractors."""
        self.register_extractor(ThinkToolExtractor())
        self.register_extractor(TodoExtractor())

    def register_extractor(self, extractor: ToolExtractor) -> None:
        """Register a custom tool extractor.

        Extractors process ToolMessage content and emit ToolExtractedEvent
        with the extracted data.

        Args:
            extractor: An object implementing the ToolExtractor protocol.
        """
        self._extractors[extractor.tool_name] = extractor

    def unregister_extractor(self, tool_name: str) -> None:
        """Remove a registered extractor.

        Args:
            tool_name: The tool name to unregister.
        """
        self._extractors.pop(tool_name, None)

    def parse(
        self,
        stream: Iterator[Any],
        *,
        stream_mode: str = "updates",
    ) -> Iterator[StreamEvent]:
        """Parse a LangGraph stream into typed events.

        This is the main entry point for parsing. It iterates over the
        stream, processes each chunk, and yields typed events.

        Args:
            stream: Iterator from graph.stream().
            stream_mode: The stream_mode used when calling graph.stream().
                Currently only "updates" is supported.

        Yields:
            StreamEvent objects.

        Example:
            for event in parser.parse(graph.stream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            handler = self._get_handler(stream_mode)

            for chunk in stream:
                yield from handler.process_chunk(chunk)

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    async def aparse(
        self,
        stream: AsyncIterator[Any],
        *,
        stream_mode: str = "updates",
    ) -> AsyncIterator[StreamEvent]:
        """Async version of parse().

        Args:
            stream: AsyncIterator from graph.astream().
            stream_mode: The stream_mode used when calling graph.astream().

        Yields:
            StreamEvent objects.

        Example:
            async for event in parser.aparse(graph.astream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            handler = self._get_handler(stream_mode)

            async for chunk in stream:
                for event in handler.process_chunk(chunk):
                    yield event

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    def parse_chunk(
        self, chunk: Any, stream_mode: str = "updates"
    ) -> list[StreamEvent]:
        """Parse a single chunk into events.

        Useful for manual iteration or when you need to process
        chunks individually.

        Args:
            chunk: A single chunk from graph.stream().
            stream_mode: The stream_mode used.

        Returns:
            List of events (may be empty, one, or multiple).
        """
        handler = self._get_handler(stream_mode)
        return list(handler.process_chunk(chunk))

    def _get_handler(self, stream_mode: str) -> UpdatesHandler:
        """Get the appropriate handler for the stream mode.

        Args:
            stream_mode: The stream mode.

        Returns:
            Handler instance configured for this parser.

        Raises:
            ValueError: If stream_mode is not supported.
        """
        if stream_mode == "updates":
            return UpdatesHandler(
                extractors=self._extractors,
                skip_tools=self._skip_tools,
                track_tool_lifecycle=self._track_tool_lifecycle,
                include_state_updates=self._include_state_updates,
                pending_tool_calls=self._pending_tool_calls,
            )
        else:
            raise ValueError(
                f"Unsupported stream_mode: {stream_mode}. "
                f"Currently only 'updates' is supported."
            )

    def reset(self) -> None:
        """Reset parser state.

        Clears pending tool calls. Call this when starting a new
        conversation or stream.
        """
        self._pending_tool_calls.clear()
