"""
Main StreamParser class for parsing LangGraph streaming outputs.

This is the primary interface for the langgraph-stream-parser package.
"""
from itertools import chain
from typing import Any, AsyncIterator, Iterator

from .events import (
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
    ToolCallStartEvent,
)
from .extractors.base import ToolExtractor
from .extractors.builtins import ThinkToolExtractor, TodoExtractor
from .handlers.messages import MessagesHandler
from .handlers.updates import UpdatesHandler

_VALID_MODES = {"updates", "messages"}


def _is_multi_mode(chunk: Any) -> bool:
    """Check if a chunk is from multi-mode streaming (a (str, data) tuple)."""
    return (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[0], str)
        and chunk[0] in _VALID_MODES
    )


class StreamParser:
    """Universal parser for LangGraph streaming outputs.

    Normalizes various output formats into typed StreamEvent objects
    that are easy to consume in application code.

    Example:
        parser = StreamParser(stream_mode=["updates", "messages"])

        for event in parser.parse(graph.stream(input, stream_mode=["updates", "messages"])):
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
        stream_mode: str | list[str] = "updates",
        track_tool_lifecycle: bool = True,
        skip_tools: list[str] | None = None,
        include_state_updates: bool = False,
    ):
        """Initialize the parser.

        Args:
            stream_mode: Tells the parser what stream format to expect.
                - "updates" (default): chunks are plain dicts
                - "messages": chunks are (AIMessageChunk, metadata) tuples
                - ["updates", "messages"]: chunks are (mode_name, data) tuples
                - "auto": auto-detect from the first chunk
            track_tool_lifecycle: If True, emit ToolCallStartEvent when tools
                are called and ToolCallEndEvent when results arrive.
                If False, only emit ToolExtractedEvent for registered extractors.
            skip_tools: Tool names to skip entirely (no events emitted).
                Useful for internal tools you don't want to expose in UI.
            include_state_updates: If True, emit StateUpdateEvent for non-message
                state keys in updates mode.

        Raises:
            ValueError: If stream_mode is invalid.
        """
        self._stream_mode = stream_mode
        self._validate_stream_mode(stream_mode)

        self._track_tool_lifecycle = track_tool_lifecycle
        self._skip_tools = set(skip_tools or [])
        self._include_state_updates = include_state_updates
        self._extractors: dict[str, ToolExtractor] = {}
        self._pending_tool_calls: dict[str, ToolCallStartEvent] = {}

        # Register built-in extractors
        self._register_builtin_extractors()

    @staticmethod
    def _validate_stream_mode(stream_mode: str | list[str]) -> None:
        """Validate the stream_mode parameter."""
        if isinstance(stream_mode, str):
            valid = _VALID_MODES | {"auto"}
            if stream_mode not in valid:
                raise ValueError(
                    f"Unsupported stream_mode: {stream_mode!r}. "
                    f"Must be one of {sorted(valid)} or a list of modes."
                )
        elif isinstance(stream_mode, list):
            for mode in stream_mode:
                if mode not in _VALID_MODES:
                    raise ValueError(
                        f"Unsupported mode in stream_mode list: {mode!r}. "
                        f"Each element must be one of {sorted(_VALID_MODES)}."
                    )
        else:
            raise ValueError(
                f"stream_mode must be a string or list of strings, "
                f"got {type(stream_mode).__name__}."
            )

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

    def parse(self, stream: Iterator[Any]) -> Iterator[StreamEvent]:
        """Parse a LangGraph stream into typed events.

        This is the main entry point for parsing. It iterates over the
        stream, processes each chunk, and yields typed events.

        Args:
            stream: Iterator from graph.stream().

        Yields:
            StreamEvent objects.

        Example:
            for event in parser.parse(graph.stream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            effective_mode = self._stream_mode

            if effective_mode == "auto":
                stream, effective_mode = self._peek_and_detect(stream)

            if isinstance(effective_mode, list):
                yield from self._parse_multi_mode(stream)
            else:
                handler = self._create_handler_for_mode(effective_mode)
                for chunk in stream:
                    yield from handler.process_chunk(chunk)

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    async def aparse(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        """Async version of parse().

        Args:
            stream: AsyncIterator from graph.astream().

        Yields:
            StreamEvent objects.

        Example:
            async for event in parser.aparse(graph.astream(input)):
                if isinstance(event, ContentEvent):
                    print(event.content, end="")
        """
        try:
            effective_mode = self._stream_mode

            if effective_mode == "auto":
                stream, effective_mode = await self._apeek_and_detect(stream)

            if isinstance(effective_mode, list):
                async for event in self._aparse_multi_mode(stream):
                    yield event
            else:
                handler = self._create_handler_for_mode(effective_mode)
                async for chunk in stream:
                    for event in handler.process_chunk(chunk):
                        yield event

            yield CompleteEvent()

        except Exception as e:
            yield ErrorEvent(
                error=f"Error parsing stream: {str(e)}",
                exception=e,
            )

    def parse_chunk(self, chunk: Any) -> list[StreamEvent]:
        """Parse a single chunk into events.

        Useful for manual iteration or when you need to process
        chunks individually.

        Args:
            chunk: A single chunk from graph.stream().

        Returns:
            List of events (may be empty, one, or multiple).

        Raises:
            ValueError: If stream_mode is "auto" (requires stream context).
        """
        if self._stream_mode == "auto":
            raise ValueError(
                "parse_chunk() does not support stream_mode='auto'. "
                "Use parse() or aparse() instead."
            )

        if isinstance(self._stream_mode, list):
            # Multi-mode: expect (mode_name, data) tuple
            if not (_is_multi_mode(chunk)):
                return []
            mode_name, data = chunk
            handler = self._create_handler_for_mode(
                mode_name,
                suppress_content=(mode_name == "updates"),
            )
            return list(handler.process_chunk(data))

        handler = self._create_handler_for_mode(self._stream_mode)
        return list(handler.process_chunk(chunk))

    def _create_updates_handler(
        self, suppress_content: bool = False
    ) -> UpdatesHandler:
        """Create an UpdatesHandler configured for this parser."""
        return UpdatesHandler(
            extractors=self._extractors,
            skip_tools=self._skip_tools,
            track_tool_lifecycle=self._track_tool_lifecycle,
            include_state_updates=self._include_state_updates,
            pending_tool_calls=self._pending_tool_calls,
            suppress_content=suppress_content,
        )

    def _create_messages_handler(self) -> MessagesHandler:
        """Create a MessagesHandler."""
        return MessagesHandler()

    def _create_handler_for_mode(
        self, mode: str, suppress_content: bool = False
    ) -> UpdatesHandler | MessagesHandler:
        """Create the appropriate handler for a given mode string."""
        if mode == "updates":
            return self._create_updates_handler(
                suppress_content=suppress_content
            )
        elif mode == "messages":
            return self._create_messages_handler()
        else:
            raise ValueError(f"Unsupported stream_mode: {mode!r}.")

    def _parse_multi_mode(self, stream: Iterator[Any]) -> Iterator[StreamEvent]:
        """Parse a multi-mode stream with deduplication.

        In dual mode, ContentEvent comes from "messages" (token-level)
        and tool/interrupt/state events come from "updates".
        """
        updates_handler = self._create_updates_handler(suppress_content=True)
        messages_handler = self._create_messages_handler()

        for chunk in stream:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                continue

            mode_name, data = chunk

            if mode_name == "updates":
                yield from updates_handler.process_chunk(data)
            elif mode_name == "messages":
                yield from messages_handler.process_chunk(data)

    async def _aparse_multi_mode(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[StreamEvent]:
        """Async version of _parse_multi_mode."""
        updates_handler = self._create_updates_handler(suppress_content=True)
        messages_handler = self._create_messages_handler()

        async for chunk in stream:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                continue

            mode_name, data = chunk

            if mode_name == "updates":
                for event in updates_handler.process_chunk(data):
                    yield event
            elif mode_name == "messages":
                for event in messages_handler.process_chunk(data):
                    yield event

    def _peek_and_detect(
        self, stream: Iterator[Any]
    ) -> tuple[Iterator[Any], str | list[str]]:
        """Peek at the first chunk to auto-detect stream format.

        Returns:
            (chained_stream, detected_mode) where detected_mode is
            either "updates" or ["updates", "messages"].
        """
        try:
            first_chunk = next(stream)
        except StopIteration:
            return iter([]), "updates"

        if _is_multi_mode(first_chunk):
            return chain([first_chunk], stream), ["updates", "messages"]

        return chain([first_chunk], stream), "updates"

    async def _apeek_and_detect(
        self, stream: AsyncIterator[Any]
    ) -> tuple[AsyncIterator[Any], str | list[str]]:
        """Async version of _peek_and_detect."""
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration:
            return _empty_async_iter(), "updates"

        if _is_multi_mode(first_chunk):
            return _async_chain(first_chunk, stream), ["updates", "messages"]

        return _async_chain(first_chunk, stream), "updates"

    def reset(self) -> None:
        """Reset parser state.

        Clears pending tool calls. Call this when starting a new
        conversation or stream.
        """
        self._pending_tool_calls.clear()


async def _empty_async_iter() -> AsyncIterator[Any]:
    """Empty async iterator."""
    return
    yield  # noqa: unreachable — makes this an async generator


async def _async_chain(
    first: Any, rest: AsyncIterator[Any]
) -> AsyncIterator[Any]:
    """Async equivalent of itertools.chain([first], rest)."""
    yield first
    async for item in rest:
        yield item
