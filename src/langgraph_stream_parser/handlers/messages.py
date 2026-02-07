"""
Handler for stream_mode="messages".

This mode produces token-level streaming of LLM outputs. Each chunk is a
(message_chunk, metadata) tuple where message_chunk is typically an
AIMessageChunk and metadata contains langgraph_node, langgraph_step, etc.
"""
from typing import Any, Iterator

from ..events import ContentEvent, StreamEvent
from ..extractors.messages import extract_message_content, get_message_type_name


class MessagesHandler:
    """Handler for stream_mode='messages' chunks.

    Processes (message_chunk, metadata) tuples and produces typed
    StreamEvent objects. Only yields ContentEvent from AIMessageChunk
    text content. Tool call chunks are ignored (the updates handler
    provides complete tool calls in dual mode).
    """

    def process_chunk(self, chunk: Any) -> Iterator[StreamEvent]:
        """Process a single messages-mode chunk.

        Args:
            chunk: A (message_chunk, metadata) tuple from
                graph.stream(stream_mode="messages"), or the inner
                data when unwrapped from a multi-mode tuple.

        Yields:
            StreamEvent objects (ContentEvent only).
        """
        if isinstance(chunk, tuple) and len(chunk) == 2:
            message, metadata = chunk
        else:
            message = chunk
            metadata = {}

        message_type = get_message_type_name(message)

        if message_type == "AIMessageChunk":
            yield from self._process_ai_chunk(message, metadata)

    def _process_ai_chunk(
        self, chunk: Any, metadata: dict
    ) -> Iterator[StreamEvent]:
        """Process an AIMessageChunk for text content only.

        Args:
            chunk: An AIMessageChunk object.
            metadata: Metadata dict with langgraph_node etc.

        Yields:
            ContentEvent if the chunk has non-empty text content.
        """
        content = extract_message_content(chunk)

        if not content:
            return

        node_name = None
        if isinstance(metadata, dict):
            node_name = metadata.get("langgraph_node")

        yield ContentEvent(
            content=content,
            role="assistant",
            node=node_name,
        )
