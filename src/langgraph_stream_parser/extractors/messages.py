"""
Message content extraction utilities.

These functions handle the extraction of content from LangChain message
objects (AIMessage, ToolMessage, etc.) without directly importing
LangChain types.
"""
import re
from typing import Any


# Block types in langchain-core 1.4 standard content blocks that are
# NOT plain text and should not bleed into ContentEvent. Reasoning blocks
# are surfaced via ReasoningEvent; tool/server-tool blocks are surfaced
# via tool-lifecycle events; multimodal blocks are out of scope for
# text content (consumers needing them should inspect the raw message).
_NON_TEXT_BLOCK_TYPES = frozenset({
    "reasoning",
    "thinking",
    "tool_call",
    "tool_use",
    "tool_call_chunk",
    "server_tool_call",
    "server_tool_call_chunk",
    "server_tool_result",
    "invalid_tool_call",
    "image",
    "audio",
    "video",
    "file",
})


def extract_message_content(message: Any) -> str:
    """Extract and convert message text content to string.

    Handles different content formats:
        - String content (returned as-is)
        - List of content blocks — only ``text`` / ``text-plain`` blocks
          contribute to the returned string. Reasoning, tool-call, and
          server-tool blocks are skipped (those are surfaced via
          ReasoningEvent and tool-lifecycle events respectively).
        - Other types (converted to string)

    Args:
        message: A LangChain message object with 'content' attribute.

    Returns:
        Text content as a string (non-text blocks excluded).
    """
    if not hasattr(message, 'content'):
        return ""

    content = message.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in _NON_TEXT_BLOCK_TYPES:
                    continue
                text = block.get("text")
                if text:
                    parts.append(text)
            else:
                parts.append(str(block))
        return " ".join(parts)
    else:
        return str(content)


def extract_reasoning_content(message: Any) -> str:
    """Extract reasoning / thinking text from a message.

    Returns concatenated text from reasoning blocks in the
    langchain-core standardized format. Recognizes both ``type:
    "reasoning"`` (standardized) and ``type: "thinking"`` (Anthropic).

    Args:
        message: A LangChain message object with 'content' attribute.

    Returns:
        Concatenated reasoning text, or empty string if none found.
    """
    if not hasattr(message, 'content'):
        return ""

    content = message.content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype not in ("reasoning", "thinking"):
            continue
        text = (
            block.get("reasoning")
            or block.get("thinking")
            or block.get("text")
            or ""
        )
        if text:
            parts.append(str(text))
    return "".join(parts)


def clean_tool_dict_from_content(content: str) -> str:
    """Remove tool call dictionary representations from content strings.

    Tool calls sometimes appear as stringified dicts in content:
    "{'id': '...', 'input': {...}, 'name': '...', 'type': 'tool_use'}"

    Args:
        content: The content string to clean.

    Returns:
        Cleaned content string with tool dicts removed.
    """
    # Pattern to match tool call dictionary representations
    tool_dict_pattern = (
        r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*"
        r"'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
    )
    cleaned = re.sub(tool_dict_pattern, '', content, flags=re.DOTALL)
    # Only strip if a substitution was made (to preserve leading/trailing
    # whitespace in normal content tokens like " world")
    if cleaned != content:
        return cleaned.strip()
    return content


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Extract tool calls from an AI message.

    Args:
        message: A LangChain AIMessage with optional 'tool_calls' attribute.

    Returns:
        List of tool call dicts with 'id', 'name', and 'args' keys.
    """
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        return []

    tool_calls = []
    for tc in message.tool_calls:
        if isinstance(tc, dict):
            tool_calls.append({
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args", {}),
            })
        else:
            tool_calls.append({
                "id": getattr(tc, 'id', None),
                "name": getattr(tc, 'name', None),
                "args": getattr(tc, 'args', {}),
            })

    return tool_calls


def detect_tool_error(message: Any) -> tuple[bool, str | None]:
    """Detect if a ToolMessage represents an error.

    Checks multiple indicators:
        - ToolMessage.status attribute
        - Dict content with 'error' key
        - Content starting with error patterns

    Args:
        message: A ToolMessage to check.

    Returns:
        Tuple of (is_error, error_message).
    """
    # Check explicit status attribute
    msg_status = getattr(message, 'status', None)
    if msg_status == 'error':
        content = getattr(message, 'content', '')
        return True, str(content) if content else "Unknown error"

    content = getattr(message, 'content', None)

    # Check for dict with explicit error field
    if isinstance(content, dict) and content.get("error"):
        return True, str(content.get("error"))

    # Check for common error patterns at the START of the message
    if isinstance(content, str):
        content_lower = content.lower().strip()
        error_prefixes = (
            "error:",
            "failed:",
            "exception:",
            "traceback",
        )
        if any(content_lower.startswith(prefix) for prefix in error_prefixes):
            return True, content

    return False, None


def get_message_type_name(message: Any) -> str | None:
    """Get the class name of a message object.

    Args:
        message: A LangChain message object.

    Returns:
        The class name (e.g., "AIMessage", "ToolMessage") or None.
    """
    if hasattr(message, '__class__'):
        return message.__class__.__name__
    return None
