"""
Built-in tool extractors for common LangGraph tools.

These extractors handle tools that are commonly used across
LangGraph applications, such as think_tool for reflections
and write_todos for todo list management.
"""
import ast
import json
import re
from typing import Any


class ThinkToolExtractor:
    """Extractor for think_tool reflections.

    The think_tool is commonly used to give AI agents a scratchpad
    for reasoning. This extractor pulls out the reflection text
    from the tool's output.

    Handles formats:
        - String content (returned as-is)
        - JSON with 'reflection' key
        - Dict with 'reflection' key
    """

    tool_name = "think_tool"
    extracted_type = "reflection"

    def extract(self, content: Any) -> str | None:
        """Extract reflection from think_tool content.

        Args:
            content: The content from the think_tool ToolMessage.

        Returns:
            The reflection string, or None if not found.
        """
        if isinstance(content, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed.get("reflection")
            except (json.JSONDecodeError, TypeError):
                pass
            # Return raw string if not JSON
            return content if content.strip() else None

        if isinstance(content, dict):
            return content.get("reflection")

        return None


class TodoExtractor:
    """Extractor for write_todos tool output.

    The write_todos tool is used for task management within agents.
    This extractor handles the various formats the tool might return
    its todo list in.

    Handles formats:
        - Direct list of todo items
        - JSON string containing array
        - String with embedded array (e.g., "Updated todo list to [...]")
        - Dict with 'todos' key
        - Python literal syntax (single quotes)
    """

    tool_name = "write_todos"
    extracted_type = "todos"

    def extract(self, content: Any) -> list[dict[str, Any]] | None:
        """Extract todo list from write_todos content.

        Args:
            content: The content from the write_todos ToolMessage.

        Returns:
            List of todo items, or None if parsing fails.
        """
        todos = None

        if isinstance(content, str):
            # Look for array pattern first (handles "Updated todo list to [...]" format)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                array_str = match.group(0)

                # Try parsing as Python literal first (handles single quotes)
                try:
                    todos = ast.literal_eval(array_str)
                except (ValueError, SyntaxError):
                    # Fall back to JSON parsing (requires double quotes)
                    try:
                        todos = json.loads(array_str)
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                # No array found, try parsing entire string as JSON
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        todos = parsed.get('todos')
                        # If todos is a string, parse it again
                        if isinstance(todos, str):
                            todos = json.loads(todos)
                    elif isinstance(parsed, list):
                        # Content is directly a list
                        todos = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

        elif isinstance(content, dict):
            todos = content.get('todos')
            if isinstance(todos, str):
                try:
                    todos = json.loads(todos)
                except (json.JSONDecodeError, TypeError):
                    pass

        elif isinstance(content, list):
            # Content is directly a list
            todos = content

        return todos if isinstance(todos, list) else None
