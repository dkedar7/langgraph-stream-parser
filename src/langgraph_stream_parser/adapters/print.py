"""
Print adapter for plain text display of LangGraph stream events.

Provides simple, universal output that works in any Python environment
without dependencies on rich, IPython, or other display libraries.
"""
from typing import Any

from ..events import InterruptEvent, ToolExtractedEvent
from .base import BaseAdapter, ToolState, ToolStatus


class PrintAdapter(BaseAdapter):
    """Plain text display for LangGraph stream events.

    Outputs simple formatted text using print(). Works universally
    in any Python environment - scripts, notebooks, REPL, etc.

    This is the default adapter when no specialized rendering is needed.

    Example:
        from langgraph_stream_parser.adapters.print import PrintAdapter

        adapter = PrintAdapter()
        adapter.run(
            graph=agent,
            input_data={"messages": [("user", "Hello")]},
            config={"configurable": {"thread_id": "my-thread"}}
        )

        # For custom event processing:
        from langgraph_stream_parser import StreamParser
        parser = StreamParser()
        for event in parser.parse(graph.stream(...)):
            adapter.update(event)
    """

    def __init__(
        self,
        *,
        show_timestamps: bool = False,
        show_tool_args: bool = True,
        max_content_preview: int = 200,
        verbose: bool = False,
        reflection_types: set[str] | list[str] | None = None,
        todo_types: set[str] | list[str] | None = None,
    ):
        """Initialize the print adapter.

        Args:
            show_timestamps: Show timestamps on events.
            show_tool_args: Show tool arguments in tool status lines.
            max_content_preview: Max characters for extracted content preview.
            verbose: If True, print more detailed output.
            reflection_types: Set of extracted_type values to render as reflections.
                Defaults to {"reflection"}.
            todo_types: Set of extracted_type values to render as todo lists.
                Defaults to {"todos"}.
        """
        super().__init__(
            show_timestamps=show_timestamps,
            show_tool_args=show_tool_args,
            max_content_preview=max_content_preview,
            reflection_types=reflection_types,
            todo_types=todo_types,
        )
        self._verbose = verbose
        self._last_rendered_count = 0

    def render(self) -> None:
        """Render new items since last render."""
        # Only render new items (incremental output)
        items_to_render = self._display_items[self._last_rendered_count:]

        for item_type, item_data in items_to_render:
            if item_type == "message":
                role, content = item_data
                self._print_message(role, content)
            elif item_type == "tool":
                self._print_tool(item_data)
            elif item_type == "extraction":
                self._print_extraction(item_data)

        self._last_rendered_count = len(self._display_items)

        # Render current in-progress content (streaming)
        # Note: For print adapter, we accumulate and show on completion
        # to avoid partial line issues

        # Render interrupt
        if self._interrupt:
            self._print_interrupt(self._interrupt)

        # Render error
        if self._error:
            print(f"ERROR: {self._error.error}")

        # Render completion
        if self._complete and not self._error and self._verbose:
            print("--- Done ---")

    def reset(self) -> None:
        """Reset state for a new stream."""
        super().reset()
        self._last_rendered_count = 0

    def prompt_interrupt(self, event: InterruptEvent) -> list[dict[str, Any]] | None:
        """Prompt user for interrupt decision using input().

        Args:
            event: The InterruptEvent requiring a decision.

        Returns:
            List of decision dicts, or None if cancelled.
        """
        allowed = self.get_allowed_decisions(event)
        options = sorted(allowed)
        options_str = "/".join(options)

        try:
            response = input(f"Decision ({options_str}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if not response or response not in allowed:
            response = "reject" if "reject" in allowed else options[0]

        # Handle edit with args prompt
        args_modifier = None
        if response == "edit":
            try:
                import json
                new_args_str = input("New args (JSON): ").strip()
                if new_args_str:
                    new_args = json.loads(new_args_str)
                    args_modifier = lambda _: new_args  # noqa: E731
            except (EOFError, KeyboardInterrupt, json.JSONDecodeError):
                response = "reject"

        return self.build_decisions(event, response, args_modifier)

    def _print_message(self, role: str, content: str) -> None:
        """Print a message."""
        label = "User" if role == "human" else "Assistant"
        print(f"\n[{label}]")
        print(content)

    def _print_tool(self, tool: ToolState) -> None:
        """Print tool status."""
        status_str = self._get_status_str(tool.status)
        time_str = ""
        if tool.duration_ms:
            time_str = f" ({self.format_duration(tool.duration_ms)})"

        args_str = ""
        if self._show_tool_args and tool.args:
            args_str = f" {self.format_args(tool.args)}"

        print(f"{status_str} {tool.name}{args_str}{time_str}")

        if tool.status == ToolStatus.ERROR and tool.error_message:
            print(f"   Error: {tool.error_message}")

    def _print_extraction(self, event: ToolExtractedEvent) -> None:
        """Print extracted content."""
        data_str = str(event.data)
        if len(data_str) > self._max_content_preview:
            data_str = data_str[:self._max_content_preview] + "..."

        # Special handling for todo types
        if event.extracted_type in self._todo_types and isinstance(event.data, list):
            todos = self.format_todos(event.data)
            if todos:
                print(f"{event.extracted_type}:")
                for status, content in todos:
                    if status == "completed":
                        icon = "[x]"
                    elif status == "in_progress":
                        icon = "[>]"
                    else:
                        icon = "[ ]"
                    print(f"  {icon} {content}")
                return

        print(f"{event.extracted_type}: {data_str}")

    def _print_interrupt(self, event: InterruptEvent) -> None:
        """Print interrupt information."""
        print("\n--- INTERRUPT ---")

        for action in event.action_requests:
            tool = action.get("tool", "unknown")
            args = action.get("args", {})
            args_str = self.format_args(args) if args else ""
            if args_str:
                print(f"  Tool: {tool}({args_str})")
            else:
                print(f"  Tool: {tool}")

        decisions = sorted(self.get_allowed_decisions(event))
        print(f"  Options: {', '.join(decisions)}")

    def _get_status_str(self, status: ToolStatus) -> str:
        """Get string representation of tool status."""
        status_map = {
            ToolStatus.PENDING: "[...]",
            ToolStatus.RUNNING: "[...]",
            ToolStatus.SUCCESS: "[ OK]",
            ToolStatus.ERROR: "[ERR]",
        }
        return status_map.get(status, "[???]")
