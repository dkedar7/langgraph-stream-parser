"""
Jupyter adapter for rich live display of LangGraph stream events.

Provides real-time, visually polished display that updates in-place
to show tool progress, streaming content, and interrupts.
"""
from typing import Any

from ..events import InterruptEvent, ToolExtractedEvent
from .base import BaseAdapter, ToolState, ToolStatus


class JupyterDisplay(BaseAdapter):
    """Live updating display for LangGraph stream events in Jupyter notebooks.

    Uses rich for rendering and IPython.display for live updates.
    Updates in place to show tool progress, streaming content, and interrupts.

    Example:
        from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

        display = JupyterDisplay()
        display.run(
            graph=agent,
            input_data={"messages": [("user", "Hello")]},
            config={"configurable": {"thread_id": "my-thread"}}
        )

        # For custom event processing, use StreamParser directly:
        from langgraph_stream_parser import StreamParser
        parser = StreamParser()
        for event in parser.parse(graph.stream(...)):
            if should_display(event):  # custom filtering
                display.update(event)
    """

    def __init__(
        self,
        *,
        show_timestamps: bool = False,
        show_tool_args: bool = True,
        max_content_preview: int = 200,
    ):
        """Initialize the Jupyter display.

        Args:
            show_timestamps: Show timestamps on events.
            show_tool_args: Show tool arguments in tool status lines.
            max_content_preview: Max characters for extracted content preview.
        """
        super().__init__(
            show_timestamps=show_timestamps,
            show_tool_args=show_tool_args,
            max_content_preview=max_content_preview,
        )

        # Lazy imports
        self._rich_available: bool | None = None
        self._ipython_available: bool | None = None

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if self._rich_available is None:
            try:
                import rich  # noqa: F401
                self._rich_available = True
            except ImportError:
                self._rich_available = False

        if self._ipython_available is None:
            try:
                from IPython.display import display, clear_output  # noqa: F401
                self._ipython_available = True
            except ImportError:
                self._ipython_available = False

        if not self._rich_available:
            raise ImportError(
                "rich is required for JupyterDisplay. "
                "Install with: pip install 'langgraph-stream-parser[jupyter]'"
            )

        if not self._ipython_available:
            raise ImportError(
                "IPython is required for JupyterDisplay. "
                "This adapter is designed for Jupyter notebooks."
            )

    def run(self, *args, **kwargs) -> None:
        """Run with dependency check."""
        self._check_dependencies()
        super().run(*args, **kwargs)

    def update(self, event) -> None:
        """Update with dependency check."""
        self._check_dependencies()
        super().update(event)

    def render(self) -> None:
        """Render the current state to the notebook."""
        from IPython.display import clear_output
        from rich.console import Console

        # Clear previous output
        clear_output(wait=True)

        # Check if there's anything to render
        has_content = (
            self._display_items or
            self._current_content or
            self._interrupt or
            self._error or
            self._complete
        )

        if not has_content:
            return

        # Create a compact console
        console = Console(force_jupyter=True, width=80)

        # Render items in chronological order
        for item_type, item_data in self._display_items:
            if item_type == "message":
                role, content = item_data
                self._render_message(console, role, content)
            elif item_type == "tool":
                self._render_tool(console, item_data)
            elif item_type == "extraction":
                self._render_extraction(console, item_data)

        # Render current in-progress message
        if self._current_content:
            self._render_message(console, self._current_role or "assistant", self._current_content)

        # Render interrupt
        if self._interrupt:
            self._render_interrupt(console, self._interrupt)

        # Render error
        if self._error:
            console.print(f"[red]ERR:[/red] {self._error.error}")

        # Render completion
        if self._complete and not self._error:
            console.print("[dim]done[/dim]")

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

        # Use input for user prompt
        try:
            response = input(f"Decision ({options_str}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if not response or response not in allowed:
            # Default to reject or first option
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

    def _render_message(self, console: Any, role: str, content: str) -> None:
        """Render a message in a compact panel."""
        from rich.panel import Panel
        from rich import box

        # Truncate very long content for display
        display_content = content[:500] + "..." if len(content) > 500 else content

        if role == "human":
            console.print(Panel(
                display_content,
                title="[green]user[/green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            ))
        else:
            console.print(Panel(
                display_content,
                title="[blue]assistant[/blue]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            ))

    def _render_tool(self, console: Any, tool: ToolState) -> None:
        """Render a single tool call inline."""
        status = self._get_status_icon(tool.status)
        time_str = f" {self.format_duration(tool.duration_ms)}" if tool.duration_ms else ""
        args_str = ""
        if self._show_tool_args and tool.args:
            args_str = f" [dim]{self.format_args(tool.args)}[/dim]"
        console.print(f"{status} [cyan]{tool.name}[/cyan]{args_str}{time_str}")

    def _render_extraction(self, console: Any, event: ToolExtractedEvent) -> None:
        """Render extraction inline."""
        data_str = str(event.data)
        if len(data_str) > self._max_content_preview:
            data_str = data_str[:self._max_content_preview] + "..."

        if event.extracted_type == "todos" and isinstance(event.data, list):
            todos = self.format_todos(event.data)
            if todos:
                lines = []
                for status, content in todos:
                    if status == "completed":
                        icon = "[green]✓[/green]"
                    elif status == "in_progress":
                        icon = "[yellow]▶[/yellow]"
                    else:  # pending
                        icon = "[dim]○[/dim]"
                    lines.append(f"{icon} {content}")
                data_str = "\n  " + "\n  ".join(lines)

        if event.extracted_type == "reflection":
            console.print(f"[magenta]{event.extracted_type}:[/magenta] [italic]{data_str}[/italic]")
        else:
            console.print(f"[magenta]{event.extracted_type}:[/magenta] {data_str}")

    def _render_interrupt(self, console: Any, event: InterruptEvent) -> None:
        """Render interrupt compactly in a panel."""
        from rich.panel import Panel
        from rich import box

        actions = []
        for action in event.action_requests:
            tool = action.get("tool", "unknown")
            args = action.get("args", {})
            args_str = self.format_args(args) if args else ""
            actions.append(f"[cyan]{tool}[/cyan]({args_str})" if args_str else f"[cyan]{tool}[/cyan]")

        actions_str = ", ".join(actions) if actions else "none"

        # Get allowed decisions
        decisions = sorted(self.get_allowed_decisions(event))
        decisions_str = "/".join(decisions) if decisions else "?"

        console.print(Panel(
            f"{actions_str}\n[dim]options: {decisions_str}[/dim]",
            title="[bold white on red]interrupt[/bold white on red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    def _get_status_icon(self, status: ToolStatus) -> str:
        """Get icon for tool status."""
        icons = {
            ToolStatus.PENDING: "[dim]...[/dim]",
            ToolStatus.RUNNING: "[yellow]...[/yellow]",
            ToolStatus.SUCCESS: "[green]OK[/green]",
            ToolStatus.ERROR: "[red]ERR[/red]",
        }
        return icons.get(status, "?")
