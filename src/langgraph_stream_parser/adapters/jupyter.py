"""
Jupyter adapter for rich live display of LangGraph stream events.

Provides real-time, visually polished display that updates in-place
to show tool progress, streaming content, and interrupts.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator

from ..events import (
    StreamEvent,
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
)
from ..parser import StreamParser


class ToolStatus(Enum):
    """Status of a tool in the lifecycle."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ToolState:
    """Tracks the state of a single tool call."""
    id: str
    name: str
    args: dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    result: Any = None
    error_message: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class JupyterDisplay:
    """Live updating display for LangGraph stream events in Jupyter notebooks.

    Uses rich for rendering and IPython.display for live updates.
    Updates in place to show tool progress, streaming content, and interrupts.

    Example:
        from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

        display = JupyterDisplay()
        display.stream(graph.stream(input, stream_mode="updates"))

        # Or with manual control
        for event in parser.parse(graph.stream(...)):
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
        self._show_timestamps = show_timestamps
        self._show_tool_args = show_tool_args
        self._max_content_preview = max_content_preview

        # State tracking - chronological list of display items
        # Each item is a tuple: (type, data) where type is "message", "tool", "extraction", "interrupt", "error"
        self._display_items: list[tuple[str, Any]] = []

        # Current message being accumulated
        self._current_role: str | None = None
        self._current_content: str = ""

        # Tool state (for updating status) - maps tool_id to index in _display_items
        self._tool_indices: dict[str, int] = {}

        # Final state
        self._interrupt: InterruptEvent | None = None
        self._error: ErrorEvent | None = None
        self._complete: bool = False

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

    def reset(self) -> None:
        """Reset display state for a new stream."""
        self._display_items.clear()
        self._current_role = None
        self._current_content = ""
        self._tool_indices.clear()
        self._interrupt = None
        self._error = None
        self._complete = False

    def stream(
        self,
        stream: Iterator[Any],
        *,
        parser: StreamParser | None = None,
        stream_mode: str = "updates",
    ) -> None:
        """Stream and display events from a LangGraph stream.

        This is the main entry point. It creates a parser (if not provided),
        parses the stream, and displays events in real-time.

        Args:
            stream: Iterator from graph.stream().
            parser: Optional pre-configured StreamParser.
            stream_mode: The stream mode used.
        """
        self._check_dependencies()
        self.reset()

        if parser is None:
            parser = StreamParser()

        for event in parser.parse(stream, stream_mode=stream_mode):
            self.update(event)

    def update(self, event: StreamEvent) -> None:
        """Update display with a single event.

        Args:
            event: A StreamEvent to display.
        """
        self._check_dependencies()
        self._process_event(event)
        self._render()

    def _flush_current_message(self) -> None:
        """Flush current message buffer to display items list.

        Merges with the previous message if it has the same role.
        """
        if self._current_content and self._current_role:
            # Check if we can merge with the last message of same role
            if self._display_items:
                last_type, last_data = self._display_items[-1]
                if last_type == "message":
                    last_role, last_content = last_data
                    if last_role == self._current_role:
                        # Merge with previous message
                        self._display_items[-1] = ("message", (self._current_role, last_content + "\n" + self._current_content))
                        self._current_content = ""
                        return

            # Otherwise add as new message
            self._display_items.append(("message", (self._current_role, self._current_content)))
            self._current_content = ""

    def _process_event(self, event: StreamEvent) -> None:
        """Process an event and update internal state."""
        match event:
            case ContentEvent(content=text, role=role):
                # If role changes, flush the previous message
                if self._current_role is not None and self._current_role != role:
                    self._flush_current_message()
                self._current_role = role
                self._current_content += text

            case ToolCallStartEvent(id=tool_id, name=name, args=args):
                # Flush any pending content before tool
                self._flush_current_message()

                # Create tool state and add to display items
                tool_state = ToolState(
                    id=tool_id,
                    name=name,
                    args=args,
                    status=ToolStatus.RUNNING,
                )
                self._tool_indices[tool_id] = len(self._display_items)
                self._display_items.append(("tool", tool_state))

            case ToolCallEndEvent(
                id=tool_id,
                result=result,
                status=status,
                error_message=error_msg,
            ):
                if tool_id in self._tool_indices:
                    idx = self._tool_indices[tool_id]
                    _, tool = self._display_items[idx]
                    tool.end_time = datetime.now()
                    tool.result = result
                    if status == "success":
                        tool.status = ToolStatus.SUCCESS
                    else:
                        tool.status = ToolStatus.ERROR
                        tool.error_message = error_msg

            case ToolExtractedEvent():
                self._display_items.append(("extraction", event))

            case InterruptEvent():
                self._flush_current_message()
                self._interrupt = event

            case ErrorEvent():
                self._error = event

            case CompleteEvent():
                self._flush_current_message()
                self._complete = True

            case StateUpdateEvent():
                pass  # Ignored in display

    def _render(self) -> None:
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
        time_str = f" {self._format_duration(tool.duration_ms)}" if tool.duration_ms else ""
        args_str = ""
        if self._show_tool_args and tool.args:
            args_str = f" [dim]{self._format_args(tool.args)}[/dim]"
        console.print(f"{status} [cyan]{tool.name}[/cyan]{args_str}{time_str}")

    def _render_extraction(self, console: Any, event: ToolExtractedEvent) -> None:
        """Render extraction inline."""
        data_str = str(event.data)
        if len(data_str) > self._max_content_preview:
            data_str = data_str[:self._max_content_preview] + "..."

        if event.extracted_type == "todos" and isinstance(event.data, list):
            tasks = []
            for item in event.data:
                if isinstance(item, dict):
                    # Support both formats: {status, content} and {done, task}
                    status = item.get("status", "pending")
                    content = item.get("content") or item.get("task", str(item))

                    # Handle done field as fallback
                    if "done" in item:
                        status = "completed" if item["done"] else "pending"

                    # Status icons
                    if status == "completed":
                        icon = "[green]✓[/green]"
                    elif status == "in_progress":
                        icon = "[yellow]▶[/yellow]"
                    else:  # pending
                        icon = "[dim]○[/dim]"

                    tasks.append(f"{icon} {content}")
            if tasks:
                data_str = "\n  " + "\n  ".join(tasks)

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
            args_str = self._format_args(args) if args else ""
            actions.append(f"[cyan]{tool}[/cyan]({args_str})" if args_str else f"[cyan]{tool}[/cyan]")

        actions_str = ", ".join(actions) if actions else "none"

        # Get allowed decisions
        decisions = []
        for config in event.review_configs:
            decisions.extend(config.get("allowed_decisions", []))
        decisions_str = "/".join(sorted(set(decisions))) if decisions else "?"

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

    def _format_duration(self, duration_ms: float | None) -> str:
        """Format duration for display."""
        if duration_ms is None:
            return ""
        if duration_ms < 1000:
            return f"{duration_ms:.0f}ms"
        return f"{duration_ms / 1000:.1f}s"

    def _format_args(self, args: dict[str, Any]) -> str:
        """Format tool arguments for display."""
        if not args:
            return ""
        parts = []
        for key, value in args.items():
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            parts.append(f"{key}={value_str}")
        result = ", ".join(parts)
        if len(result) > 40:
            result = result[:37] + "..."
        return result
