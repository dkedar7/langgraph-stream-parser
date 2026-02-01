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

        # State tracking - messages as list of (role, content) tuples
        self._messages: list[tuple[str, str]] = []
        self._current_role: str | None = None
        self._current_content: str = ""
        self._tools: dict[str, ToolState] = {}
        self._extractions: list[ToolExtractedEvent] = []
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
        self._messages.clear()
        self._current_role = None
        self._current_content = ""
        self._tools.clear()
        self._extractions.clear()
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
        """Flush current message buffer to messages list."""
        if self._current_content and self._current_role:
            self._messages.append((self._current_role, self._current_content))
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
                self._tools[tool_id] = ToolState(
                    id=tool_id,
                    name=name,
                    args=args,
                    status=ToolStatus.RUNNING,
                )

            case ToolCallEndEvent(
                id=tool_id,
                result=result,
                status=status,
                error_message=error_msg,
            ):
                if tool_id in self._tools:
                    tool = self._tools[tool_id]
                    tool.end_time = datetime.now()
                    tool.result = result
                    if status == "success":
                        tool.status = ToolStatus.SUCCESS
                    else:
                        tool.status = ToolStatus.ERROR
                        tool.error_message = error_msg

            case ToolExtractedEvent():
                self._extractions.append(event)

            case InterruptEvent():
                self._interrupt = event

            case ErrorEvent():
                self._error = event

            case CompleteEvent():
                self._complete = True

            case StateUpdateEvent():
                pass  # Ignored in display

    def _render(self) -> None:
        """Render the current state to the notebook."""
        from IPython.display import clear_output
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box

        # Clear previous output
        clear_output(wait=True)

        # Check if there's anything to render
        has_content = (
            self._messages or
            self._current_content or
            self._tools or
            self._extractions or
            self._interrupt or
            self._error or
            self._complete
        )

        if not has_content:
            return

        # Create a console that outputs directly to Jupyter
        console = Console(force_jupyter=True, width=100)

        # Render completed messages
        for role, content in self._messages:
            self._render_message_panel(console, role, content)

        # Render current in-progress message
        if self._current_content:
            self._render_message_panel(console, self._current_role or "assistant", self._current_content)

        # Render tools table if we have tools
        if self._tools:
            table = Table(
                show_header=True,
                header_style="bold",
                box=box.SIMPLE,
                padding=(0, 1),
            )
            table.add_column("Status", width=8)
            table.add_column("Tool", style="cyan")
            if self._show_tool_args:
                table.add_column("Args", max_width=40)
            table.add_column("Time", justify="right", width=10)

            for tool in self._tools.values():
                status_icon = self._get_status_icon(tool.status)
                time_str = self._format_duration(tool.duration_ms)
                args_str = self._format_args(tool.args) if self._show_tool_args else ""

                row = [status_icon, tool.name]
                if self._show_tool_args:
                    row.append(args_str)
                row.append(time_str)
                table.add_row(*row)

            console.print(Panel(
                table,
                title="[bold yellow]Tools[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        # Render extractions
        for extraction in self._extractions:
            self._render_extraction(console, extraction)

        # Render interrupt with high visibility
        if self._interrupt:
            self._render_interrupt(console, self._interrupt)

        # Render error
        if self._error:
            error_text = Text(self._error.error, style="bold red")
            console.print(Panel(
                error_text,
                title="[bold red]Error[/bold red]",
                border_style="red",
                box=box.HEAVY,
            ))

        # Render completion
        if self._complete and not self._error:
            console.print(Text("Stream completed", style="dim green"))

    def _render_message_panel(self, console: Any, role: str, content: str) -> None:
        """Render a message panel for the given role."""
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        if role == "human":
            console.print(Panel(
                Text(content),
                title="[bold green]Human[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            ))
        else:
            console.print(Panel(
                Text(content),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue",
                box=box.ROUNDED,
            ))

    def _render_extraction(self, console: Any, event: ToolExtractedEvent) -> None:
        """Render a tool extraction event."""
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich import box

        title = f"[bold magenta]{event.extracted_type.title()}[/bold magenta] from {event.tool_name}"

        if event.extracted_type == "reflection":
            # Show reflection text
            text = str(event.data)
            if len(text) > self._max_content_preview:
                text = text[:self._max_content_preview] + "..."
            console.print(Panel(
                Text(text, style="italic"),
                title=title,
                border_style="magenta",
                box=box.ROUNDED,
            ))

        elif event.extracted_type == "todos":
            # Show todos as a table
            if isinstance(event.data, list):
                table = Table(show_header=True, box=box.SIMPLE)
                table.add_column("Done", width=4)
                table.add_column("Task")

                for item in event.data:
                    if isinstance(item, dict):
                        done = item.get("done", False)
                        task = item.get("task", str(item))
                        done_icon = "[green]" if done else "[ ]"
                        table.add_row(done_icon, task)

                console.print(Panel(
                    table,
                    title=title,
                    border_style="magenta",
                    box=box.ROUNDED,
                ))
        else:
            # Generic extraction - show as formatted data
            text = str(event.data)
            if len(text) > self._max_content_preview:
                text = text[:self._max_content_preview] + "..."
            console.print(Panel(
                Text(text),
                title=title,
                border_style="magenta",
                box=box.ROUNDED,
            ))

    def _render_interrupt(self, console: Any, event: InterruptEvent) -> None:
        """Render an interrupt event with high visibility."""
        from rich.panel import Panel
        from rich.text import Text
        from rich.table import Table
        from rich import box

        # Create content for the interrupt panel
        content_parts = []

        if event.action_requests:
            table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
            table.add_column("#", width=3)
            table.add_column("Tool", style="cyan")
            table.add_column("Arguments", max_width=50)

            for i, action in enumerate(event.action_requests, 1):
                tool = action.get("tool", "unknown")
                args = action.get("args", {})
                args_str = self._format_args(args)
                table.add_row(str(i), tool, args_str)

            content_parts.append(table)

            # Show allowed decisions if available
            if event.review_configs:
                decisions = []
                for config in event.review_configs:
                    allowed = config.get("allowed_decisions", [])
                    decisions.extend(allowed)
                if decisions:
                    decisions_text = Text(
                        f"\nAllowed decisions: {', '.join(set(decisions))}",
                        style="dim"
                    )
                    content_parts.append(decisions_text)

        console.print(Panel(
            *content_parts if content_parts else [Text("Interrupt received")],
            title="[bold white on red] INTERRUPT - Action Required [/bold white on red]",
            border_style="red",
            box=box.HEAVY,
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
