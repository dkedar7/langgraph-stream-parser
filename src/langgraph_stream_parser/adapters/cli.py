"""
CLI adapter for styled terminal display of LangGraph stream events.

Provides rich terminal formatting with ANSI colors, spinners, and
interactive interrupt handling. Inspired by Claude Code / nanocode styling.
"""
import sys
import threading
import time
from typing import Any

from ..events import InterruptEvent, ToolExtractedEvent
from .base import BaseAdapter, ToolState, ToolStatus


# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Standard colors
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

# Bright variants
BRIGHT_CYAN = "\033[96m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"

# Spinner frames (braille animation)
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """Terminal spinner with elapsed time display."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.running = False
        self.thread: threading.Thread | None = None
        self.frame_idx = 0
        self.start_time: float = 0

    def _spin(self) -> None:
        """Run the spinner animation."""
        while self.running:
            frame = SPINNER_FRAMES[self.frame_idx % len(SPINNER_FRAMES)]
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed)}s"
            print(
                f"\r{CYAN}{frame}{RESET} {DIM}{self.message}... {elapsed_str}{RESET}",
                end="",
                flush=True,
            )
            self.frame_idx += 1
            time.sleep(0.08)

    def start(self) -> None:
        """Start the spinner."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        # Clear the spinner line
        print("\r\033[2K", end="", flush=True)


class CLIAdapter(BaseAdapter):
    """Styled terminal display for LangGraph stream events.

    Provides rich CLI output with:
    - ANSI color formatting
    - Spinner animation during tool execution
    - Interactive interrupt handling with arrow-key navigation
    - Styled tool calls, messages, and extractions

    Example:
        from langgraph_stream_parser.adapters.cli import CLIAdapter

        adapter = CLIAdapter()
        adapter.run(
            graph=agent,
            input_data={"messages": [("user", "Hello")]},
            config={"configurable": {"thread_id": "my-thread"}}
        )
    """

    def __init__(
        self,
        *,
        show_timestamps: bool = False,
        show_tool_args: bool = True,
        max_content_preview: int = 200,
        reflection_types: set[str] | list[str] | None = None,
        todo_types: set[str] | list[str] | None = None,
        use_spinner: bool = True,
        use_colors: bool = True,
    ):
        """Initialize the CLI adapter.

        Args:
            show_timestamps: Show timestamps on events.
            show_tool_args: Show tool arguments in tool status lines.
            max_content_preview: Max characters for extracted content preview.
            reflection_types: Set of extracted_type values to render as reflections.
            todo_types: Set of extracted_type values to render as todo lists.
            use_spinner: Show spinner animation during tool execution.
            use_colors: Use ANSI color codes (disable for non-color terminals).
        """
        super().__init__(
            show_timestamps=show_timestamps,
            show_tool_args=show_tool_args,
            max_content_preview=max_content_preview,
            reflection_types=reflection_types,
            todo_types=todo_types,
        )
        self._use_spinner = use_spinner
        self._use_colors = use_colors
        self._last_rendered_count = 0
        self._spinner: Spinner | None = None
        self._active_tools: set[str] = set()

    def _c(self, code: str) -> str:
        """Return color code if colors enabled, else empty string."""
        return code if self._use_colors else ""

    def reset(self) -> None:
        """Reset state for a new stream."""
        super().reset()
        self._last_rendered_count = 0
        self._stop_spinner()
        self._active_tools.clear()

    def _start_spinner(self, message: str = "Working") -> None:
        """Start spinner if enabled."""
        if self._use_spinner and self._spinner is None:
            self._spinner = Spinner(message)
            self._spinner.start()

    def _stop_spinner(self) -> None:
        """Stop spinner if running."""
        if self._spinner:
            self._spinner.stop()
            self._spinner = None

    def render(self) -> None:
        """Render new items since last render."""
        items_to_render = self._display_items[self._last_rendered_count:]

        for item_type, item_data in items_to_render:
            # Stop spinner before rendering output
            self._stop_spinner()

            if item_type == "message":
                role, content = item_data
                self._print_message(role, content)
            elif item_type == "tool":
                self._print_tool_start(item_data)
            elif item_type == "extraction":
                self._print_extraction(item_data)

        self._last_rendered_count = len(self._display_items)

        # Check for tool completions and update display
        self._update_tool_status()

        # Render interrupt
        if self._interrupt:
            self._stop_spinner()
            self._print_interrupt(self._interrupt)

        # Render error
        if self._error:
            self._stop_spinner()
            c = self._c
            print(f"\n{c(RED)}✗ Error: {self._error.error}{c(RESET)}")

        # Render completion
        if self._complete and not self._error:
            self._stop_spinner()

    def _update_tool_status(self) -> None:
        """Update status of running tools."""
        for tool_id, idx in self._tool_indices.items():
            _, tool = self._display_items[idx]
            if tool.status == ToolStatus.RUNNING:
                if tool_id not in self._active_tools:
                    self._active_tools.add(tool_id)
                    self._start_spinner(f"Running {tool.name}")
            elif tool_id in self._active_tools:
                # Tool just completed
                self._active_tools.discard(tool_id)
                self._stop_spinner()
                self._print_tool_result(tool)

    def prompt_interrupt(self, event: InterruptEvent) -> list[dict[str, Any]] | None:
        """Prompt user for interrupt decision with interactive selection.

        Args:
            event: The InterruptEvent requiring a decision.

        Returns:
            List of decision dicts, or None if cancelled.
        """
        allowed = self.get_allowed_decisions(event)
        num_actions = len(event.action_requests)

        # Build options based on allowed decisions
        options = []
        option_map = {}

        if "approve" in allowed:
            options.append("Approve all actions")
            option_map[len(options) - 1] = "approve"
        if "reject" in allowed:
            options.append("Reject all actions")
            option_map[len(options) - 1] = "reject"
        if "edit" in allowed:
            options.append("Edit args (JSON)")
            option_map[len(options) - 1] = "edit"

        options.append("Cancel")
        option_map[len(options) - 1] = "cancel"

        # Try interactive selection, fall back to text input
        try:
            choice = self._select_option(options, "How would you like to proceed?")
        except (EOFError, KeyboardInterrupt):
            return None

        decision_type = option_map.get(choice, "cancel")

        if decision_type == "cancel":
            return None

        # Handle edit with args prompt
        args_modifier = None
        if decision_type == "edit":
            try:
                import json
                c = self._c
                new_args_str = input(f"{c(BLUE)}New args (JSON): {c(RESET)}").strip()
                if new_args_str:
                    new_args = json.loads(new_args_str)
                    args_modifier = lambda _: new_args  # noqa: E731
                else:
                    decision_type = "reject"
            except (EOFError, KeyboardInterrupt, json.JSONDecodeError):
                decision_type = "reject"

        return self.build_decisions(event, decision_type, args_modifier)

    def _select_option(self, options: list[str], prompt: str) -> int:
        """Interactive option selector using arrow keys.

        Falls back to numbered input if interactive mode fails.
        """
        c = self._c

        # Try interactive mode first
        if sys.stdin.isatty():
            try:
                return self._interactive_select(options, prompt)
            except Exception:
                pass  # Fall back to simple input

        # Simple numbered input fallback
        print(f"\n{c(BOLD)}{prompt}{c(RESET)}")
        for i, opt in enumerate(options):
            print(f"  {i + 1}. {opt}")

        while True:
            try:
                choice = input(f"{c(BLUE)}Enter number: {c(RESET)}").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
            except (ValueError, EOFError, KeyboardInterrupt):
                return len(options) - 1  # Default to last (cancel)

    def _interactive_select(self, options: list[str], prompt: str) -> int:
        """Arrow-key based option selection."""
        import sys

        # Platform-specific imports
        if sys.platform == "win32":
            import msvcrt

            def get_key():
                ch = msvcrt.getch()
                if ch in (b'\x00', b'\xe0'):
                    ch2 = msvcrt.getch()
                    if ch2 == b'H':
                        return 'up'
                    elif ch2 == b'P':
                        return 'down'
                elif ch == b'\r':
                    return 'enter'
                elif ch == b'\x03':
                    raise KeyboardInterrupt
                return None
        else:
            import termios
            import tty

            def get_key():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A':
                                return 'up'
                            elif ch3 == 'B':
                                return 'down'
                    elif ch in ('\r', '\n'):
                        return 'enter'
                    elif ch == '\x03':
                        raise KeyboardInterrupt
                    return None
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        c = self._c
        selected = 0
        num_options = len(options)

        # Hide cursor
        print("\033[?25l", end="")

        try:
            print(f"\n{c(BOLD)}{prompt}{c(RESET)}")

            # Print initial options
            for i, opt in enumerate(options):
                if i == selected:
                    print(f"  {c(CYAN)}❯ {opt}{c(RESET)}")
                else:
                    print(f"    {c(DIM)}{opt}{c(RESET)}")

            while True:
                key = get_key()

                if key == 'up' and selected > 0:
                    selected -= 1
                elif key == 'down' and selected < num_options - 1:
                    selected += 1
                elif key == 'enter':
                    break

                # Move cursor up and redraw
                print(f"\033[{num_options}A", end="")
                for i, opt in enumerate(options):
                    print("\033[2K", end="")
                    if i == selected:
                        print(f"  {c(CYAN)}❯ {opt}{c(RESET)}")
                    else:
                        print(f"    {c(DIM)}{opt}{c(RESET)}")

            return selected
        finally:
            # Show cursor
            print("\033[?25h", end="")

    def _print_message(self, role: str, content: str) -> None:
        """Print a message with styled formatting."""
        c = self._c

        if role == "human":
            # User messages in green
            print(f"\n{c(GREEN)}● You{c(RESET)}")
            print(f"  {content}")
        else:
            # Assistant messages with cyan bullet
            print(f"\n{c(CYAN)}⏺{c(RESET)} {content}")

    def _print_tool_start(self, tool: ToolState) -> None:
        """Print tool call start with styled formatting."""
        c = self._c

        print(f"\n{c(GREEN)}● {tool.name}{c(RESET)}")

        if self._show_tool_args and tool.args:
            arg_preview = self._get_arg_preview(tool.args)
            if arg_preview:
                print(f"  {c(DIM)}└─ {arg_preview}{c(RESET)}")

    def _print_tool_result(self, tool: ToolState) -> None:
        """Print tool result with status indicator."""
        c = self._c

        if tool.status == ToolStatus.SUCCESS:
            status_icon = f"{c(GREEN)}✓{c(RESET)}"
            time_str = ""
            if tool.duration_ms:
                time_str = f" {c(DIM)}({self.format_duration(tool.duration_ms)}){c(RESET)}"
            print(f"  {status_icon} {c(DIM)}{tool.name} completed{c(RESET)}{time_str}")
        elif tool.status == ToolStatus.ERROR:
            print(f"  {c(RED)}✗ {tool.name} failed{c(RESET)}")
            if tool.error_message:
                print(f"    {c(DIM)}{tool.error_message}{c(RESET)}")

    def _print_extraction(self, event: ToolExtractedEvent) -> None:
        """Print extracted content with styled formatting."""
        c = self._c

        data_str = str(event.data)
        if len(data_str) > self._max_content_preview:
            data_str = data_str[:self._max_content_preview] + "..."

        # Special handling for todo types
        if event.extracted_type in self._todo_types and isinstance(event.data, list):
            todos = self.format_todos(event.data)
            if todos:
                print(f"\n{c(MAGENTA)}● {event.extracted_type}{c(RESET)}")
                for status, content in todos:
                    if status == "completed":
                        icon = f"{c(GREEN)}✓{c(RESET)}"
                    elif status == "in_progress":
                        icon = f"{c(YELLOW)}▶{c(RESET)}"
                    else:
                        icon = f"{c(DIM)}○{c(RESET)}"
                    print(f"  {icon} {content}")
                return

        # Special handling for reflection types
        if event.extracted_type in self._reflection_types:
            print(f"\n{c(MAGENTA)}● {event.extracted_type}{c(RESET)}")
            print(f"  {c(ITALIC)}{data_str}{c(RESET)}")
        else:
            print(f"\n{c(MAGENTA)}● {event.extracted_type}:{c(RESET)} {data_str}")

    def _print_interrupt(self, event: InterruptEvent) -> None:
        """Print interrupt information with styled formatting."""
        c = self._c

        print(f"\n{c(YELLOW)}⚠ Action Required{c(RESET)}")

        for i, action in enumerate(event.action_requests):
            tool = action.get("tool", "unknown")
            args = action.get("args", {})
            print(f"  {c(DIM)}{i + 1}. {tool}{c(RESET)}")
            if args:
                arg_preview = self._get_arg_preview(args)
                if arg_preview:
                    print(f"     {c(DIM)}└─ {arg_preview}{c(RESET)}")

    def _get_arg_preview(self, args: dict[str, Any], max_len: int = 50) -> str:
        """Get a preview of the first argument value."""
        if not args:
            return ""
        first_val = str(list(args.values())[0])
        if len(first_val) > max_len:
            return first_val[:max_len] + "..."
        return first_val
