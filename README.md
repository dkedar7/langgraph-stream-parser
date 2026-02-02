# langgraph-stream-parser

Universal parser for LangGraph streaming outputs. Normalizes complex, variable output shapes from `graph.stream()` and `graph.astream()` into consistent, typed event objects.

## Installation

```bash
pip install langgraph-stream-parser
```

## Quick Start

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import ContentEvent, ToolCallStartEvent, InterruptEvent

parser = StreamParser()

for event in parser.parse(graph.stream(input_data, stream_mode="updates")):
    match event:
        case ContentEvent(content=text):
            print(text, end="")
        case ToolCallStartEvent(name=name):
            print(f"\nCalling {name}...")
        case InterruptEvent(action_requests=actions):
            # Handle human-in-the-loop
            decision = get_user_decision(actions)
            # Resume with create_resume_input()
```

## Features

- **Typed Events**: All stream outputs normalized to dataclass events with full type hints
- **Tool Lifecycle Tracking**: Automatic tracking of tool calls from start to completion
- **Interrupt Handling**: Parse and resume from human-in-the-loop interrupts
- **Extensible Extractors**: Register custom extractors for domain-specific tools
- **Async Support**: Both sync and async parsing via `parse()` and `aparse()`
- **Zero Dependencies**: LangGraph/LangChain imported dynamically only when needed
- **Backward Compatible**: Legacy dict-based API available for gradual migration

## Event Types

| Event | Description |
|-------|-------------|
| `ContentEvent` | Text content from AI messages |
| `ToolCallStartEvent` | Tool call initiated by AI |
| `ToolCallEndEvent` | Tool call completed with result |
| `ToolExtractedEvent` | Special content extracted from tool (e.g., reflections, todos) |
| `InterruptEvent` | Human-in-the-loop interrupt requiring decision |
| `StateUpdateEvent` | Non-message state updates (opt-in) |
| `CompleteEvent` | Stream finished successfully |
| `ErrorEvent` | Error during streaming |

## Usage Examples

### Basic Parsing

```python
from langgraph_stream_parser import StreamParser

parser = StreamParser()

for event in parser.parse(graph.stream({"messages": [...]}, stream_mode="updates")):
    print(event)
```

### Pattern Matching (Python 3.10+)

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import *

parser = StreamParser()

for event in parser.parse(stream):
    match event:
        case ContentEvent(content=text, node=node):
            print(f"[{node}] {text}", end="")

        case ToolCallStartEvent(name=name, args=args):
            print(f"\n⏳ Calling {name}...")

        case ToolCallEndEvent(name=name, status="success"):
            print(f"✅ {name} completed")

        case ToolCallEndEvent(name=name, status="error", error_message=err):
            print(f"❌ {name} failed: {err}")

        case InterruptEvent() as interrupt:
            if interrupt.needs_approval:
                handle_approval(interrupt.action_requests)

        case CompleteEvent():
            print("\n✓ Done")

        case ErrorEvent(error=err):
            print(f"⚠️ Error: {err}")
```

### Handling Interrupts

```python
from langgraph_stream_parser import StreamParser, create_resume_input
from langgraph_stream_parser.events import InterruptEvent

parser = StreamParser()
config = {"configurable": {"thread_id": "my-thread"}}

for event in parser.parse(graph.stream(input_data, config=config)):
    if isinstance(event, InterruptEvent):
        # Show user the pending actions
        for action in event.action_requests:
            print(f"Tool: {action['tool']}")
            print(f"Args: {action['args']}")

        # Get user decision
        decision = input("Approve? (y/n): ")

        # Resume
        resume_input = create_resume_input(
            decisions=[{"type": "approve" if decision == "y" else "reject"}]
        )

        for resume_event in parser.parse(graph.stream(resume_input, config=config)):
            handle_event(resume_event)
        break
```

### Custom Tool Extractors

```python
from langgraph_stream_parser import StreamParser, ToolExtractor
from langgraph_stream_parser.events import ToolExtractedEvent

class CanvasExtractor:
    tool_name = "add_to_canvas"
    extracted_type = "canvas_item"

    def extract(self, content):
        if isinstance(content, dict):
            return content
        return {"type": "text", "data": str(content)}

parser = StreamParser()
parser.register_extractor(CanvasExtractor())

for event in parser.parse(stream):
    if isinstance(event, ToolExtractedEvent) and event.extracted_type == "canvas_item":
        add_to_canvas_ui(event.data)
```

### Async Support

```python
from langgraph_stream_parser import StreamParser

parser = StreamParser()

async def stream_agent():
    async for event in parser.aparse(graph.astream(input_data)):
        handle_event(event)
```

### Configuration Options

```python
parser = StreamParser(
    # Track tool call lifecycle (start -> end)
    track_tool_lifecycle=True,

    # Skip these tools entirely (no events emitted)
    skip_tools=["internal_tool"],

    # Include StateUpdateEvent for non-message state keys
    include_state_updates=False,
)
```

## Legacy Dict-Based API

For backward compatibility or simpler use cases:

```python
from langgraph_stream_parser import stream_graph_updates, resume_graph_from_interrupt

for update in stream_graph_updates(agent, input_data, config=config):
    if update.get("status") == "interrupt":
        interrupt = update["interrupt"]
        # Handle interrupt...
    elif "chunk" in update:
        print(update["chunk"], end="")
    elif "tool_calls" in update:
        print(f"Calling tools: {update['tool_calls']}")
    elif update.get("status") == "complete":
        break

# Resume from interrupt
for update in resume_graph_from_interrupt(agent, decisions=[{"type": "approve"}], config=config):
    handle_update(update)
```

## Display Adapters

Pre-built adapters for rendering stream events in different environments:

### CLIAdapter - Styled Terminal Output

```python
from langgraph_stream_parser.adapters import CLIAdapter

adapter = CLIAdapter()
adapter.run(
    graph=agent,
    input_data={"messages": [("user", "Hello")]},
    config={"configurable": {"thread_id": "my-thread"}}
)
```

Features:
- ANSI color formatting
- Spinner animation during tool execution
- Interactive arrow-key interrupt handling

### PrintAdapter - Plain Text Output

```python
from langgraph_stream_parser.adapters import PrintAdapter

adapter = PrintAdapter()
adapter.run(graph=agent, input_data=input_data, config=config)
```

Universal output that works in any Python environment without dependencies.

### JupyterDisplay - Rich Notebook Display

```python
from langgraph_stream_parser.adapters.jupyter import JupyterDisplay

display = JupyterDisplay()
display.run(graph=agent, input_data=input_data, config=config)
```

Requires: `pip install langgraph-stream-parser[jupyter]`

### Adapter Options

All adapters support:

```python
adapter = CLIAdapter(
    show_tool_args=True,           # Show tool arguments
    max_content_preview=200,       # Max chars for extracted content
    reflection_types={"thinking"}, # Custom reflection type names
    todo_types={"tasks"},          # Custom todo type names
)
```

### Custom Adapters

Extend `BaseAdapter` for custom rendering:

```python
from langgraph_stream_parser.adapters import BaseAdapter

class MyAdapter(BaseAdapter):
    def render(self):
        # Implement your rendering logic
        pass

    def prompt_interrupt(self, event):
        # Handle interrupt prompts
        return [{"type": "approve"}]
```

## Built-in Extractors

The package includes extractors for common LangGraph tools:

- **ThinkToolExtractor**: Extracts reflections from `think_tool`
- **TodoExtractor**: Extracts todo lists from `write_todos`

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=langgraph_stream_parser
```

## License

MIT
