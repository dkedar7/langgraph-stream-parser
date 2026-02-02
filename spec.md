# langgraph-stream-parser: Technical Specification

## Executive Summary

Build a Python package called `langgraph-stream-parser` that provides a universal, typed interface for parsing streaming outputs from LangGraph agents. The package normalizes the complex, variable output shapes from `graph.stream()` and `graph.astream()` into consistent, typed event objects that application developers can easily consume.

**Target Users**: Python developers building applications on top of LangGraph agents (Streamlit apps, FastAPI backends, CLI tools, Dash dashboards).

**Core Value Proposition**: Eliminate repetitive, error-prone parsing logic that developers currently copy-paste across projects.

---

## Problem Statement

### The Pain Point

LangGraph's streaming API (`graph.stream()`) produces outputs in multiple formats depending on:
- Stream mode (`values`, `updates`, `messages`, `custom`, `debug`)
- Whether subgraphs are enabled
- Whether interrupts occur (human-in-the-loop)
- The message types flowing through the graph (AIMessage, ToolMessage, HumanMessage)
- Custom tools that embed structured data in their responses

Developers building UIs or APIs on top of LangGraph agents must write parsing logic to:
1. Detect what type of chunk they received
2. Extract relevant content (text, tool calls, tool results)
3. Handle special cases (interrupts, errors, custom tools)
4. Track tool call lifecycles (pending → running → success/error)
5. Resume from interrupts with properly formatted `Command` objects

This parsing logic is **repeated across every project** with slight variations, leading to bugs and wasted effort.

### Evidence of Need

- LangGraph GitHub Issue #95 explicitly requests output parsing utilities
- No existing package addresses this gap
- The user has implemented similar parsing logic in three separate projects (code provided below as reference)

---

## LangGraph Streaming Behavior

### Stream Modes

LangGraph supports these stream modes via `graph.stream(input, stream_mode=...)`:

| Mode | Output Shape | Use Case |
|------|--------------|----------|
| `"values"` | Full state dict after each step | Simple consumption, see complete state |
| `"updates"` | `{node_name: {key: value}}` deltas | Granular updates, recommended for UIs |
| `"messages"` | `(AIMessageChunk, metadata)` tuples | Token-level streaming from LLMs |
| `"custom"` | User-defined via `get_stream_writer()` | Application-specific data |
| `"debug"` | Maximum execution info | Debugging only |

**Multi-mode streaming**: When `stream_mode=["updates", "messages"]`, outputs become `(mode_name, chunk)` tuples.

**Subgraph streaming**: When `subgraphs=True`, outputs become `(namespace_tuple, data)` where namespace identifies the subgraph.

### Message Types in Updates Mode

When using `stream_mode="updates"`, the output shape is:
```python
{
    "node_name": {
        "messages": [MessageObject, ...]  # Usually contains one message per update
    }
}
```

Message objects are LangChain message types:
- `AIMessage` - Model responses, may contain `tool_calls` attribute
- `AIMessageChunk` - Streaming token chunks
- `HumanMessage` - User inputs
- `ToolMessage` - Tool execution results, has `name`, `content`, `tool_call_id`
- `SystemMessage` - System prompts

### Interrupt Format

When a graph hits an interrupt (human-in-the-loop), the update contains:
```python
{"__interrupt__": (Interrupt(value=..., resumable=True), ...)}
```

The `Interrupt.value` structure varies by implementation. Common patterns:
```python
# Pattern 1: Action requests with review configs (deepagents style)
{
    "action_requests": [
        {"name": "bash", "args": {"command": "rm -rf /"}, "tool_call_id": "call_123"}
    ],
    "review_configs": [
        {"allowed_decisions": ["approve", "reject", "edit"]}
    ]
}

# Pattern 2: Simple value
"Please confirm you want to proceed"

# Pattern 3: Tool call dict
{"tool": "dangerous_action", "args": {...}}
```

### Resuming from Interrupt

To resume after an interrupt:
```python
from langgraph.types import Command

# Resume with decisions
graph.stream(Command(resume={"decisions": [{"type": "approve"}]}), config=config)

# Or with simple value
graph.stream(Command(resume=True), config=config)
```

---

## Reference Implementation

The following code is from the user's existing projects. Use this as the authoritative reference for patterns that work in production.

### langgraph_utils.py (Reusable Utilities Module)

This is the user's existing utility module used across multiple projects:

```python
"""
Reusable utilities for LangGraph agents.

This module provides generic functions for streaming from LangGraph agents,
handling interrupts, and processing various message types. It can be used
across different applications that use LangGraph.
"""
from typing import Any, Dict, Iterator, Optional, List, AsyncIterator
import json
import re
import ast


def parse_interrupt_value(interrupt_value: Any) -> tuple[List[Any], List[Any]]:
    """
    Parse interrupt value into action_requests and review_configs.

    Handles different interrupt value formats from LangGraph:
    - Tuple formats (single element, two elements)
    - Object formats with attributes

    Args:
        interrupt_value: The interrupt value from LangGraph

    Returns:
        Tuple of (action_requests, review_configs)
    """
    action_requests = []
    review_configs = []

    if isinstance(interrupt_value, tuple):
        if len(interrupt_value) == 1:
            # Single-element tuple containing Interrupt object
            interrupt_obj = interrupt_value[0]
            if hasattr(interrupt_obj, 'value') and isinstance(interrupt_obj.value, dict):
                action_requests = interrupt_obj.value.get('action_requests', [])
                review_configs = interrupt_obj.value.get('review_configs', [])
            else:
                action_requests = getattr(interrupt_obj, 'action_requests', [])
                review_configs = getattr(interrupt_obj, 'review_configs', [])
        elif len(interrupt_value) == 2:
            # Two-element tuple: (action_requests, review_configs)
            action_requests, review_configs = interrupt_value
    else:
        # Handle object format
        action_requests = getattr(interrupt_value, 'action_requests', [])
        review_configs = getattr(interrupt_value, 'review_configs', [])

    return action_requests, review_configs


def serialize_action_request(action: Any, index: int) -> Dict[str, Any]:
    """
    Serialize an action request to a dictionary.

    Handles both dict and object formats, and both 'name' and 'tool' field names.

    Args:
        action: The action request object or dict
        index: The index of this action (used for fallback tool_call_id)

    Returns:
        Dictionary with tool, tool_call_id, args, and description
    """
    if isinstance(action, dict):
        tool_name = action.get('tool') or action.get('name')
        tool_call_id = action.get('tool_call_id', f"call_{index}")
        args = action.get('args', {})
        description = action.get('description')
    else:
        tool_name = getattr(action, 'tool', None) or getattr(action, 'name', None)
        tool_call_id = getattr(action, 'tool_call_id', f"call_{index}")
        args = getattr(action, 'args', {})
        description = getattr(action, 'description', None)

    return {
        "tool": tool_name,
        "tool_call_id": tool_call_id,
        "args": args,
        "description": description
    }


def serialize_review_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a review config to a dictionary.

    Args:
        config: The review config object or dict

    Returns:
        Dictionary with allowed_decisions
    """
    if isinstance(config, dict):
        allowed_decisions = config.get('allowed_decisions', [])
    else:
        allowed_decisions = getattr(config, 'allowed_decisions', [])

    return {
        "allowed_decisions": allowed_decisions
    }


def process_interrupt(interrupt_value: Any) -> Dict[str, Any]:
    """
    Process a LangGraph interrupt value and convert to serializable format.

    Args:
        interrupt_value: The interrupt value from the update

    Returns:
        Dictionary containing action_requests and review_configs
    """
    action_requests, review_configs = parse_interrupt_value(interrupt_value)

    interrupt_data = {
        "action_requests": [],
        "review_configs": []
    }

    # Extract action requests
    for i, action in enumerate(action_requests):
        interrupt_data["action_requests"].append(
            serialize_action_request(action, i)
        )

    # Extract review configs
    for config in review_configs:
        interrupt_data["review_configs"].append(
            serialize_review_config(config)
        )

    return interrupt_data


def extract_todos_from_content(tool_content: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract todos list from write_todos tool content.

    Handles multiple formats:
    - String with embedded JSON/list
    - Dict with 'todos' key
    - Direct list

    Args:
        tool_content: The content from the write_todos tool message

    Returns:
        List of todo items or None if parsing fails
    """
    todos = None

    if isinstance(tool_content, str):
        # Look for array pattern first (handles "Updated todo list to [...]" format)
        match = re.search(r'\[.*\]', tool_content, re.DOTALL)
        if match:
            array_str = match.group(0)

            # Try parsing as Python literal first (handles single quotes)
            try:
                todos = ast.literal_eval(array_str)
            except:
                # Fall back to JSON parsing (requires double quotes)
                try:
                    todos = json.loads(array_str)
                except:
                    pass
        else:
            # No array found, try parsing entire string as JSON
            try:
                parsed = json.loads(tool_content)
                if isinstance(parsed, dict):
                    todos = parsed.get('todos')
                    # If todos is a string, parse it again
                    if isinstance(todos, str):
                        todos = json.loads(todos)
                elif isinstance(parsed, list):
                    # Content is directly a list
                    todos = parsed
            except:
                pass
    elif isinstance(tool_content, dict):
        todos = tool_content.get('todos')
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except:
                pass
    elif isinstance(tool_content, list):
        # Content is directly a list
        todos = tool_content

    return todos if isinstance(todos, list) else None


def extract_reflection_from_content(tool_content: Any) -> Optional[str]:
    """
    Extract reflection from think_tool content.

    Args:
        tool_content: The content from the think_tool message

    Returns:
        Reflection string or None
    """
    reflection = None

    if isinstance(tool_content, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(tool_content)
            reflection = parsed.get('reflection')
        except:
            reflection = tool_content
    elif isinstance(tool_content, dict):
        reflection = tool_content.get('reflection')

    return reflection


def serialize_tool_calls(tool_calls: List[Any], skip_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Serialize tool calls to dictionaries, optionally skipping certain tools.

    Args:
        tool_calls: List of tool call objects or dicts
        skip_tools: Optional list of tool names to skip (e.g., ['think_tool', 'write_todos'])

    Returns:
        List of serialized tool calls
    """
    skip_tools = skip_tools or []
    serialized = []

    for tc in tool_calls:
        tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None)

        # Skip specified tools
        if tool_name in skip_tools:
            continue

        serialized.append({
            "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
            "name": tool_name,
            "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {})
        })

    return serialized


def clean_content_from_tool_dicts(content: str) -> str:
    """
    Remove tool call dictionary representations from content strings.

    Tool calls often appear as strings like:
    "{'id': '...', 'input': {...}, 'name': '...', 'type': 'tool_use'}"

    Args:
        content: The content string to clean

    Returns:
        Cleaned content string
    """
    # Pattern to match tool call dictionary representations
    tool_dict_pattern = r"\{'id':\s*'[^']+',\s*'input':\s*\{.*?\},\s*'name':\s*'[^']+',\s*'type':\s*'tool_use'\}"
    content = re.sub(tool_dict_pattern, '', content, flags=re.DOTALL)
    return content.strip()


def process_message_content(message: Any) -> str:
    """
    Extract and convert message content to string.

    Handles different content formats:
    - String content
    - List of content blocks
    - Other types (converted to string)

    Args:
        message: The message object

    Returns:
        Content as string
    """
    if not hasattr(message, 'content'):
        return ""

    content = message.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of content blocks (e.g., [{"text": "...", "type": "text"}])
        return " ".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        return str(content)


def process_tool_message(message: Any) -> Optional[Dict[str, Any]]:
    """
    Process a ToolMessage and extract special content if applicable.

    Handles special tools:
    - think_tool: Extracts and returns reflection
    - write_todos: Extracts and returns todo list

    Args:
        message: The ToolMessage to process

    Returns:
        Dictionary with chunk/todo_list and status, or None if no special handling
    """
    if not hasattr(message, 'name'):
        return None

    tool_name = message.name
    tool_content = message.content

    if tool_name == 'think_tool':
        reflection = extract_reflection_from_content(tool_content)
        if reflection:
            return {
                "chunk": reflection,
                "status": "streaming"
            }
    elif tool_name == 'write_todos':
        todos = extract_todos_from_content(tool_content)
        if todos:
            return {
                "todo_list": todos,
                "status": "streaming"
            }

    return None


def process_ai_message(message: Any, node_name: str, skip_tools: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
    """
    Process an AI message and yield content and tool calls.

    Args:
        message: The AI message to process
        node_name: Name of the graph node
        skip_tools: Optional list of tool names to skip when serializing tool calls

    Yields:
        Dictionaries with tool_calls or chunk content
    """
    skip_tools = skip_tools or ['think_tool', 'write_todos']

    # Extract content
    content_str = process_message_content(message)

    # Check for tool calls
    tool_calls = None
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = serialize_tool_calls(message.tool_calls, skip_tools=skip_tools)

    # Clean content: strip whitespace and remove tool call dicts
    content_str = content_str.strip() if content_str else ""

    # Filter out tool call dictionaries from content
    if content_str and hasattr(message, 'tool_calls') and message.tool_calls:
        content_str = clean_content_from_tool_dicts(content_str)

    # Yield tool calls (if any)
    if tool_calls and len(tool_calls) > 0:
        yield {
            "tool_calls": tool_calls,
            "node": node_name,
            "status": "streaming"
        }

    # Yield content separately, only if non-empty
    if content_str:
        yield {
            "chunk": content_str,
            "node": node_name,
            "status": "streaming"
        }


def prepare_agent_input(
    message: Optional[str] = None,
    decisions: Optional[List[Dict[str, Any]]] = None,
    raw_input: Optional[Any] = None
) -> Any:
    """
    Prepare input for a LangGraph agent.

    This function handles different input types:
    - message: Regular user message (converted to message dict)
    - decisions: Resume decisions (converted to Command)
    - raw_input: Raw input passed directly (for custom formats)

    Args:
        message: Optional user message string
        decisions: Optional list of interrupt decisions
        raw_input: Optional raw input (bypasses message/decisions processing)

    Returns:
        Prepared input for the agent

    Raises:
        ValueError: If no input is provided or multiple inputs are provided
    """
    # Count how many inputs are provided
    inputs_provided = sum([
        message is not None,
        decisions is not None,
        raw_input is not None
    ])

    if inputs_provided == 0:
        raise ValueError("Must provide one of: message, decisions, or raw_input")
    if inputs_provided > 1:
        raise ValueError("Can only provide one of: message, decisions, or raw_input")

    # Handle raw input (pass through)
    if raw_input is not None:
        return raw_input

    # Handle regular message
    if message is not None:
        return {"messages": [{"role": "user", "content": message}]}

    # Handle resume from interrupt
    if decisions is not None:
        from langgraph.types import Command
        return Command(resume={"decisions": decisions})


def stream_graph_updates(
    agent,
    input_data: Any,
    config: Optional[Dict[str, Any]] = None,
    stream_mode: str = "updates"
) -> Iterator[Dict[str, Any]]:
    """
    Stream updates from a LangGraph agent.

    This is a generic function that handles:
    - Regular message streaming
    - Interrupt detection and processing
    - Special tool handling (think_tool, write_todos)
    - Tool call serialization

    Args:
        agent: The LangGraph agent/graph instance
        input_data: Input data for the agent (can be dict, Command, or any agent input)
        config: Optional configuration for the agent
        stream_mode: Stream mode for LangGraph (default: "updates")

    Yields:
        Dictionaries containing:
        - {"chunk": str, "status": "streaming"} for text content
        - {"tool_calls": list, "status": "streaming"} for tool calls
        - {"todo_list": list, "status": "streaming"} for todos
        - {"interrupt": dict, "status": "interrupt"} for interrupts
        - {"status": "complete"} when finished
        - {"error": str, "status": "error"} on errors
    """
    try:
        for update in agent.stream(input_data, config=config, stream_mode=stream_mode):
            # Check for interrupts
            if isinstance(update, dict) and "__interrupt__" in update:
                interrupt_data = process_interrupt(update["__interrupt__"])
                yield {
                    "interrupt": interrupt_data,
                    "status": "interrupt"
                }
                continue

            # Process regular updates
            if isinstance(update, dict):
                for node_name, state_data in update.items():
                    # Extract message content from the state update
                    if isinstance(state_data, dict) and "messages" in state_data:
                        messages = state_data["messages"]
                        if not messages:
                            continue

                        # Get the last message in this update
                        last_message = messages[-1] if isinstance(messages, list) else messages
                        message_type = last_message.__class__.__name__ if hasattr(last_message, '__class__') else None

                        # Handle ToolMessage (tool outputs)
                        if message_type == 'ToolMessage':
                            result = process_tool_message(last_message)
                            if result:
                                yield result

                        # Handle regular messages (including AIMessage with tool calls)
                        elif hasattr(last_message, 'content'):
                            for chunk in process_ai_message(last_message, node_name):
                                yield chunk

        yield {"status": "complete"}

    except Exception as e:
        yield {
            "error": f"Error streaming from agent: {str(e)}",
            "status": "error"
        }


def resume_graph_from_interrupt(
    agent,
    decisions: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    stream_mode: str = "updates"
) -> Iterator[Dict[str, Any]]:
    """
    Resume a LangGraph agent from an interrupt.

    This is a convenience wrapper around stream_graph_updates that prepares
    the resume input automatically.

    Args:
        agent: The LangGraph agent/graph instance
        decisions: List of decision objects with 'type' and optional fields
        config: Optional configuration for the agent
        stream_mode: Stream mode for LangGraph (default: "updates")

    Yields:
        Same format as stream_graph_updates
    """
    try:
        # Prepare resume input using the generic function
        resume_input = prepare_agent_input(decisions=decisions)

        # Use the same streaming logic as regular streaming
        for chunk in stream_graph_updates(agent, resume_input, config=config, stream_mode=stream_mode):
            yield chunk

    except Exception as e:
        yield {
            "error": f"Error resuming from interrupt: {str(e)}",
            "status": "error"
        }


# ============================================================================
# ASYNC VARIANTS
# ============================================================================


async def astream_graph_updates(
    agent,
    input_data: Any,
    config: Optional[Dict[str, Any]] = None,
    stream_mode: str = "updates"
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async version of stream_graph_updates.
    """
    try:
        async for update in agent.astream(input_data, config=config, stream_mode=stream_mode):
            # Check for interrupts
            if isinstance(update, dict) and "__interrupt__" in update:
                interrupt_data = process_interrupt(update["__interrupt__"])
                yield {
                    "interrupt": interrupt_data,
                    "status": "interrupt"
                }
                continue

            # Process regular updates
            if isinstance(update, dict):
                for node_name, state_data in update.items():
                    if isinstance(state_data, dict) and "messages" in state_data:
                        messages = state_data["messages"]
                        if not messages:
                            continue

                        last_message = messages[-1] if isinstance(messages, list) else messages
                        message_type = last_message.__class__.__name__ if hasattr(last_message, '__class__') else None

                        if message_type == 'ToolMessage':
                            result = process_tool_message(last_message)
                            if result:
                                yield result
                        elif hasattr(last_message, 'content'):
                            for chunk in process_ai_message(last_message, node_name):
                                yield chunk

        yield {"status": "complete"}

    except Exception as e:
        yield {
            "error": f"Error streaming from agent: {str(e)}",
            "status": "error"
        }
```

### Additional Patterns from Dash App (app.py)

The user's Dash application contains additional patterns not yet in the utils module. These should be incorporated into the package:

#### Tool Call Lifecycle Tracking

```python
# Track tool calls by their ID for updating status
tool_call_map = {}

def _serialize_tool_call(tc) -> Dict:
    """Serialize a tool call to a dictionary."""
    if isinstance(tc, dict):
        return {
            "id": tc.get("id"),
            "name": tc.get("name"),
            "args": tc.get("args", {}),
            "status": "running",
            "result": None
        }
    else:
        return {
            "id": getattr(tc, 'id', None),
            "name": getattr(tc, 'name', None),
            "args": getattr(tc, 'args', {}),
            "status": "running",
            "result": None
        }

def _update_tool_call_result(tool_call_id: str, result: Any, status: str = "success"):
    """Update a tool call with its result."""
    # ... updates tool_call in state by matching tool_call_id
```

#### ToolMessage Error Detection

```python
# Determine status - check message status attribute first
content = last_msg.content
status = "success"

# Check if ToolMessage has explicit status (e.g., from LangGraph)
msg_status = getattr(last_msg, 'status', None)
if msg_status == 'error':
    status = "error"
# Check for dict with explicit error field
elif isinstance(content, dict) and content.get("error"):
    status = "error"
# Check for common error patterns at the START of the message
elif isinstance(content, str):
    content_lower = content.lower().strip()
    if (content_lower.startswith("error:") or
        content_lower.startswith("failed:") or
        content_lower.startswith("exception:") or
        content_lower.startswith("traceback")):
        status = "error"
```

#### Canvas Tool Handling (Example of Domain-Specific Tools)

```python
elif last_msg.name == 'add_to_canvas':
    content = last_msg.content
    # Canvas tool returns the parsed canvas object
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            canvas_item = parsed
        except:
            canvas_item = {"type": "markdown", "data": content}
    elif isinstance(content, dict):
        canvas_item = content
    else:
        canvas_item = {"type": "markdown", "data": str(content)}
    
    # Append to canvas state...
```

---

## Package Architecture

### Directory Structure

```
langgraph-stream-parser/
├── src/
│   └── langgraph_stream_parser/
│       ├── __init__.py           # Public API exports
│       ├── parser.py             # Main StreamParser class
│       ├── events.py             # Event dataclasses (typed outputs)
│       ├── detectors.py          # Chunk type detection logic
│       ├── extractors/
│       │   ├── __init__.py
│       │   ├── base.py           # ToolExtractor protocol
│       │   ├── messages.py       # AIMessage, ToolMessage extraction
│       │   ├── interrupts.py     # Interrupt parsing
│       │   └── builtins.py       # think_tool, write_todos extractors
│       ├── handlers/
│       │   ├── __init__.py
│       │   ├── updates.py        # stream_mode="updates" handler
│       │   ├── values.py         # stream_mode="values" handler
│       │   ├── messages.py       # stream_mode="messages" handler
│       │   └── multi.py          # Multi-mode handler
│       └── resume.py             # Interrupt resume utilities
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_events.py
│   ├── test_extractors.py
│   ├── test_interrupts.py
│   └── fixtures/                 # Sample LangGraph outputs for testing
│       ├── updates_mode.py
│       ├── interrupt_formats.py
│       └── tool_messages.py
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

### Core Classes and Types

#### Event Types (events.py)

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Union
from datetime import datetime

@dataclass
class ContentEvent:
    """Text content from AI message."""
    content: str
    node: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass  
class ToolCallStartEvent:
    """Tool call initiated by AI."""
    id: str
    name: str
    args: dict[str, Any]
    node: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ToolCallEndEvent:
    """Tool call completed with result."""
    id: str
    name: str
    result: Any
    status: Literal["success", "error"]
    error_message: str | None = None
    duration_ms: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ToolExtractedEvent:
    """Special content extracted from a tool (e.g., reflection, todos)."""
    tool_name: str
    extracted_type: str  # e.g., "reflection", "todos", "canvas_item"
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InterruptEvent:
    """Human-in-the-loop interrupt requiring user decision."""
    action_requests: list[dict[str, Any]]
    review_configs: list[dict[str, Any]]
    raw_value: Any = None  # Original interrupt value for custom handling
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def needs_approval(self) -> bool:
        """Check if this interrupt has action requests needing approval."""
        return len(self.action_requests) > 0

@dataclass
class StateUpdateEvent:
    """Raw state update (for non-message state keys)."""
    node: str
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CompleteEvent:
    """Stream completed successfully."""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ErrorEvent:
    """Error occurred during streaming."""
    error: str
    exception: Exception | None = None
    timestamp: datetime = field(default_factory=datetime.now)

# Union type for all events
StreamEvent = Union[
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolExtractedEvent,
    InterruptEvent,
    StateUpdateEvent,
    CompleteEvent,
    ErrorEvent,
]
```

#### Tool Extractor Protocol (extractors/base.py)

```python
from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class ToolExtractor(Protocol):
    """Protocol for custom tool content extractors."""
    
    @property
    def tool_name(self) -> str:
        """The name of the tool this extractor handles."""
        ...
    
    @property
    def extracted_type(self) -> str:
        """The type name for extracted content (e.g., 'reflection', 'todos')."""
        ...
    
    def extract(self, content: Any) -> Any | None:
        """
        Extract meaningful data from tool content.
        
        Args:
            content: The raw content from ToolMessage.content
            
        Returns:
            Extracted data, or None if extraction fails/not applicable
        """
        ...
```

#### Main Parser Class (parser.py)

```python
from typing import Iterator, AsyncIterator, Any, Callable
from .events import StreamEvent, InterruptEvent
from .extractors.base import ToolExtractor

class StreamParser:
    """
    Universal parser for LangGraph streaming outputs.
    
    Normalizes various output formats into typed StreamEvent objects.
    
    Example:
        parser = StreamParser()
        
        # Register custom tool extractor
        parser.register_extractor(MyCanvasExtractor())
        
        # Parse stream
        for event in parser.parse(graph.stream(input, stream_mode="updates")):
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
        track_tool_lifecycle: bool = True,
        skip_tools: list[str] | None = None,
        include_state_updates: bool = False,
    ):
        """
        Initialize the parser.
        
        Args:
            track_tool_lifecycle: If True, emit ToolCallStartEvent and ToolCallEndEvent.
                                  If False, only emit ToolExtractedEvent for special tools.
            skip_tools: Tool names to skip entirely (no events emitted).
            include_state_updates: If True, emit StateUpdateEvent for non-message state keys.
        """
        self._track_tool_lifecycle = track_tool_lifecycle
        self._skip_tools = set(skip_tools or [])
        self._include_state_updates = include_state_updates
        self._extractors: dict[str, ToolExtractor] = {}
        self._pending_tool_calls: dict[str, ToolCallStartEvent] = {}
        
        # Register built-in extractors
        self._register_builtin_extractors()
    
    def register_extractor(self, extractor: ToolExtractor) -> None:
        """Register a custom tool extractor."""
        self._extractors[extractor.tool_name] = extractor
    
    def unregister_extractor(self, tool_name: str) -> None:
        """Remove a registered extractor."""
        self._extractors.pop(tool_name, None)
    
    def parse(
        self,
        stream: Iterator[Any],
        *,
        stream_mode: str = "updates",
    ) -> Iterator[StreamEvent]:
        """
        Parse a LangGraph stream into typed events.
        
        Args:
            stream: Iterator from graph.stream()
            stream_mode: The stream_mode used (affects parsing logic)
            
        Yields:
            StreamEvent objects
        """
        ...
    
    async def aparse(
        self,
        stream: AsyncIterator[Any],
        *,
        stream_mode: str = "updates",
    ) -> AsyncIterator[StreamEvent]:
        """Async version of parse()."""
        ...
    
    def parse_chunk(self, chunk: Any, stream_mode: str = "updates") -> list[StreamEvent]:
        """
        Parse a single chunk into events.
        
        Useful for manual iteration or when you need to process chunks individually.
        
        Args:
            chunk: A single chunk from graph.stream()
            stream_mode: The stream_mode used
            
        Returns:
            List of events (may be empty, one, or multiple)
        """
        ...
    
    def _register_builtin_extractors(self) -> None:
        """Register the built-in tool extractors."""
        from .extractors.builtins import ThinkToolExtractor, TodoExtractor
        self.register_extractor(ThinkToolExtractor())
        self.register_extractor(TodoExtractor())
```

---

## API Design

### Basic Usage

```python
from langgraph_stream_parser import StreamParser

# Create parser
parser = StreamParser()

# Parse a stream
for event in parser.parse(graph.stream({"messages": [...]}, stream_mode="updates")):
    print(event)
```

### Event Handling with Pattern Matching

```python
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.events import (
    ContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    InterruptEvent,
    CompleteEvent,
    ErrorEvent,
)

parser = StreamParser()

for event in parser.parse(graph.stream(input_data)):
    match event:
        case ContentEvent(content=text, node=node):
            # Stream text to UI
            print(text, end="", flush=True)
        
        case ToolCallStartEvent(name=name, args=args):
            # Show loading indicator
            print(f"\n⏳ Calling {name}...")
        
        case ToolCallEndEvent(name=name, status="success", result=result):
            print(f"✅ {name} completed")
        
        case ToolCallEndEvent(name=name, status="error", error_message=err):
            print(f"❌ {name} failed: {err}")
        
        case InterruptEvent() as interrupt:
            # Handle human-in-the-loop
            decision = get_user_decision(interrupt.action_requests)
            # Resume handled separately
        
        case CompleteEvent():
            print("\n✓ Done")
        
        case ErrorEvent(error=err):
            print(f"\n⚠️ Error: {err}")
```

### Custom Tool Extractors

```python
from langgraph_stream_parser import StreamParser, ToolExtractor
from typing import Any

class CanvasExtractor(ToolExtractor):
    tool_name = "add_to_canvas"
    extracted_type = "canvas_item"
    
    def extract(self, content: Any) -> dict | None:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except:
                return {"type": "markdown", "data": content}
        return None

# Register
parser = StreamParser()
parser.register_extractor(CanvasExtractor())

# Now ToolExtractedEvent will be emitted for add_to_canvas
for event in parser.parse(stream):
    if isinstance(event, ToolExtractedEvent) and event.extracted_type == "canvas_item":
        add_to_canvas_ui(event.data)
```

### Resuming from Interrupts

```python
from langgraph_stream_parser import StreamParser, create_resume_input
from langgraph_stream_parser.events import InterruptEvent

parser = StreamParser()
config = {"configurable": {"thread_id": "my-thread"}}

# Initial stream
for event in parser.parse(graph.stream(input_data, config=config)):
    if isinstance(event, InterruptEvent):
        # Get user decision
        decision = prompt_user(event.action_requests)
        
        # Create resume input
        resume_input = create_resume_input(
            decisions=[{"type": "approve"}]  # or "reject", "edit"
        )
        
        # Continue streaming from interrupt
        for resume_event in parser.parse(graph.stream(resume_input, config=config)):
            handle_event(resume_event)
        break
```

### Async Support

```python
from langgraph_stream_parser import StreamParser

parser = StreamParser()

async def stream_agent():
    async for event in parser.aparse(graph.astream(input_data)):
        handle_event(event)
```

### Convenience Functions (Backwards Compatible)

For users who want simple dict-based outputs (matching the reference implementation):

```python
from langgraph_stream_parser import stream_graph_updates, resume_graph_from_interrupt

# These return the same dict format as the reference implementation
for update in stream_graph_updates(agent, input_data, config=config):
    if update.get("status") == "interrupt":
        interrupt = update["interrupt"]
        # ...
    elif "chunk" in update:
        print(update["chunk"], end="")
    elif update.get("status") == "complete":
        break
```

---

## Implementation Guidelines

### Priority Order

1. **Start with `stream_mode="updates"`** - This is the most common mode and what the reference implementation uses
2. **Implement interrupt handling first** - This is the most complex and valuable feature
3. **Add tool lifecycle tracking** - Important for UIs showing tool status
4. **Add built-in extractors** - think_tool, write_todos
5. **Add other stream modes** - values, messages, multi-mode
6. **Add convenience functions** - For backwards compatibility

### Code Quality Requirements

1. **Type hints everywhere** - Use modern Python typing (3.10+ syntax acceptable)
2. **Dataclasses for events** - Immutable, typed, easy to pattern match
3. **Protocol for extensibility** - ToolExtractor as a Protocol, not ABC
4. **No LangChain/LangGraph imports at module level** - Import inside functions to avoid hard dependency
5. **Comprehensive docstrings** - Google style, with examples
6. **100% test coverage for core parsing logic**

### Error Handling

1. **Never raise exceptions during parsing** - Yield ErrorEvent instead
2. **Graceful degradation** - If a tool extractor fails, log warning and continue
3. **Preserve raw data** - InterruptEvent.raw_value allows custom handling

### Testing Strategy

Create fixture files with real LangGraph outputs captured from various scenarios:

```python
# tests/fixtures/updates_mode.py

SIMPLE_AI_MESSAGE = {
    "agent": {
        "messages": [
            AIMessage(content="Hello, how can I help?", id="msg_123")
        ]
    }
}

AI_MESSAGE_WITH_TOOL_CALLS = {
    "agent": {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"query": "weather"}}
                ]
            )
        ]
    }
}

TOOL_MESSAGE_SUCCESS = {
    "tools": {
        "messages": [
            ToolMessage(
                content="The weather is sunny",
                name="search",
                tool_call_id="call_1"
            )
        ]
    }
}

INTERRUPT_SIMPLE = {
    "__interrupt__": (
        Interrupt(value="Please confirm", resumable=True),
    )
}

INTERRUPT_WITH_ACTIONS = {
    "__interrupt__": (
        Interrupt(
            value={
                "action_requests": [
                    {"name": "bash", "args": {"command": "ls"}, "tool_call_id": "call_1"}
                ],
                "review_configs": [
                    {"allowed_decisions": ["approve", "reject"]}
                ]
            },
            resumable=True
        ),
    )
}
```

### Performance Considerations

1. **Lazy imports** - Don't import LangGraph/LangChain until needed
2. **Minimal copying** - Don't deep copy large state objects
3. **Generator-based** - Use iterators throughout, not lists

### Backwards Compatibility

The package should provide both:
1. **New typed API** - `StreamParser` class with `StreamEvent` types
2. **Legacy dict API** - `stream_graph_updates()` function returning dicts

This allows gradual migration for existing codebases.

---

## Package Metadata (pyproject.toml)

```toml
[project]
name = "langgraph-stream-parser"
version = "0.1.0"
description = "Universal parser for LangGraph streaming outputs"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your@email.com"}
]
keywords = [
    "langgraph",
    "langchain",
    "streaming",
    "parser",
    "ai",
    "agents",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = []  # No hard dependencies!

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "langgraph>=0.2.0",  # For testing only
    "langchain-core>=0.2.0",  # For testing only
]

[project.urls]
Homepage = "https://github.com/yourusername/langgraph-stream-parser"
Documentation = "https://github.com/yourusername/langgraph-stream-parser#readme"
Repository = "https://github.com/yourusername/langgraph-stream-parser"
Issues = "https://github.com/yourusername/langgraph-stream-parser/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/langgraph_stream_parser"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.coverage.run]
source = ["src/langgraph_stream_parser"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## Success Criteria

The package is complete when:

1. ✅ Can parse `stream_mode="updates"` output into typed events
2. ✅ Correctly detects and parses all interrupt formats from the reference implementation
3. ✅ Tracks tool call lifecycle (start → end with result/error)
4. ✅ Provides extensible tool extractor system
5. ✅ Includes built-in extractors for `think_tool` and `write_todos`
6. ✅ Supports both sync and async iteration
7. ✅ Provides backwards-compatible dict-based API
8. ✅ Has comprehensive test coverage using fixtures
9. ✅ Zero hard dependencies (LangGraph/LangChain imported dynamically)
10. ✅ Published to PyPI with proper metadata

---

## Out of Scope (Future Versions)

- `stream_mode="messages"` token-level streaming (v0.2.0)
- `stream_mode="values"` support (v0.2.0)
- Multi-mode parsing (v0.3.0)
- Subgraph namespace handling (v0.3.0)
- Framework integrations (Streamlit, FastAPI helpers) (v0.4.0)
- Checkpoint/state persistence utilities (v0.5.0)

---

## Questions for Implementer

If anything is unclear, prioritize:
1. Matching the behavior of the reference `langgraph_utils.py` implementation
2. Using the interrupt formats shown in the reference `app.py`
3. Keeping the API simple for common cases, extensible for advanced cases

The reference implementation is battle-tested in production. When in doubt, follow its patterns.