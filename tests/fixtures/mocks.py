"""Mock LangChain/LangGraph objects for testing without dependencies."""
from dataclasses import dataclass, field
from typing import Any


# Create mock classes with the actual LangChain class names
# This is done dynamically to ensure __class__.__name__ returns the right value

def _create_mock_class(name: str, fields: dict, defaults: dict = None):
    """Create a mock class with a specific __name__."""
    defaults = defaults or {}

    def __init__(self, **kwargs):
        for field_name, field_type in fields.items():
            default_val = defaults.get(field_name)
            if callable(default_val):
                default_val = default_val()
            setattr(self, field_name, kwargs.get(field_name, default_val))

    cls = type(name, (), {"__init__": __init__})
    return cls


# Create mock classes with proper names
AIMessage = _create_mock_class(
    "AIMessage",
    {"content": Any, "id": str, "tool_calls": list, "usage_metadata": dict},
    {"id": "msg_123", "tool_calls": list, "usage_metadata": None}
)

AIMessageChunk = _create_mock_class(
    "AIMessageChunk",
    {"content": Any, "id": str, "tool_calls": list, "tool_call_chunks": list},
    {"id": "chunk_123", "tool_calls": list, "tool_call_chunks": list}
)

ToolMessage = _create_mock_class(
    "ToolMessage",
    {"content": Any, "name": str, "tool_call_id": str, "status": str, "artifact": Any},
    {"status": None, "artifact": None}
)

HumanMessage = _create_mock_class(
    "HumanMessage",
    {"content": str, "id": str},
    {"id": "human_123"}
)


@dataclass
class MockInterrupt:
    """Mock LangGraph Interrupt for testing."""
    value: Any
    resumable: bool = True


# Sample fixtures

SIMPLE_AI_MESSAGE = {
    "agent": {
        "messages": [
            AIMessage(content="Hello, how can I help?")
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

AI_MESSAGE_WITH_CONTENT_AND_TOOLS = {
    "agent": {
        "messages": [
            AIMessage(
                content="Let me search for that.",
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
                content="The weather is sunny and 72F",
                name="search",
                tool_call_id="call_1"
            )
        ]
    }
}

TOOL_MESSAGE_ERROR = {
    "tools": {
        "messages": [
            ToolMessage(
                content="Error: API rate limit exceeded",
                name="search",
                tool_call_id="call_1",
                status="error"
            )
        ]
    }
}

TOOL_MESSAGE_ERROR_PREFIX = {
    "tools": {
        "messages": [
            ToolMessage(
                content="Failed: Connection timeout",
                name="api_call",
                tool_call_id="call_2"
            )
        ]
    }
}

THINK_TOOL_MESSAGE = {
    "tools": {
        "messages": [
            ToolMessage(
                content='{"reflection": "I should search for more recent data."}',
                name="think_tool",
                tool_call_id="call_think"
            )
        ]
    }
}

THINK_TOOL_STRING_CONTENT = {
    "tools": {
        "messages": [
            ToolMessage(
                content="This is my reflection about the problem.",
                name="think_tool",
                tool_call_id="call_think2"
            )
        ]
    }
}

WRITE_TODOS_MESSAGE = {
    "tools": {
        "messages": [
            ToolMessage(
                content='[{"task": "Research topic", "done": false}, {"task": "Write draft", "done": false}]',
                name="write_todos",
                tool_call_id="call_todos"
            )
        ]
    }
}

WRITE_TODOS_EMBEDDED = {
    "tools": {
        "messages": [
            ToolMessage(
                content='Updated todo list to [{"task": "First", "done": false}, {"task": "Second", "done": true}]',
                name="write_todos",
                tool_call_id="call_todos2"
            )
        ]
    }
}

DISPLAY_INLINE_ARTIFACT_MESSAGE = {
    "tools": {
        "messages": [
            ToolMessage(
                content="Displayed dataframe inline: Sales Data",
                name="display_inline",
                tool_call_id="call_display",
                artifact={
                    "type": "display_inline",
                    "display_type": "dataframe",
                    "title": "Sales Data",
                    "data": "<table><tr><td>A</td></tr></table>",
                    "status": "success",
                    "error": None,
                },
            )
        ]
    }
}

DISPLAY_INLINE_CONTENT_MESSAGE = {
    "tools": {
        "messages": [
            ToolMessage(
                content='{"type": "display_inline", "display_type": "image", "title": "Chart", "data": "base64data", "status": "success", "error": null}',
                name="display_inline",
                tool_call_id="call_display2",
            )
        ]
    }
}

INTERRUPT_SIMPLE = {
    "__interrupt__": (
        MockInterrupt(value="Please confirm you want to proceed"),
    )
}

INTERRUPT_WITH_ACTIONS = {
    "__interrupt__": (
        MockInterrupt(
            value={
                "action_requests": [
                    {"name": "bash", "args": {"command": "ls -la"}, "tool_call_id": "call_1"}
                ],
                "review_configs": [
                    {"allowed_decisions": ["approve", "reject", "edit"]}
                ]
            }
        ),
    )
}

INTERRUPT_MULTIPLE_ACTIONS = {
    "__interrupt__": (
        MockInterrupt(
            value={
                "action_requests": [
                    {"name": "bash", "args": {"command": "rm file.txt"}, "tool_call_id": "call_1"},
                    {"name": "write_file", "args": {"path": "/etc/hosts"}, "tool_call_id": "call_2"}
                ],
                "review_configs": [
                    {"allowed_decisions": ["approve", "reject"]},
                    {"allowed_decisions": ["approve", "reject", "edit"]}
                ]
            }
        ),
    )
}

STATE_UPDATE_WITH_EXTRA_KEYS = {
    "agent": {
        "messages": [
            AIMessage(content="Processing...")
        ],
        "current_step": 3,
        "total_steps": 5
    }
}

MULTI_MESSAGE_CONTENT = {
    "agent": {
        "messages": [
            AIMessage(
                content=[
                    {"type": "text", "text": "Here is the result:"},
                    {"type": "text", "text": " The answer is 42."}
                ]
            )
        ]
    }
}

AI_MESSAGE_WITH_USAGE = {
    "agent": {
        "messages": [
            AIMessage(
                content="Done.",
                usage_metadata={
                    "input_tokens": 150,
                    "output_tokens": 42,
                    "total_tokens": 192,
                }
            )
        ]
    }
}

# --- Dual / Messages mode fixtures ---

MESSAGES_METADATA = {"langgraph_node": "agent", "langgraph_step": 1}

# Messages-mode chunks: (AIMessageChunk, metadata) tuples
MESSAGES_CHUNK_TOKEN_1 = (AIMessageChunk(content="Hello"), MESSAGES_METADATA)
MESSAGES_CHUNK_TOKEN_2 = (AIMessageChunk(content=" world"), MESSAGES_METADATA)
MESSAGES_CHUNK_EMPTY = (AIMessageChunk(content=""), MESSAGES_METADATA)
MESSAGES_CHUNK_WITH_TOOL_CALL_CHUNKS = (
    AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "search", "args": "", "id": "call_1", "index": 0}
        ],
    ),
    MESSAGES_METADATA,
)

# Real-world scenario: tool call chunk with content containing stringified tool dict
MESSAGES_CHUNK_TOOL_WITH_CONTENT = (
    AIMessageChunk(
        content="{'id': 'toolu_013sLF47f2hfJaysysskcFjK', 'input': {}, 'name': 'ls', 'type': 'tool_use'}",
        tool_call_chunks=[
            {"name": "ls", "args": "", "id": "toolu_013sLF47f2hfJaysysskcFjK", "index": 0}
        ],
    ),
    MESSAGES_METADATA,
)

# Tool call chunk with tool_calls list (not just chunks)
MESSAGES_CHUNK_WITH_TOOL_CALLS = (
    AIMessageChunk(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "search", "args": {"query": "weather"}}
        ],
    ),
    MESSAGES_METADATA,
)

# Dual-mode stream chunks: (mode_name, data) tuples
DUAL_MESSAGES_TOKEN_1 = ("messages", MESSAGES_CHUNK_TOKEN_1)
DUAL_MESSAGES_TOKEN_2 = ("messages", MESSAGES_CHUNK_TOKEN_2)
DUAL_MESSAGES_EMPTY = ("messages", MESSAGES_CHUNK_EMPTY)
DUAL_MESSAGES_TOOL_CHUNK = ("messages", MESSAGES_CHUNK_WITH_TOOL_CALL_CHUNKS)
DUAL_MESSAGES_TOOL_WITH_CONTENT = ("messages", MESSAGES_CHUNK_TOOL_WITH_CONTENT)
DUAL_MESSAGES_TOOL_CALLS = ("messages", MESSAGES_CHUNK_WITH_TOOL_CALLS)

DUAL_UPDATES_SIMPLE = ("updates", SIMPLE_AI_MESSAGE)
DUAL_UPDATES_TOOL_CALL = ("updates", AI_MESSAGE_WITH_TOOL_CALLS)
DUAL_UPDATES_TOOL_RESULT = ("updates", TOOL_MESSAGE_SUCCESS)
DUAL_UPDATES_INTERRUPT = ("updates", INTERRUPT_WITH_ACTIONS)

# Backward compatibility aliases
MockAIMessage = AIMessage
MockToolMessage = ToolMessage
