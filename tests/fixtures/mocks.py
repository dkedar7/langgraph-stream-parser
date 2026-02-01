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
    {"content": Any, "id": str, "tool_calls": list},
    {"id": "msg_123", "tool_calls": list}
)

AIMessageChunk = _create_mock_class(
    "AIMessageChunk",
    {"content": Any, "id": str, "tool_calls": list},
    {"id": "chunk_123", "tool_calls": list}
)

ToolMessage = _create_mock_class(
    "ToolMessage",
    {"content": Any, "name": str, "tool_call_id": str, "status": str},
    {"status": None}
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

# Backward compatibility aliases
MockAIMessage = AIMessage
MockToolMessage = ToolMessage
