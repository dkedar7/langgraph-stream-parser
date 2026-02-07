"""Tests for tool extractors."""
import pytest
import json

from langgraph_stream_parser.extractors.builtins import (
    ThinkToolExtractor,
    TodoExtractor,
    DisplayInlineExtractor,
)
from langgraph_stream_parser.extractors.messages import (
    extract_message_content,
    clean_tool_dict_from_content,
    extract_tool_calls,
    detect_tool_error,
    get_message_type_name,
)
from langgraph_stream_parser.extractors.interrupts import (
    parse_interrupt_value,
    serialize_action_request,
    serialize_review_config,
    process_interrupt,
)

from .fixtures.mocks import (
    AIMessage,
    ToolMessage,
    MockInterrupt,
)


class TestThinkToolExtractor:
    def setup_method(self):
        self.extractor = ThinkToolExtractor()

    def test_tool_name(self):
        assert self.extractor.tool_name == "think_tool"
        assert self.extractor.extracted_type == "reflection"

    def test_extract_from_json(self):
        content = '{"reflection": "I need to think more"}'
        result = self.extractor.extract(content)
        assert result == "I need to think more"

    def test_extract_from_dict(self):
        content = {"reflection": "My thoughts"}
        result = self.extractor.extract(content)
        assert result == "My thoughts"

    def test_extract_from_plain_string(self):
        content = "Just a plain reflection"
        result = self.extractor.extract(content)
        assert result == "Just a plain reflection"

    def test_extract_empty_string(self):
        content = "   "
        result = self.extractor.extract(content)
        assert result is None

    def test_extract_none_value(self):
        content = {"reflection": None}
        result = self.extractor.extract(content)
        assert result is None

    def test_extract_missing_key(self):
        content = {"other_key": "value"}
        result = self.extractor.extract(content)
        assert result is None


class TestTodoExtractor:
    def setup_method(self):
        self.extractor = TodoExtractor()

    def test_tool_name(self):
        assert self.extractor.tool_name == "write_todos"
        assert self.extractor.extracted_type == "todos"

    def test_extract_json_array(self):
        content = '[{"task": "Do A", "done": false}, {"task": "Do B", "done": true}]'
        result = self.extractor.extract(content)
        assert len(result) == 2
        assert result[0]["task"] == "Do A"

    def test_extract_embedded_array(self):
        # Note: Python literal syntax uses True/False, not true/false
        content = "Updated todo list to [{'task': 'First', 'done': False}]"
        result = self.extractor.extract(content)
        assert len(result) == 1
        assert result[0]["task"] == "First"

    def test_extract_from_dict(self):
        content = {"todos": [{"task": "A"}, {"task": "B"}]}
        result = self.extractor.extract(content)
        assert len(result) == 2

    def test_extract_from_list(self):
        content = [{"task": "Direct list"}]
        result = self.extractor.extract(content)
        assert len(result) == 1

    def test_extract_nested_json_string(self):
        content = {"todos": '[{"task": "Nested"}]'}
        result = self.extractor.extract(content)
        assert len(result) == 1
        assert result[0]["task"] == "Nested"

    def test_extract_invalid_returns_none(self):
        content = "No array here"
        result = self.extractor.extract(content)
        assert result is None


class TestDisplayInlineExtractor:
    def setup_method(self):
        self.extractor = DisplayInlineExtractor()

    def test_tool_name(self):
        assert self.extractor.tool_name == "display_inline"
        assert self.extractor.extracted_type == "display_inline"

    def test_extract_from_json_string(self):
        content = '{"type": "display_inline", "display_type": "image", "title": "Chart", "data": "base64data", "status": "success"}'
        result = self.extractor.extract(content)
        assert result is not None
        assert result["display_type"] == "image"
        assert result["title"] == "Chart"

    def test_extract_from_dict(self):
        """When artifact is a dict (content_and_artifact pattern)."""
        content = {
            "type": "display_inline",
            "display_type": "dataframe",
            "title": "Sales Data",
            "data": "<table>...</table>",
            "status": "success",
            "error": None,
        }
        result = self.extractor.extract(content)
        assert result is not None
        assert result["display_type"] == "dataframe"
        assert result["title"] == "Sales Data"

    def test_extract_from_dict_without_display_type_returns_none(self):
        content = {"some_key": "value"}
        result = self.extractor.extract(content)
        assert result is None

    def test_extract_from_plain_string_returns_none(self):
        content = "Displayed dataframe inline: Sales Data"
        result = self.extractor.extract(content)
        assert result is None

    def test_extract_from_invalid_json_returns_none(self):
        content = "not json at all"
        result = self.extractor.extract(content)
        assert result is None

    def test_extract_none_returns_none(self):
        result = self.extractor.extract(None)
        assert result is None


class TestExtractMessageContent:
    def test_string_content(self):
        msg = AIMessage(content="Hello world")
        result = extract_message_content(msg)
        assert result == "Hello world"

    def test_list_content(self):
        msg = AIMessage(content=[
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"}
        ])
        result = extract_message_content(msg)
        assert result == "Hello  world"

    def test_no_content_attribute(self):
        class NoContent:
            pass
        result = extract_message_content(NoContent())
        assert result == ""


class TestCleanToolDictFromContent:
    def test_removes_tool_dict(self):
        content = "Result: {'id': 'abc', 'input': {'x': 1}, 'name': 'tool', 'type': 'tool_use'}"
        result = clean_tool_dict_from_content(content)
        assert result == "Result:"

    def test_preserves_normal_content(self):
        content = "Just normal content"
        result = clean_tool_dict_from_content(content)
        assert result == "Just normal content"


class TestExtractToolCalls:
    def test_extract_from_message(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search", "args": {"q": "test"}}
            ]
        )
        result = extract_tool_calls(msg)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["name"] == "search"
        assert result[0]["args"] == {"q": "test"}

    def test_empty_tool_calls(self):
        msg = AIMessage(content="No tools")
        result = extract_tool_calls(msg)
        assert result == []

    def test_no_tool_calls_attr(self):
        class NoToolCalls:
            content = "hi"
        result = extract_tool_calls(NoToolCalls())
        assert result == []


class TestDetectToolError:
    def test_explicit_error_status(self):
        msg = ToolMessage(
            content="Something failed",
            name="tool",
            tool_call_id="call_1",
            status="error"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is True
        assert error_msg == "Something failed"

    def test_error_prefix(self):
        msg = ToolMessage(
            content="Error: Connection failed",
            name="tool",
            tool_call_id="call_1"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is True

    def test_failed_prefix(self):
        msg = ToolMessage(
            content="Failed: Timeout",
            name="tool",
            tool_call_id="call_1"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is True

    def test_traceback_prefix(self):
        msg = ToolMessage(
            content="Traceback (most recent call last)...",
            name="tool",
            tool_call_id="call_1"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is True

    def test_dict_with_error_key(self):
        msg = ToolMessage(
            content={"error": "Something broke"},
            name="tool",
            tool_call_id="call_1"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is True

    def test_success_message(self):
        msg = ToolMessage(
            content="Operation completed successfully",
            name="tool",
            tool_call_id="call_1"
        )
        is_error, error_msg = detect_tool_error(msg)
        assert is_error is False
        assert error_msg is None


class TestGetMessageTypeName:
    def test_returns_class_name(self):
        msg = AIMessage(content="hi")
        result = get_message_type_name(msg)
        assert result == "AIMessage"

    def test_no_class(self):
        # None has a class (NoneType), so it returns "NoneType"
        result = get_message_type_name(None)
        assert result == "NoneType"


class TestParseInterruptValue:
    def test_single_element_tuple_with_dict_value(self):
        interrupt = MockInterrupt(value={
            "action_requests": [{"name": "bash"}],
            "review_configs": [{"allowed_decisions": ["approve"]}]
        })
        actions, configs = parse_interrupt_value((interrupt,))
        assert len(actions) == 1
        assert len(configs) == 1

    def test_dict_format(self):
        value = {
            "action_requests": [{"name": "tool"}],
            "review_configs": []
        }
        actions, configs = parse_interrupt_value(value)
        assert len(actions) == 1
        assert len(configs) == 0

    def test_empty_value(self):
        actions, configs = parse_interrupt_value({})
        assert actions == []
        assert configs == []


class TestSerializeActionRequest:
    def test_dict_with_name(self):
        action = {"name": "bash", "args": {"cmd": "ls"}, "tool_call_id": "call_1"}
        result = serialize_action_request(action, 0)
        assert result["tool"] == "bash"
        assert result["tool_call_id"] == "call_1"
        assert result["args"] == {"cmd": "ls"}

    def test_dict_with_tool_key(self):
        action = {"tool": "search", "args": {}}
        result = serialize_action_request(action, 5)
        assert result["tool"] == "search"
        assert result["tool_call_id"] == "call_5"  # fallback

    def test_object_format(self):
        class ActionObj:
            name = "mytool"
            args = {"a": 1}
            tool_call_id = "obj_call"

        result = serialize_action_request(ActionObj(), 0)
        assert result["tool"] == "mytool"
        assert result["tool_call_id"] == "obj_call"


class TestSerializeReviewConfig:
    def test_dict_format(self):
        config = {"allowed_decisions": ["approve", "reject"]}
        result = serialize_review_config(config)
        assert result["allowed_decisions"] == ["approve", "reject"]

    def test_empty_decisions(self):
        config = {}
        result = serialize_review_config(config)
        assert result["allowed_decisions"] == []


class TestProcessInterrupt:
    def test_full_interrupt(self):
        interrupt = MockInterrupt(value={
            "action_requests": [
                {"name": "bash", "args": {"cmd": "ls"}, "tool_call_id": "c1"}
            ],
            "review_configs": [
                {"allowed_decisions": ["approve", "reject"]}
            ]
        })
        result = process_interrupt((interrupt,))

        assert len(result["action_requests"]) == 1
        assert result["action_requests"][0]["tool"] == "bash"
        assert len(result["review_configs"]) == 1
        assert "approve" in result["review_configs"][0]["allowed_decisions"]
