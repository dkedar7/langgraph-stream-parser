"""Tests for resume utilities."""
import pytest

from langgraph_stream_parser.resume import (
    create_resume_input,
    prepare_agent_input,
)


class TestCreateResumeInput:
    def test_with_decisions(self):
        # Uses the real Command class from langgraph
        result = create_resume_input(decisions=[{"type": "approve"}])

        # Command objects have a resume attribute
        assert hasattr(result, 'resume')
        assert result.resume == {"decisions": [{"type": "approve"}]}

    def test_with_simple_value(self):
        result = create_resume_input(value=True)

        assert hasattr(result, 'resume')
        assert result.resume is True

    def test_no_input_raises(self):
        with pytest.raises(ValueError) as exc_info:
            create_resume_input()

        assert "Must provide either" in str(exc_info.value)

    def test_both_inputs_raises(self):
        with pytest.raises(ValueError) as exc_info:
            create_resume_input(decisions=[{"type": "approve"}], value=True)

        assert "Cannot provide both" in str(exc_info.value)


class TestPrepareAgentInput:
    def test_with_message(self):
        result = prepare_agent_input(message="Hello!")

        assert result == {
            "messages": [{"role": "user", "content": "Hello!"}]
        }

    def test_with_raw_input(self):
        custom = {"custom": "format", "data": [1, 2, 3]}
        result = prepare_agent_input(raw_input=custom)

        assert result == custom

    def test_with_decisions(self):
        result = prepare_agent_input(decisions=[{"type": "reject"}])

        # Should return a Command object
        assert hasattr(result, 'resume')
        assert result.resume == {"decisions": [{"type": "reject"}]}

    def test_no_input_raises(self):
        with pytest.raises(ValueError) as exc_info:
            prepare_agent_input()

        assert "Must provide one of" in str(exc_info.value)

    def test_multiple_inputs_raises(self):
        with pytest.raises(ValueError) as exc_info:
            prepare_agent_input(message="Hi", raw_input={"x": 1})

        assert "Can only provide one of" in str(exc_info.value)
