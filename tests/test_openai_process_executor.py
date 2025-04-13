"""Tests for the OpenAI process executor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor


class TestOpenAIProcessExecutor:
    """Tests for the OpenAI process executor."""

    def test_openai_with_tools_raises_error(self):
        """Test that using OpenAI with tools raises an error."""
        # Create a program with tools
        program = LLMProgram(
            model_name="gpt-4",
            provider="openai",
            system_prompt="Test system prompt",
            tools={"enabled": ["spawn"]},  # Enable a tool
        )

        # Import the test helper
        from tests.conftest import create_test_llmprocess_directly

        # Creating a process should raise an error
        with pytest.raises(ValueError) as excinfo:
            process = create_test_llmprocess_directly(program=program)

        # Check error message
        assert "Tool usage is not yet supported for OpenAI" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test the run method of OpenAIProcessExecutor."""
        # Create a mock process
        process = MagicMock()
        process.model_name = "gpt-4"
        process.provider = "openai"
        process.enriched_system_prompt = "Test system prompt"
        process.state = []
        process.tools = []
        process.api_params = {"temperature": 0.7}

        # Create mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_response.id = "test-id"

        # Mock the API call
        process.client = MagicMock()
        process.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create the executor
        executor = OpenAIProcessExecutor()

        # Test the run method
        result = await executor.run(process, "Test input")

        # Verify the API call was made
        process.client.chat.completions.create.assert_called_once()

        # Verify the state was updated
        assert len(process.state) == 2
        assert process.state[0] == {"role": "user", "content": "Test input"}
        assert process.state[1] == {"role": "assistant", "content": "Test response"}

        # Verify the run result
        assert result.api_calls == 1
        assert len(result.api_call_infos) == 1
        assert result.api_call_infos[0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the run method."""
        # Create a mock process
        process = MagicMock()
        process.model_name = "gpt-4"
        process.provider = "openai"
        process.enriched_system_prompt = "Test system prompt"
        process.state = []
        process.tools = []
        process.api_params = {"temperature": 0.7}

        # Mock the API call to raise an exception
        process.client = MagicMock()
        process.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        # Create the executor
        executor = OpenAIProcessExecutor()

        # Test the run method with exception
        with pytest.raises(Exception) as excinfo:
            result = await executor.run(process, "Test input")

        # Check error message
        assert "API error" in str(excinfo.value)

        # Check that the run_stop_reason was set
        assert process.run_stop_reason == "error"
