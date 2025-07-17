"""Tests for the OpenAI process executor."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.providers.openai_process_executor import OpenAIProcessExecutor
from llmproc.plugin.plugin_event_runner import PluginEventRunner


class TestOpenAIProcessExecutor:
    """Tests for the OpenAI process executor."""

    def test_openai_with_tools_supported(self):
        """Test that OpenAI process creation works even with tools configured."""
        # Create a program with tools
        program = LLMProgram(
            model_name="gpt-4o-mini",
            provider="openai",
            system_prompt="Test system prompt",
            tools=["spawn"],  # Enable a tool
        )

        # Import the test helper
        from tests.conftest import create_test_llmprocess_directly

        # Creating a process should succeed without error
        process = create_test_llmprocess_directly(program=program)
        assert isinstance(process, LLMProcess)

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
        from llmproc.plugin.plugin_event_runner import PluginEventRunner
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner
        process.trigger_event = AsyncMock()
        process.get_last_message = MagicMock(return_value="Test response")

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
        assert result.api_call_count == 1
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
        from llmproc.plugin.plugin_event_runner import PluginEventRunner
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner

        # Mock the API call to raise an exception
        process.client = MagicMock()
        process.client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        # Mock process methods that are called during execution
        process.trigger_event = AsyncMock()
        process.get_last_message = MagicMock(return_value="")

        # Create the executor
        executor = OpenAIProcessExecutor()

        # Test the run method with exception
        with pytest.raises(Exception) as excinfo:
            result = await executor.run(process, "Test input")

        # Check error message
        assert "API error" in str(excinfo.value)

        # Test passes if the exception was properly handled and re-raised

    @pytest.mark.asyncio
    async def test_tool_usage(self):
        """Verify tools can be invoked through OpenAI responses."""
        process = MagicMock()
        process.model_name = "gpt-4o"
        process.provider = "openai"
        process.enriched_system_prompt = "system"
        process.state = []
        process.api_params = {}
        from llmproc.plugin.plugin_event_runner import PluginEventRunner
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner

        # Mock tool call in first response, then a final response with no tools
        tool_call = SimpleNamespace(
            id="call1",
            type="function",
            function=SimpleNamespace(name="echo", arguments="{\"text\": \"hi\"}")
        )
        mock_resp_with_tools = MagicMock()
        mock_resp_with_tools.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tool_call]), finish_reason="tool_calls")]
        mock_resp_with_tools.usage = {}
        mock_resp_with_tools.id = "r1"

        # Second response with no tool calls to end the conversation
        mock_resp_final = MagicMock()
        mock_resp_final.choices = [MagicMock(message=MagicMock(content="Done!", tool_calls=[]), finish_reason="stop")]
        mock_resp_final.usage = {}
        mock_resp_final.id = "r2"

        process.client = MagicMock()
        process.client.chat.completions.create = AsyncMock(side_effect=[mock_resp_with_tools, mock_resp_final])

        # Mock tool execution
        process.call_tool = AsyncMock(return_value=ToolResult.from_success("ok"))
        process.trigger_event = AsyncMock()

        executor = OpenAIProcessExecutor()
        result = await executor.run(process, "hi")

        process.call_tool.assert_awaited_once_with("echo", {"text": "hi"})
        assert any(msg.get("role") == "tool" for msg in process.state)
        assert result.tool_call_count == 1
        # Should have made 2 API calls: one with tool call, one final response
        assert result.api_call_count == 2

    @pytest.mark.asyncio
    async def test_tools_included_in_api_call(self):
        """Verify tools are properly converted and included in OpenAI API calls."""
        process = MagicMock()
        process.model_name = "gpt-4o"
        process.provider = "openai"
        process.enriched_system_prompt = "system"
        process.state = []
        process.api_params = {}
        from llmproc.plugin.plugin_event_runner import PluginEventRunner
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner

        # Mock tools in internal format
        process.tools = [
            {
                "name": "calculator",
                "description": "Calculate expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        ]

        # Mock API response
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Result", tool_calls=[]), finish_reason="stop")]
        mock_resp.usage = {}
        mock_resp.id = "r1"

        process.client = MagicMock()
        process.client.chat.completions.create = AsyncMock(return_value=mock_resp)
        process.trigger_event = AsyncMock()

        executor = OpenAIProcessExecutor()
        await executor.run(process, "test")

        # Verify API was called with tools in OpenAI format
        call_args = process.client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        tools = call_args.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "calculator"
        assert tools[0]["function"]["description"] == "Calculate expressions"
        assert "parameters" in tools[0]["function"]
        assert tools[0]["function"]["parameters"]["type"] == "object"
