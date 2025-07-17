"""Tests for the Gemini process executor tool support."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json

import pytest

from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.providers.gemini_process_executor import GeminiProcessExecutor
from llmproc.plugin.plugin_event_runner import PluginEventRunner


class TestGeminiProcessExecutorToolSupport:
    """Tests for the Gemini process executor tool support."""

    @pytest.mark.asyncio
    async def test_tool_usage(self):
        """Verify tools can be invoked through Gemini responses."""
        process = MagicMock()
        process.model_name = "gemini-1.5-flash"
        process.provider = "gemini"
        process.enriched_system_prompt = "system"
        process.state = []
        process.api_params = {}
        process.tools = []
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner

        # Mock tool call in first response, then a final response with no tools
        tool_call_part = SimpleNamespace(
            function_call=SimpleNamespace(name="echo", args={"text": "hi"}),
            text=None
        )
        mock_resp_with_tools = MagicMock()
        mock_resp_with_tools.candidates = [
            MagicMock(content=MagicMock(parts=[tool_call_part]))
        ]
        mock_resp_with_tools.text = ""

        # Second response with no tool calls to end the conversation
        final_part = SimpleNamespace(text="Done!", function_call=None)
        mock_resp_final = MagicMock()
        mock_resp_final.candidates = [
            MagicMock(content=MagicMock(parts=[final_part]))
        ]
        mock_resp_final.text = "Done!"

        # Mock the new API structure used in the implementation
        process.client = MagicMock()
        process.client.aio.models.generate_content = AsyncMock(side_effect=[mock_resp_with_tools, mock_resp_final])

        # Mock tool execution
        process.call_tool = AsyncMock(return_value=ToolResult.from_success("ok"))
        process.trigger_event = AsyncMock()

        executor = GeminiProcessExecutor()
        result = await executor.run(process, "hi")

        process.call_tool.assert_awaited_once_with("echo", {"text": "hi"})
        assert any(msg.get("role") == "tool" for msg in process.state)
        assert result.tool_call_count == 1
        # Should have made 2 API calls: one with tool call, one final response
        assert result.api_call_count == 2

    @pytest.mark.asyncio
    async def test_tools_included_in_api_call(self):
        """Verify tools are properly converted and included in Gemini API calls."""
        process = MagicMock()
        process.model_name = "gemini-1.5-flash"
        process.provider = "gemini"
        process.enriched_system_prompt = "system"
        process.state = []
        process.api_params = {}
        process._submit_to_loop = lambda coro: asyncio.get_running_loop().create_task(coro)
        runner = PluginEventRunner(process._submit_to_loop, ())
        process.plugins = runner
        process.hooks = runner

        # Mock tools in internal format
        mock_tool = {
            "name": "calculator",
            "description": "Calculate expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
        process.tools = [mock_tool]

        # Mock API response
        final_part = SimpleNamespace(text="Result", function_call=None)
        mock_resp = MagicMock()
        mock_resp.candidates = [
            MagicMock(content=MagicMock(parts=[final_part]))
        ]
        mock_resp.text = "Result"

        process.client = MagicMock()
        process.client.aio.models.generate_content = AsyncMock(return_value=mock_resp)
        process.trigger_event = AsyncMock()

        executor = GeminiProcessExecutor()
        await executor.run(process, "test")

        # Verify API was called with tools in Gemini format
        call_args = process.client.aio.models.generate_content.call_args
        assert "tools" in call_args.kwargs
        tools = call_args.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function_declarations"][0]["name"] == "calculator"
        assert tools[0]["function_declarations"][0]["description"] == "Calculate expressions"
        assert "parameters" in tools[0]["function_declarations"][0]
        assert tools[0]["function_declarations"][0]["parameters"]["type"] == "object"
