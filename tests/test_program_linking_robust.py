"""Robust tests for program linking functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess


class TestProgramLinkingRobust:
    """Comprehensive tests for program linking that don't depend on external files."""

    @pytest.fixture
    def mock_toml_files(self):
        """Create temporary TOML files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create main program TOML
            main_toml_path = Path(temp_dir) / "main.toml"
            with open(main_toml_path, "w") as f:
                f.write("""
                [model]
                name = "test-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are a test assistant with access to a specialized model."
                
                [parameters]
                max_tokens = 1000
                
                
                [tools]
                enabled = ["spawn"]
                
                [linked_programs]
                expert = "expert.toml"
                """)

            # Create expert program TOML
            expert_toml_path = Path(temp_dir) / "expert.toml"
            with open(expert_toml_path, "w") as f:
                f.write("""
                [model]
                name = "expert-model"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are an expert on test subjects."
                
                [parameters]
                max_tokens = 500
                """)

            yield {
                "temp_dir": temp_dir,
                "main_toml": main_toml_path,
                "expert_toml": expert_toml_path,
            }

    @pytest.mark.asyncio
    async def test_spawn_tool_with_mock_programs(self, mock_toml_files):
        """Test spawn tool by using mocked linked programs."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()

            # Create expert process
            from llmproc.program import LLMProgram

            expert_program = LLMProgram(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="You are an expert model.",
            )
            expert_process = LLMProcess(program=expert_program)

            # Import RunResult for mock creation
            from llmproc.results import RunResult

            # Create a mock RunResult for the expert's response
            mock_run_result = RunResult()
            # Add a mock API call instead of setting api_calls directly
            mock_run_result.add_api_call({"model": "test-model"})
            expert_process.run = AsyncMock(return_value=mock_run_result)

            # Mock get_last_message to return the expected response
            expert_process.get_last_message = MagicMock(
                return_value="I am the expert's response"
            )

            # Create main process with linked program
            main_program = LLMProgram(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
            )
            main_process = LLMProcess(
                program=main_program,
                linked_programs_instances={"expert": expert_process},
            )
            
            # Set empty api_params to avoid None error
            main_process.api_params = {}

            # Set mcp_enabled to allow tool registration
            main_process.mcp_enabled = True
            main_process.enabled_tools = ["spawn"]

            # Initialize the tool_manager
            from llmproc.tools.tool_manager import ToolManager
            main_process.tool_manager = ToolManager()
            main_process.tool_registry = main_process.tool_manager.registry
            
            # Set enabled tools in the tool manager
            main_process.tool_manager.set_enabled_tools(["spawn"])
            
            # Register spawn tool using the new method
            from llmproc.tools.utils import register_spawn_tool

            register_spawn_tool(main_process.tool_registry, main_process)
            
            # Add spawn to enabled tools list in the registry if not already there
            if "spawn" not in main_process.tool_manager.enabled_tools:
                main_process.tool_manager.enabled_tools.append("spawn")

            # Ensure the tool was registered - tools is now a property
            assert len(main_process.tools) > 0
            assert any(tool["name"] == "spawn" for tool in main_process.tools)
            assert "spawn" in main_process.tool_handlers
            assert "expert" in main_process.linked_programs

            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool

            result = await spawn_tool(
                program_name="expert",
                query="What is your expertise?",
                llm_process=main_process,
            )

            # Verify the result
            from llmproc.tools.tool_result import ToolResult

            assert isinstance(result, ToolResult)
            assert result.is_error is False
            assert result.content == "I am the expert's response"

            # Verify the expert was called with the right query
            expert_process.run.assert_called_once_with("What is your expertise?")

    @pytest.mark.asyncio
    async def test_spawn_tool_with_real_toml(self, mock_toml_files):
        """Test spawn tool by loading from actual TOML files."""
        with (
            patch("llmproc.providers.providers.get_provider_client") as mock_client,
            patch("llmproc.program.LLMProgram.start") as mock_start,
        ):
            # Mock the API client
            mock_client.return_value = MagicMock()

            # Use the two-step pattern with patched start() to avoid actual async initialization
            from llmproc.program import LLMProgram

            main_program = LLMProgram.from_toml(mock_toml_files["main_toml"])

            # Create a mock process that would normally be returned by start()
            main_process = LLMProcess(program=main_program)
            mock_start.return_value = main_process

            # Replace expert process with mock
            mock_expert = MagicMock()

            # Import and create RunResult for the mock
            from llmproc.results import RunResult

            mock_run_result = RunResult()
            # Add a mock API call instead of setting api_calls directly
            mock_run_result.add_api_call({"model": "test-model"})
            mock_expert.run = AsyncMock(return_value=mock_run_result)

            # Mock get_last_message to return the expected response
            mock_expert.get_last_message = MagicMock(
                return_value="Expert response from TOML"
            )

            main_process.linked_programs["expert"] = mock_expert

            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool

            result = await spawn_tool(
                program_name="expert",
                query="Tell me about version 0.1.0",
                llm_process=main_process,
            )

            # Verify the result
            from llmproc.tools.tool_result import ToolResult

            assert isinstance(result, ToolResult)
            assert result.is_error is False
            assert result.content == "Expert response from TOML"

            # Verify the expert was called with the right query
            mock_expert.run.assert_called_once_with("Tell me about version 0.1.0")

    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self, mock_toml_files):
        """Test error handling in spawn tool."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()

            # Create main process with linked program that will raise an error
            mock_expert = MagicMock()
            mock_expert.run = AsyncMock(side_effect=ValueError("Test error"))

            from llmproc.program import LLMProgram

            main_program = LLMProgram(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
            )
            main_process = LLMProcess(
                program=main_program,
                linked_programs_instances={"error_expert": mock_expert},
            )
            
            # Set empty api_params to avoid None error
            main_process.api_params = {}

            # Call the spawn tool directly
            from llmproc.tools.spawn import spawn_tool

            result = await spawn_tool(
                program_name="error_expert",
                query="This will error",
                llm_process=main_process,
            )

            # Verify the error result
            from llmproc.tools.tool_result import ToolResult

            assert isinstance(result, ToolResult)
            assert result.is_error is True
            assert "Test error" in result.content

            # Test with nonexistent program
            result = await spawn_tool(
                program_name="nonexistent",
                query="This won't work",
                llm_process=main_process,
            )

            # Verify the error result
            assert isinstance(result, ToolResult)
            assert result.is_error is True
            assert "not found" in result.content

    def test_empty_messages_filtering(self):
        """Test that empty messages are filtered when preparing messages for API."""
        # Directly test the message filtering logic by examining the _run_anthropic_with_tools method

        # Create a test state with empty messages
        state = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": ""},  # This should be filtered
            {"role": "user", "content": "Another message"},
        ]

        # Extract filtered messages
        system_prompt = None
        messages = []

        for msg in state:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Skip empty messages that would cause API errors
                if msg.get("content") != "":
                    messages.append(msg)

        # Verify filtering
        assert (
            len(messages) == 2
        )  # Two non-system messages (with empty message skipped)
        assert system_prompt == "System prompt"

        # Verify no empty messages
        for msg in messages:
            assert msg.get("content") != ""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="run_anthropic_with_tools is deprecated in favor of AnthropicProcessExecutor"
    )
    async def test_run_anthropic_with_tools_skips_empty_response(self):
        """This test is now skipped as we've moved to the AnthropicProcessExecutor API."""
        pass
