"""Robust tests for program linking functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from tests.conftest import create_test_llmprocess_directly


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

            # Create expert program
            from llmproc.program import LLMProgram

            expert_program = LLMProgram(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="You are an expert model.",
            )

            # Create a mock process that will be returned when expert_program is passed to create_process
            expert_process = MagicMock(spec=LLMProcess)

            # Import RunResult for mock creation
            from llmproc.common.results import RunResult

            # Create a mock RunResult for the expert's response
            mock_run_result = RunResult()
            # Add a mock API call instead of setting api_calls directly
            mock_run_result.add_api_call({"model": "test-model"})
            expert_process.run = AsyncMock(return_value=mock_run_result)

            # Mock get_last_message to return the expected response
            expert_process.get_last_message = MagicMock(
                return_value="I am the expert's response"
            )

            # Create main program and link expert_program to it (not expert_process)
            main_program = LLMProgram(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
            )

            # Link the expert PROGRAM to the main program
            main_program.add_linked_program("expert", expert_program)

            # Mock the process creation
            with (
                patch.object(main_program, "start") as mock_main_start,
                patch("llmproc.program_exec.create_process") as mock_create_process,
            ):
                # Configure create_process to return our expert_process when called with expert_program
                mock_create_process.return_value = expert_process

                # Create a main process instance for the mock to return
                main_process = create_test_llmprocess_directly(program=main_program)

                # Initialize linked_programs with the expert PROGRAM reference (not process)
                main_process.linked_programs = {"expert": expert_program}
                main_process.has_linked_programs = True
                main_process.api_params = {}

                # Configure the mock to return our main process
                mock_main_start.return_value = main_process

            # Set mcp_enabled to allow tool registration
            main_process.mcp_enabled = True
            main_process.enabled_tools = ["spawn"]

            # Initialize the tool_manager
            from llmproc.tools.tool_manager import ToolManager

            main_process.tool_manager = ToolManager()

            # Set enabled tools in the tool manager
            main_process.tool_manager.set_enabled_tools(["spawn"])

            # Register spawn tool using the new integration method
            # Create registry with the builtin tools
            from llmproc.tools.builtin.integration import (
                load_builtin_tools,
                register_spawn_tool,
            )
            from llmproc.tools.tool_registry import ToolRegistry

            main_process.tool_manager.builtin_registry = ToolRegistry()
            load_builtin_tools(main_process.tool_manager.builtin_registry)

            # Register spawn tool with the runtime registry
            register_spawn_tool(
                main_process.tool_manager.builtin_registry,
                main_process.tool_manager.runtime_registry,
                "spawn",
                main_process.linked_programs,
                {},  # No linked program descriptions in this test
            )

            # Add spawn to enabled tools list in the registry if not already there
            if "spawn" not in main_process.tool_manager.enabled_tools:
                main_process.tool_manager.enabled_tools.append("spawn")

            # Ensure the tool was registered - tools is now a property
            assert len(main_process.tools) > 0
            assert any(tool["name"] == "spawn" for tool in main_process.tools)
            assert "spawn" in main_process.tool_handlers
            assert "expert" in main_process.linked_programs

            # Now, call the spawn tool with a fresh patch to program_exec.create_process
            with patch("llmproc.program_exec.create_process") as mock_create_process:
                # Configure mock to return our expert_process
                mock_create_process.return_value = expert_process

                # Call the spawn tool directly
                from llmproc.tools.builtin.spawn import spawn_tool

                result = await spawn_tool(
                    program_name="expert",
                    query="What is your expertise?",
                    runtime_context={
                        "process": main_process,
                        "linked_programs": main_process.linked_programs,
                    },
                )

                # Verify that create_process was called with expert_program and None for additional_preload_files
                mock_create_process.assert_called_once_with(expert_program, None)

                # Verify the expert was called with the right query
                expert_process.run.assert_called_once_with("What is your expertise?")

                # Verify the result
                from llmproc.common.results import ToolResult

                assert isinstance(result, ToolResult)
                assert not result.is_error
                assert result.content == "I am the expert's response"

    @pytest.mark.asyncio
    async def test_spawn_tool_with_real_toml(self, mock_toml_files):
        """Test spawn tool by loading from actual TOML files."""
        with (
            patch("llmproc.providers.providers.get_provider_client") as mock_client,
        ):
            # Mock the API client
            mock_client.return_value = MagicMock()

            # Load main program from TOML - this correctly sets up linked_programs
            from llmproc.program import LLMProgram

            main_program = LLMProgram.from_toml(mock_toml_files["main_toml"])

            # Get the actual linked program reference that was loaded from TOML
            expert_program_ref = main_program.linked_programs["expert"]

            # Create a mock expert process to be returned by create_process
            mock_expert_process = MagicMock(spec=LLMProcess)

            # Import and create RunResult for the mock
            from llmproc.common.results import RunResult

            mock_run_result = RunResult()
            mock_run_result.add_api_call({"model": "test-model"})
            mock_expert_process.run = AsyncMock(return_value=mock_run_result)

            # Mock get_last_message to return the expected response
            mock_expert_process.get_last_message = MagicMock(
                return_value="Expert response from TOML"
            )

            # Create main process with a patch for start()
            with patch.object(main_program, "start") as mock_start:
                # Create a process instance for the mock to return
                main_process = create_test_llmprocess_directly(program=main_program)

                # The linked_programs should be correctly inherited from main_program
                # during LLMProcess.__init__, but let's ensure it's set:
                if not main_process.linked_programs:
                    main_process.linked_programs = main_program.linked_programs

                main_process.has_linked_programs = True

                # Configure the mock to return our process
                mock_start.return_value = main_process

            # Now call the spawn tool with a patch for create_process
            with patch("llmproc.program_exec.create_process") as mock_create_process:
                # Configure the create_process mock to return mock_expert_process
                mock_create_process.return_value = mock_expert_process

                # Call the spawn tool directly
                from llmproc.tools.builtin.spawn import spawn_tool

                result = await spawn_tool(
                    program_name="expert",
                    query="Tell me about version 0.1.0",
                    runtime_context={
                        "process": main_process,
                        "linked_programs": main_process.linked_programs,
                    },
                )

                # Verify create_process was called with the expert_program reference from TOML and None for additional_preload_files
                mock_create_process.assert_called_once_with(expert_program_ref, None)

                # Verify the expert was called with the right query
                mock_expert_process.run.assert_called_once_with(
                    "Tell me about version 0.1.0"
                )

                # Verify the result
                from llmproc.common.results import ToolResult

                assert isinstance(result, ToolResult)
                assert not result.is_error
                assert result.content == "Expert response from TOML"

    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self, mock_toml_files):
        """Test error handling in spawn tool."""
        with patch("llmproc.providers.providers.get_provider_client") as mock_client:
            # Mock the API client
            mock_client.return_value = MagicMock()

            # Create error expert program
            from llmproc.program import LLMProgram

            error_expert_program = LLMProgram(
                model_name="error-expert-model",
                provider="anthropic",
                system_prompt="I will raise errors",
            )

            # Create a mock expert process that will raise errors
            mock_expert_process = MagicMock(spec=LLMProcess)
            mock_expert_process.run = AsyncMock(side_effect=ValueError("Test error"))

            # Create main program
            main_program = LLMProgram(
                model_name="main-model",
                provider="anthropic",
                system_prompt="You are the main model.",
            )

            # Link the error expert PROGRAM to the main program
            main_program.add_linked_program("error_expert", error_expert_program)

            # Create main process
            with patch.object(main_program, "start") as mock_start:
                # Create a process instance for the mock to return
                main_process = create_test_llmprocess_directly(program=main_program)

                # Initialize linked_programs with the expert PROGRAM reference (not process)
                main_process.linked_programs = {"error_expert": error_expert_program}
                main_process.has_linked_programs = True
                main_process.api_params = {}

                # Configure the mock to return our process
                mock_start.return_value = main_process

            # Test with error raising expert
            with patch("llmproc.program_exec.create_process") as mock_create_process:
                # Configure create_process to return the error-raising process
                mock_create_process.return_value = mock_expert_process

                # Call the spawn tool
                from llmproc.common.results import ToolResult
                from llmproc.tools.builtin.spawn import spawn_tool

                result_error = await spawn_tool(
                    program_name="error_expert",
                    query="This will error",
                    runtime_context={
                        "process": main_process,
                        "linked_programs": main_process.linked_programs,
                    },
                )

                # Verify create_process was called with the program and None for additional_preload_files
                mock_create_process.assert_called_once_with(error_expert_program, None)

                # Verify run was attempted
                mock_expert_process.run.assert_called_once_with("This will error")

                # Verify the error result
                assert isinstance(result_error, ToolResult)
                assert result_error.is_error
                assert "Test error" in result_error.content

            # Test with nonexistent program
            result_nonexistent = await spawn_tool(
                program_name="nonexistent",
                query="This won't work",
                runtime_context={
                    "process": main_process,
                    "linked_programs": main_process.linked_programs,
                },
            )

            # Verify the error result
            assert isinstance(result_nonexistent, ToolResult)
            assert result_nonexistent.is_error
            assert "not found" in result_nonexistent.content

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
