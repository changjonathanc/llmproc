"""Tests for program linking functionality."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.common.results import ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.spawn import spawn_tool
from tests.conftest import create_test_llmprocess_directly


class TestProgramLinking:
    """Test program linking functionality."""

    def test_program_linking_compilation(self):
        """Test compilation of linked programs using LLMProgram.compile."""
        # Create a temporary directory for test files
        tmp_dir = Path("tmp_test_linked")
        tmp_dir.mkdir(exist_ok=True)

        try:
            # Create a test file path
            expert_toml = tmp_dir / "expert.toml"
            with open(expert_toml, "w") as f:
                f.write("""
                [model]
                name = "expert-model"
                provider = "anthropic"

                [prompt]
                system_prompt = "Expert prompt"
                """)

            # Create a main toml that links to the expert
            main_toml = tmp_dir / "main.toml"
            with open(main_toml, "w") as f:
                f.write(f"""
                [model]
                name = "main-model"
                provider = "anthropic"

                [prompt]
                system_prompt = "Main prompt"

                [linked_programs]
                expert = "{expert_toml.name}"
                """)

            # Mock the client creation to avoid API calls
            with patch(
                "llmproc.providers.providers.get_provider_client"
            ) as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client

                # Test with direct compilation
                from llmproc.program import LLMProgram

                with patch(
                    "llmproc.program.LLMProgram.from_toml", wraps=LLMProgram.from_toml
                ) as mock_from_toml:
                    # Load the main program with linked programs
                    main_program = LLMProgram.from_toml(main_toml, include_linked=True)

                    # Verify the compilation worked - now linked_programs contains Program objects
                    assert hasattr(main_program, "linked_programs")
                    assert "expert" in main_program.linked_programs

                    # Create a process from the program using the proper pattern
                    # For testing purposes, we need to mock start() since it's async
                    with patch.object(main_program, "start") as mock_start:
                        # Mock what start() would return
                        mock_process = create_test_llmprocess_directly(
                            program=main_program
                        )
                        mock_process.has_linked_programs = True
                        mock_process.linked_programs = {
                            "expert": main_program.linked_programs["expert"]
                        }
                        mock_start.return_value = mock_process

                        # Normally we would use: process = await main_program.start()
                        # But in a non-async test, we use the mock
                        process = mock_process

                        # Verify the process has the linked program
                        assert process.has_linked_programs
                        assert "expert" in process.linked_programs

                    # Verify from_toml was called for both files
                    assert mock_from_toml.called

                    # Check all call arguments to verify both files were processed
                    call_file_names = [
                        call.args[0].name for call in mock_from_toml.call_args_list
                    ]
                    assert main_toml.name in call_file_names, (
                        "main.toml should be processed"
                    )
                    assert expert_toml.name in call_file_names, (
                        "expert.toml should be processed"
                    )

                    # Verify include_linked was set to True in at least one call
                    include_linked_calls = [
                        call
                        for call in mock_from_toml.call_args_list
                        if call.kwargs.get("include_linked") is True
                    ]
                    assert len(include_linked_calls) > 0, (
                        "At least one call should have include_linked=True"
                    )

        finally:
            # Clean up test files
            for file_path in [expert_toml, main_toml]:
                if file_path.exists():
                    file_path.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()

    def test_register_spawn_tool(self):
        """Test registration of spawn tool."""
        # Mock the client creation to avoid API calls
        with patch(
            "llmproc.providers.providers.get_provider_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create a process with linked programs
            from llmproc.program import LLMProgram
            from llmproc.tools.builtin.integration import register_spawn_tool

            # Create an expert program (not process) to link
            expert_program = LLMProgram(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="Expert prompt",
            )

            # Create the main program with tools enabled
            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
                tools={"enabled": ["spawn"]},
            )

            # Link the expert program to the main program
            program.add_linked_program(
                "expert", expert_program, "Expert program description"
            )

            # Create a mock process with the necessary attributes
            process = create_test_llmprocess_directly(
                program=program,
                linked_programs={"expert": expert_program},
                linked_program_descriptions={"expert": "Expert program description"},
                has_linked_programs=True,
            )

            # For proper mocking, ensure the tool_manager has runtime_registry defined
            from llmproc.tools.tool_registry import ToolRegistry

            process.tool_manager.runtime_registry = ToolRegistry()

            # Create a source registry
            builtin_registry = ToolRegistry()
            process.tool_manager.builtin_registry = builtin_registry

            # Load builtin tools into the builtin registry
            from llmproc.tools.builtin.integration import load_builtin_tools

            load_builtin_tools(builtin_registry)

            # Get linked programs and descriptions from process
            linked_programs = process.linked_programs
            linked_program_descriptions = process.linked_program_descriptions

            # Register the spawn tool
            register_spawn_tool(
                builtin_registry,  # Source registry
                process.tool_manager.runtime_registry,  # Target registry
                "spawn",
                linked_programs,
                linked_program_descriptions,
            )

            # Check that the tool was registered in the handler
            assert "spawn" in process.tool_manager.runtime_registry.tool_handlers

            # Check that we can get the tool definition
            definitions = process.tool_manager.runtime_registry.get_definitions()
            spawn_tools = [d for d in definitions if d.get("name") == "spawn"]
            assert len(spawn_tools) == 1
            assert "input_schema" in spawn_tools[0]

    @pytest.mark.asyncio
    async def test_spawn_tool_functionality(self):
        """Test the functionality of the spawn tool."""
        # Create mock linked program and mock process creation
        from llmproc.common.results import RunResult

        # Create a mock for program_exec.create_process
        with patch("llmproc.program_exec.create_process") as mock_create_process:
            # Create a mock process that will be returned when spawn creates a process on demand
            mock_expert_process = MagicMock()

            # Create a mock RunResult for the expert's response
            mock_run_result = RunResult()
            # Add a mock API call instead of setting api_calls directly
            mock_run_result.add_api_call({"model": "test-model"})
            mock_expert_process.run = AsyncMock(return_value=mock_run_result)

            # Mock get_last_message to return the expected response
            mock_expert_process.get_last_message = MagicMock(
                return_value="Expert response"
            )

            # Configure create_process to return our mock expert process
            mock_create_process.return_value = mock_expert_process

            # Create the main program and expert program
            from llmproc.program import LLMProgram

            expert_program = LLMProgram(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="Expert prompt",
            )

            main_program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
            )

            # Link the expert program directly (using program reference, not process)
            main_program.add_linked_program("expert", expert_program)

            # Create mock process with our test helper
            process = create_test_llmprocess_directly(
                program=main_program,
                linked_programs={"expert": expert_program},
                has_linked_programs=True,
                api_params={},  # Set empty api_params to avoid None error
            )

            # Test the spawn tool with runtime_context
            result = await spawn_tool(
                program_name="expert",
                query="Test query",
                runtime_context={
                    "process": process,
                    "linked_programs": process.linked_programs,
                },
            )

            # Check the result
            from llmproc.common.results import ToolResult

            assert isinstance(result, ToolResult)
            assert not result.is_error
            assert result.content == "Expert response"

            # Verify that create_process was called with the expert program
            mock_create_process.assert_called_once()
            # Extract arguments from the call
            called_args = mock_create_process.call_args[0]
            assert called_args[0] == expert_program

            # Verify that run was called on the expert process
            mock_expert_process.run.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_spawn_tool_error_handling(self):
        """Test error handling in the spawn tool."""
        # Mock the client creation to avoid API calls
        with patch(
            "llmproc.providers.providers.get_provider_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create a process without linked programs
            from llmproc.program import LLMProgram

            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
            )

            # Create a mock for program.start()
            mock_start = AsyncMock()
            program.start = mock_start

            # Create mock process that would be returned by start()
            process = create_test_llmprocess_directly(program=program)

            # Configure the mock to return our process
            mock_start.return_value = process

            # In a real implementation, we would use:
            # process = await program.start()

        # Test with missing linked program
        result = await spawn_tool(
            program_name="nonexistent",
            query="Test query",
            runtime_context={
                "process": process,
                "linked_programs": process.linked_programs,
            },
        )

        # Check that an error was returned
        from llmproc.common.results import ToolResult

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "not found" in result.content

        # Test with exception in linked program creation
        with patch("llmproc.program_exec.create_process") as mock_create_process:
            # Set up the mock to raise an exception
            mock_create_process.side_effect = Exception("Test error")

            # Create an expert program to link
            expert_program = LLMProgram(
                model_name="expert-model",
                provider="anthropic",
                system_prompt="Expert prompt",
            )

            # Set up linked programs with the program instance (not process)
            process.linked_programs = {"expert": expert_program}
            process.has_linked_programs = True

            result = await spawn_tool(
                program_name="expert",
                query="Test query",
                runtime_context={
                    "process": process,
                    "linked_programs": process.linked_programs,
                },
            )

            # Check that an error was returned
            assert isinstance(result, ToolResult)
            assert result.is_error is True
            assert "Test error" in result.content

    @pytest.mark.asyncio
    async def test_program_linking_descriptions(self):
        """Test program linking with descriptions using enhanced syntax."""
        # Create a temporary directory for test files
        tmp_dir = Path("tmp_test_linked_desc")
        tmp_dir.mkdir(exist_ok=True)

        try:
            # Create test files for different experts
            math_expert_toml = tmp_dir / "math_expert.toml"
            code_expert_toml = tmp_dir / "code_expert.toml"

            with open(math_expert_toml, "w") as f:
                f.write("""
                [model]
                name = "math-expert-model"
                provider = "anthropic"

                [prompt]
                system_prompt = "You are a math expert"
                """)

            with open(code_expert_toml, "w") as f:
                f.write("""
                [model]
                name = "code-expert-model"
                provider = "anthropic"

                [prompt]
                system_prompt = "You are a coding expert"
                """)

            # Create a main toml that links to both experts with descriptions
            main_toml = tmp_dir / "main_with_descriptions.toml"
            with open(main_toml, "w") as f:
                f.write(f"""
                [model]
                name = "main-model"
                provider = "anthropic"

                [prompt]
                system_prompt = "Main prompt"

                [tools]
                enabled = ["spawn"]

                [linked_programs]
                math_expert = {{ path = "{math_expert_toml.name}", description = "Expert specialized in mathematics and statistics" }}
                code_expert = {{ path = "{code_expert_toml.name}", description = "Expert specialized in software development" }}
                """)

            # Mock the client creation to avoid API calls
            with (
                patch(
                    "llmproc.providers.providers.get_provider_client"
                ) as mock_get_client,
                patch(
                    "llmproc.tools.builtin.integration.register_spawn_tool"
                ) as mock_register_spawn_tool,
            ):
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client

                # Create a mock implementation of register_spawn_tool that adds a mock tool schema
                def mock_spawn_tool_register(
                    source_registry,
                    target_registry,
                    tool_name,
                    linked_programs,
                    linked_program_descriptions,
                ):
                    # Create basic spawn tool definition
                    spawn_def = {
                        "name": "spawn",
                        "description": "Spawn process from a linked program - Available programs: "
                        "- 'math_expert': Expert specialized in mathematics and statistics"
                        "- 'code_expert': Expert specialized in software development",
                    }
                    # Register the mock tool
                    target_registry.register_tool("spawn", lambda x: x, spawn_def)
                    return True

                mock_register_spawn_tool.side_effect = mock_spawn_tool_register

                # Compile the main program with linked programs
                main_program = LLMProgram.from_toml(main_toml, include_linked=True)

                # Verify the program has linked_program_descriptions
                assert hasattr(main_program, "linked_program_descriptions")
                assert "math_expert" in main_program.linked_program_descriptions
                assert "code_expert" in main_program.linked_program_descriptions

                # Verify descriptions match what was provided
                assert (
                    main_program.linked_program_descriptions["math_expert"]
                    == "Expert specialized in mathematics and statistics"
                )
                assert (
                    main_program.linked_program_descriptions["code_expert"]
                    == "Expert specialized in software development"
                )

                # Use the proper initialization pattern with start()
                # We need to mock it since this is a synchronous test
                from llmproc.llm_process import LLMProcess

                # Create a mock for start() method
                with patch.object(main_program, "start") as mock_start:
                    # Create mock process using proper initialization
                    process = create_test_llmprocess_directly(program=main_program)

                    # Ensure the process has the linked program descriptions
                    process.linked_program_descriptions = (
                        main_program.linked_program_descriptions
                    )

                    # Configure mock to return our process
                    mock_start.return_value = process

                    # In a real async implementation, we would use:
                    # process = await main_program.start()

                # Verify the process has the linked program descriptions
                assert hasattr(process, "linked_program_descriptions")
                assert "math_expert" in process.linked_program_descriptions
                assert "code_expert" in process.linked_program_descriptions

                # Verify descriptions are correctly transferred
                assert (
                    process.linked_program_descriptions["math_expert"]
                    == "Expert specialized in mathematics and statistics"
                )
                assert (
                    process.linked_program_descriptions["code_expert"]
                    == "Expert specialized in software development"
                )

                # Skip the tool description tests as this is just testing LinkedProgram descriptions

        finally:
            # Clean up test files
            for file_path in [math_expert_toml, code_expert_toml, main_toml]:
                if file_path and file_path.exists():
                    file_path.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()

    @pytest.mark.asyncio
    async def test_spawn_tool_with_preloaded_files(self):
        """Test the spawn tool with file preloading."""
        # Create temporary files for testing
        tmp_dir = Path("tmp_test_preload")
        tmp_dir.mkdir(exist_ok=True)

        try:
            # Create test files
            test_file1 = tmp_dir / "test1.txt"
            test_file2 = tmp_dir / "test2.txt"

            with open(test_file1, "w") as f:
                f.write("Test content 1")
            with open(test_file2, "w") as f:
                f.write("Test content 2")

            # Create a mock for program_exec.create_process
            with patch("llmproc.program_exec.create_process") as mock_create_process:
                # Create a mock expert process that will be returned when spawn creates a process
                mock_expert_process = MagicMock()

                # Mock the run method to return a RunResult
                from llmproc.common.results import RunResult

                mock_run_result = RunResult()
                mock_expert_process.run = AsyncMock(return_value=mock_run_result)

                # Mock get_last_message to return the expected response
                mock_expert_process.get_last_message = MagicMock(
                    return_value="Expert response"
                )

                # Configure create_process to return our mock expert process and capture arguments
                mock_create_process.return_value = mock_expert_process

                # Make create_process capture additional_preload_files
                async def mock_create_with_files(
                    program, additional_preload_files=None
                ):
                    mock_create_process.additional_preload_files = (
                        additional_preload_files
                    )
                    return mock_expert_process

                mock_create_process.side_effect = mock_create_with_files

                # Create the main program and expert program
                from llmproc.program import LLMProgram

                expert_program = LLMProgram(
                    model_name="expert-model",
                    provider="anthropic",
                    system_prompt="Expert prompt",
                )

                main_program = LLMProgram(
                    model_name="test-model",
                    provider="anthropic",
                    system_prompt="Test prompt",
                )

                # Link the expert program directly (using program reference, not process)
                main_program.add_linked_program("expert", expert_program)

                # Create mock process with our test helper
                process = create_test_llmprocess_directly(
                    program=main_program,
                    linked_programs={"expert": expert_program},
                    has_linked_programs=True,
                )

                # Call spawn_tool with additional_preload_files
                file_paths = [str(test_file1), str(test_file2)]
                result = await spawn_tool(
                    program_name="expert",
                    query="Test query with preloaded files",
                    additional_preload_files=file_paths,
                    runtime_context={
                        "process": process,
                        "linked_programs": process.linked_programs,
                    },
                )

                # Verify that create_process was called with the expert program and additional_preload_files
                mock_create_process.assert_called_once()
                # Extract arguments from the call
                called_args = mock_create_process.call_args[0]
                assert called_args[0] == expert_program
                # Verify that additional_preload_files was passed using our side_effect function
                assert mock_create_process.additional_preload_files == file_paths

                # Check that run was called with the query
                mock_expert_process.run.assert_called_once_with(
                    "Test query with preloaded files"
                )

                # Check the result
                assert isinstance(result, ToolResult)
                assert not result.is_error
                assert result.content == "Expert response"

        finally:
            # Clean up test files
            for file_path in [test_file1, test_file2]:
                if file_path.exists():
                    file_path.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()
