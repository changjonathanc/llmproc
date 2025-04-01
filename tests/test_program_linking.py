"""Tests for program linking functionality."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.spawn import spawn_tool
from llmproc.tools.tool_result import ToolResult


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

                    # Create a process from the program
                    process = LLMProcess(program=main_program)

                    # Verify the process has the linked program
                    assert process.has_linked_programs
                    assert "expert" in process.linked_programs

                    # Verify compile was called
                    mock_compile.assert_called_with(main_toml, include_linked=True)

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
            from llmproc.tools import register_spawn_tool

            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
                tools={"enabled": ["spawn"]},
            )
            process = LLMProcess(
                program=program, linked_programs_instances={"expert": MagicMock()}
            )

            # Registry should already contain the spawn tool from initialization
            # but we'll register it directly for testing
            register_spawn_tool(process.tool_registry, process)

        # Check that the tool was registered
        assert len(process.tools) >= 1
        assert any(tool["name"] == "spawn" for tool in process.tools)
        assert "spawn" in process.tool_handlers
        assert process.tools[0]["name"] == "spawn"
        assert "input_schema" in process.tools[0]
        # Handler is stored separately in tool_handlers
        assert "spawn" in process.tool_handlers

    @pytest.mark.asyncio
    async def test_spawn_tool_functionality(self):
        """Test the functionality of the spawn tool."""
        # Create mock linked program
        # Import RunResult for mock creation
        from llmproc.results import RunResult

        # Create mock linked program
        mock_expert = MagicMock()

        # Create a mock RunResult for the expert's response
        mock_run_result = RunResult()
        # Add a mock API call instead of setting api_calls directly
        mock_run_result.add_api_call({"model": "test-model"})
        mock_expert.run = AsyncMock(return_value=mock_run_result)

        # Mock get_last_message to return the expected response
        mock_expert.get_last_message = MagicMock(return_value="Expert response")

        # Mock the client creation to avoid API calls
        with patch(
            "llmproc.providers.providers.get_provider_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create a process with linked programs
            from llmproc.program import LLMProgram

            program = LLMProgram(
                model_name="test-model",
                provider="anthropic",
                system_prompt="Test prompt",
            )
            process = LLMProcess(
                program=program, linked_programs_instances={"expert": mock_expert}
            )
            
            # Set empty api_params to avoid None error
            process.api_params = {}

        # Test the spawn tool
        result = await spawn_tool(
            program_name="expert", query="Test query", llm_process=process
        )

        # Check the result
        from llmproc.tools.tool_result import ToolResult

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.content == "Expert response"
        mock_expert.run.assert_called_once_with("Test query")

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
            process = LLMProcess(program=program)

        # Test with missing linked program
        result = await spawn_tool(
            program_name="nonexistent", query="Test query", llm_process=process
        )

        # Check that an error was returned
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "not found" in result.content

        # Test with exception in linked program
        mock_expert = MagicMock()
        mock_expert.run = AsyncMock(side_effect=Exception("Test error"))
        process.linked_programs = {"expert": mock_expert}
        process.has_linked_programs = True

        result = await spawn_tool(
            program_name="expert", query="Test query", llm_process=process
        )

        # Check that an error was returned
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "Test error" in result.content
        
    def test_program_linking_descriptions(self):
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
            with patch(
                "llmproc.providers.providers.get_provider_client"
            ) as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client

                # Compile the main program with linked programs
                main_program = LLMProgram.compile(main_toml, include_linked=True)

                # Verify the program has linked_program_descriptions
                assert hasattr(main_program, "linked_program_descriptions")
                assert "math_expert" in main_program.linked_program_descriptions
                assert "code_expert" in main_program.linked_program_descriptions
                
                # Verify descriptions match what was provided
                assert main_program.linked_program_descriptions["math_expert"] == "Expert specialized in mathematics and statistics"
                assert main_program.linked_program_descriptions["code_expert"] == "Expert specialized in software development"
                
                # Create a process from the program
                process = LLMProcess(program=main_program)
                
                # Verify the process has the linked program descriptions
                assert hasattr(process, "linked_program_descriptions")
                assert "math_expert" in process.linked_program_descriptions
                assert "code_expert" in process.linked_program_descriptions
                
                # Check that tool descriptions include the program descriptions
                assert any("spawn" in tool["name"] for tool in process.tools)
                spawn_tool_desc = next(tool["description"] for tool in process.tools if tool["name"] == "spawn")
                assert "math_expert" in spawn_tool_desc
                assert "code_expert" in spawn_tool_desc
                assert "mathematics" in spawn_tool_desc
                assert "software" in spawn_tool_desc

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
            
            # Create mock linked program
            mock_expert = MagicMock()
            
            # Create a mock for preload_files to verify it's called correctly
            mock_expert.preload_files = MagicMock()
            
            # Mock the run method to return a RunResult
            from llmproc.results import RunResult
            mock_run_result = RunResult()
            mock_expert.run = AsyncMock(return_value=mock_run_result)
            
            # Mock get_last_message to return the expected response
            mock_expert.get_last_message = MagicMock(return_value="Expert response")
            
            # Mock the client creation to avoid API calls
            with patch(
                "llmproc.providers.providers.get_provider_client"
            ) as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                
                # Create a process with linked programs
                from llmproc.program import LLMProgram
                
                program = LLMProgram(
                    model_name="test-model",
                    provider="anthropic",
                    system_prompt="Test prompt",
                )
                process = LLMProcess(
                    program=program, linked_programs_instances={"expert": mock_expert}
                )
            
            # Call spawn_tool with additional_preload_files
            file_paths = [str(test_file1), str(test_file2)]
            result = await spawn_tool(
                program_name="expert", 
                query="Test query with preloaded files", 
                additional_preload_files=file_paths,
                llm_process=process
            )
            
            # Verify preload_files was called with the correct paths
            mock_expert.preload_files.assert_called_once_with(file_paths)
            
            # Check that run was called with the query
            mock_expert.run.assert_called_once_with("Test query with preloaded files")
            
            # Check the result
            assert isinstance(result, ToolResult)
            assert result.is_error is False
            assert result.content == "Expert response"
            
        finally:
            # Clean up test files
            for file_path in [test_file1, test_file2]:
                if file_path.exists():
                    file_path.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()
