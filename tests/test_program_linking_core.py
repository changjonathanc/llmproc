"""Core tests for program linking functionality.

This file consolidates the core program linking tests from:
- test_program_linking.py
- test_program_linking_compiler.py
- test_program_linking_robust.py
- test_program_linking_descriptions_specific.py
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.common.results import RunResult, ToolResult
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.spawn import spawn_tool

from tests.conftest import create_test_llmprocess_directly


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing program linking."""
    with tempfile.TemporaryDirectory() as temp_dir_path:
        yield Path(temp_dir_path)


@pytest.fixture
def mock_linked_programs(temp_dir):
    """Create a set of linked program TOML files for testing."""
    # Create main program TOML
    main_toml = temp_dir / "main.toml"
    with open(main_toml, "w") as f:
        f.write(
            """
        [model]
        name = "main-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Main program"

        [tools]
        builtin = ["spawn"]

        [linked_programs]
        helper = "helper.toml"
        expert = "expert.toml"
        """
        )

    # Create helper program TOML
    helper_toml = temp_dir / "helper.toml"
    with open(helper_toml, "w") as f:
        f.write(
            """
        [model]
        name = "helper-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Helper program"
        """
        )

    # Create expert program TOML
    expert_toml = temp_dir / "expert.toml"
    with open(expert_toml, "w") as f:
        f.write(
            """
        [model]
        name = "expert-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Expert program"
        """
        )

    return {
        "main_toml": main_toml,
        "helper_toml": helper_toml,
        "expert_toml": expert_toml,
    }


@pytest.fixture
def mock_linked_programs_with_descriptions(temp_dir):
    """Create linked program TOML files with descriptions."""
    # Create main program TOML with descriptions
    main_toml = temp_dir / "main_with_desc.toml"
    with open(main_toml, "w") as f:
        f.write(
            """
        [model]
        name = "main-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Main program"

        [tools]
        builtin = ["spawn"]

        [linked_programs]
        helper = { path = "helper_with_desc.toml", description = "A helper program that provides assistance" }
        expert = { path = "expert_with_desc.toml", description = "An expert program with specialized knowledge" }
        """
        )

    # Create helper program TOML
    helper_toml = temp_dir / "helper_with_desc.toml"
    with open(helper_toml, "w") as f:
        f.write(
            """
        [model]
        name = "helper-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Helper program"
        """
        )

    # Create expert program TOML
    expert_toml = temp_dir / "expert_with_desc.toml"
    with open(expert_toml, "w") as f:
        f.write(
            """
        [model]
        name = "expert-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Expert program"
        """
        )

    return {
        "main_toml": main_toml,
        "helper_toml": helper_toml,
        "expert_toml": expert_toml,
    }


@pytest.fixture
def mock_nested_linked_programs(temp_dir):
    """Create a set of nested linked program TOML files for testing."""
    # Create main program TOML
    main_toml = temp_dir / "main_nested.toml"
    with open(main_toml, "w") as f:
        f.write(
            """
        [model]
        name = "main-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Main program"

        [tools]
        builtin = ["spawn"]

        [linked_programs]
        helper = "helper_nested.toml"
        expert = "expert_nested.toml"
        """
        )

    # Create helper program TOML that links to utility
    helper_toml = temp_dir / "helper_nested.toml"
    with open(helper_toml, "w") as f:
        f.write(
            """
        [model]
        name = "helper-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Helper program"

        [linked_programs]
        utility = "utility.toml"
        """
        )

    # Create expert program TOML
    expert_toml = temp_dir / "expert_nested.toml"
    with open(expert_toml, "w") as f:
        f.write(
            """
        [model]
        name = "expert-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Expert program"
        """
        )

    # Create utility program TOML
    utility_toml = temp_dir / "utility.toml"
    with open(utility_toml, "w") as f:
        f.write(
            """
        [model]
        name = "utility-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Utility program"
        """
        )

    return {
        "main_toml": main_toml,
        "helper_toml": helper_toml,
        "expert_toml": expert_toml,
        "utility_toml": utility_toml,
    }


def test_compile_basic_program_linking(mock_linked_programs):
    """Test compiling a program with basic linked programs."""
    # Compile the main program
    program = LLMProgram.from_toml(mock_linked_programs["main_toml"])
    program.compile()

    # Verify linked programs were loaded
    assert "helper" in program.linked_programs
    assert "expert" in program.linked_programs

    # Verify linked programs are LLMProgram instances
    assert isinstance(program.linked_programs["helper"], LLMProgram)
    assert isinstance(program.linked_programs["expert"], LLMProgram)

    # Verify linked program attributes
    assert program.linked_programs["helper"].model_name == "helper-model"
    assert program.linked_programs["expert"].model_name == "expert-model"
    assert program.linked_programs["helper"].system_prompt == "Helper program"
    assert program.linked_programs["expert"].system_prompt == "Expert program"


def test_compile_nested_programs(mock_nested_linked_programs):
    """Test compiling a program with nested linked programs."""
    # Compile the main program
    program = LLMProgram.from_toml(mock_nested_linked_programs["main_toml"])
    program.compile()

    # Verify first level linked programs were loaded
    assert "helper" in program.linked_programs
    assert "expert" in program.linked_programs

    # Verify nested linked programs
    helper_program = program.linked_programs["helper"]
    assert "utility" in helper_program.linked_programs
    assert helper_program.linked_programs["utility"].model_name == "utility-model"

    # Verify proper nesting structure
    assert program.linked_programs["helper"].linked_programs["utility"].system_prompt == "Utility program"


def test_program_linking_descriptions(mock_linked_programs_with_descriptions):
    """Test program linking descriptions functionality."""
    # Compile the main program with descriptions
    program = LLMProgram.from_toml(mock_linked_programs_with_descriptions["main_toml"])
    program.compile()

    # Verify descriptions were loaded
    assert hasattr(program, "linked_program_descriptions")
    assert "helper" in program.linked_program_descriptions
    assert "expert" in program.linked_program_descriptions

    # Verify description content
    assert program.linked_program_descriptions["helper"] == "A helper program that provides assistance"
    assert program.linked_program_descriptions["expert"] == "An expert program with specialized knowledge"


def test_program_linking_paths(temp_dir):
    """Test path resolution in program linking."""
    # Create subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir(exist_ok=True)

    # Create main program in root dir
    main_toml = temp_dir / "main_path.toml"
    with open(main_toml, "w") as f:
        f.write(
            """
        [model]
        name = "main-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Main program"

        [linked_programs]
        subdir_expert = "subdir/expert_path.toml"
        """
        )

    # Create expert program in subdirectory
    expert_toml = subdir / "expert_path.toml"
    with open(expert_toml, "w") as f:
        f.write(
            """
        [model]
        name = "expert-model"
        provider = "anthropic"

        [prompt]
        system_prompt = "Expert program"
        """
        )

    # Compile the main program
    program = LLMProgram.from_toml(main_toml)
    program.compile()

    # Verify linked program was found despite being in a subdirectory
    assert "subdir_expert" in program.linked_programs
    assert program.linked_programs["subdir_expert"].model_name == "expert-model"


@pytest.mark.asyncio
async def test_program_start_with_linked_programs(mock_linked_programs):
    """Test starting a process with linked programs."""
    # Compile the main program
    program = LLMProgram.from_toml(mock_linked_programs["main_toml"])
    program.compile()

    # Patch the get_provider_client function to avoid actual API calls
    with patch("llmproc.program_exec.get_provider_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Patch asyncio.create_subprocess_exec to avoid actual subprocess calls
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = AsyncMock()

            # Start the process
            process = await program.start()

            # Verify process has linked programs
            assert hasattr(process, "linked_programs")
            assert "helper" in process.linked_programs
            assert "expert" in process.linked_programs


@pytest.mark.asyncio
async def test_spawn_tool_in_linked_programs():
    """Test the spawn tool functionality for program linking."""
    # Create a mock for program_exec.create_process
    with patch("llmproc.program_exec.create_process") as mock_create_process:
        # Create a mock child process
        child_process = MagicMock()
        mock_run_result = RunResult()
        child_process.run = AsyncMock(return_value=mock_run_result)
        child_process.get_last_message = MagicMock(return_value="Child response")

        # Configure create_process to return our mock child process
        mock_create_process.return_value = child_process

        # Create a child program to link
        child_program = LLMProgram(model_name="child-model", provider="anthropic", system_prompt="Child system prompt")

        # Create a parent program
        parent_program = LLMProgram(
            model_name="parent-model", provider="anthropic", system_prompt="Parent system prompt"
        )

        # Import spawn tool function
        from llmproc.tools.builtin import spawn_tool

        # Link the programs and enable spawn tool (using function reference)
        parent_program.add_linked_program("expert", child_program)
        parent_program.register_tools([spawn_tool])

        # Create mock parent process with our test helper
        process = create_test_llmprocess_directly(
            program=parent_program, linked_programs={"expert": child_program}, has_linked_programs=True
        )

        # Test spawn_tool with runtime_context
        from llmproc.tools.builtin.spawn import spawn_tool

        result = await spawn_tool(
            program_name="expert",
            prompt="Test message",
            runtime_context={
                "process": process,
                "linked_programs": process.linked_programs,
            },
        )

        # Verify the spawn tool worked correctly
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert result.content == "Child response"

        # Verify create_process was called with the correct program
        mock_create_process.assert_called_once()
        assert mock_create_process.call_args[0][0] == child_program

        # Verify run was called on the child process with the query
        child_process.run.assert_called_once_with("Test message")


@pytest.mark.asyncio
async def test_process_with_linked_programs():
    """Test creating and using processes with linked programs."""
    # Create program
    program = LLMProgram(model_name="parent-model", provider="anthropic", system_prompt="Parent system prompt")

    # Create linked program
    linked_program = LLMProgram(model_name="child-model", provider="anthropic", system_prompt="Child system prompt")

    # Link the programs
    program.add_linked_program("child", linked_program)

    # Mock response from provider client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [{"type": "text", "text": "Mock response"}]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    # Start the process with mocked client
    with patch("llmproc.program_exec.get_provider_client", return_value=mock_client):
        process = await program.start()

        # Verify that the process has the linked program
        assert "child" in process.linked_programs

        # Verify access to the parent and child system prompts
        assert program.system_prompt == "Parent system prompt"
        assert linked_program.system_prompt == "Child system prompt"

        # We can't check process.linked_programs["child"].system_prompt directly
        # because linked_programs contains the program instances, not process instances
        # Instead, verify that the linked program reference is correct
        assert process.linked_programs["child"] == linked_program
