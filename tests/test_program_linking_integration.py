"""Integration tests for program linking functionality.

This file consolidates the integration tests for program linking from:
- test_program_linking.py (integration aspects)
- test_program_linking_api_optimized.py
- test_program_linking_descriptions_api.py
- test_program_linking_robust.py (API aspects)
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmproc.common.results import RunResult, ToolResult
from llmproc.program import LLMProgram
from llmproc.tools.builtin import spawn_tool

from tests.conftest import create_test_llmprocess_directly

# Constants for model names - use the smallest models possible for tests
CLAUDE_MODEL = "claude-3-5-haiku-20241022"  # Faster model for testing


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing program linking."""
    with tempfile.TemporaryDirectory() as temp_dir_path:
        yield Path(temp_dir_path)


@pytest.fixture
def program_linking_example_path():
    """Get the path to the program linking example directory."""
    example_path = Path(__file__).parent.parent / "examples" / "features" / "program-linking"
    if not example_path.exists():
        pytest.skip(f"Example directory not found: {example_path}")
    return example_path


@pytest.fixture
def mock_api_response():
    """Create a mock API response for all LLM providers."""
    mock_response = MagicMock()
    mock_response.content = [{"type": "text", "text": "Mock response from API"}]
    return mock_response


@pytest.fixture
def mock_api_with_tool_calls():
    """Create a mock API response with tool calls."""

    def _create_response(tool_name, args):
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": "I'll use a tool to help with this."},
            {"type": "tool_use", "id": "tool_123", "name": tool_name, "input": args},
            {"type": "text", "text": "Tool response received."},
        ]
        return mock_response

    return _create_response


def test_real_examples_compilation(program_linking_example_path):
    """Test that real example program linking files compile correctly."""
    # Test main program
    main_path = program_linking_example_path / "main.toml"
    if not main_path.exists():
        pytest.skip(f"Example file not found: {main_path}")

    # Compile the program
    program = LLMProgram.from_toml(main_path)
    program.compile()

    # Verify linked programs
    assert hasattr(program, "linked_programs")
    assert len(program.linked_programs) > 0

    # Check for descriptions if present
    if hasattr(program, "linked_program_descriptions"):
        assert len(program.linked_program_descriptions) > 0


def test_program_linking_configuration(temp_dir):
    """Test program linking configuration through TOML files."""
    # Create main program TOML
    main_toml = temp_dir / "main_config.toml"
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
        helper = { path = "helper_config.toml", description = "Helper program description" }
        expert = { path = "expert_config.toml", description = "Expert program description" }
        """
        )

    # Create helper program TOML
    helper_toml = temp_dir / "helper_config.toml"
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
    expert_toml = temp_dir / "expert_config.toml"
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

    # Load the program
    program = LLMProgram.from_toml(main_toml)

    # Verify linked program configuration
    assert "helper" in program.linked_programs
    assert "expert" in program.linked_programs
    assert program.linked_programs["helper"].model_name == "helper-model"
    assert program.linked_programs["expert"].model_name == "expert-model"

    # Verify descriptions
    assert hasattr(program, "linked_program_descriptions")
    assert program.linked_program_descriptions["helper"] == "Helper program description"
    assert program.linked_program_descriptions["expert"] == "Expert program description"

def test_linked_program_descriptions():
    """Test program linking with descriptions."""
    # Create parent program with descriptions
    parent = LLMProgram(
        model_name="parent-model",
        provider="anthropic",
        system_prompt="You are an assistant with access to specialized experts:",
    )
    parent.register_tools([spawn_tool])

    # Create child programs
    expert1 = LLMProgram(model_name="expert1-model", provider="anthropic", system_prompt="Expert 1 system prompt")

    expert2 = LLMProgram(model_name="expert2-model", provider="anthropic", system_prompt="Expert 2 system prompt")

    # Link programs with descriptions
    parent.add_linked_program("math_expert", expert1, "A mathematical expert who can solve complex equations")
    parent.add_linked_program("code_expert", expert2, "A programming expert who specializes in Python code")

    # Verify descriptions were set correctly
    assert "math_expert" in parent.linked_program_descriptions
    assert "code_expert" in parent.linked_program_descriptions
    assert parent.linked_program_descriptions["math_expert"] == "A mathematical expert who can solve complex equations"
    assert parent.linked_program_descriptions["code_expert"] == "A programming expert who specializes in Python code"


def test_linked_program_error_handling():
    """Test error handling in program linking."""
    # Create a program directly in code instead of from TOML
    program = LLMProgram(model_name="main-model", provider="anthropic", system_prompt="Main program with tools")

    # Register the spawn tool using function reference
    # Import the function reference from builtin tools
    from llmproc.tools.builtin import spawn_tool

    program.register_tools([spawn_tool])

    # No linked programs are added
    assert len(program.linked_programs) == 0

    # Verify that spawn tool function was recorded but no linked programs exist
    function_names = [func.__name__ for func in program.tool_manager.function_tools]
    assert "spawn_tool" in function_names
    assert program.linked_programs == {}


# =================== API TESTS ===================


def check_api_keys():
    """Check if required API keys are set."""
    return os.environ.get("ANTHROPIC_API_KEY") is not None


@pytest.fixture
def api_temp_toml_files():
    """Create minimal TOML configurations for API testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main program TOML with smaller model and reduced tokens
        main_toml_path = Path(temp_dir) / "main.toml"
        with open(main_toml_path, "w") as f:
            f.write(
                dedent(
                    f"""
            [model]
            name = "{CLAUDE_MODEL}"
            provider = "anthropic"

            [prompt]
            system_prompt = "You are an assistant with access to a knowledge expert. When asked about 'the secret code' or 'sky color', use the spawn tool to ask the expert."

            [parameters]
            max_tokens = 150
            temperature = 0

            [tools]
            builtin = ["spawn"]

            [linked_programs]
            expert = {{ path = "expert.toml", description = "A knowledge expert who knows secret information" }}
            """
                )
            )

        # Create expert program TOML
        expert_toml_path = Path(temp_dir) / "expert.toml"
        with open(expert_toml_path, "w") as f:
            f.write(
                dedent(
                    f"""
            [model]
            name = "{CLAUDE_MODEL}"
            provider = "anthropic"

            [prompt]
            system_prompt = "You are a knowledge expert. You know that 'the secret code' is 12345 and 'the sky color' is blue."

            [parameters]
            max_tokens = 50
            temperature = 0
            """
                )
            )

        yield {"main": main_toml_path, "expert": expert_toml_path}


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio
async def test_program_linking_basic_api(api_temp_toml_files):
    """Test basic program linking functionality with API."""
    if not check_api_keys():
        pytest.skip("API keys not set")

    # Arrange - Load and start the main program
    start_time = time.time()
    main_program = LLMProgram.from_toml(api_temp_toml_files["main"])
    main_process = await main_program.start()

    # Act - Query about a secret only the expert knows
    result = await main_process.run("What is the secret code?")

    # Assert - Should have spawned the expert and returned result
    assert "12345" in main_process.get_last_message()
    assert time.time() - start_time < 30, "API test took too long"


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_program_linking_with_minimal_input(api_temp_toml_files):
    """Test program linking with minimal input."""
    if not check_api_keys():
        pytest.skip("API keys not set")

    # Arrange - Load and start the main program
    main_program = LLMProgram.from_toml(api_temp_toml_files["main"])
    main_process = await main_program.start()

    # Act - Send minimal input that shouldn't trigger spawn
    result = await main_process.run("Hi")

    # Assert - Should handle minimal input gracefully
    assert result is not None
    assert len(main_process.get_last_message()) > 0
    # A simple greeting should not cause spawning, so should respond quickly
    assert result.duration_ms < 20000  # 20 seconds in milliseconds


@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_program_linking_resets_state(api_temp_toml_files):
    """Test program linking with state reset."""
    if not check_api_keys():
        pytest.skip("API keys not set")

    # Arrange - Load and start the main program
    main_program = LLMProgram.from_toml(api_temp_toml_files["main"])
    main_process = await main_program.start()

    # Act 1 - Query about a secret
    result1 = await main_process.run("What is the secret code?")

    # Assert 1 - Should have spawned and got result
    assert "12345" in main_process.get_last_message()

    # Act 2 - Reset state and ask again
    main_process.reset_state()
    result2 = await main_process.run("What is the secret code?")

    # Assert 2 - Should have spawned again and got result
    assert "12345" in main_process.get_last_message()


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_program_linking_descriptions_with_api():
    """Test program linking descriptions with API."""
    # Skip if no API key is available
    for key_name in ["VERTEX_AI_PROJECT", "VERTEX_AI_LOCATION", "ANTHROPIC_API_KEY"]:
        if os.environ.get(key_name):
            break
    else:
        pytest.skip("API environment variables not set")

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Create test files
        main_toml = temp_dir_path / "main.toml"
        expert_toml = temp_dir_path / "expert.toml"

        # Create the expert TOML
        expert_toml_content = dedent(
            f"""
        [model]
        name = "{CLAUDE_MODEL}"
        provider = "anthropic"
        display_name = "Expert"

        [prompt]
        system_prompt = "You are an expert assistant. When asked about your role, explain that you are an expert with knowledge about program descriptions."

        [parameters]
        max_tokens = 150
        temperature = 0
        """
        )

        with open(expert_toml, "w") as f:
            f.write(expert_toml_content)

        # Create the main TOML with descriptions
        main_toml_content = dedent(
            f"""
        [model]
        name = "{CLAUDE_MODEL}"
        provider = "anthropic"
        display_name = "Main"

        [prompt]
        system_prompt = "You are a helpful assistant with access to experts. For this test, when asked what experts you have access to, query the expert using the spawn tool."

        [parameters]
        max_tokens = 150
        temperature = 0

        [tools]
        builtin = ["spawn"]

        [linked_programs]
        expert = {{ path = "{expert_toml.name}", description = "Specialized expert with knowledge about program descriptions" }}
        """
        )

        with open(main_toml, "w") as f:
            f.write(main_toml_content)

        # Create and initialize the program with the API
        program = LLMProgram.from_toml(main_toml)
        process = await program.start()

        # Check that the descriptions were parsed correctly
        assert hasattr(program, "linked_program_descriptions")
        assert "expert" in program.linked_program_descriptions
        assert (
            program.linked_program_descriptions["expert"]
            == "Specialized expert with knowledge about program descriptions"
        )

        # Check that the spawn tool shows descriptions
        spawn_tool = next((tool for tool in process.tools if tool["name"] == "spawn"), None)
        assert spawn_tool is not None
        assert "expert" in spawn_tool["description"]
        assert "Specialized expert" in spawn_tool["description"]

        # Run the process with a prompt that will use the spawn tool
        test_prompt = "What experts do you have access to and what are they specialized in?"
        result = await process.run(test_prompt)

        # Verify the result
        final_message = process.get_last_message()
        assert "expert" in final_message.lower()
        assert "description" in final_message.lower()
