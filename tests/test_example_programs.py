"""Tests that exercise each example program file with actual LLM APIs."""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import tomli

from llmproc import LLMProcess, LLMProgram


def get_example_programs():
    """Get all example TOML program files as test params."""
    base_dir = Path(__file__).parent.parent / "examples"
    programs = []

    # Get all .toml files recursively
    for program_file in base_dir.glob("**/*.toml"):
        # Skip reference file and any scripts
        if program_file.name not in ["reference.toml", "scripts"]:
            programs.append(program_file.relative_to(base_dir.parent))

    return programs


def api_keys_available():
    """Check if required API keys are available."""
    has_openai = "OPENAI_API_KEY" in os.environ
    has_anthropic = "ANTHROPIC_API_KEY" in os.environ
    has_vertex = (
        "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        or "GOOGLE_CLOUD_PROJECT" in os.environ
    )

    return has_openai and has_anthropic and has_vertex


def test_test_structure():
    """Test the test structure itself to verify program paths."""
    # Verify example programs exist
    example_programs = get_example_programs()
    assert len(example_programs) > 5, "Expected at least 5 example programs"

    # Verify paths exist
    for program_path in example_programs:
        full_path = Path(__file__).parent.parent / program_path
        assert full_path.exists(), f"Example program {program_path} does not exist"

    # Known files with special syntax that aren't standard TOML
    skip_files = [
        "claude-code.toml",  # Uses a complex linked_programs syntax
        "main.toml",  # Uses a complex linked_programs syntax in program-linking folder
    ]

    # Check that each program is valid TOML
    for program_path in example_programs:
        # Skip known problematic files
        if program_path.name in skip_files:
            continue

        full_path = Path(__file__).parent.parent / program_path
        with open(full_path, "rb") as f:
            try:
                program = tomli.load(f)
                # Basic validation of required fields
                # Check for model configuration in either the root or in a model section
                has_model_info = (
                    "model_name" in program and "provider" in program
                ) or (
                    "model" in program
                    and "name" in program["model"]
                    and "provider" in program["model"]
                )
                assert has_model_info, (
                    f"Program {program_path} missing model information"
                )
            except tomli.TOMLDecodeError as e:
                pytest.fail(f"Invalid TOML in {program_path}: {e}")


def get_provider_from_program(program_path):
    """Extract provider from a TOML program file."""
    with open(program_path, "rb") as f:
        program = tomli.load(f)

    # Check in both root level and model section
    if "provider" in program:
        return program.get("provider")
    elif "model" in program and "provider" in program["model"]:
        return program["model"].get("provider")
    return ""


# Mark tests as requiring API access
@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
@pytest.mark.parametrize("program_path", get_example_programs())
async def test_example_program(program_path):
    """Test an example program with the actual LLM API."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Skip certain providers if you need to
    provider = get_provider_from_program(program_path)
    if (
        provider in ["anthropic_vertex", "gemini_vertex"]
        and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    ):
        pytest.skip("Vertex AI credentials not available")

    # Create and start process using two-step pattern
    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Send a simple test query
    test_query = (
        "Respond with a short one-sentence confirmation that you received this message."
    )

    # Run the process and get the response
    result = await process.run(test_query)

    # Get response from RunResult
    response = process.get_last_message()

    # Verify we got a response (we don't check exact content as it varies by model)
    assert response
    assert isinstance(response, str)
    assert len(response) > 10


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
async def test_minimal_functionality():
    """Test basic LLMProcess functionality with any model."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use the more reliable example program
    program_path = (
        Path(__file__).parent.parent / "examples" / "openai" / "gpt-4o-mini.toml"
    )
    if not program_path.exists():
        program_path = (
            Path(__file__).parent.parent
            / "examples"
            / "anthropic"
            / "claude-3-5-haiku.toml"
        )

    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Test with a simple, deterministic prompt
    result = await process.run("Please respond with exactly: 'Hello, World!'")
    response = process.get_last_message()
    assert "Hello, World" in response, "Expected model to respond with greeting"


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
async def test_mcp_tool_functionality():
    """Test mcp.toml with tool execution functionality."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use the MCP example
    program_path = Path(__file__).parent.parent / "examples" / "features" / "mcp.toml"
    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Send a simple request that should trigger a tool
    result = await process.run(
        "Tell me what tools you have access to. Just list the tool names."
    )
    response = process.get_last_message()

    # Check if response includes tool information
    assert any(
        term in response.lower() for term in ["tool", "function", "capability"]
    ), "Expected model to mention tools in response"


@pytest.mark.llm_api
@pytest.mark.release_api  # This test is already covered by test_program_linking_api_optimized.py in the extended tier
@pytest.mark.skip(
    reason="Covered by test_program_linking_basic in test_program_linking_api_optimized.py"
)
@pytest.mark.asyncio
async def test_program_linking_functionality():
    """Test program_linking/main.toml with spawn tool execution.

    NOTE: This test is redundant and is skipped. The same functionality is tested more
    efficiently in test_program_linking_api_optimized.py. See STRATEGIC_TESTING.md for details.
    """
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent
        / "examples"
        / "features"
        / "program-linking"
        / "main.toml"
    )
    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Send a simple query that should use the spawn tool to delegate to repo_expert
    result = await process.run("Ask the repo expert to say 'Hello, World!'")
    response = process.get_last_message()

    # Check for common terms that should appear in a response
    assert "hello" in response.lower() and "world" in response.lower(), (
        "Expected response to include 'Hello, World!'"
    )

    # We don't need to verify the actual tool usage in the message content
    # Just check that we got a valid response that matches our expected text
    assert response, "Expected a non-empty response"


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
async def test_file_preload_functionality():
    """Test preload.toml with file preloading functionality."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent / "examples" / "features" / "preload.toml"
    )
    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Ask about content that should be in the preloaded files
    result = await process.run(
        "Based on the information preloaded from the README.md file, what is the purpose of the llmproc library? Keep your response under 50 words."
    )
    response = process.get_last_message()

    # Check for terms that should be in the response based on README content
    assert any(
        term in response.lower() for term in ["api", "llm", "process", "interface"]
    ), "Expected response to reference content from preloaded README.md"


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
async def test_claude_code_comprehensive():
    """Test claude-code.toml with comprehensive features including tools and preloaded content."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent / "examples" / "claude-code" / "claude-code.toml"
    )
    program = LLMProgram.from_toml(program_path)
    process = await program.start()

    # Use a simple prompt to test basic functionality
    preload_result = await process.run("Please say 'Hello, Claude!'")

    # Get response from RunResult
    preload_response = process.get_last_message()

    # Check for expected greeting
    assert "hello" in preload_response.lower(), "Expected response to include greeting"

    # Test a simpler tool execution - just mention that tools exist
    tool_result = await process.run(
        "What tools do you have access to? Just list them briefly."
    )

    # Get response from RunResult
    tool_response = process.get_last_message()

    # Just check for a non-empty response since tools may be named differently
    assert tool_response and len(tool_response) > 10, (
        "Expected a substantive response about tools"
    )


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.asyncio
async def test_provider_specific_functionality():
    """Test each provider with their specific example programs."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # List of example programs to test
    provider_programs = [
        "openai/gpt-4o-mini.toml",
        "anthropic/claude-3-5-haiku.toml",
        "anthropic/claude-3-5-haiku-vertex.toml",
    ]

    for program_name in provider_programs:
        # Skip anthropic_vertex test if credentials aren't available
        if (
            "vertex" in program_name
            and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
        ):
            continue

        program_path = Path(__file__).parent.parent / "examples" / program_name
        program = LLMProgram.from_toml(program_path)
        process = await program.start()

        # Use a simple test prompt that should work with any provider
        test_prompt = "Echo this unique identifier: TEST_CONNECTION_SUCCESS_12345"
        result = await process.run(test_prompt)
        response = process.get_last_message()

        # Verify that we got a response (not empty) and it contains our test identifier
        assert response, f"Expected non-empty response from {program_name}"
        assert "TEST_CONNECTION_SUCCESS_12345" in response, (
            f"Expected echo response from {program_name} to include the test identifier"
        )


def run_cli_with_input(
    program_path: Path, input_text: str, timeout: int | None = 30
) -> str:
    """Run the llmproc-demo CLI with a program file and input text.

    Args:
        program_path: Path to the TOML program file
        input_text: Text to send to the CLI as user input
        timeout: Maximum time to wait for the process to complete

    Returns:
        The CLI output as a string
    """
    # Create a temporary file for the input
    with tempfile.NamedTemporaryFile("w+") as input_file:
        # Write the input text followed by the exit command
        input_file.write(f"{input_text}\nexit\n")
        input_file.flush()
        input_file.seek(0)

        # Use subprocess to run the CLI with the input file
        cmd = [sys.executable, "-m", "llmproc.cli", str(program_path)]
        result = subprocess.run(
            cmd,
            stdin=input_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        # If the command failed, print the error
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
            print(f"STDERR: {result.stderr}")

        return result.stdout


@pytest.mark.llm_api
@pytest.mark.release_api
def test_cli_with_minimal_example():
    """Test the CLI with a simple example program."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent / "examples" / "openai" / "gpt-4o-mini.toml"
    )
    if not program_path.exists():
        program_path = (
            Path(__file__).parent.parent
            / "examples"
            / "anthropic"
            / "claude-3-5-haiku.toml"
        )

    # Use a unique test string that the model should echo back
    unique_test_string = "TEST_STRING_XYZ123_ECHO_THIS_BACK"
    prompt = f"Please echo back this exact string: {unique_test_string}"

    # Run the CLI with the test prompt
    output = run_cli_with_input(program_path, prompt)

    # Check if the CLI ran successfully and echoed back our test string
    assert unique_test_string in output, (
        f"Expected CLI output to echo back the test string: {unique_test_string}"
    )

    # Check if program information is shown
    assert any(term in output for term in ["Program Summary", "Configuration"]), (
        "Expected CLI to show program information"
    )


@pytest.mark.llm_api
@pytest.mark.release_api
def test_cli_with_program_linking():
    """Test the CLI with program linking example."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent
        / "examples"
        / "features"
        / "program-linking"
        / "main.toml"
    )

    # Run the CLI with a simple query that should trigger the spawn tool
    output = run_cli_with_input(
        program_path,
        "Ask the repo expert to say 'Hello, World!'",
        timeout=60,  # Longer timeout for program linking
    )

    # Check if the response includes the expected greeting
    assert "hello" in output.lower() and "world" in output.lower(), (
        "Expected CLI output to include greeting"
    )


@pytest.mark.llm_api
@pytest.mark.release_api
@pytest.mark.parametrize(
    "program_name",
    [
        "anthropic/claude-3-5-haiku.toml",
        "openai/gpt-4o-mini.toml",
        "claude-code/claude-code.toml",
        "features/mcp.toml",
        "features/preload.toml",
        pytest.param(
            "anthropic/claude-3-5-haiku-vertex.toml",
            marks=pytest.mark.skipif(
                "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ,
                reason="Vertex AI credentials not available",
            ),
        ),
    ],
)
def test_cli_with_all_programs(program_name):
    """Test CLI with all example programs."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / program_name

    try:
        # Create a unique identifier for this test run
        unique_id = f"ECHO_TEST_{program_name.replace('.', '_').upper()}_ID123"

        # Run with a command for the model to echo our unique string
        output = run_cli_with_input(
            program_path,
            f"Respond with exactly this string: '{unique_id}' - nothing else.",
            timeout=45,
        )

        # Check for our unique test string
        assert unique_id in output, (
            f"Expected CLI using {program_name} to echo back '{unique_id}'"
        )
        assert any(term in output for term in ["Program Summary", "Configuration"]), (
            f"Expected CLI using {program_name} to show program information"
        )

    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI with {program_name} timed out")
    except subprocess.SubprocessError as e:
        pytest.fail(f"CLI with {program_name} failed: {e}")


@pytest.mark.llm_api
@pytest.mark.release_api
def test_error_handling_and_recovery():
    """Test error handling and recovery with an invalid and valid program."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # First create a temporary invalid program
    with tempfile.NamedTemporaryFile("w+", suffix=".toml") as invalid_program:
        invalid_program.write("""
        [invalid]
        this_is_not_valid = true

        model_name = "nonexistent-model"
        provider = "unknown"
        """)
        invalid_program.flush()

        # Try to run with invalid program (should return non-zero)
        cmd = [sys.executable, "-m", "llmproc.cli", invalid_program.name]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            input="test\nexit\n",
            timeout=10,
        )

        # Verify error is reported
        assert result.returncode != 0, (
            "Expected non-zero return code for invalid program"
        )
        assert "error" in result.stderr.lower() or "error" in result.stdout.lower(), (
            "Expected error message for invalid program"
        )

    # Now test with a valid program to make sure the system recovers
    program_path = (
        Path(__file__).parent.parent / "examples" / "openai" / "gpt-4o-mini.toml"
    )
    if not program_path.exists():
        program_path = (
            Path(__file__).parent.parent
            / "examples"
            / "anthropic"
            / "claude-3-5-haiku.toml"
        )

    output = run_cli_with_input(program_path, "Say hello.")

    # Check for success
    assert "hello" in output.lower(), (
        "Expected successful response after error recovery"
    )
