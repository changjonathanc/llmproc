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

from llmproc import LLMProcess


def get_example_programs():
    """Get all example TOML program files as test params."""
    base_dir = Path(__file__).parent.parent / "examples"
    programs = []

    # Add standalone example programs
    for program_file in base_dir.glob("*.toml"):
        if program_file.name != "reference.toml":  # Skip reference file
            programs.append(program_file.relative_to(base_dir.parent))

    # Add program linking example programs
    linking_dir = base_dir / "program_linking"
    for program_file in linking_dir.glob("*.toml"):
        if program_file.name != "README.md":  # Skip README
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

    # Check that each program is valid TOML
    for program_path in example_programs:
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
@pytest.mark.asyncio
@pytest.mark.parametrize("program_path", get_example_programs())
async def test_example_program(program_path):
    """Test an example program with the actual LLM API."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Skip certain providers if you need to
    provider = get_provider_from_program(program_path)
    if provider == "anthropic_vertex" and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        pytest.skip("Vertex AI credentials not available")

    # Create LLMProcess from TOML program file
    process = LLMProcess.from_toml(program_path)

    # Send a simple test query
    test_query = (
        "Respond with a short one-sentence confirmation that you received this message."
    )

    # Run the process and get the response
    response = await process.run(test_query)

    # Verify we got a response (we don't check exact content as it varies by model)
    assert response
    assert isinstance(response, str)
    assert len(response) > 10


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_minimal_functionality():
    """Test minimal.toml with basic functionality checks."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / "minimal.toml"
    process = LLMProcess.from_toml(program_path)

    # Test basic Q&A functionality
    response = await process.run("What is 2+2?")
    assert "4" in response.lower(), "Expected model to answer basic math question"

    # Test conversation continuity
    response = await process.run("What was my previous question?")
    assert "2" in response and "+" in response, (
        "Expected model to remember previous question"
    )

    # Test reset functionality
    process.reset_state()
    response = await process.run("What was my previous question?")
    assert (
        "previous question" not in response.lower()
        or "don't" in response.lower()
        or "no" in response.lower()
    ), "Expected model to not remember questions after reset"


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_mcp_tool_functionality():
    """Test mcp.toml with tool execution functionality."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use the more reliable time-only example
    program_path = Path(__file__).parent.parent / "examples" / "mcp_time.toml"
    process = LLMProcess.from_toml(program_path)

    # Send a request that should trigger the time tool
    response = await process.run(
        "What is the current time? Use the time tool to tell me."
    )

    # Check if response includes time information
    assert any(
        term in response.lower() for term in ["utc", "gmt", "time", "hour", "minute"]
    ), "Expected model to use the time tool to get current time information"


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_program_linking_functionality():
    """Test program_linking/main.toml with spawn tool execution."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent / "examples" / "program_linking" / "main.toml"
    )
    process = LLMProcess.from_toml(program_path)

    # Send a query that should use the spawn tool to delegate to repo_expert
    response = await process.run(
        "Ask the repo expert to describe what the llmproc package does. "
        "Keep the response under 100 words."
    )

    # Check for common terms that should appear in a description of llmproc
    assert "llm" in response.lower() and (
        "process" in response.lower() or "api" in response.lower()
    ), "Expected response about llmproc package functionality"

    # Verify the spawn tool was used
    state = process.get_state()
    tool_usage_found = False
    for message in state:
        if message.get("role") == "assistant" and message.get("content"):
            if (
                "spawn" in message.get("content").lower()
                and "repo_expert" in message.get("content").lower()
            ):
                tool_usage_found = True
                break

    assert tool_usage_found, "Expected to find evidence of spawn tool usage in state"


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_file_preload_functionality():
    """Test preload.toml with file preloading functionality."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / "preload.toml"
    process = LLMProcess.from_toml(program_path)

    # Ask about content that should be in the preloaded files
    response = await process.run(
        "Based on the information preloaded from the README.md file, what is the purpose of the llmproc library? "
        "Keep your response under 50 words."
    )

    # Check for terms that should be in the response based on README content
    assert any(
        term in response.lower() for term in ["api", "llm", "process", "interface"]
    ), "Expected response to reference content from preloaded README.md"


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_claude_code_comprehensive():
    """Test claude_code.toml with comprehensive features including tools and preloaded content."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / "claude_code.toml"
    process = LLMProcess.from_toml(program_path)

    # Test preloaded file knowledge
    preload_response = await process.run(
        "Based only on the preloaded content in your system prompt, "
        "what are the main components of the llmproc package? "
        "Keep your answer short and reference specific files."
    )

    # Check for specific components from the codebase files
    assert any(
        term in preload_response.lower()
        for term in ["llm_process", "providers", "tools"]
    ), "Expected response to reference components from preloaded content"

    # Test tool execution
    tool_response = await process.run(
        "Use the dispatch_agent tool to search for the classes defined in the llm_process.py file. "
        "Provide just the class names found, separated by commas."
    )

    # Check for LLMProcess class in the response
    assert "LLMProcess" in tool_response, (
        "Expected response to list the LLMProcess class from using tools"
    )

    # Test combined functionality - using preloaded content to drive tool usage
    combined_response = await process.run(
        "Based on what you know about the repository structure (from preloaded content), "
        "use the dispatch_agent tool to look at one of the example TOML programs and tell me "
        "what type of provider it configures. Keep your response under 30 words."
    )

    # Either the model references an example or identifies a provider
    assert any(
        term in combined_response.lower()
        for term in ["anthropic", "openai", "vertex", "claude", "gpt", "example"]
    ), "Expected response to identify provider from example programs using tools"


@pytest.mark.llm_api
@pytest.mark.asyncio
async def test_provider_specific_functionality():
    """Test each provider with their specific example programs."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Dictionary mapping provider programs to expected terms in responses
    provider_programs = {
        "openai.toml": ["gpt", "openai"],
        "anthropic.toml": ["claude", "anthropic", "haiku"],
        "anthropic_vertex.toml": ["vertex", "claude", "google"],
    }

    for program_name, expected_terms in provider_programs.items():
        # Skip anthropic_vertex test if credentials aren't available
        if (
            program_name == "anthropic_vertex.toml"
            and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
        ):
            continue

        program_path = Path(__file__).parent.parent / "examples" / program_name
        process = LLMProcess.from_toml(program_path)

        # Ask about the model identity
        response = await process.run(
            "Please identify yourself: what model are you and what provider are you from? "
            "Answer in a single short sentence."
        )

        # Check if any expected terms are in the response
        assert any(term in response.lower() for term in expected_terms), (
            f"Expected response from {program_name} to reference {expected_terms}"
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
def test_cli_with_minimal_example():
    """Test the CLI with the minimal example program."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / "minimal.toml"

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
    assert "Configuration" in output, "Expected CLI to show program information"
    assert "GPT-4o-mini" in output, "Expected CLI to show model name"


@pytest.mark.llm_api
def test_cli_with_program_linking():
    """Test the CLI with program linking example."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = (
        Path(__file__).parent.parent / "examples" / "program_linking" / "main.toml"
    )

    # Run the CLI with a query that should trigger the spawn tool
    output = run_cli_with_input(
        program_path,
        "Ask the repo expert: what is the main class in llmproc?",
        timeout=60,  # Longer timeout for program linking
    )

    # Check if the response includes information about LLMProcess
    assert "LLMProcess" in output, "Expected CLI output to mention LLMProcess class"


@pytest.mark.llm_api
@pytest.mark.parametrize(
    "program_name",
    [
        "minimal.toml",
        "anthropic.toml",
        "openai.toml",
        "claude_code.toml",
        "mcp.toml",
        "preload.toml",
        pytest.param(
            "anthropic_vertex.toml",
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
        assert "Configuration" in output, (
            f"Expected CLI using {program_name} to show program information"
        )

    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI with {program_name} timed out")
    except subprocess.SubprocessError as e:
        pytest.fail(f"CLI with {program_name} failed: {e}")


@pytest.mark.llm_api
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
            stdin=subprocess.PIPE,
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
    program_path = Path(__file__).parent.parent / "examples" / "minimal.toml"
    output = run_cli_with_input(program_path, "Say hello.")

    # Check for success
    assert "hello" in output.lower(), (
        "Expected successful response after error recovery"
    )
