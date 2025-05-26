#!/usr/bin/env python3
"""Consolidated CLI tests for the llmproc-demo command.

This file combines tests from test_cli_non_interactive.py and test_direct_cli_commands.py
to provide comprehensive CLI testing with reduced redundancy.
"""

import os
import subprocess

# Import path helpers from conftest using absolute import
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from conftest import get_example_file_path, get_repo_root


def api_keys_available():
    """Check if required API keys are available."""
    # Check for presence of keys
    has_openai = "OPENAI_API_KEY" in os.environ
    has_anthropic = "ANTHROPIC_API_KEY" in os.environ
    has_vertex = "GOOGLE_APPLICATION_CREDENTIALS" in os.environ or "GOOGLE_CLOUD_PROJECT" in os.environ

    # Additionally check if the keys are valid (not "None" or empty)
    if has_openai and (not os.environ["OPENAI_API_KEY"] or "None" in os.environ["OPENAI_API_KEY"]):
        has_openai = False
    if has_anthropic and (not os.environ["ANTHROPIC_API_KEY"] or "None" in os.environ["ANTHROPIC_API_KEY"]):
        has_anthropic = False

    # For test purposes, if environment has any valid key, consider it available
    return has_openai or has_anthropic or has_vertex


def get_available_program_path():
    """Get a path to an available program based on API keys."""
    if "ANTHROPIC_API_KEY" in os.environ and "None" not in os.environ["ANTHROPIC_API_KEY"]:
        return Path(get_example_file_path("anthropic.toml"))
    elif "OPENAI_API_KEY" in os.environ and "None" not in os.environ["OPENAI_API_KEY"]:
        return Path(get_example_file_path("openai.toml"))
    else:
        return None


def run_cli_command(args, input_text=None, timeout=45):
    """Run the CLI command with specified arguments.

    Args:
        args: List of command arguments
        input_text: Optional text to send to stdin
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            args,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"


def run_prompt_option(program_path, prompt, timeout=45):
    """Run CLI with --prompt option.

    Args:
        program_path: Path to the program file
        prompt: The prompt to send
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [sys.executable, "-m", "llmproc.cli.run", str(program_path), "--prompt", prompt]
    return run_cli_command(cmd, timeout=timeout)


def run_non_interactive_option(program_path, input_text="Hello from stdin\n", timeout=45):
    """Run CLI with stdin input.

    Args:
        program_path: Path to the program file
        input_text: Text to send to stdin
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [sys.executable, "-m", "llmproc.cli.run", str(program_path)]
    return run_cli_command(cmd, input_text=input_text, timeout=timeout)


def run_exact_command(command, timeout=45):
    """Run an exact command string.

    Args:
        command: The command string to run
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    parts = command.split()

    # Replace 'llmproc-demo' with the actual module invocation
    if parts[0] == "llmproc-demo":
        parts[0] = sys.executable
        parts.insert(1, "-m")
        parts.insert(2, "llmproc.cli")
    elif parts[0] == "llmproc":
        parts[0] = sys.executable
        parts.insert(1, "-m")
        parts.insert(2, "llmproc.cli.run")

    return run_cli_command(parts, timeout=timeout)


# Basic interactive mode tests
@pytest.mark.llm_api
@pytest.mark.essential_api
def test_cli_prompt_option_outputs_marker():
    """Test the --prompt option with an example program."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Create a unique test marker
    unique_marker = f"UNIQUE_TEST_MARKER_{Path(program_path).stem.upper()}"
    prompt = f"Reply with this exact marker: {unique_marker}"

    # Run CLI with --prompt option
    return_code, stdout, stderr = run_prompt_option(program_path, prompt)

    # Check for successful execution
    assert return_code == 0, f"CLI exited with error code {return_code}. Stderr: {stderr}"

    # Check for the unique marker in the output
    assert unique_marker in stdout, f"Expected unique marker '{unique_marker}' in output, but it wasn't found"


@pytest.mark.llm_api
@pytest.mark.essential_api
def test_cli_reads_stdin():
    """Test piping input to the non-interactive CLI."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Run CLI with stdin input
    return_code, stdout, stderr = run_non_interactive_option(program_path)

    # Check for successful execution
    assert return_code == 0, f"CLI exited with error code {return_code}. Stderr: {stderr}"

    # Check for some output (can't check specific content as it depends on the model)
    assert len(stdout) > 0, "Expected some output, but got empty response"


# CLI format tests
@pytest.mark.llm_api
@pytest.mark.essential_api
def test_complex_prompt_with_quotes():
    """Test a complex prompt with quotes in it."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Use direct list of arguments to properly handle complex quoting
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli.run",
        str(program_path),
        "--prompt",
        'Define the term "machine learning" in one sentence.',
    ]

    # Run command directly
    return_code, stdout, stderr = run_cli_command(cmd)

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"

    # Look for expected terms in the response
    expected_terms = ["machine", "learning", "algorithm"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]
    assert len(found_terms) > 0, f"Expected output to contain at least one of {expected_terms}"


@pytest.mark.llm_api
@pytest.mark.essential_api
def test_stdin_pipe_with_stdin():
    """Test piping input to the CLI."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Create command for piping input
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli.run",
        str(program_path),
    ]

    # Run with simulated stdin pipe using a simple, deterministic prompt
    return_code, stdout, stderr = run_cli_command(
        cmd,
        input_text="Say 'Hello world' exactly like that.",
    )

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"

    # Check for exact expected phrase
    assert "Hello world" in stdout, "Expected output to contain 'Hello world'"


# Feature tests
@pytest.mark.llm_api
@pytest.mark.extended_api
def test_tool_usage():
    """Test that tools work correctly through CLI."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Simple prompt that should use calculator tool
    prompt = "What is 2+2?"

    # Run CLI with --prompt option
    return_code, stdout, stderr = run_prompt_option(program_path, prompt, timeout=90)

    # Check for successful execution
    assert return_code == 0, f"CLI exited with error code {return_code}. Stderr: {stderr}"

    # Check for expected answer
    assert "4" in stdout, "Expected output to contain the answer '4'"


@pytest.mark.llm_api
@pytest.mark.release_api
def test_program_linking():
    """Test program linking through CLI."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Test with program-linking/main.toml
    program_path = Path(get_example_file_path("program-linking/main.toml"))

    # Prompt that should trigger the spawn tool
    prompt = "Ask the repo expert what files are in src/llmproc"

    # Run CLI with --prompt option (longer timeout for program linking)
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli.run",
        str(program_path),
        "--prompt",
        prompt,
    ]

    return_code, stdout, stderr = run_cli_command(cmd, timeout=120)

    # Check for successful execution
    assert return_code == 0, f"CLI exited with error code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"


# Error handling tests
@pytest.mark.llm_api
@pytest.mark.essential_api
def test_cli_handles_invalid_program():
    """Test handling of invalid program."""
    # Create a temporary invalid program file
    with tempfile.NamedTemporaryFile("w+", suffix=".toml") as invalid_program:
        invalid_program.write(
            """
        [invalid]
        model_name = "nonexistent"
        provider = "unknown"
        """
        )
        invalid_program.flush()

        # Run CLI with invalid program
        return_code, stdout, stderr = run_prompt_option(invalid_program.name, "test")

        # Should exit with non-zero return code
        assert return_code != 0, "Expected non-zero return code for invalid program"

        # Should provide an error message
        assert "error" in (stdout + stderr).lower(), "Expected error message for invalid program"


@pytest.mark.llm_api
@pytest.mark.essential_api
def test_empty_prompt_error():
    """Test that empty prompts cause appropriate error message and exit code."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = get_available_program_path()
    if not program_path:
        pytest.skip("No API keys available for testing")

    # Run CLI with empty prompt
    return_code, stdout, stderr = run_prompt_option(program_path, "")

    # Should exit with error code when given empty prompt
    assert return_code != 0, "CLI should exit with error when given empty prompt"

    # Should provide error message in either stdout or stderr
    combined_output = (stdout + stderr).lower()
    assert "empty prompt" in combined_output, "Error message should mention empty prompt"


# Utility tests (no API required)
def test_help_option():
    """Test the help command for the non-interactive CLI."""
    cmd = [sys.executable, "-m", "llmproc.cli.run", "--help"]
    return_code, stdout, stderr = run_cli_command(cmd)

    # Check execution
    assert return_code == 0, f"Help command failed with code {return_code}. Stderr: {stderr}"

    # Check for expected help terms
    expected_terms = ["prompt"]
    for term in expected_terms:
        assert term in stdout, f"Expected help output to mention '{term}'"
