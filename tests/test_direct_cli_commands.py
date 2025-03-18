#!/usr/bin/env python3
"""Tests for direct CLI command invocations matching real user commands."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def api_keys_available():
    """Check if required API keys are available."""
    has_openai = "OPENAI_API_KEY" in os.environ
    has_anthropic = "ANTHROPIC_API_KEY" in os.environ
    has_vertex = (
        "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        or "GOOGLE_CLOUD_PROJECT" in os.environ
    )

    return has_openai and has_anthropic and has_vertex


def run_exact_cli_command(command, timeout=45):
    """Run the exact CLI command as provided.

    Args:
        command: The full command to run
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        parts = command.split()

        # Replace 'llmproc-demo' with the actual module invocation
        if parts[0] == "llmproc-demo":
            parts[0] = sys.executable
            parts.insert(1, "-m")
            parts.insert(2, "llmproc.cli")

        # Run the command
        result = subprocess.run(
            parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"


@pytest.mark.llm_api
@pytest.mark.parametrize(
    "command",
    [
        "llmproc-demo examples/anthropic.toml -p 'hi'",
        "llmproc-demo examples/openai.toml -p 'hello there'",
        "llmproc-demo examples/minimal.toml -p 'what is 2+2?'",
        "llmproc-demo examples/claude_code.toml -p 'say hello'",
        pytest.param(
            "llmproc-demo examples/vertex.toml -p 'hello'",
            marks=pytest.mark.skipif(
                "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ,
                reason="Vertex AI credentials not available",
            ),
        ),
    ],
)
def test_exact_cli_commands(command):
    """Test exact CLI commands that users would type."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Run the exact CLI command
    return_code, stdout, stderr = run_exact_cli_command(command)

    # Check command executed successfully
    assert return_code == 0, (
        f"Command '{command}' failed with code {return_code}. Stderr: {stderr}"
    )

    # Check there was output
    assert len(stdout) > 0, f"Command '{command}' produced no output"

    # Check for common error strings
    error_terms = ["error", "exception", "traceback", "failed"]
    for term in error_terms:
        assert term.lower() not in stdout.lower(), (
            f"Output contains error term '{term}'"
        )
        assert term.lower() not in stderr.lower(), (
            f"Error output contains error term '{term}'"
        )


@pytest.mark.llm_api
def test_complex_prompt_with_quotes():
    """Test a complex prompt with quotes in it."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    command = 'llmproc-demo examples/minimal.toml -p "Define the term \\"machine learning\\" in one sentence."'
    return_code, stdout, stderr = run_exact_cli_command(command)

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"

    # Look for expected terms in the response
    expected_terms = ["machine", "learning", "algorithm"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]
    assert len(found_terms) > 0, (
        f"Expected output to contain at least one of {expected_terms}"
    )


@pytest.mark.llm_api
def test_tool_usage_direct_command():
    """Test a command that triggers tool usage."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    command = 'llmproc-demo examples/claude_code.toml -p "List the files in the src directory using the dispatch_agent tool"'
    return_code, stdout, stderr = run_exact_cli_command(command, timeout=90)

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"

    # Check for expected tool output terms
    expected_terms = ["llm_process.py", "program.py", "cli.py"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]
    assert len(found_terms) > 0, (
        f"Expected output to mention at least one of {expected_terms}"
    )


@pytest.mark.llm_api
def test_program_linking_direct_command():
    """Test a command with program linking."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    command = 'llmproc-demo examples/program_linking/main.toml -p "Ask the repo expert what files are in src/llmproc"'
    return_code, stdout, stderr = run_exact_cli_command(command, timeout=120)

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"


@pytest.mark.llm_api
def test_stdin_pipe_with_n_flag():
    """Test piping input to the CLI with -n flag."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # We can't directly test piping with subprocess.run, so we'll simulate it
    # by using input parameter
    cmd = [sys.executable, "-m", "llmproc.cli", "examples/minimal.toml", "-n"]

    try:
        # Run with simulated stdin pipe
        result = subprocess.run(
            cmd,
            input="Tell me the current year",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )

        # Check execution
        assert result.returncode == 0, (
            f"Command failed with code {result.returncode}. Stderr: {result.stderr}"
        )
        assert len(result.stdout) > 0, "Command produced no output"

        # Check for year in response
        expected_terms = ["2025", "twenty", "two thousand"]
        found_terms = [
            term for term in expected_terms if term.lower() in result.stdout.lower()
        ]
        assert len(found_terms) > 0, (
            f"Expected output to mention the year using one of {expected_terms}"
        )

    except subprocess.TimeoutExpired:
        pytest.fail("Command timed out")


def test_direct_cli_help():
    """Test the help command (does not require API keys)."""
    command = "llmproc-demo --help"
    return_code, stdout, stderr = run_exact_cli_command(command)

    # Check execution
    assert return_code == 0, (
        f"Help command failed with code {return_code}. Stderr: {stderr}"
    )

    # Check for expected help terms
    expected_terms = ["prompt", "non-interactive"]
    for term in expected_terms:
        assert term in stdout, f"Expected help output to mention '{term}'"
