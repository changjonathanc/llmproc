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
@pytest.mark.release_api
@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "llmproc-demo examples/anthropic/claude-3-5-sonnet.toml -p hi",
            marks=pytest.mark.skipif(
                "ANTHROPIC_API_KEY" not in os.environ
                or "None" in os.environ["ANTHROPIC_API_KEY"],
                reason="Anthropic API key not available or invalid",
            ),
        ),
        pytest.param(
            "llmproc-demo examples/anthropic/claude-3-5-haiku.toml -p hello",
            marks=pytest.mark.skipif(
                "ANTHROPIC_API_KEY" not in os.environ
                or "None" in os.environ["ANTHROPIC_API_KEY"],
                reason="Anthropic API key not available or invalid",
            ),
        ),
        pytest.param(
            "llmproc-demo examples/openai/gpt-4o-mini.toml -p welcome",
            marks=pytest.mark.skipif(
                "OPENAI_API_KEY" not in os.environ
                or "None" in os.environ["OPENAI_API_KEY"],
                reason="OpenAI API key not available or invalid",
            ),
        ),
        pytest.param(
            "llmproc-demo examples/claude-code/claude-code.toml -p hello",
            marks=pytest.mark.skipif(
                "ANTHROPIC_API_KEY" not in os.environ
                or "None" in os.environ["ANTHROPIC_API_KEY"],
                reason="Anthropic API key not available or invalid",
            ),
        ),
        pytest.param(
            "llmproc-demo examples/anthropic/claude-3-5-haiku-vertex.toml -p hello",
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
@pytest.mark.release_api
def test_complex_prompt_with_quotes():
    """Test a complex prompt with quotes in it."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use subprocess.run directly to properly handle complex quoting
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli",
        str(
            Path(__file__).parent.parent
            / "examples"
            / "anthropic"
            / "claude-3-5-sonnet.toml"
        ),
        "--prompt",
        'Define the term "machine learning" in one sentence.',
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45
        )
        return_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return_code, stdout, stderr = -1, "", "Command timed out"

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
@pytest.mark.release_api
def test_tool_usage_direct_command():
    """Test a command that triggers tool usage."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use a different approach for this test to handle quotes properly
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli",
        str(
            Path(__file__).parent.parent
            / "examples"
            / "claude-code"
            / "claude-code.toml"
        ),
        "-p",
        "Say 'Hello, world!' and also tell me what tools you have available.",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
        )
        return_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return_code, stdout, stderr = -1, "", "Command timed out after 90 seconds"
    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"

    # Check for very basic expected output terms that should always be present
    expected_terms = ["Hello", "world", "tool"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]
    assert len(found_terms) > 0, (
        f"Expected output to mention at least one of {expected_terms}"
    )


@pytest.mark.llm_api
@pytest.mark.release_api
def test_program_linking_direct_command():
    """Test a command with program linking."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Use a different approach for this test to handle quotes properly
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli",
        "examples/features/program-linking/main.toml",
        "-p",
        "Ask the repo expert what files are in src/llmproc",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        return_code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return_code, stdout, stderr = -1, "", "Command timed out after 120 seconds"

    # Check execution
    assert return_code == 0, f"Command failed with code {return_code}. Stderr: {stderr}"
    assert len(stdout) > 0, "Command produced no output"


@pytest.mark.llm_api
@pytest.mark.release_api
def test_stdin_pipe_with_n_flag():
    """Test piping input to the CLI with -n flag."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # We can't directly test piping with subprocess.run, so we'll simulate it
    # by using input parameter
    cmd = [
        sys.executable,
        "-m",
        "llmproc.cli",
        "examples/anthropic/claude-3-5-haiku.toml",
        "-n",
    ]

    try:
        # Run with simulated stdin pipe using a simple, deterministic prompt
        result = subprocess.run(
            cmd,
            input="Say 'Hello world' exactly like that.",
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

        # Check for exact expected phrase
        assert "Hello world" in result.stdout, (
            "Expected output to contain 'Hello world'"
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
