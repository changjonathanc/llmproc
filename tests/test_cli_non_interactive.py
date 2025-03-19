#!/usr/bin/env python3
"""Tests for the non-interactive mode of the llmproc-demo CLI."""

import os
import subprocess
import sys
import tempfile
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


def run_cli_non_interactive(program_path, prompt=None, timeout=45):
    """Run the llmproc-demo CLI in non-interactive mode.

    Args:
        program_path: Path to the TOML program file
        prompt: Text to send using --prompt option (if None, use --non-interactive with stdin)
        timeout: Maximum time to wait for the process to complete

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [sys.executable, "-m", "llmproc.cli", str(program_path)]

    # Use --prompt if provided, otherwise use --non-interactive
    if prompt is not None:
        cmd.extend(["--prompt", prompt])
    else:
        cmd.append("--non-interactive")

    try:
        if prompt is None:
            # When using --non-interactive, read from stdin
            result = subprocess.run(
                cmd,
                input="Hello from stdin\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        else:
            # When using --prompt, no stdin needed
            result = subprocess.run(
                cmd,
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
    "program_name",
    [
        "minimal.toml",
        "anthropic.toml",
        "openai.toml",
        pytest.param(
            "anthropic_vertex.toml",
            marks=pytest.mark.skipif(
                "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ,
                reason="Vertex AI credentials not available",
            ),
        ),
    ],
)
def test_prompt_option(program_name):
    """Test the --prompt option with various example programs."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / program_name

    # Create a unique test marker
    unique_marker = f"UNIQUE_TEST_MARKER_{program_name.replace('.', '_').upper()}"
    prompt = f"Reply with this exact marker: {unique_marker}"

    # Run CLI with --prompt option
    return_code, stdout, stderr = run_cli_non_interactive(program_path, prompt=prompt)

    # Check for successful execution
    assert return_code == 0, (
        f"CLI exited with error code {return_code}. Stderr: {stderr}"
    )

    # Check for the unique marker in the output
    assert unique_marker in stdout, (
        f"Expected unique marker '{unique_marker}' in output, but it wasn't found"
    )


@pytest.mark.llm_api
@pytest.mark.parametrize(
    "program_name",
    [
        "minimal.toml",
        "anthropic.toml",
        "openai.toml",
    ],
)
def test_non_interactive_option(program_name):
    """Test the --non-interactive option with stdin input."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / program_name

    # Run CLI with --non-interactive option
    return_code, stdout, stderr = run_cli_non_interactive(program_path, prompt=None)

    # Check for successful execution
    assert return_code == 0, (
        f"CLI exited with error code {return_code}. Stderr: {stderr}"
    )

    # Check for some output (can't check specific content as it depends on the model)
    assert len(stdout) > 0, "Expected some output, but got empty response"


@pytest.mark.llm_api
def test_tool_usage_in_non_interactive_mode():
    """Test that tools work correctly in non-interactive mode."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Test with claude_code.toml which has tools enabled
    program_path = Path(__file__).parent.parent / "examples" / "claude_code.toml"

    # Prompt that should trigger a tool use
    prompt = "What directories are in the current project? Use the dispatch_agent tool to find out."

    # Run CLI with --prompt option
    return_code, stdout, stderr = run_cli_non_interactive(
        program_path, prompt=prompt, timeout=90
    )

    # Check for successful execution
    assert return_code == 0, (
        f"CLI exited with error code {return_code}. Stderr: {stderr}"
    )

    # Check that output contains likely directory names
    expected_terms = ["src", "examples", "tests"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]

    assert len(found_terms) > 0, (
        f"Expected output to mention at least one of {expected_terms}, but found none"
    )


@pytest.mark.llm_api
def test_program_linking_in_non_interactive_mode():
    """Test program linking in non-interactive mode."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Test with program_linking/main.toml
    program_path = (
        Path(__file__).parent.parent / "examples" / "program_linking" / "main.toml"
    )

    # Prompt that should trigger the spawn tool
    prompt = "Ask the repo expert: what is the purpose of LLMProcess?"

    # Run CLI with --prompt option (longer timeout for program linking)
    return_code, stdout, stderr = run_cli_non_interactive(
        program_path, prompt=prompt, timeout=120
    )

    # Check for successful execution
    assert return_code == 0, (
        f"CLI exited with error code {return_code}. Stderr: {stderr}"
    )

    # Check that output mentions LLMProcess functionality
    expected_terms = ["llm", "process", "interface", "class"]
    found_terms = [term for term in expected_terms if term.lower() in stdout.lower()]

    assert len(found_terms) > 0, (
        f"Expected output to mention at least one of {expected_terms}, but found none"
    )


@pytest.mark.llm_api
def test_invalid_program_handling():
    """Test handling of invalid program with --prompt option."""
    # Create a temporary invalid program file
    with tempfile.NamedTemporaryFile("w+", suffix=".toml") as invalid_program:
        invalid_program.write("""
        [invalid]
        model_name = "nonexistent"
        provider = "unknown"
        """)
        invalid_program.flush()

        # Run CLI with invalid program
        return_code, stdout, stderr = run_cli_non_interactive(
            invalid_program.name, prompt="test"
        )

        # Should exit with non-zero return code
        assert return_code != 0, "Expected non-zero return code for invalid program"

        # Should provide an error message
        assert "error" in (stdout + stderr).lower(), (
            "Expected error message for invalid program"
        )


@pytest.mark.llm_api
def test_empty_prompt_handling():
    """Test handling of empty prompt."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    program_path = Path(__file__).parent.parent / "examples" / "minimal.toml"

    # Run CLI with empty prompt
    return_code, stdout, stderr = run_cli_non_interactive(program_path, prompt="")

    # Should execute successfully
    assert return_code == 0, (
        f"CLI exited with error code {return_code}. Stderr: {stderr}"
    )

    # Should provide some response
    assert len(stdout) > 0, "Expected some output even with empty prompt"
