"""Optimized version of program linking API tests with faster execution.

This version implements the recommendations from RFC027:
1. Uses smaller/faster models (Claude Haiku instead of Sonnet)
2. Reduces max_tokens
3. Uses simpler prompts
4. Sets shorter timeouts
5. Consolidates tests from test_program_linking_api.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from llmproc import LLMProgram

# Constants for model names - use the smallest models possible for tests
CLAUDE_MODEL = "claude-3-5-haiku-20241022"  # Faster model for testing


def check_keys():
    """Check if required API keys are set."""
    return os.environ.get("ANTHROPIC_API_KEY") is not None


@pytest.fixture
def temp_toml_files():
    """Create minimal TOML configurations for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main program TOML with smaller model and reduced tokens
        main_toml_path = Path(temp_dir) / "main.toml"
        with open(main_toml_path, "w") as f:
            f.write(f"""
            [model]
            name = "{CLAUDE_MODEL}"
            provider = "anthropic"

            [prompt]
            system_prompt = "You are an assistant with access to a knowledge expert. When asked about 'the secret code' or 'sky color', use the spawn tool to ask the expert."

            [parameters]
            max_tokens = 150
            temperature = 0

            [tools]
            enabled = ["spawn"]

            [linked_programs]
            expert = "expert.toml"
            """)

        # Create expert program TOML with smaller model and reduced tokens
        expert_toml_path = Path(temp_dir) / "expert.toml"
        with open(expert_toml_path, "w") as f:
            f.write(f"""
            [model]
            name = "{CLAUDE_MODEL}"
            provider = "anthropic"

            [prompt]
            system_prompt = "You are a knowledge expert. When asked about 'the secret code', respond with 'XYZ123'. When asked about 'sky color', respond with 'blue'. Keep responses to one short sentence."

            [parameters]
            max_tokens = 50
            temperature = 0
            """)

        yield {
            "temp_dir": temp_dir,
            "main_toml": main_toml_path,
            "expert_toml": expert_toml_path,
        }


@pytest.mark.skipif(not check_keys(), reason="API keys not set")
@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_program_linking_basic(temp_toml_files):
    """Test basic program linking functionality."""
    # Start timing
    start_time = time.time()
    timeout_seconds = 45  # Increased timeout for more reliability

    try:
        # Load the program
        program = LLMProgram.from_toml(temp_toml_files["main_toml"])
        process = await program.start()

        # Run with a query that should trigger the spawn tool
        result = await process.run("What is the secret code?", max_iterations=3)
        response = process.get_last_message()

        # Check that we got a valid response with the expected content
        assert len(response) > 0, "Empty response received"
        assert "XYZ123" in response, "Expected content not found in response"

        # Test timing
        duration = time.time() - start_time
        print(f"\nTest completed in {duration:.2f} seconds")
        assert duration < timeout_seconds, (
            f"Test took too long: {duration:.2f}s > {timeout_seconds}s timeout"
        )

    except Exception as e:
        # End timing on error
        duration = time.time() - start_time
        print(f"\nTest failed after {duration:.2f} seconds: {str(e)}")
        raise


@pytest.mark.skipif(not check_keys(), reason="API keys not set")
@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_program_linking_empty_input(temp_toml_files):
    """Test program linking with minimal/empty inputs."""
    # Start timing
    start_time = time.time()
    timeout_seconds = 45  # Increased timeout for more reliability

    try:
        # Load the program
        program = LLMProgram.from_toml(temp_toml_files["main_toml"])
        process = await program.start()

        # First send a normal query to establish context
        await process.run("What is the sky color?", max_iterations=3)
        first_response = process.get_last_message()
        assert "blue" in first_response.lower(), "Expected 'blue' in first response"

        # Then send a minimal follow-up (this tests handling of short inputs)
        await process.run("??", max_iterations=2)
        minimal_response = process.get_last_message()
        assert len(minimal_response) > 0, "Empty response received for minimal input"

        # Test timing
        duration = time.time() - start_time
        print(f"\nTest completed in {duration:.2f} seconds")
        assert duration < timeout_seconds, (
            f"Test took too long: {duration:.2f}s > {timeout_seconds}s timeout"
        )

    except Exception as e:
        # End timing on error
        duration = time.time() - start_time
        print(f"\nTest failed after {duration:.2f} seconds: {str(e)}")
        raise


@pytest.mark.skipif(not check_keys(), reason="API keys not set")
@pytest.mark.llm_api
@pytest.mark.extended_api
@pytest.mark.asyncio
async def test_program_linking_state_reset(temp_toml_files):
    """Test program linking after state reset."""
    # Start timing
    start_time = time.time()
    timeout_seconds = 45  # Increased timeout for more reliability

    try:
        # Load the program
        program = LLMProgram.from_toml(temp_toml_files["main_toml"])
        process = await program.start()

        # Send an initial query
        await process.run("What is the sky color?", max_iterations=3)
        first_response = process.get_last_message()
        assert "blue" in first_response.lower(), "Expected 'blue' in first response"

        # Reset the state
        process.reset_state()

        # Send a query after reset - should still work
        await process.run("What is the sky color?", max_iterations=3)
        reset_response = process.get_last_message()
        assert "blue" in reset_response.lower(), (
            "Expected 'blue' in response after reset"
        )

        # Test timing
        duration = time.time() - start_time
        print(f"\nTest completed in {duration:.2f} seconds")
        assert duration < timeout_seconds, (
            f"Test took too long: {duration:.2f}s > {timeout_seconds}s timeout"
        )

    except Exception as e:
        # End timing on error
        duration = time.time() - start_time
        print(f"\nTest failed after {duration:.2f} seconds: {str(e)}")
        raise
