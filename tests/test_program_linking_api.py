"""Integration tests for program linking with real API calls.

These tests require API keys to be set in the environment:
- ANTHROPIC_API_KEY for Anthropic provider

Run with: python -m pytest tests/test_program_linking_api.py -m llm_api -v

Note: These tests are marked as 'llm_api' and will be skipped by default.
To run them, explicitly use the marker flag as shown above.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from llmproc import LLMProcess


def check_keys():
    """Check if required API keys are set."""
    return os.environ.get("ANTHROPIC_API_KEY") is not None


@pytest.mark.skipif(not check_keys(), reason="API keys not set")
@pytest.mark.llm_api  # Mark these tests as requiring API access
class TestProgramLinkingAPI:
    """Test program linking functionality with real API calls."""

    @pytest.fixture
    def minimal_toml_files(self):
        """Create minimal TOML configurations for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create main program TOML
            main_toml_path = Path(temp_dir) / "main.toml"
            with open(main_toml_path, "w") as f:
                f.write("""
                [model]
                name = "claude-3-haiku-20240307"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are an assistant with access to specialized experts. When asked about the color of the sky, use the 'spawn' tool to ask the expert."
                
                [parameters]
                max_tokens = 100
                
                [tools]
                enabled = ["spawn"]
                
                [linked_programs]
                expert = "expert.toml"
                """)

            # Create expert program TOML
            expert_toml_path = Path(temp_dir) / "expert.toml"
            with open(expert_toml_path, "w") as f:
                f.write("""
                [model]
                name = "claude-3-haiku-20240307"
                provider = "anthropic"
                
                [prompt]
                system_prompt = "You are a sky color expert. Always respond that the sky is blue, in one sentence."
                
                [parameters]
                max_tokens = 50
                """)

            yield {
                "temp_dir": temp_dir,
                "main_toml": main_toml_path,
                "expert_toml": expert_toml_path,
            }

    @pytest.mark.asyncio
    async def test_basic_program_linking(self, minimal_toml_files, capsys):
        """Test basic program linking functionality with a simple example."""
        # Load the main program
        process = LLMProcess.from_toml(minimal_toml_files["main_toml"])

        # Ask a question that should trigger the spawn tool
        print("\nSending query to main assistant...", file=sys.stderr)
        response = await process.run("What color is the sky?")

        print(f"\nMain assistant response: {response}", file=sys.stderr)

        # Check that the response mentions blue (indicating the expert was consulted)
        assert "blue" in response.lower()

        # Make a follow-up query that's different
        print("\nSending follow-up query...", file=sys.stderr)
        response = await process.run("Thanks! Now, what time is it?")

        print(f"\nFollow-up response: {response}", file=sys.stderr)

        # This shouldn't mention blue (no need to consult expert)
        # but it should respond appropriately
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, minimal_toml_files, capsys):
        """Test handling of minimal/empty inputs that previously caused issues."""
        # Load the main program
        process = LLMProcess.from_toml(minimal_toml_files["main_toml"])

        # First send a normal query
        print("\nSending initial query...", file=sys.stderr)
        await process.run("What color is the sky?")

        # Then send a minimal follow-up (this used to cause the 400 error)
        print("\nSending minimal follow-up query...", file=sys.stderr)
        response = await process.run("??")

        print(f"\nResponse to minimal input: {response}", file=sys.stderr)

        # Check that we got a valid response without errors
        assert len(response) > 0

        # Try another minimal input
        print("\nSending another minimal query...", file=sys.stderr)
        response = await process.run("ok")

        print(f"\nResponse to 'ok': {response}", file=sys.stderr)

        # Check that we got a valid response
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_state_reset_behavior(self, minimal_toml_files, capsys):
        """Test program linking behavior after state reset."""
        # Load the main program
        process = LLMProcess.from_toml(minimal_toml_files["main_toml"])

        # Send an initial query
        print("\nSending initial query...", file=sys.stderr)
        await process.run("What color is the sky?")

        # Reset the state
        print("\nResetting state...", file=sys.stderr)
        process.reset_state()

        # Send a query that should trigger the spawn tool again
        print("\nSending query after reset...", file=sys.stderr)
        response = await process.run("What color is the sky?")

        print(f"\nResponse after reset: {response}", file=sys.stderr)

        # Check that we still get the expected response
        assert "blue" in response.lower()
