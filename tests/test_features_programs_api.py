"""Test that all feature program examples can run with a single prompt."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from llmproc.program import LLMProgram

# Define base paths for examples to make future path changes easier
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
FEATURES_DIR = EXAMPLES_DIR / "features"
ANTHROPIC_DIR = EXAMPLES_DIR / "anthropic"
OPENAI_DIR = EXAMPLES_DIR / "openai"
CLAUDE_CODE_DIR = EXAMPLES_DIR / "claude-code"

# Mark this test module as requiring API access
pytestmark = pytest.mark.llm_api


def api_keys_available():
    """Check if any necessary API keys are available."""
    for key_name in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "OPENAI_API_KEY"]:
        if os.environ.get(key_name):
            return True
    return False


async def run_program_with_prompt(program_path, test_prompt="Hi"):
    """Run a program with a very simple test prompt.

    Just checks if the program initializes and processes a minimal prompt.
    """
    try:
        # Load and start the program
        program = LLMProgram.from_toml(program_path, include_linked=False)
        process = await program.start()

        # Run the process with the minimal test prompt
        result = await process.run(test_prompt)

        # Get the response text
        response = process.get_last_message()

        # No cleanup needed in current version

        return True, response
    except Exception as e:
        return False, str(e)


@pytest.mark.asyncio
async def test_all_feature_programs():
    """Test all programs in the features directory with a simple prompt."""
    if not api_keys_available():
        pytest.skip("API keys not available for testing")

    # Get all TOML files from the features directory using the constant
    toml_files = []
    for root, _, files in os.walk(FEATURES_DIR):
        for file in files:
            if file.endswith(".toml"):
                toml_files.append(Path(root) / file)

    # Skip only files that absolutely cannot work in a test environment
    skip_files = [
        # Skip files that require registry configuration
        "mcp.toml",  # Requires MCP registry integration and external dependencies
    ]

    # Track results
    successful = []
    failed = []

    # Try each program
    for toml_file in toml_files:
        # Skip files known to cause issues
        if toml_file.name in skip_files:
            print(f"Skipping {toml_file.name} (in skip list)")
            continue

        print(f"Testing {toml_file}...", end=" ")
        sys.stdout.flush()

        # Run with a simple test prompt
        success, response = await run_program_with_prompt(toml_file)

        if success:
            print("SUCCESS")
            successful.append(toml_file)
            # Print the first 80 characters of the response
            print(f"  Response: {response[:80]}...")
        else:
            print("FAILED")
            failed.append((toml_file, response))
            print(f"  Error: {response}")

    # Report results
    print(f"\nSuccessfully ran {len(successful)} out of {len(toml_files)} files")

    # If any failed, report them
    if failed:
        for file, error in failed:
            print(f"Failed to run {file}: {error}")

        pytest.fail(f"Failed to run {len(failed)} out of {len(toml_files)} files")


if __name__ == "__main__":
    asyncio.run(test_all_feature_programs())
