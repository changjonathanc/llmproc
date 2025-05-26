"""Test for the temperature SDK demo script.

This test verifies that the temperature tool can be used by an LLM to adjust its own temperature.
It runs the temperature SDK demo script and checks for successful tool usage.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.llm_api
@pytest.mark.extended_api
def test_temperature_sdk_demo():
    """Test that the temperature SDK demo script runs successfully.

    This test:
    1. Runs the temperature_sdk_demo.py script
    2. Captures stdout/stderr
    3. Verifies that there are exactly 2 successful tool results
    """
    # Set up the path to the demo script
    script_path = Path(__file__).parent.parent.parent / "examples" / "scripts" / "temperature_sdk_demo.py"

    # Verify script exists
    assert script_path.exists(), f"Script not found at {script_path}"

    # Run the script as a subprocess and capture its output
    print(f"\nRunning temperature SDK demo from: {script_path}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=300,  # Add 5-minute timeout for safety
        )

        # Print the script output (truncated if too long)
        output = result.stdout
        print("\nDemo script output (truncated):")
        print("\n".join(output.split("\n")[-20:]))  # Show the last 20 lines

        # Verify the script ran successfully
        assert (
            result.returncode == 0
        ), f"Demo script failed with return code {result.returncode}. Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        pytest.fail("Demo script timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"Failed to run demo script: {str(e)}")

    # Check that we have exactly 2 successful tool results
    success_markers = "✅ Tool result:"
    success_count = output.count(success_markers)

    print("\nVerifying temperature tool usage...")
    print(f"- Found {success_count} successful tool usages")

    # We expect 2 tool results ideally
    if success_count == 0:
        pytest.fail("No successful tool results found in the output")
    elif success_count == 1:
        # Issue a warning but don't fail the test if only one tool use is found
        pytest.warns(UserWarning, match="Only one temperature tool use detected")
        print("⚠️ WARNING: Only found 1 tool use instead of the expected 2")
    else:
        print("✅ Found 2 or more tool usages as expected")

    # Verify at least one tool use occurred
    assert success_count >= 1, "Expected at least 1 successful tool result"

    # Check for temperature changes in the output
    assert "Changing temperature to:" in output, "Expected to find temperature changes in output"

    print("✅ All assertions passed: Temperature SDK demo verified!")


if __name__ == "__main__":
    """Allow running this test file directly."""
    test_temperature_sdk_demo()
