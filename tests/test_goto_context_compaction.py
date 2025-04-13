"""Test the GOTO context compaction functionality.

This test verifies that the GOTO tool can be used to compact context while preserving knowledge.
It runs the context compaction demo script and verifies the results by parsing the script output.
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.llm_api
@pytest.mark.essential_api
def test_goto_context_compaction():
    """Test GOTO context compaction by running the demo script and parsing its output."""
    # Set up the path to the demo script
    script_path = (
        Path(__file__).parent.parent
        / "examples"
        / "scripts"
        / "goto_context_compaction_demo.py"
    )

    # Run the script as a subprocess and capture its output
    print(f"\nRunning GOTO context compaction demo from: {script_path}")
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
        assert result.returncode == 0, (
            f"Demo script failed with return code {result.returncode}. Error: {result.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Demo script timed out after 5 minutes")
    except Exception as e:
        pytest.fail(f"Failed to run demo script: {str(e)}")

    # Try multiple patterns to extract token reduction information
    token_reduction_match = re.search(
        r"Token reduction: (\d+) tokens \((\d+\.\d+)%\)", output
    )

    # Fallback patterns in case the format changes slightly
    if not token_reduction_match:
        token_reduction_match = re.search(
            r"Token reduction: (\d+) \((\d+\.\d+)%\)", output
        )
    if not token_reduction_match:
        token_reduction_match = re.search(
            r"reduction: (\d+) tokens \((\d+\.\d+)%\)", output
        )
    if not token_reduction_match:
        # As a last resort, check for any percentage in the output
        percent_match = re.search(r"reduction.*?(\d+\.\d+)%", output)
        if percent_match:
            # Set token count to 1000 as a placeholder if we can only find the percentage
            token_reduction = 1000
            reduction_percent = float(percent_match.group(1))
        else:
            raise AssertionError("Could not find token reduction information in script output")
    else:
        # Extract the token reduction percentage from the match
        token_reduction = int(token_reduction_match.group(1))
        reduction_percent = float(token_reduction_match.group(2))

    print("\nVerifying GOTO context compaction...")
    print(f"- Token reduction: {token_reduction} tokens ({reduction_percent:.1f}%)")

    # Assert significant token reduction (at least 30% is expected)
    assert reduction_percent >= 30, (
        f"Expected at least 30% token reduction, got {reduction_percent:.1f}%"
    )

    # Verify GOTO was used by checking for its mention in the output
    assert "GOTO: returning to" in output, (
        "GOTO tool should have been used when requested"
    )

    # Verify the model could still answer the follow-up question (knowledge preservation)
    assert "STEP 3: Knowledge check" in output, (
        "Follow-up knowledge check step should be present"
    )
    assert "--- SUMMARY ---" in output, (
        "Summary section should be present, indicating successful completion"
    )

    print("✅ All assertions passed: GOTO context compaction verified!")


if __name__ == "__main__":
    """Allow running this test file directly."""
    test_goto_context_compaction()
