import os

import pytest

pytest_plugins = ["tests.conftest_api"]

from tests.patterns import assert_successful_response, timed_test


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
def test_sync_claude_minimal(minimal_claude_program):
    """Verify start_sync and run with a minimal Claude program."""
    if os.environ.get("ANTHROPIC_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing ANTHROPIC_API_KEY environment variable")
    # Arrange
    program = minimal_claude_program

    # Act
    with timed_test(timeout_seconds=8.0):
        process = program.start_sync()
        result = process.run("Hello! Who are you?")

    # Assert
    assert_successful_response(result)
    assert len(process.get_state()) >= 2


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.openai_api
def test_sync_openai_minimal(minimal_openai_program):
    """Verify start_sync and run with a minimal OpenAI program."""
    if os.environ.get("OPENAI_API_KEY") in (None, "API_KEY", ""):
        pytest.skip("Missing OPENAI_API_KEY environment variable")
    # Arrange
    program = minimal_openai_program

    # Act
    with timed_test(timeout_seconds=8.0):
        process = program.start_sync()
        result = process.run("Hello! Who are you?")

    # Assert
    assert_successful_response(result)
    assert len(process.get_state()) >= 2
