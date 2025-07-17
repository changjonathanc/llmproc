import pytest

pytest_plugins = ["tests.conftest_api"]

from tests.patterns import assert_successful_response, timed_test


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.asyncio
@pytest.mark.claude_code_api
async def test_claude_code_hello_world(minimal_claude_code_process):
    """Send a simple prompt using the Claude Code provider."""
    process = minimal_claude_code_process
    with timed_test(timeout_seconds=8.0):
        result = await process.run("Hello, world!")

    assert_successful_response(result)
    assert "Hello" in process.get_last_message()
