"""Tests for the standardized fixtures in conftest.py.

This file provides examples of how to use the standardized fixtures
for unit, integration, and API testing.

The fixtures demonstrated here include:
- base_program: A minimal LLMProgram instance for non-API tests
- program_with_tools: An LLMProgram instance with common tools enabled
- mocked_llm_process: A fully mocked LLMProcess instance for non-API tests

These fixtures help standardize test setup and make tests more consistent.
"""

import pytest


def test_base_program_fixture(base_program):
    """Test the base_program fixture provides a correctly configured LLMProgram."""
    assert base_program is not None
    assert base_program.model_name == "test-fixture-model"
    assert base_program.provider == "test-fixture-provider"
    assert base_program.system_prompt == "Fixture system prompt."
    assert base_program.parameters == {}


def test_program_with_tools_fixture(program_with_tools):
    """Test the program_with_tools fixture configures tools correctly."""
    assert program_with_tools is not None
    # Compile program to register tools
    program_with_tools.compile()
    assert "calculator" in program_with_tools.tool_manager.get_registered_tools()
    assert "read_file" in program_with_tools.tool_manager.get_registered_tools()
    assert program_with_tools.model_name == "test-fixture-model"


@pytest.mark.asyncio
async def test_mocked_llm_process_fixture(mocked_llm_process):
    """Test the mocked_llm_process fixture provides a working mocked process."""
    assert mocked_llm_process is not None

    # The process should be an instance of LLMProcess
    from llmproc.llm_process import LLMProcess

    assert isinstance(mocked_llm_process, LLMProcess)

    # Should have expected attributes
    assert mocked_llm_process.provider == "anthropic"
    assert mocked_llm_process.model_name == "claude-3-5-sonnet-20240620"

    # Test that running works with the mock
    result = await mocked_llm_process.run("test prompt")
    assert result.content == "Mocked LLM response"


@pytest.mark.asyncio
async def test_typical_integration_test_pattern(mocked_llm_process):
    """Example of a typical integration test using the mocked_llm_process fixture.

    This demonstrates the recommended pattern for integration tests that
    test component interactions without making real API calls.
    """
    # Arrange - The fixture provides a ready-to-use mocked process
    process = mocked_llm_process
    initial_state_length = len(process.state)
    user_prompt = "Test integration pattern"

    # Act - Use the mocked process (API calls are automatically mocked)
    await process.run(user_prompt)

    # Assert - Verify expected state changes
    # After a run() call, state should have 2 more messages (user + assistant)
    assert len(process.state) == initial_state_length + 2
    assert process.state[-2]["role"] == "user"
    assert process.state[-2]["content"] == user_prompt
    assert process.state[-1]["role"] == "assistant"
