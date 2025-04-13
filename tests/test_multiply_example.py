"""Test for the multiply example featured in the README."""

import asyncio

import pytest

from llmproc import LLMProgram, register_tool


# Register the same multiply tool as in the example
@register_tool()
def multiply(a: float, b: float) -> dict:
    """Multiply two numbers and return the result."""
    return {"result": a * b}


@pytest.fixture
async def multiply_process():
    """Create a process with the multiply tool for testing."""
    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        system_prompt="You're a helpful assistant.",
        parameters={"max_tokens": 1024},
    )

    program.set_enabled_tools([multiply])

    process = await program.start()
    yield process


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_multiply_pi_e(multiply_process):
    """Test multiplying π and e using the example from README."""
    # Run with the same prompt as in the README example
    result = await multiply_process.run(
        "Can you multiply 3.14159265359 by 2.71828182846?"
    )

    # Get the response
    response = multiply_process.get_last_message()

    # Assert that the response includes the result
    assert len(response) > 0
    assert (
        "8.539734222" in response
    )  # First part of π*e to ensure match regardless of rounding


@pytest.mark.llm_api
@pytest.mark.essential_api
@pytest.mark.anthropic_api
@pytest.mark.asyncio
async def test_multiply_large_numbers(multiply_process):
    """Test multiplying large numbers to verify the tool works correctly with any inputs."""
    # Test with larger numbers
    result = await multiply_process.run("What is 1234.5678 * 9876.5432?")

    # Get the response
    response = multiply_process.get_last_message()

    # The expected result is approximately 12193262.21
    assert len(response) > 0
    assert any(x in response for x in ["12193262", "12,193,262"])


@pytest.mark.asyncio
async def test_multiply_function_directly():
    """Test the multiply function directly without using an LLM."""
    # Test the function itself
    result = multiply(3.14159265359, 2.71828182846)

    # Verify the result
    assert result["result"] == pytest.approx(8.539734222677128)
