"""Integration tests for Claude 3.7 Sonnet thinking models with different thinking levels.

These tests validate that Claude 3.7 models with different thinking budgets
function correctly and produce outputs reflecting their reasoning capabilities.
"""

import os
import pytest
import time
import asyncio
from typing import Dict, List, Tuple

from llmproc import LLMProcess, LLMProgram


async def load_thinking_model(config_path: str) -> LLMProcess:
    """Load a thinking model from a TOML configuration file."""
    program = LLMProgram.from_toml(config_path)
    return await program.start()


@pytest.mark.llm_api
def test_thinking_models_configuration():
    """Test that thinking model configurations load correctly with proper parameters."""
    # Load all three thinking model configurations
    high_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-high.toml")
    medium_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-medium.toml")
    low_program = LLMProgram.from_toml("examples/basic/claude-3-7-thinking-low.toml")
    
    # Verify high thinking configuration
    assert high_program.model_name == "claude-3-7-sonnet-20250219"
    assert high_program.provider == "anthropic"
    assert high_program.parameters["thinking_budget"] == 16000
    assert high_program.parameters["max_tokens"] == 32768
    
    # Verify medium thinking configuration
    assert medium_program.model_name == "claude-3-7-sonnet-20250219"
    assert medium_program.provider == "anthropic"
    assert medium_program.parameters["thinking_budget"] == 4000
    assert medium_program.parameters["max_tokens"] == 16384
    
    # Verify low thinking configuration
    assert low_program.model_name == "claude-3-7-sonnet-20250219"
    assert low_program.provider == "anthropic"
    assert low_program.parameters["thinking_budget"] == 1024
    assert low_program.parameters["max_tokens"] == 8192


@pytest.mark.llm_api
async def test_thinking_models_parameter_transformation():
    """Test that the API parameters are correctly transformed for thinking models."""
    # Skip if no Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Load all three thinking model configurations
    high_process = await load_thinking_model("examples/basic/claude-3-7-thinking-high.toml")
    medium_process = await load_thinking_model("examples/basic/claude-3-7-thinking-medium.toml")
    low_process = await load_thinking_model("examples/basic/claude-3-7-thinking-low.toml")
    
    # Verify high thinking API parameters
    assert "thinking_budget" in high_process.api_params
    assert high_process.api_params["thinking_budget"] == 16000
    assert "max_tokens" in high_process.api_params
    
    # Verify medium thinking API parameters
    assert "thinking_budget" in medium_process.api_params
    assert medium_process.api_params["thinking_budget"] == 4000
    assert "max_tokens" in medium_process.api_params
    
    # Verify low thinking API parameters
    assert "thinking_budget" in low_process.api_params
    assert low_process.api_params["thinking_budget"] == 1024
    assert "max_tokens" in low_process.api_params


@pytest.mark.llm_api
async def test_thinking_models_response_quality():
    """Test that thinking models with different budgets produce 
    different quality responses for complex problems."""
    # Skip if no Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Complex problem requiring multi-step reasoning
    complex_problem = """
    A farmer has a rectangular field with a perimeter of 100 meters. 
    If the width of the field is 20 meters, what is its area in square meters?
    
    Work through this step-by-step.
    """
    
    # Load all three thinking model configurations
    high_process = await load_thinking_model("examples/basic/claude-3-7-thinking-high.toml")
    medium_process = await load_thinking_model("examples/basic/claude-3-7-thinking-medium.toml")
    low_process = await load_thinking_model("examples/basic/claude-3-7-thinking-low.toml")
    
    # Run the models with the same prompt
    high_result = await high_process.run(complex_problem)
    medium_result = await medium_process.run(complex_problem)
    low_result = await low_process.run(complex_problem)
    
    # Verify we got responses from all models
    assert high_result
    assert medium_result
    assert low_result
    
    # Get the text content from the RunResult objects
    high_text = high_process.get_last_message()
    medium_text = medium_process.get_last_message()
    low_text = low_process.get_last_message()
    
    # Check that high thinking model provides more detailed reasoning
    # (We can't check exact content, but we can check length as a proxy for detail)
    high_length = len(high_text)
    medium_length = len(medium_text)
    low_length = len(low_text)
    
    # Log the response lengths for diagnostics
    print(f"High thinking response length: {high_length}")
    print(f"Medium thinking response length: {medium_length}")
    print(f"Low thinking response length: {low_length}")
    
    # The actual response lengths will vary, but we verify that all responses
    # contain the correct answer (600 square meters)
    assert "600" in high_text
    assert "600" in medium_text
    assert "600" in low_text


@pytest.mark.llm_api
async def test_thinking_models_response_time():
    """Test the response time differences between different thinking budgets."""
    # Skip if no Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Simple problem that should be quick to solve
    simple_problem = "What is 24 * 7?"
    
    # Load all three thinking model configurations
    high_process = await load_thinking_model("examples/basic/claude-3-7-thinking-high.toml")
    medium_process = await load_thinking_model("examples/basic/claude-3-7-thinking-medium.toml")
    low_process = await load_thinking_model("examples/basic/claude-3-7-thinking-low.toml")
    
    # Measure response times
    high_times = []
    medium_times = []
    low_times = []
    
    # Run each model once (to reduce test time)
    # High thinking
    start_time = time.time()
    await high_process.run(simple_problem)
    high_times.append(time.time() - start_time)
    
    # Medium thinking
    start_time = time.time()
    await medium_process.run(simple_problem)
    medium_times.append(time.time() - start_time)
    
    # Low thinking
    start_time = time.time()
    await low_process.run(simple_problem)
    low_times.append(time.time() - start_time)
    
    # Calculate average times
    avg_high_time = sum(high_times) / len(high_times)
    avg_medium_time = sum(medium_times) / len(medium_times)
    avg_low_time = sum(low_times) / len(low_times)
    
    # Log the average response times
    print(f"Average high thinking response time: {avg_high_time:.2f}s")
    print(f"Average medium thinking response time: {avg_medium_time:.2f}s")
    print(f"Average low thinking response time: {avg_low_time:.2f}s")
    
    # Note: We don't assert specific response times as they can vary significantly
    # based on network conditions, API server load, etc. This test is primarily
    # for diagnostic information.


@pytest.mark.llm_api
async def test_thinking_models_complex_coding():
    """Test thinking models with a complex coding task."""
    # Skip if no Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    
    # Complex coding problem requiring reasoning
    coding_problem = """
    Write a Python function to find all prime numbers in the Fibonacci sequence 
    up to the 100th Fibonacci number. Make sure your solution is optimized for 
    performance and memory usage.
    
    Show your reasoning process as comments in the code.
    """
    
    # Load high thinking model (we only use high for complex coding)
    high_process = await load_thinking_model("examples/basic/claude-3-7-thinking-high.toml")
    
    # Run the model
    high_result = await high_process.run(coding_problem)
    
    # Verify we got a response
    assert high_result
    
    # Get the text content from the RunResult object
    high_text = high_process.get_last_message()
    
    # Check for key indicators of a valid solution
    assert "def" in high_text  # Function definition
    assert "fibonacci" in high_text.lower()  # Fibonacci logic
    assert "prime" in high_text.lower()  # Prime checking logic
    assert "#" in high_text  # Comments for reasoning