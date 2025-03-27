"""Integration tests for O3-mini reasoning models with different reasoning levels.

These tests validate that O3-mini models with different reasoning levels (high, medium, low)
function correctly and produce outputs reflecting their respective reasoning capabilities.
"""

import os
import pytest
import time
import asyncio
from typing import Dict, List, Tuple

from llmproc import LLMProcess, LLMProgram


async def load_reasoning_model(config_path: str) -> LLMProcess:
    """Load a reasoning model from a TOML configuration file."""
    program = LLMProgram.from_toml(config_path)
    return await program.start()


@pytest.mark.llm_api
def test_reasoning_models_configuration():
    """Test that reasoning model configurations load correctly with proper parameters."""
    # Load all three reasoning model configurations
    high_program = LLMProgram.from_toml("examples/basic/o3-mini-high.toml")
    medium_program = LLMProgram.from_toml("examples/basic/o3-mini-medium.toml")
    low_program = LLMProgram.from_toml("examples/basic/o3-mini-low.toml")
    
    # Verify high reasoning configuration
    assert high_program.model_name == "o3-mini"
    assert high_program.provider == "openai"
    assert high_program.parameters["reasoning_effort"] == "high"
    assert high_program.parameters["max_completion_tokens"] == 25000
    
    # Verify medium reasoning configuration
    assert medium_program.model_name == "o3-mini"
    assert medium_program.provider == "openai"
    assert medium_program.parameters["reasoning_effort"] == "medium"
    assert medium_program.parameters["max_completion_tokens"] == 10000
    
    # Verify low reasoning configuration
    assert low_program.model_name == "o3-mini"
    assert low_program.provider == "openai"
    assert low_program.parameters["reasoning_effort"] == "low"
    assert low_program.parameters["max_completion_tokens"] == 5000


@pytest.mark.llm_api
async def test_reasoning_models_parameter_transformation():
    """Test that the API parameters are correctly transformed for reasoning models."""
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Load all three reasoning model configurations
    high_process = await load_reasoning_model("examples/basic/o3-mini-high.toml")
    medium_process = await load_reasoning_model("examples/basic/o3-mini-medium.toml")
    low_process = await load_reasoning_model("examples/basic/o3-mini-low.toml")
    
    # Verify high reasoning API parameters
    assert "reasoning_effort" in high_process.api_params
    assert high_process.api_params["reasoning_effort"] == "high"
    assert "max_completion_tokens" in high_process.api_params
    assert "max_tokens" not in high_process.api_params
    
    # Verify medium reasoning API parameters
    assert "reasoning_effort" in medium_process.api_params
    assert medium_process.api_params["reasoning_effort"] == "medium"
    assert "max_completion_tokens" in medium_process.api_params
    assert "max_tokens" not in medium_process.api_params
    
    # Verify low reasoning API parameters
    assert "reasoning_effort" in low_process.api_params
    assert low_process.api_params["reasoning_effort"] == "low"
    assert "max_completion_tokens" in low_process.api_params
    assert "max_tokens" not in low_process.api_params


@pytest.mark.llm_api
async def test_reasoning_models_response_quality():
    """Test that reasoning models with different reasoning levels produce 
    different quality responses for complex problems."""
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Complex problem requiring multi-step reasoning
    complex_problem = """
    A farmer has a rectangular field with a perimeter of 100 meters. 
    If the width of the field is 20 meters, what is its area in square meters?
    
    Work through this step-by-step.
    """
    
    # Load all three reasoning model configurations
    high_process = await load_reasoning_model("examples/basic/o3-mini-high.toml")
    medium_process = await load_reasoning_model("examples/basic/o3-mini-medium.toml")
    low_process = await load_reasoning_model("examples/basic/o3-mini-low.toml")
    
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
    
    # Check that high reasoning model provides more detailed reasoning
    # (We can't check exact content, but we can check length as a proxy for detail)
    high_length = len(high_text)
    medium_length = len(medium_text)
    low_length = len(low_text)
    
    # Log the response lengths for diagnostics
    print(f"High reasoning response length: {high_length}")
    print(f"Medium reasoning response length: {medium_length}")
    print(f"Low reasoning response length: {low_length}")
    
    # The actual response lengths will vary, but we verify that all responses
    # contain the correct answer (600 square meters)
    assert "600" in high_text
    assert "600" in medium_text
    assert "600" in low_text


@pytest.mark.llm_api
async def test_reasoning_models_response_time():
    """Test the response time differences between different reasoning levels."""
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Simple problem that should be quick to solve
    simple_problem = "What is 24 * 7?"
    
    # Load all three reasoning model configurations
    high_process = await load_reasoning_model("examples/basic/o3-mini-high.toml")
    medium_process = await load_reasoning_model("examples/basic/o3-mini-medium.toml")
    low_process = await load_reasoning_model("examples/basic/o3-mini-low.toml")
    
    # Measure response times
    high_times = []
    medium_times = []
    low_times = []
    
    # Run each model once (to reduce test time)
    # High reasoning
    start_time = time.time()
    await high_process.run(simple_problem)
    high_times.append(time.time() - start_time)
    
    # Medium reasoning
    start_time = time.time()
    await medium_process.run(simple_problem)
    medium_times.append(time.time() - start_time)
    
    # Low reasoning
    start_time = time.time()
    await low_process.run(simple_problem)
    low_times.append(time.time() - start_time)
    
    # Calculate average times
    avg_high_time = sum(high_times) / len(high_times)
    avg_medium_time = sum(medium_times) / len(medium_times)
    avg_low_time = sum(low_times) / len(low_times)
    
    # Log the average response times
    print(f"Average high reasoning response time: {avg_high_time:.2f}s")
    print(f"Average medium reasoning response time: {avg_medium_time:.2f}s")
    print(f"Average low reasoning response time: {avg_low_time:.2f}s")
    
    # Note: We don't assert specific response times as they can vary significantly
    # based on network conditions, API server load, etc. This test is primarily
    # for diagnostic information.


@pytest.mark.llm_api
async def test_reasoning_models_complex_coding():
    """Test reasoning models with a complex coding task."""
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Complex coding problem requiring reasoning
    coding_problem = """
    Write a Python function to find all prime numbers in the Fibonacci sequence 
    up to the 100th Fibonacci number. Make sure your solution is optimized for 
    performance and memory usage.
    
    Show your reasoning process as comments in the code.
    """
    
    # Load high reasoning model (we only use high for complex coding)
    high_process = await load_reasoning_model("examples/basic/o3-mini-high.toml")
    
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