#!/usr/bin/env python3
"""Example program demonstrating Claude 3.7 Sonnet with different thinking budgets.

This example compares the responses from three Claude 3.7 Sonnet configurations:
- High thinking budget: 16,000 tokens for thorough reasoning
- Medium thinking budget: 4,000 tokens for balanced approach
- Low thinking budget: 1,024 tokens (minimum) for faster responses

Usage:
    python claude_thinking_comparison.py [--problem PROBLEM_TYPE]

Arguments:
    --problem PROBLEM_TYPE    Type of problem to solve: 'math', 'code', or 'science'
                             (default: 'math')
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Dict, List, Tuple

from llmproc import LLMProcess, LLMProgram


# Problem prompts for different domains
PROBLEMS = {
    "math": """
    A farmer has a rectangular field with a perimeter of 100 meters. 
    If the width of the field is 20 meters, what is its area in square meters?
    
    Work through this step-by-step.
    """,
    
    "code": """
    Write a Python function to find the longest common subsequence of two strings.
    Include comments explaining your approach and time complexity analysis.
    """,
    
    "science": """
    Explain the greenhouse effect and its role in climate change. 
    Include the key chemical processes involved and how human activities 
    contribute to the enhancement of this effect.
    """
}


async def load_model(config_path: str) -> LLMProcess:
    """Load a model from a TOML configuration file."""
    program = LLMProgram.from_toml(config_path)
    return await program.start()


async def compare_thinking_levels(problem_type: str):
    """Compare responses from models with different thinking budgets."""
    # Verify environment
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)
    
    # Get the problem prompt
    if problem_type not in PROBLEMS:
        print(f"Error: Problem type '{problem_type}' not recognized.")
        print(f"Available types: {', '.join(PROBLEMS.keys())}")
        sys.exit(1)
    
    problem = PROBLEMS[problem_type]
    print(f"\n=== Problem ({problem_type}) ===")
    print(problem.strip())
    print("\n" + "=" * 50 + "\n")
    
    # Load the three thinking models
    print("Loading models...")
    high_process = await load_model("examples/basic/claude-3-7-thinking-high.toml")
    medium_process = await load_model("examples/basic/claude-3-7-thinking-medium.toml")
    low_process = await load_model("examples/basic/claude-3-7-thinking-low.toml")
    
    # Run the models with timing
    print("\n=== Running high thinking budget model (16,000 tokens) ===")
    high_start = time.time()
    high_result = await high_process.run(problem)
    high_time = time.time() - high_start
    print(f"Time: {high_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(high_process.get_last_message())
    print("-" * 50)
    
    print("\n=== Running medium thinking budget model (4,000 tokens) ===")
    medium_start = time.time()
    medium_result = await medium_process.run(problem)
    medium_time = time.time() - medium_start
    print(f"Time: {medium_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(medium_process.get_last_message())
    print("-" * 50)
    
    print("\n=== Running low thinking budget model (1,024 tokens) ===")
    low_start = time.time()
    low_result = await low_process.run(problem)
    low_time = time.time() - low_start
    print(f"Time: {low_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(low_process.get_last_message())
    print("-" * 50)
    
    # Compare response times
    print("\n=== Performance Comparison ===")
    print(f"High thinking (16K): {high_time:.2f}s")
    print(f"Medium thinking (4K): {medium_time:.2f}s")
    print(f"Low thinking (1K): {low_time:.2f}s")
    
    # Compare response lengths as a proxy for detail
    high_length = len(high_process.get_last_message())
    medium_length = len(medium_process.get_last_message())
    low_length = len(low_process.get_last_message())
    
    print("\n=== Response Detail Comparison ===")
    print(f"High thinking (16K): {high_length} characters")
    print(f"Medium thinking (4K): {medium_length} characters")
    print(f"Low thinking (1K): {low_length} characters")


def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(
        description="Compare Claude 3.7 Sonnet models with different thinking budgets."
    )
    parser.add_argument(
        "--problem",
        choices=["math", "code", "science"],
        default="math",
        help="Type of problem to solve"
    )
    args = parser.parse_args()
    
    # Run the comparison
    asyncio.run(compare_thinking_levels(args.problem))


if __name__ == "__main__":
    main()