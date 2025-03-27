#!/usr/bin/env python3
"""Example program demonstrating O3-mini with different reasoning levels.

This example compares the responses from three O3-mini configurations:
- High reasoning effort: Thoroughness prioritized over speed
- Medium reasoning effort: Balanced approach
- Low reasoning effort: Speed prioritized over thoroughness

Usage:
    python reasoning_comparison.py [--problem PROBLEM_TYPE]

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


async def compare_reasoning_levels(problem_type: str):
    """Compare responses from models with different reasoning levels."""
    # Verify environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
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
    
    # Load the three reasoning models
    print("Loading models...")
    high_process = await load_model("examples/basic/o3-mini-high.toml")
    medium_process = await load_model("examples/basic/o3-mini-medium.toml")
    low_process = await load_model("examples/basic/o3-mini-low.toml")
    
    # Run the models with timing
    print("\n=== Running high reasoning effort model ===")
    high_start = time.time()
    high_result = await high_process.run(problem)
    high_time = time.time() - high_start
    print(f"Time: {high_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(high_result)
    print("-" * 50)
    
    print("\n=== Running medium reasoning effort model ===")
    medium_start = time.time()
    medium_result = await medium_process.run(problem)
    medium_time = time.time() - medium_start
    print(f"Time: {medium_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(medium_result)
    print("-" * 50)
    
    print("\n=== Running low reasoning effort model ===")
    low_start = time.time()
    low_result = await low_process.run(problem)
    low_time = time.time() - low_start
    print(f"Time: {low_time:.2f} seconds")
    print("\nResponse:")
    print("-" * 50)
    print(low_result)
    print("-" * 50)
    
    # Compare response times
    print("\n=== Performance Comparison ===")
    print(f"High reasoning: {high_time:.2f}s")
    print(f"Medium reasoning: {medium_time:.2f}s")
    print(f"Low reasoning: {low_time:.2f}s")
    
    # Compare response lengths as a proxy for detail
    high_length = len(high_result)
    medium_length = len(medium_result)
    low_length = len(low_result)
    
    print("\n=== Response Detail Comparison ===")
    print(f"High reasoning: {high_length} characters")
    print(f"Medium reasoning: {medium_length} characters")
    print(f"Low reasoning: {low_length} characters")


def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(
        description="Compare O3-mini models with different reasoning levels."
    )
    parser.add_argument(
        "--problem",
        choices=["math", "code", "science"],
        default="math",
        help="Type of problem to solve"
    )
    args = parser.parse_args()
    
    # Run the comparison
    asyncio.run(compare_reasoning_levels(args.problem))


if __name__ == "__main__":
    main()