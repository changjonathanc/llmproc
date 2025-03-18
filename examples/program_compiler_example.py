#!/usr/bin/env python3
"""
Example demonstrating how to use the LLMProgram compiler
"""

import asyncio
import os
from pathlib import Path

from llmproc import LLMProcess, LLMProgram

async def main():
    """Run the program compiler example."""
    print("LLMProgram Compiler Example")
    print("==========================")
    
    # Specify the TOML file path
    toml_path = Path("examples/minimal.toml")
    if not toml_path.exists():
        print(f"Error: TOML file not found at {toml_path}")
        return
    
    print(f"Compiling program from: {toml_path}")
    
    # Method 1: Use LLMProgram.compile directly
    program = LLMProgram.compile(toml_path)
    print("\nCompiled Program Details:")
    print(f"  Model: {program.provider} / {program.model_name}")
    print(f"  Display Name: {program.display_name}")
    print(f"  System Prompt: {program.system_prompt[:50]}...")
    print(f"  API Parameters: {program.api_params}")
    
    # Method 2: Use start() to create and initialize an LLMProcess
    print("\nCreating and initializing LLMProcess from compiled program...")
    process = await program.start()
    
    # Method 3: Two-step pattern: from_toml followed by start()
    print("\nCreating LLMProcess using two-step pattern...")
    program2 = LLMProgram.from_toml(toml_path)
    process2 = await program2.start()
    
    # Compare the process details
    print("\nVerifying both processes are equivalent:")
    print(f"  Display Names Match: {process.display_name == process2.display_name}")
    print(f"  Model Names Match: {process.model_name == process2.model_name}")
    print(f"  Providers Match: {process.provider == process2.provider}")
    
    # Test with a sample query if API keys are available
    required_env_var = f"{process.provider.upper()}_API_KEY"
    if os.environ.get(required_env_var):
        print("\nRunning a test query...")
        query = "Tell me a short joke."
        response = await process.run(query)
        print(f"\n{process.display_name}> {response}")
    else:
        print(f"\nSkipping test query (no {required_env_var} environment variable found)")
    
    print("\nProgram compiler example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())