#!/usr/bin/env python3

"""
MCP Example Script

This script demonstrates the use of MCP (Model Context Protocol) tools
with the LLMProcess class. It loads the MCP configuration from a TOML file
and runs a simple conversation with tool usage.
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llmproc import LLMProcess

def main():
    """Run a simple example of MCP functionality."""
    # Load the MCP-enabled LLMProcess from TOML
    llm_process = LLMProcess.from_toml("examples/mcp.toml")
    
    print(f"Initialized {llm_process.display_name}")
    print("------------------------------------------------------------")
    
    # Run a simple example that uses the search_repositories tool
    if "github" in llm_process.mcp_tools:
        response = llm_process.run(
            "Please search for popular Python repositories on GitHub that focus on machine learning."
        )
        print("Response:")
        print(response)
        print("------------------------------------------------------------")
    
    # Run a simple example that uses the ReadFile tool
    if "codemcp" in llm_process.mcp_tools:
        # Create a temporary file to read
        with open("temp_file.txt", "w") as f:
            f.write("This is a test file created by the mcp_example.py script.\n")
            f.write("It demonstrates the use of the codemcp.ReadFile tool.\n")
        
        response = llm_process.run(
            "Please read the file 'temp_file.txt' and summarize its contents."
        )
        print("Response:")
        print(response)
        print("------------------------------------------------------------")
        
        # Clean up
        os.unlink("temp_file.txt")
    
    # Example of a conversation without tool usage
    response = llm_process.run(
        "What is the capital of France?"
    )
    print("Response:")
    print(response)
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()