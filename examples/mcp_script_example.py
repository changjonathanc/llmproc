#!/usr/bin/env python3

"""
MCP Async Script Example

This script demonstrates the use of the async_run method with MCP tools,
showcasing the proper way to use asynchronous tool execution with the
newly implemented async_run method in LLMProcess.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llmproc import LLMProcess

async def run_examples():
    """Run example queries to demonstrate the async_run method with MCP tools."""
    # Make sure the config directory exists
    os.makedirs("config", exist_ok=True)
    
    # Create the MCP config file if it doesn't exist
    config_path = Path("config/mcp_servers.json")
    if not config_path.exists():
        config = {
            "mcpServers": {
                "github": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
                },
                "codemcp": {
                    "type": "stdio",
                    "command": "/bin/zsh",
                    "args": ["-c", "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "]
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    # Check for required environment variables
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        print("Please set it with: export ANTHROPIC_API_KEY=your-key-here")
        return
        
    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN environment variable is not set")
        print("GitHub tools may not work correctly")
        print("Please set it with: export GITHUB_TOKEN=your-token-here")
    
    # Initialize the LLMProcess with MCP tools
    print("Initializing LLMProcess with MCP tools...")
    process = LLMProcess(
        model_name="claude-3-haiku-20240307",
        provider="anthropic",
        system_prompt="You are a helpful assistant with access to tools. Use tools when appropriate to answer user queries accurately and concisely.",
        mcp_config_path=str(config_path),
        mcp_tools={"github": ["search_repositories", "get_file_contents"]},
        debug_tools=True,  # Enable debugging for tool calls
        temperature=0.7,
        max_tokens=1000
    )
    
    # Define example queries
    examples = [
        # Example 1: Ask about available tools
        "What tools are available to you? Please describe each one briefly.",
        
        # Example 2: Try to use a specific tool
        "Search for popular Python repositories on GitHub. Summarize the top results.",
        
        # Example 3: Try to use multiple tools
        "First search for information about 'transformer models', then find a popular GitHub repository related to transformers."
    ]
    
    # Run each example query
    for i, example in enumerate(examples, 1):
        print(f"\n=== Example {i}: {example} ===")
        print(f"User> {example}")
        
        # Use the run method which properly supports async tool execution
        response = await process.run(example)
        
        print(f"\n{process.display_name}> {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    # Run the async examples
    asyncio.run(run_examples())