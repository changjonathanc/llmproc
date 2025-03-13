#!/usr/bin/env python3

"""
Simple MCP Demo

This script provides a simpler way to demo MCP with our LLMProcess implementation
by using the TOML configuration approach.
"""

import os
import json
import sys
from pathlib import Path

# Get the repository root directory (where the script is running from)
REPO_ROOT = Path(__file__).parent.absolute()

# Add the package to the Python path if needed
sys.path.insert(0, str(REPO_ROOT))

from llmproc import LLMProcess

def setup_mcp_config():
    """Set up the MCP config file if it doesn't exist."""
    # Make sure the config directory exists
    config_dir = REPO_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create the MCP config file if it doesn't exist
    config_path = config_dir / "mcp_servers.json"
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
    
    return config_path

def create_mcp_toml():
    """Create a TOML file for MCP if it doesn't exist."""
    # Ensure examples directory exists
    examples_dir = REPO_ROOT / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Set up paths with absolute references
    toml_path = examples_dir / "simple_mcp_demo.toml"
    config_path = REPO_ROOT / "config" / "mcp_servers.json"
    
    # Remove any existing file to ensure we have the correct config path
    if toml_path.exists():
        toml_path.unlink()
        
    # Create the TOML file with the absolute path to the config file
    toml_content = f"""[model]
name = "claude-3-5-haiku-20241022"
provider = "anthropic"
display_name = "Claude MCP Assistant"

[parameters]
temperature = 0.7
max_tokens = 1000

[prompt]
system_prompt = \"\"\"You are a helpful assistant with access to tools.
When you need to perform specific tasks or look up information, use the appropriate tool.
Be concise and clear in your responses.
\"\"\"

[mcp]
config_path = "{config_path.absolute()}"

[mcp.tools]
# Using 'all' to capture any tools available from any servers
unknown = "all" 
"""
    toml_path.write_text(toml_content)
    
    return toml_path

def run_examples(process):
    """Run example queries with the LLMProcess."""
    examples = [
        "What tools are available to you? Explain them briefly.",
        "Can you search for information about 'machine learning frameworks'?",
        "Can you find popular Python repositories on GitHub?",
        "Search for information about 'transformer models' and find a GitHub repository related to transformers."
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n=== Example {i}: {example} ===")
        response = process.run(example)
        print(f"\n{process.display_name}> {response}\n")
        print("-" * 80)

def main():
    """Main function to run the MCP demo."""
    print(f"Working in repository root: {REPO_ROOT}")
    
    # Set up the MCP config
    config_path = setup_mcp_config()
    print(f"Using MCP config: {config_path.absolute()}")
    
    # Create the TOML file
    toml_path = create_mcp_toml()
    print(f"Created TOML config: {toml_path.absolute()}")
    
    # Check for required environment variables
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it with: export ANTHROPIC_API_KEY=your-key-here")
        return
    
    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN environment variable is not set.")
        print("GitHub tools may not work correctly.")
        print("Please set it with: export GITHUB_TOKEN=your-token-here")
    
    try:
        # Create the LLMProcess from the TOML file
        print(f"Loading configuration from {toml_path.absolute()}...")
        process = LLMProcess.from_toml(toml_path)
        print(f"Loaded {process.display_name} with MCP support.")
        
        # Debug MCP tools
        if hasattr(process, 'tools') and process.tools:
            print("\nRegistered MCP tools:")
            for i, tool in enumerate(process.tools, 1):
                print(f"  {i}. {tool['name']} - {tool['description'][:60]}...")
        else:
            print("\nWarning: No MCP tools were registered!")
        
        # Run the examples
        print("\nRunning examples...\n")
        run_examples(process)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
