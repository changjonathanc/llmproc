#!/bin/bash

# Check for Anthropic API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable is not set."
  echo "Please set your Anthropic API key with: export ANTHROPIC_API_KEY=your-key-here"
  exit 1
fi

# Check for GitHub token (for GitHub tools)
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Warning: GITHUB_TOKEN environment variable is not set."
  echo "GitHub tools may not work correctly."
  echo "Set your GitHub token with: export GITHUB_TOKEN=your-token-here"
fi

# Ensure config directory exists
mkdir -p config

# Create MCP servers config file if it doesn't exist
if [ ! -f "config/mcp_servers.json" ]; then
  echo "Creating MCP servers configuration..."
  cat > config/mcp_servers.json << EOF
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "\${GITHUB_TOKEN}"
      }
    },
    "codemcp": {
      "type": "stdio",
      "command": "/bin/zsh",
      "args": [
        "-c",
        "uvx --from git+https://github.com/cccntu/codemcp@main codemcp "
      ]
    }
  }
}
EOF
  echo "MCP servers configuration created."
fi

# Install the package in development mode if needed
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment and installing the package..."
  uv venv
  source .venv/bin/activate
  uv pip install -e .
  echo "Package installed."
else
  source .venv/bin/activate
fi

# Run the CLI with our MCP demo configuration
python -m llmproc.cli examples/mcp.toml

echo "Demo completed."