# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

## Features

- Load configurations from TOML files
- Maintain conversation state
- Support for different LLM providers (OpenAI, Anthropic, Vertex)
- Extensive parameter customization
- Simple API for easy integration
- Command-line interface for interactive chat sessions
- Comprehensive documentation for all parameters
- File preloading for context preparation
- Model Context Protocol (MCP) support for tool usage *(in development)*

## Installation

```bash
# recommended
uv pip install -e .
# or
pip install -e .

# Set up environment variables
# supports .env file
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Example

```python
from llmproc import LLMProcess

# Load configuration from TOML
process = LLMProcess.from_toml('examples/minimal.toml')

# Run the process with user input
output = process.run('Hello!')
print(output)

# Continue the conversation
output = process.run('Tell me more about that.')
print(output)

# Reset the conversation state
process.reset_state()
```

### Example with MCP Tools (Experimental)

```python
import asyncio
from llmproc import LLMProcess

async def main():
    # Load configuration with MCP tools enabled
    process = LLMProcess.from_toml('examples/mcp.toml')
    
    # Run the process with tool support
    output = await process.run('Search for popular Python repositories on GitHub')
    print(output)
    
    # Continue with follow-up questions that may use tools
    output = await process.run('Which of these has the most stars?')
    print(output)

# Run the async example
asyncio.run(main())
```

The `run` method also works in synchronous code and will automatically handle the event loop creation:

```python
from llmproc import LLMProcess

# Load configuration with MCP tools
process = LLMProcess.from_toml('examples/mcp.toml')

# This will automatically create an event loop if needed
output = process.run('Search for Python repositories on GitHub')
print(output)
```

### TOML Configuration

Minimal example:

```toml
[model]
name = "gpt-4o-mini"
provider = "openai"

[prompt]
system_prompt = "You are a helpful assistant."
```

See `examples/reference.toml` for a comprehensive reference with comments for all supported parameters.

## Command-Line Demo

LLMProc includes a simple command-line demo for interacting with LLM models:

```bash
# Start the interactive demo (select from examples)
llmproc-demo

# Start with a specific TOML configuration file
llmproc-demo path/to/your/config.toml
```

The demo will:
1. If no config is specified, show a list of available TOML configurations from the examples directory
2. Let you select a configuration by number, or use the specified config file
3. Start an interactive chat session with the selected model

### Interactive Commands

In the interactive session, you can use the following commands:

- Type `exit` or `quit` to end the session
- Type `reset` to reset the conversation state

## License

MIT