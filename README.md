# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

## Features

- Load configurations from TOML files
- Maintain conversation state
- Support for different LLM providers (OpenAI initially)
- Extensive parameter customization
- Simple API for easy integration
- Comprehensive documentation for all parameters

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

## License

MIT