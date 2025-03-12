# LLMProc

A simple, flexible framework for building LLM-powered applications with a standardized configuration approach.

## Features

- Load configurations from TOML files
- Maintain conversation state
- Support for different LLM providers (OpenAI initially)
- Parameter customization
- Simple API for easy integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llmproc.git
cd llmproc

# Install as a development package using uv
uv pip install -e .

# Managing dependencies with uv
uv add pandas        # Add a package to dependencies
uv add --dev pytest  # Add a package to dev dependencies
uv remove pandas     # Remove a package
uv pip freeze > requirements.txt  # Update requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
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

Complex example:

```toml
[model]
name = "gpt-4o"
provider = "openai"
temperature = 0.7
max_tokens = 1000

[prompt]
system_prompt_file = "prompts/example_prompt.md"

[parameters]
top_p = 0.95
frequency_penalty = 0.0
presence_penalty = 0.0
```

## License

MIT