# API Parameters for LLM Providers

## OpenAI API Parameters

- **temperature** (0-2): Controls randomness in responses. Higher values (e.g., 0.8) make output more random, while lower values (e.g., 0.2) make it more deterministic and focused.

- **max_tokens**: Maximum number of tokens to generate in the response. One token is roughly 4 characters for English text.

- **top_p** (0-1): Controls diversity via nucleus sampling. Set to 0.5 to consider only tokens comprising the top 50% probability mass. Lower values increase focus, higher values increase randomness.

- **frequency_penalty** (-2 to 2): Reduces repetition by penalizing tokens based on how frequently they've appeared in the text so far. Higher values discourage repetition.

- **presence_penalty** (-2 to 2): Reduces repetition by penalizing tokens that have appeared at all, regardless of frequency. Higher values encourage the model to talk about new topics.

- **stop**: Sequences where the API will stop generating further tokens. For example, ["\n", "Human:", "AI:"] would stop generation at a new line or when the tokens "Human:" or "AI:" are generated.

## Anthropic (Claude) API Parameters

- **temperature** (0-1): Controls randomness in responses. Values closer to 0 produce more deterministic responses, while values closer to 1 produce more creative responses. Unlike OpenAI, Anthropic caps temperature at 1.0.

- **max_tokens**: Maximum number of tokens to generate before stopping. The equivalent of OpenAI's max_tokens parameter.

- **top_p** (0-1): Uses nucleus sampling to limit token selection based on cumulative probability threshold.

- **top_k**: Only samples from the top K options for each subsequent token. Used to remove "long tail" low probability responses.

- **thinking**: Configure thinking capabilities (Claude 3.7+ only): `{type = "enabled", budget_tokens = 4000}`

## Claude on Vertex AI (anthropic_vertex) Parameters

- **temperature** (0-1): Controls randomness in responses, with values closer to 0 being more deterministic.

- **max_tokens**: Maximum number of tokens to generate. Same as direct Anthropic API.

- **top_p** (0-1): Controls diversity via nucleus sampling, similar to other providers.

- **top_k**: Only considers the top K tokens when generating each token in the response.

- **thinking**: Configure thinking capabilities (Claude 3.7+ only): `{type = "enabled", budget_tokens = 4000}`

### Using Claude Models on Vertex AI

When using Claude models through Vertex AI, the model name format includes the version with @ symbol:

- **claude-3-5-sonnet@20241022**: Claude 3.5 Sonnet model with timestamp
- **claude-3-5-haiku@20241022**: Claude 3.5 Haiku model with timestamp
- **claude-3-7-sonnet@20250219**: Claude 3.7 Sonnet model with timestamp

## Usage in TOML Configuration

In this project, all model generation parameters should be placed in the `[parameters]` section of the TOML configuration file, not in the `[model]` section:

```toml
[model]
name = "gpt-4o"
provider = "openai"

[parameters]
temperature = 0.7
max_tokens = 150
top_p = 0.95
frequency_penalty = 0.0
presence_penalty = 0.0
```
