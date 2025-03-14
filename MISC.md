# LLMProc - Miscellaneous Notes & Details

This document contains additional information, tips, and implementation details that supplement the main README but aren't essential for getting started.

## Environment Variables

LLMProc supports the following environment variables:

- `OPENAI_API_KEY`: Required for OpenAI providers (GPT models)
- `ANTHROPIC_API_KEY`: Required for Anthropic providers (Claude models)
- `VERTEX_API_KEY`: Required for Google Vertex AI providers
- `LLMPROC_DEBUG`: Set to "true" to enable detailed debug output for tools and program linking

You can set these in your environment or include them in a `.env` file.

## Program Linking Details

### Feature Comparison

The program linking feature (via the spawn tool) is conceptually similar to the `dispatch_agent` tool in Claude Code. Both allow delegating specific tasks to specialized agents/processes. However, LLMProc's implementation gives you more control over configuration and enables fully custom, specialized LLMs.

### Example TOML Configurations

A complete program linking setup involves at least two TOML files:

**Main Process (main.toml):**
```toml
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"
display_name = "Main Assistant"

[prompt]
system_prompt = """You are a helpful assistant with access to specialized experts.
When users ask detailed questions about specific domains, use the 'spawn' tool to 
delegate to the appropriate expert."""

[parameters]
max_tokens = 1000

[tools]
enabled = ["spawn"]

[linked_programs]
finance_expert = "./finance_expert.toml"
code_expert = "./code_expert.toml"
```

**Expert Process (finance_expert.toml):**
```toml
[model]
name = "gpt-4o"
provider = "openai"

[prompt]
system_prompt = """You are a financial expert specializing in market analysis and investment strategies.
Provide detailed, accurate financial advice based on your expertise."""

[parameters]
max_tokens = 2000
temperature = 0.1

[preload]
files = [
  "./data/financial_terms.md",
  "./data/market_overview.md"
]
```

### Error Handling

When using program linking, consider these error handling approaches:

```python
from llmproc import LLMProcess

async def run_with_expert(query):
    try:
        main_process = LLMProcess.from_toml('main.toml')
        response = await main_process.run(query)
        return response
    except Exception as e:
        # Handle errors from either main process or linked processes
        print(f"Error during expert consultation: {str(e)}")
        return "I encountered an error while consulting with experts."
```

## Performance Considerations

- **Resource Usage**: Each linked program creates a separate LLMProcess instance, which increases memory usage
- **API Costs**: Using program linking will result in additional API calls to the model providers
- **Latency**: There is additional latency when delegating to linked programs due to the extra API calls
- **Optimization**: For frequent queries to the same expert, consider caching common responses

## Current Limitations

- Limited to Anthropic provider for the main process (due to MCP tool support)
- No automatic fallback if a linked program fails
- State is not shared between linked programs
- No automatic batching of related queries to the same expert
- Limited debugging information for tool execution

## Advanced Configurations

### Enabling Debug Mode

```bash
# Enable detailed debug output
export LLMPROC_DEBUG=true
llmproc-demo ./examples/program_linking/main.toml
```

### Chaining Multiple Experts

You can create chains of experts by configuring linked programs that themselves have linked programs:

```
Main Assistant → Domain Expert → Specialized Sub-Expert
```

### Multi-turn Conversations with Experts

To enable multi-turn conversations with the same expert, you'll need to implement a session management pattern that maintains state between calls to the expert.