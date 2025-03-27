# RFC008: OpenAI Reasoning Model Support

## Terminology and Conventions

This document uses the following standardized terminology and conventions:

- **Reasoning Models**: OpenAI models that explicitly use chain-of-thought reasoning, including the o1-mini, o1, o3-mini, and o3 model families.
- **Reasoning Effort**: The parameter controlling how much "thinking" a reasoning model performs before responding, with options of "low", "medium", or "high".
- **Code Examples**: All code examples use Python syntax with 4-space indentation. String literals use triple quotes for multi-line strings.
- **API Parameters**: All parameter names use snake_case. Parameter names are consistent across all RFCs.
- **Implementation Status**: Features are marked with ✅ (implemented), 🔄 (in progress), or ❌ (not implemented).

## 1. Background & Problem Statement

OpenAI has introduced a new series of reasoning models (o1, o1-mini, o3, o3-mini) that have fundamentally different capabilities from their earlier GPT models. These models:

- Use explicit chain-of-thought reasoning to solve complex problems
- Support a reasoning_effort parameter to control the thinking-response tradeoff
- Excel at STEM reasoning tasks including science, mathematics, and coding
- Have significantly larger context windows (up to 200,000 tokens) and outputs (up to 100,000 tokens)

The current LLMProc implementation has limited OpenAI support. As reasoning models become production-ready, we need to properly integrate them into our framework to leverage their enhanced capabilities, particularly their reasoning abilities.

While these models also support function calling (tools), this RFC is intentionally limited in scope to focus only on basic reasoning model support. Tool functionality will be addressed in a future RFC after we have established stable support for the basic reasoning capabilities.

## 2. Goals

- Implement support for OpenAI reasoning models in the LLMProc framework
- Enable configuration of reasoning-specific parameters like reasoning_effort
- Maintain a consistent interface between providers
- Provide clear examples of how to use reasoning models effectively
- Explicitly exclude tool support in this phase of implementation

## 3. Solution: OpenAI Reasoning Model Support

The proposal introduces support for OpenAI reasoning models, enabling their unique reasoning capabilities while maintaining compatibility with the existing LLMProc architecture:

1. Add reasoning model support to the OpenAI process executor
2. Add support for the reasoning_effort parameter
3. Provide examples demonstrating effective use of reasoning models

## 4. Architecture

### 4.1 OpenAI Process Executor Enhancements

The OpenAI process executor will be enhanced to support reasoning models:

1. **Reasoning Effort Configuration**:
   - Add reasoning_effort parameter to API calls
   - Support configuration via TOML

### 4.2 Configuration Schema

The configuration schema will be extended to support reasoning model specific parameters:

```python
class OpenAIAPIParams(BaseModel):
    """OpenAI API parameters."""
    
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    reasoning_effort: Optional[str] = None  # New parameter for reasoning models
    # Additional parameters...
```

## 5. API Design

The existing OpenAI process executor API will remain largely unchanged, with the addition of support for the reasoning_effort parameter.

## 6. Implementation

The implementation will focus on adding support for the reasoning_effort parameter to the OpenAI process executor.

## 7. Configuration

Configuration will use the existing TOML pattern with additions for reasoning models:

```toml
[model]
name = "o3-mini"
provider = "openai"

[prompt]
system_prompt = "You are a helpful assistant."

[parameters]
temperature = 0.7
reasoning_effort = "medium"  # Options: "low", "medium", "high"
```

## 8. Example Usage

### Basic Usage

```python
import asyncio
from llmproc import LLMProgram

async def main():
    # Load program from TOML
    program = LLMProgram.from_toml('examples/openai_reasoning.toml')

    # Start the process
    process = await program.start()

    # Run the process with user input
    run_result = await process.run('Solve this math problem: What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?')
    response = process.get_last_message()
    print(response)

# Run the async example
asyncio.run(main())
```

## 9. Testing and Quality Assurance

Tests will be added to verify:

1. Basic functionality with reasoning models
2. Reasoning effort parameter effects
3. Proper error handling

## 10. Future Considerations

1. **Chain-of-Thought Visualization**: Add support for visualizing the chain-of-thought reasoning process
2. **Reasoning Effort Optimization**: Develop guidance on optimal reasoning_effort settings for different tasks
3. **Cross-Provider Reasoning**: Standardize reasoning across providers
4. **Enhanced Monitoring**: Add metrics for tracking reasoning performance
5. **Tool Support Implementation**: Implement function calling (tool) support for OpenAI reasoning models in a future RFC
6. **Fork Functionality**: Add fork support for OpenAI models in a separate implementation phase

## 11. References

- [OpenAI o3-mini Announcement](https://community.openai.com/t/launching-o3-mini-in-the-api/1109387)
- [OpenAI API Documentation for Reasoning Models](https://platform.openai.com/docs/models#o3-mini)
- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)