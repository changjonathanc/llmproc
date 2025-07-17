# Program Linking Example

This example demonstrates how to use the program linking feature with specialized models.

## Files

- `main.yaml`: Primary LLM configuration with spawn tool enabled
- `repo_expert.yaml`: Specialized LLM with LLMProc project knowledge pre-loaded into the enriched system prompt.
- Links to external models:
  - Claude 3.7 with high thinking budget
  - GPT-4.5 model for different insights

## Usage

Run the example:

```bash
llmproc-demo ./examples/features/program-linking/main.yaml
```

## Implementation Details

The program linking feature uses:

1. A `[plugins.spawn]` section with descriptions in TOML
2. The `spawn` tool for LLM-to-LLM communication
3. Background initialization of all linked programs

## Configuration Example

```yaml
model:
  name: "claude-3-haiku-20240307"
  provider: "anthropic"
  display_name: "Claude Haiku"

prompt:
  system_prompt: |
    You are Claude, a helpful AI assistant with access to specialized thinking experts.

    You have access to the 'spawn' tool that lets you communicate with specialized experts.
    The tool will provide descriptions of available experts to help you choose the right one.

tools:
  builtin:
    - spawn

# Section-based format for linked programs
plugins:
  spawn:
    linked_programs:
      repo_expert: ./repo_expert.toml
      thinking_expert: ../../anthropic/claude-3-7-thinking-high.toml
      gpt_judge: ../../openai/gpt-4-5.toml
```

With the enhanced description system, you don't need to manually list each expert in the system prompt. The spawn tool automatically makes the descriptions available to the model.
