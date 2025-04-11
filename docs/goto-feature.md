# GOTO (Time Travel) Feature

The GOTO feature allows LLMs to reset a conversation to a previous point and start fresh from there. This enables powerful self-correction, context compression, and conversational exploration capabilities without requiring user intervention.

## Overview

LLMs often find themselves in conversational states where:
1. The discussion has gone off-track or down an unproductive path
2. A misunderstanding occurred early that affects all subsequent responses
3. The LLM realizes it needs to restart from an earlier point with improved context
4. A complex reasoning path led to a dead end, requiring a different approach

With the GOTO tool, LLMs can autonomously reset the conversation to a more productive state, much like time travel within the conversation. This follows the Unix-inspired design philosophy where processes have control over their state. For more on when to use this feature versus the fork tool, see the [API Design FAQ](../FAQ.md#when-to-use-fork-and-goto).

## Key Features

- **Self-correction**: LLMs can recover from conversational dead-ends autonomously
- **Context compression**: After gathering information through multiple tool calls, compress findings into a single message
- **Alternative approaches**: Explore different reasoning paths without losing context
- **User-directed resets**: Users can ask the LLM to restart the conversation from a specific point
- **Streaming-compatible**: Works with both streaming and non-streaming LLM implementations
- **Clear demarcation**: Uses XML tags for structured "time travel" messages

## How It Works

Each message in the conversation is assigned a unique ID (msg_0, msg_1, etc.). The LLM can use the GOTO tool to:

1. Specify a target message ID to reset to (e.g., "msg_0" for the very beginning)
2. Provide a new message to add at that point (explaining the reason for the reset)
3. The system then truncates the conversation history at the target point
4. The new message is added with special `<time_travel>` XML tags
5. The LLM then continues the conversation from this new state

## Usage Example

### Configuration

To enable the GOTO tool in your LLMProcess configuration:

```toml
[tools]
enabled = ["goto", "read_file", "calculator"]  # Include "goto" in the enabled tools list
```

### Conversation Example

Here's a conversation example showing the GOTO tool in action:

```
[msg_0] User: Hello, what can you help me with?
[msg_1] Assistant: I can help with reading files, calculations, and more.
[msg_2] User: Can you explain how black holes work?
[msg_3] Assistant: Black holes are regions of spacetime where gravity is so strong...

[LLM uses GOTO tool to reset to msg_0 with message:]
"Let's start over and talk about AI instead."

The conversation resets to:
[msg_0] User: Hello, what can you help me with?
[msg_1] User: [SYSTEM NOTE: Conversation reset to message msg_0. 3 messages were removed.]

<time_travel>
Let's start over and talk about AI instead.
</time_travel>

[msg_2] Assistant: I'd be happy to talk about AI. Artificial Intelligence refers to...
```

## Implementation

The GOTO tool consists of two main components:

1. **Tool Definition**: Contains detailed instructions for the LLM on when and how to use the time travel capability
2. **Handler Function**: Implements the actual state manipulation to reset the conversation

The tool definition includes guidance for the LLM on:
- When to use time travel (explicit requests, off-track conversations)
- How to use the position parameter (specifying message IDs)
- XML formatting for time travel messages
- What happens after using the tool (state reset)

## Demo Script

The repository includes a demo script that showcases the GOTO feature. You can run it with:

```bash
python examples/scripts/goto_demo.py
```

This interactive demo allows you to experience the time travel capability directly and provides visual feedback when time travel occurs.

## Testing

The GOTO feature includes comprehensive tests:
- Unit tests for core functionality (message identification, state truncation)
- API tests that verify correct behavior with actual LLM interactions
- Demo scripts that showcase the feature in interactive contexts

## Use Cases

- **Self-correction**: LLM realizes it misunderstood a question and resets to try again
- **Task switching**: Reset conversation before starting a completely new topic
- **Context compression**: After multiple information-gathering steps, reset and summarize findings
- **Exploration**: Try multiple approaches to solving a problem by branching from the same starting point
- **Recovery**: Recover from hallucinations or reasoning errors by going back to a known-good state

## Best Practices

1. **Targeted instructions**: In system prompts, clearly instruct the LLM on when time travel is appropriate
2. **User permission**: Consider requiring explicit user permission before letting the LLM time travel
3. **Monitoring**: Use callbacks to track time travel operations for debugging and analysis
4. **Clear transitions**: Encourage the LLM to explain why it's using time travel

## Future Extensions

Potential enhancements to the GOTO feature:
- Timeline branching with multiple save points
- Different traversal modes (step-by-step vs. direct jumps)
- Automatic tagging of important conversation milestones
- Selective state retention during time travel

## Related Features

- **[Fork Tool](./fork-feature.md)**: Creates process copies, complementary to GOTO
- **[File Descriptor System](./file-descriptor-system.md)**: Handles large outputs
- **[Program Linking](./program-linking.md)**: Multi-agent collaboration