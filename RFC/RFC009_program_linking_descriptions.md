# RFC009: Enhancing Program Linking with Descriptions [IMPLEMENTED]

## Summary
This RFC proposes adding a simple mechanism for parent programs to provide descriptions of linked programs, making it easier for LLMs to understand when to use each linked program.

## Motivation
The program linking feature allows a parent LLM to delegate tasks to specialized child processes. Currently, users need to manually describe each linked program's purpose in the parent's system prompt. Adding structured descriptions would:

1. Make it easier to provide standardized descriptions of linked programs
2. Allow the spawn tool to automatically include program descriptions in its help text
3. Enable better error messages that suggest appropriate programs based on descriptions

## Approach
Add support for program descriptions in the linked programs configuration:

```toml
[linked_programs]
# Simple form (backward compatible)
simple_program = "./simple.toml"

# Enhanced section-based form with description
[linked_programs.repo_expert]
path = "./repo_expert.toml"
description = "Expert specialized in repository analysis"
```

## Implementation Details

### Schema Changes
```python
class LinkedProgramItem(BaseModel):
    """Configuration for a single linked program."""
    path: str
    description: str = ""

class LinkedProgramsConfig(RootModel):
    """Linked programs configuration section."""
    root: dict[str, str | LinkedProgramItem] = {}
```

### Program Class Changes
- Add `linked_program_descriptions` dictionary to `LLMProgram` class
- Populate descriptions when parsing TOML configuration
- Pass descriptions to `LLMProcess` when starting

### Spawn Tool Changes
- Update the tool description template to include available programs with their descriptions
- Enhance error messages to include program descriptions when suggesting alternatives

## Example Usage

```toml
# main.toml
[model]
name = "claude-3-haiku-20240307"
provider = "anthropic"

[prompt]
system_prompt = """You are Claude, a helpful AI assistant.
You have access to the 'spawn' tool that lets you communicate with specialized experts."""

[tools]
enabled = ["spawn"]

# Section-based format for clearer organization
[linked_programs.repo_expert]
path = "./repo_expert.toml"
description = "Specialized in analyzing repository structure and codebase organization"

[linked_programs.math_helper]
path = "./math_helper.toml"
description = "Specialized in mathematical calculations and proofs"
```

## Backward Compatibility
This proposal maintains backward compatibility with existing configurations by supporting both string paths and structured objects in the `[linked_programs]` section. Existing configurations will continue to work with empty descriptions.

## Testing Plan
1. Unit tests for the enhanced schema parsing
2. Integration tests to verify descriptions are properly loaded
3. System tests to confirm the spawn tool correctly presents descriptions
4. Backward compatibility tests with existing TOML files

## Implementation Status
This RFC has been fully implemented on 2025-03-28. The implementation includes:

1. ✅ Updated schema to support structured linked program definitions
2. ✅ Modified the program compiler to extract and store descriptions
3. ✅ Enhanced the spawn tool to include descriptions in its help text and error messages
4. ✅ Updated documentation and examples to demonstrate the feature
5. ✅ Added tests to verify the implementation

This feature enables more intuitive program linking by allowing LLMs to better understand when to use each linked program.