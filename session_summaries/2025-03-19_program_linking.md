# Program Linking Implementation

## Overview

In this session, we implemented the program linking feature for LLMProc that allows one LLM to communicate with and delegate tasks to other specialized LLMs.

## Implementation Details

### Core Components:

1. **Linked Programs Configuration**
   - Added `[linked_programs]` section to TOML configuration
   - Created infrastructure for loading linked programs from their own TOML files
   - Implemented directory resolution for relative paths

2. **Spawn Tool**
   - Created dedicated `tools/` package with `spawn.py` implementation
   - Implemented the tool as an async function that takes program_name and query parameters
   - Designed simple error handling for missing programs and execution failures

3. **Tool Integration**
   - Added `[tools]` section for enabling built-in tools
   - Implemented automatic tool registration based on configuration
   - Created separate tool handlers storage to avoid JSON serialization issues

4. **Parameter Structure Improvements**
   - Implemented cleaner parameter handling with dedicated attributes for API parameters
   - Removed implicit behavior of filtering parameters
   - Made the code mirror the TOML structure more closely for better alignment

### Documentation:

1. Created comprehensive documentation in `docs/program-linking.md`
2. Added example configurations in `examples/program_linking/`
3. Updated `reference.toml` with new section documentation
4. Updated repository documentation (`CLAUDE.md` and `repo-map.txt`)

## Testing

Added comprehensive tests in `tests/test_program_linking.py`:
- Testing linked program initialization
- Testing spawn tool registration
- Testing spawn tool functionality with mock programs
- Testing error handling in various scenarios

## Usage Example

The feature allows creating specialized "expert" LLMs that can be consulted by a primary LLM:

```toml
# Primary LLM config
[tools]
enabled = ["spawn"]

[linked_programs]
repo_expert = "./repo_expert.toml"
```

When the primary LLM needs specialized knowledge about the repository, it can use:

```
I'll check with the repository expert about that...
```

And then invoke the spawn tool with the repo_expert program to get detailed information about the codebase.

## Future Enhancements

Potential future improvements for the program linking feature:

1. Stateful linked programs that maintain conversation history
2. Bidirectional communication between programs
3. Support for nested program linking (programs calling other programs)
4. Dedicated UI for visualizing program linking and tool usage
5. Memory sharing between linked programs