# CLAUDE.md - Session and Repository Summary

## Repository Structure
- `src/llmproc/`: Main package directory containing LLMProcess implementation
- `examples/`: TOML configuration examples (minimal.toml, complex.toml)
- `prompts/`: Contains system prompt templates for LLM configuration
- `tests/`: Test files with 90% coverage of codebase
- `worktrees/`: Development branches as worktrees

## Session Procedures
- Start of Session: Read README.md and repo-map.txt to get familiar with the codebase
- End of Session: Update repo-map.txt with any changes made during the session

## Key Commands
- Install: `uv pip install -e ".[dev,all]"`
- Start CLI: `llmproc-demo ./examples/claude_code.toml`
- Try program linking: `llmproc-demo ./examples/program_linking/main.toml`
- Manage dependencies: `uv add <package>`, `uv add --dev <package>`, `uv remove <package>`
- Create worktree: `git worktree add worktrees/feature-name feature/feature-name`

## Testing Procedures
- Run standard tests: `pytest`
- Run verbose tests: `pytest -v`
- Run specific test file: `pytest tests/test_file.py`
- Run specific test class or function: `pytest tests/test_file.py::TestClass::test_function`
- Run tests with API access (requires API keys): `pytest -m llm_api`
- Test example programs with API: `pytest tests/test_example_programs.py -m llm_api`
- Test program linking: `pytest tests/test_program_linking_robust.py`
- Test program linking with API access: `pytest tests/test_program_linking_api.py -m llm_api`
- Debug output: Configure Python logging for `llmproc` package

### Test Coverage
- Unit tests: Cover core functionality without API calls (~41% code coverage)
- Provider tests: Mock API responses to test provider integrations
- MCP tests: Verify Model Context Protocol tool implementation
- Program Linking tests: Verify LLM-to-LLM communication features
- Example configurations tests: Verify all example TOML files work with actual APIs
- CLI tests: Verify command-line interface functionality with actual APIs
- API Integration tests: Verify end-to-end functionality with real API calls
- All tests are marked appropriately to skip API tests by default

## Code Style Guidelines
- Uses absolute imports (`from llmproc import X`)
- Type hints on all functions and methods
- Google-style docstrings
- Max line length of 88 characters (Black)
- PEP8 compliant with Ruff enforcement

## LLMProcess Features
- Configurable via TOML files with validation
- Supports system prompts from strings or files
- Maintains conversation state
- Parameters configurable via TOML
- File preloading for context via system prompt (using [preload] section in TOML or preload_files() method)
- Custom display names for models in CLI interfaces
- Supports OpenAI, Anthropic, and Vertex AI models
- MCP (Model Context Protocol) support for tool usage
- Program linking for LLM-to-LLM communication via spawn tool
- Program compiler for validation and preprocessing of configurations
- Methods: run() [async], get_state(), reset_state(), from_toml(), preload_files()
- Command-line interface for interactive chat sessions
- Comprehensive error handling and diagnostics

## Session Summary (2025-03-12)
1. Restructured project to src/llmproc layout
2. Added comprehensive tests with pytest
3. Added type hints and documentation
4. Set up pre-commit hooks
5. Added dev tools configuration
6. Fixed file path issues in examples
7. Updated example script for shorter responses
8. Created and updated repository documentation

## Session Summary (2025-03-13)
1. Added file preloading feature to LLMProcess
2. Created preload.toml example to demonstrate the feature
3. Updated reference.toml with preload section documentation
4. Enhanced example.py to showcase preload functionality
5. Implemented XML-formatted warnings for missing files
6. Added XML-tagged format for preloaded file content
7. Updated reset_state to handle preloaded content consistently
8. Added preload_files() method for runtime file preloading
9. Created detailed documentation in docs/preload-feature.md
10. Added LLM evaluation test for preload feature
11. Updated pytest.ini with markers for API tests
12. Updated documentation in repo-map.txt and CLAUDE.md

## Session Summary (2025-03-14)
1. Added command-line interface (CLI) for interactive chat
2. Implemented model display name feature for better UX
3. Fixed Anthropic API system message handling 
4. Created configuration file selection functionality
5. Added configuration summary display
6. Implemented interactive chat with custom prompts
7. Added direct TOML file path specification support
8. Successfully merged CLI feature with preload functionality
9. Updated examples with display_name field
10. Updated documentation for CLI and display_name features

## Session Summary (2025-03-15)
1. Implemented async run method with proper tool execution support
2. Created unified API for both synchronous and asynchronous contexts
3. Added event loop detection and automatic handling
4. Fixed multi-turn tool execution for Anthropic models
5. Improved error handling for MCP tool responses
6. Added debug_tools parameter for detailed tool execution logging
7. Created comprehensive examples demonstrating tool usage
8. Updated documentation to reflect the new unified run method
9. Added unit tests for async tool execution
10. Updated README and feature status documentation

## Session Summary (2025-03-16)
1. Refactored provider implementations into a dedicated providers/ directory
2. Created separate anthropic_tools.py module for Anthropic-specific tool implementations
3. Fixed provider imports in test files for compatibility with new structure
4. Added debug_dumps directory for error logging and debugging
5. Fixed empty text blocks handling in Anthropic API messages
6. Completed repository cleanup and refactoring
7. Updated MCP configuration and documentation
8. Fixed test imports to work with new module structure
9. Updated repo-map.txt to reflect the new directory organization
10. Maintained backward compatibility for existing code

## Session Summary (2025-03-17)
1. Refactored MCP tools implementation for better organization and maintainability
2. Improved async handling in tool execution flow
3. Added proper type annotations throughout MCP code
4. Separated tool response processing from tool execution logic
5. Implemented more robust error handling for Anthropic API
6. Added specific tests for MCP tools functionality
7. Created mock time server test infrastructure
8. Updated pytest.ini to properly support async tests
9. Improved test isolation to avoid requiring API keys
10. Updated documentation in CONTRIBUTING.md and CLAUDE.md

## Session Summary (2025-03-18)
1. Completely rewrote _initialize_mcp_tools for clarity and efficiency
2. Implemented flexible tool name matching with case-insensitivity
3. Added support for different naming conventions (snake_case/camelCase)
4. Fixed proper namespacing of tools with server prefixes
5. Improved server configuration validation and error handling
6. Removed unpredictable automatic fallback behaviors
7. Added better diagnostic messages for tool registration
8. Created comprehensive tests for MCP tools functionality
9. Fixed mcp.toml example to use correct tool naming format
10. Optimized server initialization by checking configuration first

## Session Summary (2025-03-19)
1. Implemented program linking feature for communication between LLMs
2. Created spawn tool for delegating queries to specialized LLMs
3. Added [linked_programs] section to TOML configuration
4. Added support for [tools] section to enable built-in tools
5. Improved parameter handling with cleaner API parameters approach
6. Implemented custom tool handlers for non-MCP tools
7. Added example configurations in examples/program_linking/
8. Created detailed documentation in docs/program-linking.md
9. Added comprehensive tests for program linking functionality
10. Updated reference.toml with documentation for new sections

## Session Summary (2025-03-20)
1. Fixed critical bug in program linking with Anthropic API error handling
2. Removed unnecessary filter_empty_text_blocks function in favor of direct filtering
3. Fixed empty message handling to prevent 400 errors from Anthropic API
4. Updated LLMProcess to skip empty messages when preparing API requests
5. Modified run_anthropic_with_tools to only add messages with content
6. Added fallback message for empty responses
7. Created comprehensive test cases for empty message handling
8. Added robust test file for program linking unit tests
9. Created API integration tests for program linking with real APIs
10. Updated CLAUDE.md with detailed testing procedures
11. Added MISC.md with additional documentation and advanced usage
12. Added logging support for debugging and diagnostics

## Session Summary (2025-03-21)
1. Improved preloaded files implementation by appending to system prompt
2. Changed preloaded content from user/assistant messages to part of system prompt
3. Added tracking of original system prompt for proper reset handling
4. Updated reset_state to maintain or remove preloaded content correctly
5. Fixed bug in MCP tools response content processing
6. Added comprehensive tests for the new preload implementation
7. Updated documentation to reflect the new approach
8. Simplified preload logic to be more direct
9. Enhanced reference.toml with clearer explanation of preload functionality

## Session Summary (2025-03-15)
1. Added comprehensive API tests for all example programs
2. Created tests for CLI interface with real API calls
3. Implemented test for unique message echo verification
4. Added tests for program linking through CLI
5. Created tests for error handling and recovery
6. Improved test skip behavior when API keys aren't available
7. Added API testing documentation in docs/api_testing.md
8. Updated README and CLAUDE.md with testing information
9. Added basic validation for TOML program files
10. Enhanced test coverage for all example programs

## Session Summary (2025-03-22)
1. Standardized terminology across codebase: "configuration" → "program"
2. Added Terminology section to MISC.md defining key terms
3. Updated docstrings, function parameters, and variable names in llm_process.py
4. Renamed CLI parameters from config_* to program_*
5. Fixed documentation in README, program-linking.md, and preload-feature.md
6. Fixed test_mcp_tools.py test to match function return signature 
7. Created test_example_programs.py to test all example programs with real APIs
8. Used echo verification to ensure models respond correctly
9. Enhanced error handling in tests for invalid programs
10. Added detailed API testing guide in docs/api_testing.md
11. Identified additional terminology inconsistencies for future improvement:
    - "LLM" vs "model" vs "agent" usage
    - "prompt" vs "system prompt" vs "system message"
    - "process" vs "instance" usage
    - Inconsistent capitalization of system calls

## Session Summary (2025-03-23)
1. Implemented program compiler for robust validation and preprocessing
2. Created LLMProgram class to separate configuration parsing from LLMProcess
3. Added Pydantic models for comprehensive configuration validation
4. Simplified API parameter extraction with cleaner loop-based approach
5. Created new [debug] section in TOML for debug-related configuration
6. Moved debug_tools from parameters to dedicated debug section
7. Updated from_toml method to use the new compiler
8. Added comprehensive documentation in docs/program-compiler.md
9. Created program_compiler_example.py to demonstrate usage
10. Added test_program_compiler.py with validation tests
11. Added clear error messages for configuration issues
12. Updated all documentation to reflect the new features

## Session Summary (2025-03-24)
1. Refactored LLMProcess to accept a program parameter instead of individual fields
2. Removed create_for_testing helper to enforce the new Program API
3. Updated core tests in test_llm_process.py to use the Program-based API
4. Updated provider tests in test_llm_process_providers.py with the new pattern
5. Fixed docstrings and comments to reflect the new API patterns
6. Added clearer examples of creating processes from programs in README.md
7. Improved type annotations and imports for better static analysis

## Session Summary (2025-03-25)
1. Completed Program API refactoring for all test modules
   - Updated all tests to use the program-based API rather than direct parameter construction
   - Maintained test coverage while transitioning to new interface
   - Fixed instantiate() method in LLMProgram to use the updated API pattern

2. Implemented System Prompt & Enriched System Prompt separation
   - Added clear distinction between base system_prompt and enriched_system_prompt
   - Made enriched_system_prompt lazily initialized on first message
   - Empty initial state until first run (more efficient)
   - Preserved original system prompt for reset operations

3. Added Environment Information Framework
   - Created new [env_info] section in TOML with variables list
   - Added support for standard environment variables:
     - working_directory, platform, date, python_version, hostname, username
   - Implemented custom variables as key-value pairs in TOML
   - Made environment information disabled by default (opt-in model)
   - Added variables="all" option for comprehensive environment details
   - Created examples/env_info.toml demonstration example
   - Standardized format with <env> XML tags

4. Improved System Prompt Loading & Error Handling
   - Added PromptConfig.resolve() method to replace _load_system_prompt
   - Made system prompt file errors throw proper FileNotFoundError instead of warnings
   - Same for MCP config files - failing fast with proper errors
   - Enhanced error messages to show both specified path and resolved path
   - Improved documentation with clearer exception descriptions
   - Updated docstrings to document exceptions that might be raised

5. Renamed and Standardized Path Handling
   - Changed config_dir to base_dir throughout codebase
   - Improved handling of relative vs absolute paths
   - Added directory existence checks
   - Note: We maintain backward compatibility by still using config_dir in LLMProcess but it now maps to program.base_dir

## Session Summary (2025-03-28)
1. Improved Program Compilation Semantics
   - Added global Program Registry singleton to avoid redundant compilation
   - Unified compile() and compile_all() into a single, more powerful method
   - Implemented non-recursive BFS for safer traversal of program graphs
   - Made linked programs directly reference compiled Program objects
   - Fixed duplicate tool registration issue in linked programs
   - Added test script to verify tool registration behavior
   - Refactored code for better separation of compilation and instantiation

2. Enhanced Memory Efficiency
   - Implemented lazy instantiation of LLMProcess objects
   - Programs are now stored once in registry and reused
   - Processes are only created when needed (via spawn tool)
   - Reduced memory footprint for applications with many linked programs

3. Improved Error Handling
   - Better path resolution for linked programs
   - Enhanced error messages for missing files
   - More robust circular dependency detection
   - Fixed edge cases in tool registration

4. Updated Documentation and Tests
   - Updated CLAUDE.md with latest changes
   - Fixed all tests to work with the new compilation semantics
   - Updated test assertions to check for program objects instead of paths

## Session Summary (2025-03-27)
1. Enhanced Program Compilation and Linking System
   - Implemented recursive compilation for program graphs with `compile_all` method
   - Compiled programs are tracked by absolute file paths to avoid redundant compilation
   - Added proper handling of circular dependencies in the program graph
   - Enhanced error handling with appropriate warnings for missing files
   - Optimized for large program graphs with many shared dependencies
   - Created comprehensive tests for various program graph scenarios

2. Improved Program Linking Process
   - Enhanced `from_toml` method to compile and link an entire program graph
   - Made linked programs accessible at all levels of the hierarchy
   - Redesigned `_initialize_linked_programs` with a multi-pass approach
   - Added clear distinction between compilation and linking phases
   - Ensured program paths are properly resolved in all contexts

3. Added Comprehensive Documentation
   - Created detailed `program-compilation.md` explaining the compilation process
   - Added `program-linking-advantages.md` highlighting benefits of the enhanced system
   - Documented real-world use cases and best practices
   - Created example program graphs for different scenarios
   - Added detailed documentation of error handling and debugging

4. Fixed Critical Program Linking Issue
   - Identified and fixed a format validation issue in linked_programs section
   - Created test cases to verify correct program format
   - Updated example programs to use correct format
   - Enhanced error messages to clearly indicate format issues

## Session Summary (2025-03-29)
1. Implemented Fork System Call
   - Created `fork.py` with fork_tool function in tools directory
   - Added fork_process method to LLMProcess for creating process copies
   - Implemented proper deep copying of conversation state and context
   - Modified LLMProcess to register the fork tool when enabled
   - Ensured forks inherit full conversation history from parent

2. Added Testing and Documentation
   - Created `test_fork_tool.py` with comprehensive test cases
   - Added unit tests for fork_process method and fork tool function
   - Created API tests for real-world fork functionality
   - Added fork option in reference.toml documentation
   - Created detailed fork-feature.md documentation file

3. Added Examples and Configuration
   - Created fork.toml example with the fork system call enabled
   - Updated the repo-map.txt file to include new files
   - Added session summary to CLAUDE.md
   - Made fork tool compatible with existing code structure
   - Enhanced error handling for fork operations

4. Enhanced the Fork Implementation
   - Designed fork_tool to return results from all child processes
   - Structured fork results in a consistent format
   - Added detailed logging for debugging and diagnostics
   - Preserved all process state including preloaded content
   - Ensured tools are properly initialized in forked processes

## Session Summary (2025-03-26)
1. Improved preload file error handling with consistent warnings
   - Updated preload file warnings to use Python warnings module instead of print()
   - Added both specified and resolved paths to warning messages for clarity
   - Maintained non-breaking behavior for missing preload files (warnings vs exceptions)
   - Added comprehensive tests for preload file warnings in test_program_compiler.py
   - Improved docstrings to clarify behavior with missing files
   - Ensured compiler issues warnings but doesn't fail for missing preload files

2. Refactored tests to consistently use program-based API
   - Updated all test files to use LLMProgram + LLMProcess pattern
   - Replaced direct constructor calls with two-step initialization
   - Updated assertions to match the lazy system prompt initialization
   - Improved test readability with clear initialization steps

3. Improved linked program initialization with compilation semantics
   - Updated _initialize_linked_programs to use LLMProgram.compile()
   - Applied consistent two-step initialization to linked programs
   - Used warnings module for consistent warning messages
   - Added better error handling and more informative logging
   - Updated tests to reflect the new implementation

4. Fixed environment information handling in program linking
   - Added missing include_env parameter to LLMProgram.get_enriched_system_prompt() method
   - Made environment info inclusion optional with include_env parameter (default: True)
   - Created comprehensive tests for different environment inclusion scenarios
   - Fixed program linking demo example that was failing due to this issue

## Completed Tasks

1. **Improved Program Compilation Semantics**:
   - Added global singleton Program Registry to avoid redundant compilation
   - Unified the compile() method with non-recursive BFS approach
   - Enhanced linked programs to reference compiled Program objects directly
   - Fixed duplicate tool registration when spawning linked programs
   - Added test script to verify tool registration

## Pending Tasks (Next Session)

1. **Testing for Environment Information**:
   - Add dedicated tests for environment information
   - Test error cases for file not found errors
   - Test all environment variable configurations

2. **Path Resolution Improvements**:
   - Factor out path resolution logic to a dedicated helper
   - Make handling of linked program paths more consistent

3. **Documentation Updates**:
   - Add environment info section documentation to README.md
   - Create examples showing common environment variable uses
   - Document security implications of environment info

## Implementation Notes

Important details about our changes:

1. **Terminology**:
   - system_prompt: Original base system prompt from user
   - enriched_system_prompt: Final prompt with environment info and preloaded files
   - This distinction ensures resets can preserve the original prompt while regenerating enhanced version as needed

2. **Environment Info Format**:
   ```toml
   [env_info]
   # List form for specific variables
   variables = ["working_directory", "platform", "date"]
   
   # Or enable all standard variables
   # variables = "all"
   
   # Custom environment variables
   project_name = "My Project"
   git_branch = "main"
   ```

3. **Error Messages**:
   - Show both specified and resolved paths:
   - "System prompt file not found - Specified: 'prompt.md', Resolved: '/path/to/prompt.md'"
   - "MCP config file not found - Specified: 'config.json', Resolved: '/path/to/config.json'"

4. **API Changes**:
   - We've maintained backward compatibility across test classes
   - All tests now use the program-based construction API
   - We've simplified the internals with methods like PromptConfig.resolve()