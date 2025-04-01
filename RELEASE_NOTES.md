# Release Notes

## Version 0.3.0 (2025-04-01)

This version includes significant refactoring and improvements to the codebase organization, as well as support for Claude 3.7 thinking models.

### Major Features

1. **Code Reorganization and Refactoring**
   - Extracted configuration loading to a dedicated ProgramLoader class
   - Simplified tool API by removing dictionary-based tools
   - Added method chaining support for better developer experience
   - Improved separation of concerns throughout the codebase

2. **Claude 3.7 Thinking Models Support**
   - Added support for Claude 3.7 models with thinking capability
   - Configurable thinking budget (high, medium, low)
   - Clear parameter validation for thinking models
   - Example configurations in examples/anthropic directory

3. **Enhanced API for Tool Management**
   - Added set_enabled_tools method for easier tool enabling
   - Streamlined tool registration process
   - Better error handling for tool execution

4. **Bug Fixes**
   - Fixed cache control handling in Anthropic API for thinking models
   - Fixed program linking bug in the program loader
   - Improved error handling for API parameters

### Documentation Updates

1. **Added New RFCs:**
   - RFC023: File Descriptor Manager
   - RFC024: Configuration Manager
   - RFC025: Simplified Tool Registration
   - RFC026: Unified Tool Management

2. **Updated Examples**
   - New examples for Claude 3.7 thinking models
   - Updated documentation for tool usage

### Dependency Updates
   - Support for newer versions of the Anthropic API
   - Token-efficient tools beta header for Claude 3.7 models

## Version 0.2.0 (2025-03-19)

This version includes significant improvements and new features over the previous 0.1.0 release.

### Major Features

1. **Program Linking**
   - Support for LLM-to-LLM communication via spawn tool
   - Configurable in TOML with [linked_programs] section
   - Similar to dispatch_agent in Claude Code

2. **Fork System Call**
   - Added fork tool for process duplication
   - Allows branching conversations
   - Preserves conversation state

3. **Environment Information Support**
   - Added env_info section to share context with LLMs
   - Configurable variables: working_directory, platform, date, etc.
   - Custom environment variables support

4. **Program Compiler**
   - Robust validation and preprocessing of TOML files
   - Automatic program linking and dependency resolution
   - Improved error messages for configuration issues

### Other Improvements

1. **Standardized Terminology**
   - "Program" for TOML files (was "config" previously)
   - "State" for conversation history (without system messages) 
   - Consistent naming across API and documentation

2. **Anthropic on Vertex AI Integration**
   - Standardized environment variables
   - Comprehensive documentation in docs/anthropic.md
   - Example configuration in examples/anthropic_vertex.toml

3. **API Pattern Standardization**
   - Two-step pattern: LLMProgram.from_toml() followed by program.start()
   - Consistent method signatures
   - Better error handling and propagation

4. **Documentation**
   - Improved README with clearer examples
   - Created comprehensive reference documentation
   - Added roadmap for future development

### Bug Fixes

1. Fixed system prompt handling (now stored separately from conversation state)
2. Fixed empty message handling for Anthropic API
3. Fixed file path resolution issues with linked programs
4. Improved error handling for missing files
5. Fixed duplicate tool registration in linked programs

## Version 0.1.0 (2025-03-12)

Initial release of LLMProc framework with basic features:

- TOML configuration support
- Basic API for OpenAI and Anthropic models
- Simple CLI interface
- File preloading capability