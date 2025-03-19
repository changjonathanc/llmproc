# Release Notes

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