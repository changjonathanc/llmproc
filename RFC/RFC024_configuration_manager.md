# RFC024: Configuration Management Refactoring

## Overview
This RFC proposes extracting configuration management functionality from the `LLMProgram` class to separate Python SDK functionality from TOML-specific code, improving modularity and focusing `LLMProgram` on its core responsibilities.

## Motivation
Currently, the `LLMProgram` class handles both core program functionality and TOML configuration loading. This makes the class larger than necessary (currently ~807 lines) and mixes different concerns. By extracting configuration loading functionality, we can:

1. Focus `LLMProgram` specifically on Python SDK use cases
2. Move TOML-related functionality outside the core class
3. Improve separation of concerns
4. Reduce file size and complexity
5. Make the codebase more maintainable

## Implementation Details

### New File Structure
- Create a new file: `/src/llmproc/config/program_loader.py`
- Use the existing `/src/llmproc/config/` directory

### ProgramLoader Class
```python
class ProgramLoader:
    """Loads LLMProgram configurations from various sources.

    This class handles loading and parsing program configurations from TOML files,
    separating configuration concerns from the LLMProgram class itself.
    """

    @classmethod
    def from_toml(cls, toml_file, **kwargs):
        """Create an LLMProgram from a TOML configuration file.

        Args:
            toml_file: Path to the TOML file
            **kwargs: Additional parameters to override TOML values

        Returns:
            An initialized LLMProgram instance
        """
        # Implementation moved from LLMProgram.from_toml
```

### Methods to Extract from LLMProgram

The following methods and related code should be moved from `LLMProgram` to `ProgramLoader`:

1. **TOML Loading and Compilation**
   ```python
   @classmethod
   def from_toml(cls, toml_file, **kwargs):
       """Create a program from a TOML file."""
       # Implementation to be moved...

   @classmethod
   def _compile_single_program(cls, config, base_dir=None):
       """Compile a single program configuration."""
       # Implementation to be moved...
   ```

2. **Configuration Processing Helpers**
   ```python
   @classmethod
   def _resolve_preload_files(cls, config, base_dir=None):
       """Resolve preload files in configuration."""
       # Implementation to be moved...

   @classmethod
   def _resolve_mcp_config(cls, config, base_dir=None):
       """Resolve MCP configuration."""
       # Implementation to be moved...

   @classmethod
   def _process_config_linked_programs(cls, config, base_dir=None):
       """Process linked programs in configuration."""
       # Implementation to be moved...

   @classmethod
   def _process_toml_linked_programs(cls, toml_file, linked_programs_section):
       """Process linked programs from TOML."""
       # Implementation to be moved...
   ```

### Changes to LLMProgram

The `LLMProgram` class will need to be updated to delegate to the new `ProgramLoader` for TOML functionality:

```python
class LLMProgram:
    """Core LLM program configuration.

    This class focuses on the Python SDK interface and core functionality,
    with configuration loading delegated to specialized loaders.
    """

    def __init__(self, model_name=None, provider=None, system_prompt=None, ...):
        # Core initialization logic remains unchanged
        # ...

    # Add delegation method for backward compatibility
    @classmethod
    def from_toml(cls, toml_file, **kwargs):
        """Create a program from a TOML file.

        This method delegates to ProgramLoader.from_toml for backward compatibility.
        """
        from llmproc.config.program_loader import ProgramLoader
        return ProgramLoader.from_toml(toml_file, **kwargs)
```

### Usage Examples

The existing API will be maintained for simplicity.

```python
# Current usage (unchanged)
from llmproc import LLMProgram

program = LLMProgram.from_toml("path/to/config.toml")

# New direct loader usage (for internal use only)
from llmproc.config.program_loader import ProgramLoader

program = ProgramLoader.from_toml("path/to/config.toml")
```

### Migration Strategy
1. Create the new `ProgramLoader` class with all extracted methods
2. Update `LLMProgram.from_toml` to delegate to the new loader
3. Update tests to verify functionality
4. Update any relevant documentation

## Benefits
1. **Improved Separation of Concerns**: `LLMProgram` focuses on its core responsibilities
2. **Reduced File Size**: Both `program.py` and the extracted file are under 500 lines
3. **Better SDK Experience**: SDK-focused class without TOML-specific code
4. **Enhanced Maintainability**: Easier to modify configuration loading without affecting core functionality
5. **Clearer Architecture**: Well-defined boundaries between configuration loading and program execution

## Backward Compatibility
The `LLMProgram.from_toml` method will be maintained but will delegate to the new `ProgramLoader`. This ensures no breaking changes for existing code.

## Future Work
Future enhancements could include:
1. Support for additional configuration formats (JSON, YAML, etc.)
2. Configuration validation improvements
3. Configuration schema documentation generation
4. Dynamic configuration reloading
5. Configuration inheritance and composition patterns