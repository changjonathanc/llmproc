# Adding New Built-in Tools to LLMProc

This guide explains how to add a new built-in tool to the LLMProc library.

## Overview

Built-in tools in LLMProc use a function-based approach with the `@register_tool` decorator. This approach automatically:

1. Generates a JSON schema from the function signature and docstrings
2. Creates a proper async handler for the tool (even for sync functions)
3. Handles parameter validation and error reporting
4. Integrates with the tool registry system

## Step 1: Create a New Tool File

Create a new Python file in the `src/llmproc/tools/` directory. Follow the naming convention of `tool_name.py`.

Example: `list_dir.py` for a directory listing tool.

## Step 2: Implement the Tool Function

Use the `@register_tool` decorator to define your tool function. The decorator automatically extracts the schema from your function's type hints and docstrings.

```python
"""Description of your tool."""

import logging
from typing import Any, Dict, List  # Import whatever types you need

from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)

@register_tool(
    description="High-level description of what the tool does.",
    param_descriptions={
        "param1": "Description of the first parameter",
        "param2": "Description of the second parameter",
        # Add descriptions for all parameters
    }
)
async def your_tool_name(param1: str, param2: int = 42) -> Any:
    """Detailed description of what the tool does.

    Args:
        param1: Description of param1
        param2: Description of param2 with its default value

    Returns:
        Description of the return value
    """
    try:
        # Implement your tool logic here
        result = do_something(param1, param2)
        return result

    except Exception as e:
        # Log the error
        error_msg = f"Error in your_tool_name: {str(e)}"
        logger.error(error_msg)
        # Return error as ToolResult
        return ToolResult.from_error(error_msg)
```

### Key Points for Implementation:

1. Use descriptive docstrings and type hints
2. Return a `ToolResult.from_error()` for error conditions
3. Include proper error handling and logging
4. Keep the implementation focused and efficient
5. Use async functions when appropriate (especially for I/O operations)

## Step 3: Update the `__init__.py` File

Add your tool to the `__init__.py` file in the tools directory:

```python
# Add the import
from .your_tool_name import your_tool_name

# Register it in the central registry
registry_data.register("your_tool_name", your_tool_name)

# Add to __all__ list
__all__ = [
    # Function-based tools
    "calculator",
    "read_file",
    "list_dir",
    "your_tool_name",  # Add this line
    # Rest of existing entries...
]
```

That's it! You don't need to modify `tool_manager.py` because it now automatically discovers and registers tools from the central registry.

## Step 4: Create an Example Program

Create an example TOML file in `examples/` to demonstrate your tool:

```toml
# Example program for testing the your_tool_name tool

[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"
display_name = "Your Tool Demo"

[prompt]
system_prompt = """Tool demonstration prompt that explains:
- What the tool does
- What parameters it accepts
- How it should be used
"""
# Add a user prompt for immediate testing
user = "Please demonstrate how to use the your_tool_name tool with different parameters."

[parameters]
max_tokens = 4096
temperature = 0.7

# Enable your tool
[tools]
enabled = ["your_tool_name"]
```

## Step 5: Add Tests

Create tests for your tool in the `tests/` directory to ensure it works as expected.

## Best Practices

1. **Error Handling**: Use try-except blocks to catch exceptions and return meaningful error messages
2. **Documentation**: Include detailed docstrings explaining parameters and return values
3. **Type Hints**: Use proper type hints to ensure correct schema generation
4. **Validation**: Validate inputs before processing
5. **Logging**: Use the logger to record errors and important operations
6. **Security**: Be mindful of security implications, especially for file system operations

## Complete Example: Adding the list_dir Tool

Here's a complete example of how the `list_dir` tool was added:

1. Create the tool file `src/llmproc/tools/list_dir.py`:

```python
"""Directory listing tool for LLMProcess."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from llmproc.tools.function_tools import register_tool
from llmproc.tools.tool_result import ToolResult

# Set up logger
logger = logging.getLogger(__name__)

@register_tool(
    description="Lists directory contents with options for showing hidden files and detailed information.",
    param_descriptions={
        "directory_path": "Absolute or relative path to the directory to list. Defaults to current working directory if not specified.",
        "show_hidden": "Whether to include hidden files and directories in the listing. Defaults to False.",
        "detailed": "Whether to show detailed information (size, permissions, modification time) for each item. Defaults to False."
    },
)
async def list_dir(
    directory_path: str = ".",
    show_hidden: bool = False,
    detailed: bool = False
) -> str:
    """List directory contents with options for showing hidden files and detailed information.

    Args:
        directory_path: Path to the directory to list. Defaults to current directory.
        show_hidden: Whether to include hidden files and directories. Defaults to False.
        detailed: Whether to show detailed information for each item. Defaults to False.

    Returns:
        A formatted string of directory contents
    """
    try:
        # Implementation details...
        # ...
    except Exception as e:
        error_msg = f"Error listing directory {directory_path}: {str(e)}"
        logger.error(error_msg)
        return ToolResult.from_error(error_msg)
```

2. Update `src/llmproc/tools/__init__.py`:

```python
# Add import
from .list_dir import list_dir

# Register in central registry
registry_data.register("list_dir", list_dir)

# Add to __all__ list
__all__ = [
    # Function-based tools
    "calculator",
    "read_file",
    "list_dir",  # Add this line
    # Other exports...
]
```

3. Create an example file in `examples/builtin-tools.toml`:

```toml
# Example program for testing the list_dir tool

[model]
name = "claude-3-7-sonnet-20250219"
provider = "anthropic"
display_name = "List Directory Tool Demo"

[prompt]
system_prompt = """You are a helpful assistant that specializes in file system operations.
You have access to a list_dir tool that can list directory contents with various options.

The list_dir tool accepts these parameters:
- directory_path: Path to the directory to list (defaults to current directory ".")
- show_hidden: Boolean flag to show hidden files (defaults to false)
- detailed: Boolean flag to show detailed file information (defaults to false)

Demonstrate how to use this tool effectively with different parameter combinations.
"""

[parameters]
max_tokens = 4096
temperature = 0.7

# Enable the list_dir tool
[tools]
enabled = ["list_dir"]
```

That's it! The tool_manager.py now automatically discovers and uses the registered tool from the central registry.
