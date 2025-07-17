"""Docstrings for the LLMProgram class and its methods."""

# Class docstring
LLMPROGRAM_CLASS = """Program definition for LLM processes.

This class handles creating, configuring, and compiling LLM programs
for use with LLMProcess. It focuses on the Python SDK interface and
core functionality, with configuration loading delegated to specialized loaders.
"""

# Method docstrings
INIT = """Initialize a program.

Args:
    model_name: Name of the model to use
    provider: Provider of the model (openai, anthropic, or anthropic_vertex)
    system_prompt: System prompt text that defines the behavior of the process
    system_prompt_file: Path to a file containing the system prompt (alternative to system_prompt)
    parameters: Dictionary of API parameters
    preload_files: List of file paths to preload into the system prompt as context
    preload_relative_to: Whether preload paths are relative to the program file or CWD
    mcp_config_path: Path to MCP servers configuration file
    mcp_tools: Dictionary mapping server names to tools to enable
    tools: Dictionary from the [tools] section, or list of function-based tools
    linked_programs: Dictionary mapping program names to paths or LLMProgram objects
    linked_program_descriptions: Dictionary mapping program names to descriptions
    base_dir: Base directory for resolving relative paths in files
    project_id: Project ID for Vertex AI
    region: Region for Vertex AI
    user_prompt: User prompt to execute automatically
    max_iterations: Maximum number of iterations for tool calls
"""

COMPILE_SELF = """Internal method to validate and compile this program.

This method validates the program configuration, resolves any
system prompt files, and compiles linked programs recursively.

Returns:
    self (for method chaining)
"""

ADD_LINKED_PROGRAM = """Link another program to this one.

Args:
    name: Name to identify the linked program
    program: LLMProgram instance to link
    description: Optional description of the program's purpose

Returns:
    self (for method chaining)
"""

CONFIGURE_THINKING = """Configure Claude 3.7 thinking capability.

This method configures the thinking capability for Claude 3.7 models, allowing
the model to perform deeper reasoning on complex problems.

Args:
    enabled: Whether to enable thinking capability
    budget_tokens: Budget for thinking in tokens (1024-32768)

Returns:
    self (for method chaining)

Note:
    This only applies to Claude 3.7 models. For other models, this configuration
    will be ignored.

Examples:
    ```python
    # Enable thinking with default budget
    program.configure_thinking()

    # Enable thinking with custom budget
    program.configure_thinking(budget_tokens=8192)

    # Disable thinking
    program.configure_thinking(enabled=False)
    ```
"""

ENABLE_TOKEN_EFFICIENT_TOOLS = """Enable token-efficient tool use for Claude 3.7 models.

This method enables the token-efficient tools feature which can
significantly reduce token usage when working with tools.

Returns:
    self (for method chaining)

Note:
    This only applies to Claude 3.7 models. For other models, this configuration
    will be ignored.

Examples:
    ```python
    # Enable token-efficient tools
    program.enable_token_efficient_tools()
    ```
"""

REGISTER_TOOLS = """Register tools for use in the program.

This method allows you to enable specific built-in tools by name.
It replaces any previously enabled tools.

Args:
    tool_names: List of tool names to enable

Returns:
    self (for method chaining)

Note:
    During program compilation, dependencies between the file descriptor system
    and related tools are automatically resolved:
    - If file descriptor tools ("read_fd", "fd_to_file") are enabled, the file descriptor system
      will be automatically enabled
    - If the file descriptor system is enabled, the "read_fd" tool will be automatically added

Examples:
    ```python
    # Enable calculator and read_file tools
    program.register_tools(["calculator", "read_file"])

    # Later, replace with different tools
    program.register_tools(["calculator", "spawn"])

    # Enabling fd tools will automatically enable the file descriptor system
    program.register_tools(["calculator", "read_fd", "fd_to_file"])
    ```

Available built-in tools:
- calculator: Simple mathematical calculations
- read_file: Read local files
- fork: Create a new conversation state copy
- spawn: Call linked programs
- read_fd: Read from file descriptors (requires FD system)
- fd_to_file: Write file descriptor content to file (requires FD system)
"""


CONFIGURE_MCP = """Configure Model Context Protocol (MCP) server connection.

This method sets up the MCP server configuration for the program. You can
provide a path to a JSON configuration file or embed server definitions
directly as a dictionary. After configuring the server connection, use
``register_tools()`` with ``MCPServerTools`` to select specific tools from
MCP servers.

Args:
    config_path: Path to the MCP servers configuration file
    servers: Optional dictionary of embedded server definitions

Returns:
    self (for method chaining)

Examples:
    ```python
    # Configure using a JSON file
    program.configure_mcp(config_path="config/mcp_servers.json")

    # Or embed server definitions directly
    program.configure_mcp(servers={"calc": {"type": "stdio", "command": "echo"}})

    # Then register specific MCP tools
    from llmproc.tools.mcp import MCPServerTools
    program.register_tools([
        MCPServerTools(server="calc"),
        MCPServerTools(server="github", names="search_repositories"),
    ])
    ```
"""


COMPILE = """Validate and compile this program.

This method validates the program configuration, resolves any
system prompt files, and compiles linked programs recursively.

Returns:
    self (for method chaining)

Raises:
    ValueError: If validation fails
    FileNotFoundError: If required files cannot be found
"""

API_PARAMS = """Get API parameters for LLM API calls.

This property returns all parameters from the program configuration,
relying on the schema's validation to issue warnings for unknown parameters.

Returns:
    Dictionary of API parameters for LLM API calls
"""

FROM_DICT = """Create a program from a configuration dictionary, primarily for in-memory YAML.

Args:
    config: Dictionary containing program configuration
    base_dir: Optional base directory for resolving relative paths

Returns:
    An initialized LLMProgram instance

Useful for extracting subsections from YAML configurations:
```python
with open("config.yaml") as f:
    config = yaml.safe_load(f)
program = LLMProgram.from_dict(config["agents"]["assistant"])
```
"""


START = """Create and fully initialize an LLMProcess from this program.

✅ THIS IS THE CORRECT WAY TO CREATE AN LLMPROCESS ✅

```python
program = LLMProgram.from_toml("config.toml")
process = await program.start()  # Default ADMIN access

# Or with specific access level:
process = await program.start(access_level=AccessLevel.READ)  # Read-only process

# Register callbacks after creation:
timer = TimingCallback()
process = await program.start().add_plugins(timer)
```

This method delegates the entire program-to-process creation logic
to the `llmproc.program_exec.create_process` function, which handles
compilation, tool initialization, process instantiation, and runtime
context setup in a modular way.

Args:
    access_level: Optional access level for the process (READ, WRITE, or ADMIN).
                  Defaults to ADMIN for root processes.

⚠️ IMPORTANT: Never use direct constructor `LLMProcess(program=...)` ⚠️
Direct instantiation will result in broken context-aware tools (spawn, goto, fd_tools, etc.)
and bypass the proper tool initialization sequence.

Returns:
    A fully initialized LLMProcess ready for execution with properly configured tools
"""

START_SYNC = """Synchronously create and initialize a :class:`SyncLLMProcess`.

This method creates a synchronous process that can be used in non-async code.
It provides synchronous versions of all the async methods in LLMProcess.

Args:
    access_level: Optional access level for the process.

Returns:
    A fully initialized :class:`SyncLLMProcess`.

Example:
```python
program = LLMProgram.from_toml("config.toml")
process = program.start_sync()  # Returns SyncLLMProcess
result = process.run("Hello")   # Blocking call
process.close()                 # Blocking cleanup
```
"""
