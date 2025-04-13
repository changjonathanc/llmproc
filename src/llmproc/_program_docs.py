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
    display_name: User-facing name for the process in CLI interfaces
    preload_files: List of file paths to preload into the system prompt as context
    mcp_config_path: Path to MCP servers configuration file
    mcp_tools: Dictionary mapping server names to tools to enable
    tools: Dictionary from the [tools] section, or list of function-based tools
    linked_programs: Dictionary mapping program names to paths or LLMProgram objects
    linked_program_descriptions: Dictionary mapping program names to descriptions
    env_info: Environment information configuration
    file_descriptor: File descriptor configuration
    base_dir: Base directory for resolving relative paths in files
    disable_automatic_caching: Whether to disable automatic prompt caching for Anthropic models
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

ADD_PRELOAD_FILE = """Add a file to preload into the system prompt.

Args:
    file_path: Path to the file to preload

Returns:
    self (for method chaining)
"""

CONFIGURE_ENV_INFO = """Configure environment information sharing.

This method configures which environment variables will be included in the
system prompt for added context. For privacy/security, this is disabled by default.

Args:
    variables: List of variables to include, or "all" to include all standard variables
              Standard variables include: "working_directory", "platform", "date",
              "python_version", "hostname", "username"

Returns:
    self (for method chaining)

Examples:
    ```python
    # Include specific environment variables
    program.configure_env_info(["working_directory", "platform", "date"])

    # Include all standard environment variables
    program.configure_env_info("all")

    # Explicitly disable environment information (default)
    program.configure_env_info([])
    ```
"""

CONFIGURE_FILE_DESCRIPTOR = """Configure the file descriptor system.

The file descriptor system provides Unix-like pagination for large outputs,
allowing LLMs to handle content that would exceed context limits.

Args:
    enabled: Whether to enable the file descriptor system
    max_direct_output_chars: Threshold for FD creation
    default_page_size: Page size for pagination
    max_input_chars: Threshold for user input FD creation
    page_user_input: Whether to page user input
    enable_references: Whether to enable reference ID system

Returns:
    self (for method chaining)

Note:
    During program compilation, dependencies between the file descriptor system
    and related tools are automatically resolved:
    - If the file descriptor system is enabled, the "read_fd" tool will be automatically added
    - If file descriptor tools ("read_fd", "fd_to_file") are enabled, the file descriptor system
      will be automatically enabled

Examples:
    ```python
    # Enable with default settings
    program.configure_file_descriptor()

    # Configure with custom settings
    program.configure_file_descriptor(
        max_direct_output_chars=10000,
        default_page_size=5000,
        enable_references=True
    )

    # Disable file descriptor system
    program.configure_file_descriptor(enabled=False)
    ```
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

SET_ENABLED_TOOLS = """Set the list of enabled built-in tools.

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
    program.set_enabled_tools(["calculator", "read_file"])

    # Later, replace with different tools
    program.set_enabled_tools(["calculator", "spawn"])
    
    # Enabling fd tools will automatically enable the file descriptor system
    program.set_enabled_tools(["calculator", "read_fd", "fd_to_file"])
    ```

Available built-in tools:
- calculator: Simple mathematical calculations
- read_file: Read local files
- fork: Create a new conversation state copy
- spawn: Call linked programs
- read_fd: Read from file descriptors (requires FD system)
- fd_to_file: Write file descriptor content to file (requires FD system)
"""

SET_TOOL_ALIASES = """Set tool aliases, merging with any existing aliases.

These aliases will be used to create the tool schemas sent to the LLM API.
Aliases allow you to provide shorter, more intuitive names for tools.

Args:
    aliases: Dictionary mapping alias names to target tool names
    
Returns:
    Self for method chaining
    
Examples:
    ```python
    # Set aliases for built-in and MCP tools
    program.set_tool_aliases({
        "read": "read_file",
        "calc": "calculator",
        "add": "everything__add"
    })
    ```
    
Note:
    The original tool names should be enabled via set_enabled_tools.
    Aliases are only for the LLM's use when calling tools.
    
Raises:
    ValueError: If aliases is not a dictionary or if multiple aliases
               point to the same target tool (must be one-to-one mapping)
"""

CONFIGURE_MCP = """Configure Model Context Protocol (MCP) tools.

This method configures MCP tool access for the program.

Args:
    config_path: Path to the MCP servers configuration file
    tools: Dictionary mapping server names to lists of tools to enable,
          or "all" to enable all tools from a server

Returns:
    self (for method chaining)

Examples:
    ```python
    # Enable specific tools from servers
    program.configure_mcp(
        config_path="config/mcp_servers.json",
        tools={
            "sequential-thinking": "all",
            "github": ["search_repositories", "get_file_contents"]
        }
    )

    # Enable only MCP configuration without tools
    program.configure_mcp(config_path="config/mcp_servers.json")
    ```
"""

ADD_TOOL = """Add a function-based tool to this program.

This method allows adding function-based tools to the program:
1. Adding a function decorated with @register_tool
2. Adding a regular function (will be converted to a tool using its name and docstring)

Args:
    tool: A function to register as a tool

Returns:
    self (for method chaining)

Examples:
    ```python
    # Register a function as a tool
    @register_tool(description="Searches for weather")
    def get_weather(location: str):
        # Implementation...
        return {"temperature": 22}

    program.add_tool(get_weather)

    # Register a regular function (auto-converts to tool)
    def search_docs(query: str, limit: int = 5) -> list:
        '''Search documentation for a query.

Args:
            query: The search query
            limit: Maximum results to return

Returns:
            List of matching documents
        '''
        # Implementation...
        return [{"title": "Doc1"}]

    program.add_tool(search_docs)
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

FROM_TOML = """Create a program from a TOML file.

This method delegates to ProgramLoader.from_toml for backward compatibility.

Args:
    toml_file: Path to the TOML file
    **kwargs: Additional parameters to override TOML values

Returns:
    An initialized LLMProgram instance
"""
