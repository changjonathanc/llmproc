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
    preload_relative_to: Whether preload paths are relative to the program file or CWD
    mcp_config_path: Path to MCP servers configuration file
    mcp_tools: Dictionary mapping server names to tools to enable
    tools: Dictionary from the [tools] section, or list of function-based tools
    linked_programs: Dictionary mapping program names to paths or LLMProgram objects
    linked_program_descriptions: Dictionary mapping program names to descriptions
    env_info: Environment information configuration
    file_descriptor: File descriptor configuration
    base_dir: Base directory for resolving relative paths in files
    disable_automatic_caching: Whether to disable automatic prompt caching for Anthropic models
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
              "python_version", "hostname", "username", "file_map"
    env_vars: Optional mapping of labels to environment variable names

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

    # Append additional info from environment variables
    program.configure_env_info(env_vars={"region": "MY_ENV_INFO"})
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
