"""LLMProcess class for handling LLM interactions."""

import asyncio
import json
import os
import tomllib
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from llmproc.program import LLMProgram

try:
    from mcp_registry import MCPAggregator, ServerRegistry, get_config_path

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from llmproc.providers import get_provider_client

try:
    from llmproc.providers.anthropic_tools import (
        dump_api_error,
        run_anthropic_with_tools,
    )

    HAS_ANTHROPIC_TOOLS = True
except ImportError:
    HAS_ANTHROPIC_TOOLS = False

load_dotenv()


class LLMProcess:
    """Process for interacting with LLMs using standardized program definitions."""

    def __init__(
        self,
        program: "LLMProgram",
        linked_programs_instances: dict[str, "LLMProcess"] | None = None,
    ) -> None:
        """Initialize LLMProcess from a compiled program.

        Args:
            program: A compiled LLMProgram instance
            linked_programs_instances: Dictionary of pre-initialized LLMProcess instances

        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
            FileNotFoundError: If any of the preload files cannot be found
            ValueError: If MCP is enabled but provider is not anthropic
        """
        # Store the program reference
        self.program = program
        
        # Extract core attributes from program
        self.model_name = program.model_name
        self.provider = program.provider
        self.system_prompt = program.system_prompt
        self.display_name = program.display_name
        self.config_dir = program.config_dir
        self.api_params = program.api_params
        self.debug_tools = program.debug_tools
        self.parameters = {} # Keep empty - parameters are already processed in program
        
        # Initialize state for preloaded content
        self.preloaded_content = {}
        
        # Extract tool configuration
        self.enabled_tools = []
        if hasattr(program, 'tools') and program.tools:
            # Get enabled tools from the program
            self.enabled_tools = program.tools.get("enabled", [])
        
        # MCP Configuration
        self.mcp_enabled = False
        self.mcp_tools = {}
        self.tools = []
        
        # Extract MCP configuration if present
        self.mcp_config_path = getattr(program, "mcp_config_path", None)
        self.mcp_tools = getattr(program, "mcp_tools", {})
        
        # Linked Programs Configuration
        self.linked_programs = {}
        self.has_linked_programs = False
        
        # Initialize linked programs if provided in constructor
        if linked_programs_instances:
            self.has_linked_programs = True
            self.linked_programs = linked_programs_instances
        # Otherwise use linked programs from the program
        elif hasattr(program, "linked_programs") and program.linked_programs:
            self.has_linked_programs = True
            self._initialize_linked_programs(program.linked_programs)

        # Setup MCP if configured
        if self.mcp_config_path and self.mcp_tools:
            if not HAS_MCP:
                raise ImportError(
                    "MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'."
                )

            # Currently only support Anthropic with MCP
            if self.provider != "anthropic":
                raise ValueError(
                    "MCP features are currently only supported with the Anthropic provider"
                )

            self.mcp_enabled = True
            
            # Initialize MCP registry and tools asynchronously
            asyncio.run(self._initialize_mcp_tools())
            
        # Check if we need to register the spawn tool
        if "spawn" in self.enabled_tools and self.has_linked_programs:
            # Register the spawn tool
            self._register_spawn_tool()

        # Get project_id and region for Vertex if provided in parameters
        project_id = getattr(program, "project_id", None)
        region = getattr(program, "region", None)

        # Initialize the client
        self.client = get_provider_client(self.provider, self.model_name, project_id, region)

        # Store the original system prompt before any files are preloaded
        self.original_system_prompt = self.system_prompt
        
        # Initialize message state with system prompt
        self.state = [{"role": "system", "content": self.system_prompt}]

        # Preload files if specified
        if hasattr(program, "preload_files") and program.preload_files:
            self._preload_files(program.preload_files)

    def preload_files(self, file_paths: list[str]) -> None:
        """Add additional files to the conversation context.

        Args:
            file_paths: List of file paths to preload

        Raises:
            FileNotFoundError: If any of the files cannot be found
        """
        self._preload_files(file_paths)

    def _preload_files(self, file_paths: list[str]) -> None:
        """Preload files and add their content to the initial conversation state.

        Args:
            file_paths: List of file paths to preload

        Raises:
            FileNotFoundError: If any of the files cannot be found
        """
        # Build a single preload content string with XML tags
        preload_content = "<preload>\n"

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(
                    f"<warning>{os.path.abspath(file_path)} does not exist.</warning>"
                )
                continue

            content = path.read_text()
            self.preloaded_content[str(path)] = content

            # Add file content with filename to the preload content
            filename = path.name
            preload_content += f'<file path="{filename}">\n{content}\n</file>\n'

        preload_content += "</preload>"

        # Only add if we actually loaded some files
        if self.preloaded_content:
            # Update the system prompt to include preloaded files
            if len(self.state) > 0 and self.state[0]["role"] == "system":
                # Append to existing system prompt
                self.state[0]["content"] = f"{self.system_prompt}\n\n{preload_content}"
            else:
                # No system message yet, create one
                self.state.insert(0, {"role": "system", "content": f"{self.system_prompt}\n\n{preload_content}"})
            
            # Update the stored system prompt to include the preloaded content
            self.system_prompt = self.state[0]["content"]

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "LLMProcess":
        """Create an LLMProcess from a TOML program file.

        Args:
            toml_path: Path to the TOML program file

        Returns:
            An initialized LLMProcess instance

        Raises:
            ValueError: If program configuration is invalid
        """
        # Use the LLMProgram compiler directly
        from llmproc.program import LLMProgram
        
        # Compile the program
        program = LLMProgram.compile(toml_path)
        
        # Create the LLMProcess instance from the compiled program
        return cls(program=program)
        

    async def run(self, user_input: str, max_iterations: int = 10) -> str:
        """Run the LLM process with user input asynchronously.

        This method supports full tool execution with proper async handling.
        If used in a synchronous context, it will automatically run in a new event loop.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The model's response as a string
        """
        # Check if we're in an event loop
        try:
            asyncio.get_running_loop()
            in_event_loop = True
        except RuntimeError:
            in_event_loop = False

        # If not in an event loop, run in a new one
        if not in_event_loop:
            return asyncio.run(self._async_run(user_input, max_iterations))
        else:
            return await self._async_run(user_input, max_iterations)

    async def _async_run(self, user_input: str, max_iterations: int = 10) -> str:
        """Internal async implementation of run.

        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The model's response as a string

        Raises:
            ValueError: If user_input is empty
        """
        # Verify user input isn't empty
        if not user_input or user_input.strip() == "":
            raise ValueError("User input cannot be empty")

        self.state.append({"role": "user", "content": user_input})

        # Create provider-specific API calls
        if self.provider == "openai":
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": self.state,
                **self.api_params
            }

            try:
                # Use the async client for OpenAI
                response = await self.client.chat.completions.create(**api_params)
                output = response.choices[0].message.content.strip()

                # Update state with assistant response
                self.state.append({"role": "assistant", "content": output})
                return output
            except Exception as e:
                # Create debug dump directory if it doesn't exist
                dump_dir = Path("debug_dumps")
                dump_dir.mkdir(exist_ok=True)

                # Dump message content to file for debugging
                dump_file = dump_dir / f"openai_api_error_{id(self)}.json"
                with open(dump_file, "w") as f:
                    json.dump(
                        {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "api_params": {
                                "model": self.model_name,
                                "messages": [
                                    {
                                        "role": m["role"],
                                        "content": (
                                            m["content"][:500] + "..."
                                            if len(m["content"]) > 500
                                            else m["content"]
                                        ),
                                    }
                                    for m in self.state
                                ],
                                "temperature": self.api_params.get("temperature"),
                                "max_tokens": self.api_params.get("max_tokens"),
                            },
                        },
                        f,
                        indent=2,
                    )

                error_msg = f"Error calling OpenAI API: {str(e)}"
                print(f"ERROR: {error_msg}")
                print(f"Debug information dumped to {dump_file}")
                raise RuntimeError(f"{error_msg} (see {dump_file} for details)")

        elif self.provider == "anthropic":
            # For Anthropic with tools enabled, handle tool calls
            if self.tools:
                # Use the full async implementation for tool calls
                return await self._run_anthropic_with_tools(max_iterations)
            else:
                # Standard Anthropic call without tools
                # Extract system prompt and user/assistant messages
                system_prompt = None
                messages = []

                for msg in self.state:
                    if msg["role"] == "system":
                        system_prompt = msg["content"]
                    else:
                        # Handle potential structured content (for tools)
                        if isinstance(msg["content"], list):
                            # Filter out empty text blocks
                            empty_blocks = [
                                i
                                for i, block in enumerate(msg["content"])
                                if block.get("type") == "text" and not block.get("text")
                            ]

                            if empty_blocks:
                                # Create a safe copy without empty text blocks
                                filtered_content = [
                                    block
                                    for block in msg["content"]
                                    if not (
                                        block.get("type") == "text"
                                        and not block.get("text")
                                    )
                                ]
                                msg_copy = msg.copy()
                                msg_copy["content"] = filtered_content
                                messages.append(msg_copy)

                                # Add debugging info to console if debug is enabled
                                if self.debug_tools:
                                    print(
                                        f"WARNING: Filtered out {len(empty_blocks)} empty text blocks from message with role {msg['role']}"
                                    )
                            else:
                                messages.append(msg)
                        else:
                            messages.append(msg)

                # Prepare API parameters
                api_params = {
                    "model": self.model_name,
                    "system": system_prompt,
                    "messages": messages,
                    **self.api_params
                }
                
                # Debug: print tools if they exist
                if hasattr(self, 'tools') and self.tools:
                    print(f"DEBUG: Passing {len(self.tools)} tools to Anthropic API:")
                    for tool in self.tools:
                        print(f"  - {tool['name']}: {tool['description'][:50]}...")
                    # Add tools to API params
                    api_params["tools"] = self.tools

                try:
                    # Create the response with system prompt separate from messages
                    response = await self.client.messages.create(**api_params)
                    
                    # Check if the response is a tool use
                    if response.content[0].type == "tool_use":
                        # We need to handle tool calls here
                        print("DEBUG: Model is using a tool, redirecting to tool handling")
                        return await self._run_anthropic_with_tools(max_iterations=10)
                    else:
                        output = response.content[0].text

                    # Update state with assistant response
                    self.state.append({"role": "assistant", "content": output})
                    return output
                except Exception as e:
                    # Create debug dump directory if it doesn't exist
                    dump_dir = Path("debug_dumps")
                    dump_dir.mkdir(exist_ok=True)

                    # Dump message content to file for debugging
                    dump_file = dump_dir / f"anthropic_api_error_{id(self)}.json"
                    with open(dump_file, "w") as f:
                        json.dump(
                            {
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "api_params": {
                                    "model": self.model_name,
                                    "system": system_prompt,
                                    "messages": [
                                        {
                                            "role": m["role"],
                                            "content": (
                                                m["content"][:500] + "..."
                                                if len(m["content"]) > 500
                                                else m["content"]
                                            ),
                                        }
                                        for m in messages
                                    ],
                                    "temperature": self.api_params.get("temperature"),
                                    "max_tokens": self.api_params.get("max_tokens"),
                                },
                            },
                            f,
                            indent=2,
                        )

                    error_msg = f"Error calling Anthropic API: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    print(f"Debug information dumped to {dump_file}")
                    raise RuntimeError(f"{error_msg} (see {dump_file} for details)")

        elif self.provider == "vertex":
            # AnthropicVertex uses the same API signature as Anthropic
            # Extract system prompt and user/assistant messages
            system_prompt = None
            messages = []

            for msg in self.state:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    # Handle potential structured content (for tools)
                    if isinstance(msg["content"], list):
                        # Filter out empty text blocks
                        empty_blocks = [
                            i
                            for i, block in enumerate(msg["content"])
                            if block.get("type") == "text" and not block.get("text")
                        ]

                        if empty_blocks:
                            # Create a safe copy without empty text blocks
                            filtered_content = [
                                block
                                for block in msg["content"]
                                if not (
                                    block.get("type") == "text"
                                    and not block.get("text")
                                )
                            ]
                            msg_copy = msg.copy()
                            msg_copy["content"] = filtered_content
                            messages.append(msg_copy)

                            # Add debugging info to console if debug is enabled
                            if self.debug_tools:
                                print(
                                    f"WARNING: Filtered out {len(empty_blocks)} empty text blocks from message with role {msg['role']}"
                                )
                        else:
                            messages.append(msg)
                    else:
                        messages.append(msg)

            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": messages,
                **self.api_params,
                **self.extra_params
            }
            
            # Debug: print tools if they exist
            if hasattr(self, 'tools') and self.tools:
                print(f"DEBUG: Passing {len(self.tools)} tools to Vertex API:")
                for tool in self.tools:
                    print(f"  - {tool['name']}: {tool['description'][:50]}...")
                # Add tools to API params
                api_params["tools"] = self.tools

            try:
                # Create the response with system prompt separate from messages
                # Create the response with system prompt separate from messages
                import traceback
                
                try:
                    response = await self.client.messages.create(**api_params)
                    
                    # Check if the response is a tool use
                    if response.content[0].type == "tool_use":
                        # We need to handle tool calls here
                        print("DEBUG: Model is using a tool, redirecting to tool handling")
                        return await self._run_anthropic_with_tools(max_iterations=10)
                    else:
                        output = response.content[0].text
                except Exception as e:
                    print("FULL ERROR TRACEBACK:")
                    traceback.print_exc()
                    raise

                # Update state with assistant response
                self.state.append({"role": "assistant", "content": output})
                return output
            except Exception as e:
                # Create debug dump directory if it doesn't exist
                dump_dir = Path("debug_dumps")
                dump_dir.mkdir(exist_ok=True)

                # Dump message content to file for debugging
                dump_file = dump_dir / f"vertex_api_error_{id(self)}.json"
                with open(dump_file, "w") as f:
                    json.dump(
                        {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "api_params": {
                                "model": self.model_name,
                                "system": system_prompt,
                                "messages": [
                                    {
                                        "role": m["role"],
                                        "content": (
                                            m["content"][:500] + "..."
                                            if len(m["content"]) > 500
                                            else m["content"]
                                        ),
                                    }
                                    for m in messages
                                ],
                                "temperature": self.api_params.get("temperature"),
                                "max_tokens": self.api_params.get("max_tokens"),
                            },
                        },
                        f,
                        indent=2,
                    )

                error_msg = f"Error calling Vertex AI API: {str(e)}"
                print(f"ERROR: {error_msg}")
                print(f"Debug information dumped to {dump_file}")
                raise RuntimeError(f"{error_msg} (see {dump_file} for details)")

        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    async def _run_anthropic_with_tools(self, max_iterations: int = 10) -> str:
        """Run Anthropic with tool support.

        Handles multiple iterations of tool calls and responses in an asynchronous context,
        processing each tool and providing the result back to the model.

        Args:
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            The final model response as a string
        """
        if not HAS_ANTHROPIC_TOOLS:
            raise ImportError(
                "Anthropic tools support requires the llmproc.providers.anthropic_tools module."
            )

        # Extract system prompt and messages
        system_prompt = None
        messages = []

        for msg in self.state:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Skip empty messages that would cause API errors
                if msg.get("content") != "":
                    messages.append(msg)

        # Run the tool interaction loop through the specialized module
        # Just pass the LLMProcess instance and let the function access what it needs
        final_response = await run_anthropic_with_tools(
            llm_process=self,
            system_prompt=system_prompt,
            messages=messages,
            max_iterations=max_iterations
        )

        # Add the final response to the permanent state
        self.state.append({"role": "assistant", "content": final_response})

        return final_response

    def get_state(self) -> list[dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    async def _initialize_mcp_tools(self) -> None:
        """Initialize MCP registry and tools.

        This sets up the MCP registry and filters tools based on user configuration.
        Only tools explicitly specified in the mcp_tools configuration will be enabled.
        Only servers that have tools configured will be initialized.
        """
        if not self.mcp_enabled:
            return

        print(f"Tool configuration: {self.mcp_tools}")
        
        # Get the set of server names that are configured in mcp_tools
        configured_servers = set(self.mcp_tools.keys())
        
        # Read the MCP config file into a dictionary
        import json
        try:
            with open(self.mcp_config_path, 'r') as f:
                full_config = json.load(f)
                
            # Just check if all servers are in the config, but use original config
            # to avoid validation issues with the data structure
            mcp_servers = full_config.get("mcpServers", {})
            
            # Check which configured servers exist in the config file
            for server_name in configured_servers:
                if server_name in mcp_servers:
                    print(f"Including server '{server_name}' from config")
                else:
                    print(f"<warning>Configured server '{server_name}' not found in config file</warning>")
            
            # Skip servers not configured in mcp_tools
            for server_name in set(mcp_servers.keys()) - configured_servers:
                print(f"Skipping server '{server_name}' (not configured in mcp_tools)")
            
            # We'll use the standard method to initialize but only after checking
            self.registry = ServerRegistry.from_config(self.mcp_config_path)
            self.aggregator = MCPAggregator(self.registry)
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"<warning>Error reading MCP config: {str(e)}</warning>")
            # Fall back to the standard method in case of errors
            self.registry = ServerRegistry.from_config(self.mcp_config_path)
            self.aggregator = MCPAggregator(self.registry)

        # Get all available tools from the MCP registry
        # Use the server_mapping option to get tools grouped by server
        server_tools_map = await self.aggregator.list_tools(return_server_mapping=True)

        # Get standard list for displaying all tools
        all_tools_results = await self.aggregator.list_tools()

        # Print available tools for debugging
        print("\nAvailable MCP tools:")
        for tool in all_tools_results.tools:
            print(f" - {tool.name}")
        
        # Track registered tools to avoid duplicates
        registered_tools = set()
        
        # Process each server and tool configuration
        for server_name, tool_config in self.mcp_tools.items():
            # Check if this server exists in the tools map
            if server_name not in server_tools_map:
                print(f"<warning>Server '{server_name}' not found in available tools</warning>")
                continue
                
            server_tools = server_tools_map[server_name]
            
            # Create a mapping of tool names to tools for this server (both original and lowercase)
            server_tool_map = {}
            for tool in server_tools:
                server_tool_map[tool.name] = tool
                # Also index by lowercase name for case-insensitive matching
                server_tool_map[tool.name.lower()] = tool
                # And index by snake_case -> camelCase conversion
                if "_" in tool.name:
                    camel_case = "".join(x.capitalize() if i > 0 else x for i, x in enumerate(tool.name.split("_")))
                    server_tool_map[camel_case] = tool
                    server_tool_map[camel_case.lower()] = tool
            
            # Case 1: Register all tools for a server
            if tool_config == "all":
                if server_tools:
                    print(f"Registering all tools for server '{server_name}'")
                    for tool in server_tools:
                        namespaced_name = f"{server_name}__{tool.name}"
                        if namespaced_name not in registered_tools:
                            self.tools.append(self._format_tool_for_anthropic(tool, server_name))
                            registered_tools.add(namespaced_name)
                else:
                    print(f"<warning>No tools found for server '{server_name}'</warning>")
            
            # Case 2: Register specific tools
            elif isinstance(tool_config, list):
                for tool_name in tool_config:
                    # Try multiple variations for flexible matching
                    variations = [
                        tool_name,                    # As provided
                        tool_name.lower(),            # Lowercase
                        tool_name.replace("_", ""),   # No underscores
                        tool_name.lower().replace("_", "")  # Lowercase, no underscores
                    ]
                    
                    found = False
                    for variant in variations:
                        if variant in server_tool_map:
                            tool = server_tool_map[variant]
                            namespaced_name = f"{server_name}__{tool.name}"
                            if namespaced_name not in registered_tools:
                                self.tools.append(self._format_tool_for_anthropic(tool, server_name))
                                registered_tools.add(namespaced_name)
                                print(f"Registered tool: {namespaced_name}")
                                found = True
                            break
                    
                    if not found:
                        print(f"<warning>Tool '{tool_name}' not found for server '{server_name}'</warning>")
        
        # If no tools were registered, show a warning
        if not self.tools:
            print("<warning>No tools were registered. Check your MCP configuration.</warning>")
        
        # Show summary of registered tools
        print(f"Registered {len(self.tools)} tools from MCP registry:")
        for i, tool in enumerate(self.tools, 1):
            print(f"  {i}. {tool['name']} - {tool['description'][:60]}...")
    
    def _initialize_linked_programs(self, linked_programs: dict[str, Path | str]) -> None:
        """Initialize linked LLM programs from their TOML program files.
        
        Args:
            linked_programs: Dictionary mapping program names to TOML program paths
            
        Raises:
            FileNotFoundError: If a linked program file cannot be found
        """
        for program_name, program_path in linked_programs.items():
            path = Path(program_path)
            
            # If path is relative and we have a config_dir, resolve it
            if not path.is_absolute() and self.config_dir:
                path = self.config_dir / path
                
            if not path.exists():
                print(f"<warning>Linked program file at {path} does not exist</warning>")
                continue
                
            try:
                # Initialize the linked program using the same from_toml class method
                linked_program = LLMProcess.from_toml(path)
                self.linked_programs[program_name] = linked_program
                print(f"Initialized linked program '{program_name}' from {path}")
            except Exception as e:
                print(f"<warning>Failed to initialize linked program '{program_name}': {str(e)}</warning>")
    
    def _register_spawn_tool(self) -> None:
        """Register the spawn system call for creating new processes from linked programs."""
        from llmproc.tools import spawn_tool
        
        # Only register if we have linked programs
        if not self.linked_programs:
            print("<warning>No linked programs available. Spawn system call not registered.</warning>")
            return
            
        # Create the spawn system call definition for Anthropic API
        spawn_tool_def = {
            "name": "spawn",
            "description": "Create a new process from a linked program to handle a specific query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "program_name": {
                        "type": "string", 
                        "description": "Name of the linked program to call"
                    },
                    "query": {
                        "type": "string",
                        "description": "The query to send to the linked program"
                    }
                },
                "required": ["program_name", "query"]
            }
        }
        
        # Create a copy of the tool definition for the API
        api_tool_def = spawn_tool_def.copy()
        
        # Add the handler to the internal tool definition
        spawn_tool_def["handler"] = lambda args: spawn_tool(
            program_name=args.get("program_name"),
            query=args.get("query"),
            llm_process=self
        )
        
        # Only print linked programs info if explicitly debugging
        if self.debug_tools and os.environ.get("LLMPROC_DEBUG", "").lower() == "true":
            import sys
            print("\nLinked programs:", file=sys.stderr)
            for prog_name, prog_instance in self.linked_programs.items():
                prog_files = ""
                if hasattr(prog_instance, "preloaded_content") and prog_instance.preloaded_content:
                    prog_files = f" with {len(prog_instance.preloaded_content)} preloaded files"
                print(f"  - {prog_name}: {prog_instance.model_name} ({prog_instance.provider}){prog_files}", 
                      file=sys.stderr)
        
        # Keep the handler and tool definition separate
        self.tool_handlers = getattr(self, "tool_handlers", {})
        self.tool_handlers["spawn"] = spawn_tool_def["handler"]
        
        # Add to the tools list (API-safe version without handler)
        self.tools.append(api_tool_def)
        print(f"Registered spawn tool with access to programs: {', '.join(self.linked_programs.keys())}")
        
    def _format_tool_for_anthropic(self, tool, server_name=None):
        """Format a tool for Anthropic API.
        
        Args:
            tool: Tool object from MCP registry
            server_name: Optional server name for proper namespacing
            
        Returns:
            Dictionary with tool information formatted for Anthropic API
        """
        # Create the base tool definition with properly namespaced name
        if server_name:
            namespaced_name = f"{server_name}__{tool.name}"
        else:
            # If no server name is provided, use the tool name as is (likely already namespaced)
            namespaced_name = tool.name
            
        tool_def = {
            "name": namespaced_name,
            "description": tool.description,
            "input_schema": tool.inputSchema.copy() if tool.inputSchema else {"type": "object", "properties": {}}
        }
        
        # Ensure schema has required fields
        schema = tool_def["input_schema"]
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
            
        return tool_def

    async def _process_tool_calls(self, tool_calls):
        """Process tool calls from Anthropic response.

        Args:
            tool_calls: List of tool call objects from Anthropic response

        Returns:
            List of tool results with standardized format
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.input
            tool_id = tool_call.id

            if self.debug_tools:
                print(f"\nProcessing tool call: {tool_name}")
                print(f"  Tool ID: {tool_id}")
                print(f"  Args: {json.dumps(tool_args, indent=2)[:200]}...")

            try:
                # Call the tool through the aggregator
                start_time = asyncio.get_event_loop().time()

                # Use the async aggregator call
                result = await self.aggregator.call_tool(tool_name, tool_args)

                end_time = asyncio.get_event_loop().time()

                if self.debug_tools:
                    print(f"  Tool execution time: {end_time - start_time:.2f}s")

                # Extract content and error status based on result type
                content = None
                is_error = False

                # Check if the result has content attribute
                if hasattr(result, "content"):
                    content = result.content

                    # Try to determine if it's JSON-serializable
                    try:
                        # This will validate if the content can be serialized to JSON
                        json.dumps(content)
                    except (TypeError, OverflowError, ValueError):
                        # If it can't be serialized, convert to string
                        if self.debug_tools:
                            print(
                                "  Content is not JSON-serializable, converting to string"
                            )
                        content = str(content)
                else:
                    # If no content attribute, use the whole result
                    content = result

                    # Same serialization check
                    try:
                        json.dumps(content)
                    except (TypeError, OverflowError, ValueError):
                        if self.debug_tools:
                            print("  Content is not JSON-serializable, converting to string")
                        content = str(content)

                # Check error status
                if hasattr(result, "isError"):
                    is_error = result.isError
                elif hasattr(result, "is_error"):
                    is_error = result.is_error
                # For some tools, error might be indicated in the content
                elif isinstance(content, dict) and (
                    "error" in content or "errors" in content
                ):
                    is_error = True

                tool_result = {
                    "tool_use_id": tool_id,
                    "content": content,
                    "is_error": is_error,
                }

                if self.debug_tools:
                    print(f"  Result: {'ERROR' if is_error else 'SUCCESS'}")
                    if isinstance(content, dict):
                        print(
                            f"  Content (truncated): {json.dumps(content, indent=2)[:200]}..."
                        )
                    else:
                        print(f"  Content (truncated): {str(content)[:200]}...")

            except Exception as e:
                # For errors in the tool execution itself, create an error result
                error_message = f"Error calling MCP tool {tool_name}: {str(e)}"

                if self.debug_tools:
                    print(f"  EXCEPTION: {error_message}")

                tool_result = {
                    "tool_use_id": tool_id,
                    "content": {"error": error_message},
                    "is_error": True,
                }

            tool_results.append(tool_result)

        return tool_results

    def reset_state(
        self, keep_system_prompt: bool = True, keep_preloaded: bool = True
    ) -> None:
        """Reset the conversation state.

        Args:
            keep_system_prompt: Whether to keep the system prompt in the state
            keep_preloaded: Whether to keep preloaded file content in the state
        """
        # Access the stored original system prompt
        original_system_prompt = getattr(self, "original_system_prompt", self.system_prompt)
        
        # Reset the state
        if keep_system_prompt:
            if keep_preloaded and hasattr(self, "preloaded_content") and self.preloaded_content:
                # Keep both system prompt and preloaded content
                self.state = [{"role": "system", "content": self.system_prompt}]
            else:
                # Keep only the original system prompt without preloaded content
                self.state = [{"role": "system", "content": original_system_prompt}]
                # Update stored system prompt
                self.system_prompt = original_system_prompt
        else:
            # Remove everything
            self.state = []
            
        # If we're keeping system prompt but not preloaded content, we already handled it above
        # If we're not keeping system prompt but want to keep preloaded, re-add everything
        if not keep_system_prompt and keep_preloaded and hasattr(self, "preloaded_content") and self.preloaded_content:
            # Rebuild the preload content
            preload_content = "<preload>\n"
            for file_path, content in self.preloaded_content.items():
                filename = Path(file_path).name
                preload_content += f'<file path="{filename}">\n{content}\n</file>\n'
            preload_content += "</preload>"
            
            # Create a new system message with original prompt and preloaded content
            system_content = f"{original_system_prompt}\n\n{preload_content}"
            self.state = [{"role": "system", "content": system_content}]
            # Update stored system prompt
            self.system_prompt = system_content
