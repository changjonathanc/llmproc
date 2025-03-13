"""LLMProcess class for handling LLM interactions."""

import os
import json
import tomllib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

from dotenv import load_dotenv

try:
    from mcp_registry import ServerRegistry, MCPAggregator, get_config_path
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from llmproc.providers import get_provider_client
try:
    from llmproc.providers.anthropic_tools import (
        run_anthropic_with_tools,
        filter_empty_text_blocks,
        dump_api_error
    )
    HAS_ANTHROPIC_TOOLS = True
except ImportError:
    HAS_ANTHROPIC_TOOLS = False

load_dotenv()

class LLMProcess:
    """Process for interacting with LLMs using standardized configuration."""

    def __init__(
        self,
        model_name: str,
        provider: str,
        system_prompt: str,
        preload_files: Optional[List[str]] = None,
        display_name: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        mcp_tools: Optional[Dict[str, Union[List[str], str]]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize LLMProcess.

        Args:
            model_name: Name of the model to use
            provider: Provider of the model (openai, anthropic, or vertex)
            system_prompt: System message to provide to the model
            preload_files: List of file paths to preload as context
            display_name: User-facing name for the model in CLI interfaces
            mcp_config_path: Path to MCP servers configuration file
            mcp_tools: Dictionary mapping server names to tools to enable
                       Value can be a list of tool names or "all" to enable all tools
            **kwargs: Additional parameters to pass to the model

        Raises:
            NotImplementedError: If the provider is not implemented
            ImportError: If the required package for a provider is not installed
            FileNotFoundError: If any of the preload files cannot be found
            ValueError: If MCP is enabled but provider is not anthropic
        """
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.display_name = display_name or f"{provider.title()} {model_name}"
        self.parameters = kwargs
        self.preloaded_content = {}

        # MCP Configuration
        self.mcp_enabled = False
        self.mcp_tools = {}
        self.tools = []

        # Setup MCP if configured
        if mcp_config_path and mcp_tools:
            if not HAS_MCP:
                raise ImportError("MCP features require the mcp-registry package. Install it with 'pip install mcp-registry'.")

            # Currently only support Anthropic with MCP
            if provider != "anthropic":
                raise ValueError("MCP features are currently only supported with the Anthropic provider")

            self.mcp_enabled = True
            self.mcp_config_path = mcp_config_path
            self.mcp_tools = mcp_tools

            # Initialize MCP registry and tools asynchronously
            asyncio.run(self._initialize_mcp_tools())

        # Get project_id and region for Vertex if provided in parameters
        project_id = kwargs.pop('project_id', None)
        region = kwargs.pop('region', None)

        # Initialize the client
        self.client = get_provider_client(provider, model_name, project_id, region)

        # Initialize message state with system prompt
        self.state = [{"role": "system", "content": self.system_prompt}]

        # Preload files if specified
        if preload_files:
            self._preload_files(preload_files)

    def preload_files(self, file_paths: List[str]) -> None:
        """Add additional files to the conversation context.

        Args:
            file_paths: List of file paths to preload

        Raises:
            FileNotFoundError: If any of the files cannot be found
        """
        self._preload_files(file_paths)

    def _preload_files(self, file_paths: List[str]) -> None:
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
                print(f"<warning>{os.path.abspath(file_path)} does not exist.</warning>")
                continue

            content = path.read_text()
            self.preloaded_content[str(path)] = content

            # Add file content with filename to the preload content
            filename = path.name
            preload_content += f"<file path=\"{filename}\">\n{content}\n</file>\n"

        preload_content += "</preload>"

        # Add the combined preload content as a single user message
        if preload_content != "<preload>\n</preload>":  # Only add if there's content
            self.state.append({
                "role": "user",
                "content": f"Please read the following preloaded files:\n{preload_content}"
            })
            self.state.append({
                "role": "assistant",
                "content": "I've read all the preloaded files. I'll incorporate this information in our conversation."
            })

    @classmethod
    def from_toml(cls, toml_path: Union[str, Path]) -> "LLMProcess":
        """Create an LLMProcess from a TOML configuration file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            An initialized LLMProcess instance

        Raises:
            ValueError: If MCP configuration is invalid
        """
        path = Path(toml_path)
        with path.open('rb') as f:
            config = tomllib.load(f)

        model = config['model']
        prompt_config = config.get('prompt', {})
        parameters = config.get('parameters', {})
        preload_config = config.get('preload', {})
        mcp_config = config.get('mcp', {})

        if 'system_prompt_file' in prompt_config:
            system_prompt_path = path.parent / prompt_config['system_prompt_file']
            system_prompt = system_prompt_path.read_text()
        else:
            system_prompt = prompt_config.get('system_prompt', '')

        # Handle preload files if specified
        preload_files = None
        if 'files' in preload_config:
            # Convert all paths to be relative to the TOML file's directory
            preload_files = [str(path.parent / file_path) for file_path in preload_config['files']]
            # Check if all preloaded files exist
            for file_path in preload_files:
                if not Path(file_path).exists():
                    print(f"<warning>{os.path.abspath(file_path)} does not exist.</warning>")

        # Get display name if present
        display_name = model.get('display_name', None)

        # Handle MCP configuration if specified
        mcp_config_path = None
        mcp_tools = None

        if mcp_config:
            if 'config_path' in mcp_config:
                # Convert path to be relative to the TOML file's directory
                config_path = path.parent / mcp_config['config_path']
                if not config_path.exists():
                    print(f"<warning>MCP config file {os.path.abspath(config_path)} does not exist.</warning>")
                else:
                    mcp_config_path = str(config_path)

            # Get tool configuration
            if 'tools' in mcp_config:
                tools_config = mcp_config['tools']
                if not isinstance(tools_config, dict):
                    raise ValueError("MCP tools configuration must be a dictionary mapping server names to tool lists")

                # Process the tools configuration
                mcp_tools = {}
                for server_name, tool_config in tools_config.items():
                    # Accept either "all" or a list of tool names
                    if tool_config == "all":
                        mcp_tools[server_name] = "all"
                    elif isinstance(tool_config, list):
                        mcp_tools[server_name] = tool_config
                    else:
                        raise ValueError(f"Invalid tool configuration for server '{server_name}'. Expected 'all' or a list of tool names")

        return cls(
            model_name=model['name'],
            provider=model['provider'],
            system_prompt=system_prompt,
            preload_files=preload_files,
            display_name=display_name,
            mcp_config_path=mcp_config_path,
            mcp_tools=mcp_tools,
            **parameters
        )

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

        # Extract common parameters
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1000)

        # Create provider-specific API calls
        if self.provider == "openai":
            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "messages": self.state,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in self.parameters.items()
                   if k not in ['temperature', 'max_tokens']}
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
                    json.dump({
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "api_params": {
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": m["role"],
                                    "content": (m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"])
                                } for m in self.state
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                    }, f, indent=2)

                error_msg = f"Error calling OpenAI API: {str(e)}"
                print(f"ERROR: {error_msg}")
                print(f"Debug information dumped to {dump_file}")
                raise RuntimeError(f"{error_msg} (see {dump_file} for details)")

        elif self.provider == "anthropic":
            # For Anthropic with MCP enabled, handle tool calls
            if self.mcp_enabled and self.tools:
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
                            empty_blocks = [i for i, block in enumerate(msg["content"])
                                           if block.get("type") == "text" and not block.get("text")]

                            if empty_blocks:
                                # Create a safe copy without empty text blocks
                                filtered_content = [
                                    block for block in msg["content"]
                                    if not (block.get("type") == "text" and not block.get("text"))
                                ]
                                msg_copy = msg.copy()
                                msg_copy["content"] = filtered_content
                                messages.append(msg_copy)

                                # Add debugging info to console if debug is enabled
                                debug = self.parameters.get('debug_tools', False)
                                if debug:
                                    print(f"WARNING: Filtered out {len(empty_blocks)} empty text blocks from message with role {msg['role']}")
                            else:
                                messages.append(msg)
                        else:
                            messages.append(msg)

                # Prepare API parameters
                api_params = {
                    "model": self.model_name,
                    "system": system_prompt,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **{k: v for k, v in self.parameters.items()
                       if k not in ['temperature', 'max_tokens']}
                }

                try:
                    # Create the response with system prompt separate from messages
                    response = await self.client.messages.create(**api_params)
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
                        json.dump({
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "api_params": {
                                "model": self.model_name,
                                "system": system_prompt,
                                "messages": [
                                    {
                                        "role": m["role"],
                                        "content": (m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"])
                                    } for m in messages
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            }
                        }, f, indent=2)

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
                        empty_blocks = [i for i, block in enumerate(msg["content"])
                                       if block.get("type") == "text" and not block.get("text")]

                        if empty_blocks:
                            # Create a safe copy without empty text blocks
                            filtered_content = [
                                block for block in msg["content"]
                                if not (block.get("type") == "text" and not block.get("text"))
                            ]
                            msg_copy = msg.copy()
                            msg_copy["content"] = filtered_content
                            messages.append(msg_copy)

                            # Add debugging info to console if debug is enabled
                            debug = self.parameters.get('debug_tools', False)
                            if debug:
                                print(f"WARNING: Filtered out {len(empty_blocks)} empty text blocks from message with role {msg['role']}")
                        else:
                            messages.append(msg)
                    else:
                        messages.append(msg)

            # Prepare API parameters
            api_params = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in self.parameters.items()
                   if k not in ['temperature', 'max_tokens']}
            }

            try:
                # Create the response with system prompt separate from messages
                response = await self.client.messages.create(**api_params)
                output = response.content[0].text

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
                    json.dump({
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "api_params": {
                            "model": self.model_name,
                            "system": system_prompt,
                            "messages": [
                                {
                                    "role": m["role"],
                                    "content": (m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"])
                                } for m in messages
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                    }, f, indent=2)

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
            raise ImportError("Anthropic tools support requires the llmproc.providers.anthropic_tools module.")
            
        # Extract common parameters
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1000)
        debug = self.parameters.get('debug_tools', False)

        # Extract system prompt and filter messages
        system_prompt = None
        messages = []

        for msg in self.state:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append(msg)
                
        # Filter messages to remove empty text blocks
        messages = filter_empty_text_blocks(messages, debug)
        
        # Run the tool interaction loop through the specialized module
        final_response = await run_anthropic_with_tools(
            client=self.client,
            model_name=self.model_name,
            system_prompt=system_prompt,
            messages=messages,
            tools=self.tools,
            aggregator=self.aggregator,
            temperature=temperature,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            debug=debug,
            **{k: v for k, v in self.parameters.items()
               if k not in ['temperature', 'max_tokens', 'debug_tools']}
        )
        
        # Add the final response to the permanent state
        self.state.append({"role": "assistant", "content": final_response})
        
        return final_response

    def get_state(self) -> List[Dict[str, str]]:
        """Return the current conversation state.

        Returns:
            A copy of the current conversation state
        """
        return self.state.copy()

    async def _initialize_mcp_tools(self) -> None:
        """Initialize MCP registry and tools.

        This sets up the MCP registry and filters tools based on user configuration.
        Only tools explicitly specified in the mcp_tools configuration will be enabled.
        """
        if not self.mcp_enabled:
            return

        # Initialize MCP registry and aggregator
        self.registry = ServerRegistry.from_config(self.mcp_config_path)
        self.aggregator = MCPAggregator(self.registry)

        # Get all available tools from the MCP registry
        results = await self.aggregator.list_tools()

        # Print available tools for debugging
        print("\nAvailable MCP tools:")
        for tool in results.tools:
            print(f" - {tool.name}")

        # Organize tools by server name for easier filtering
        server_tools = {}
        for tool in results.tools:
            try:
                # Extract server name from tool name (prefix before first '.')
                parts = tool.name.split('.')
                if len(parts) > 1:
                    server_name = parts[0]
                    tool_name = parts[1]  # The actual tool name without server prefix
                else:
                    # If no prefix, assume the whole name is both server and tool name
                    server_name = "unknown"
                    tool_name = tool.name

                # Add to server_tools dictionary in multiple ways to increase matching chance
                # 1. By server name
                if server_name not in server_tools:
                    server_tools[server_name] = []
                server_tools[server_name].append(tool)

                # 2. By full name as key
                if tool.name not in server_tools:
                    server_tools[tool.name] = []
                server_tools[tool.name].append(tool)

                # 3. By tool name
                if tool_name not in server_tools:
                    server_tools[tool_name] = []
                server_tools[tool_name].append(tool)
            except Exception as e:
                print(f"<warning>Error processing tool {tool.name}: {str(e)}</warning>")

        # Register tools based on user configuration
        print(f"Tool configuration: {self.mcp_tools}")
        registered_tools = set()  # Track registered tools to avoid duplicates

        # First try the traditional filtering approach
        for server_name, tool_config in self.mcp_tools.items():
            if server_name in server_tools:
                server_tool_list = server_tools[server_name]
                print(f"Found {len(server_tool_list)} tools for server '{server_name}'")

                # Apply the filtering based on configuration
                if tool_config == "all":
                    # Enable all tools from this server
                    for tool in server_tool_list:
                        if tool.name not in registered_tools:
                            self.tools.append({
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema,
                            })
                            registered_tools.add(tool.name)
                elif isinstance(tool_config, list):
                    # Enable specific tools
                    for tool_name in tool_config:
                        # Try different name patterns
                        patterns = [
                            tool_name,  # As provided
                            f"{server_name}.{tool_name}",  # With server prefix
                            tool_name.split(".")[-1]  # Just the last part
                        ]

                        for pattern in patterns:
                            for tool in server_tool_list:
                                if (pattern in tool.name or tool.name in pattern) and tool.name not in registered_tools:
                                    self.tools.append({
                                        "name": tool.name,
                                        "description": tool.description,
                                        "input_schema": tool.inputSchema,
                                    })
                                    registered_tools.add(tool.name)
            else:
                print(f"<warning>Server '{server_name}' not found in MCP registry</warning>")

                # Fallback approach - try to match by partial name
                for tool in results.tools:
                    # Check if server name is substring of tool name
                    if server_name in tool.name and tool.name not in registered_tools:
                        self.tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        })
                        registered_tools.add(tool.name)

        # If no tools registered yet, try a more permissive approach
        if not self.tools:
            print("<warning>No tools matched specific criteria, trying more flexible matching...</warning>")

            # Register all tools as a last resort if "all" was used for any server
            all_requested = any(config == "all" for config in self.mcp_tools.values())
            if all_requested:
                for tool in results.tools:
                    if tool.name not in registered_tools:
                        self.tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        })
                        registered_tools.add(tool.name)

        # Ensure the input_schema is properly formatted for Anthropic
        for tool in self.tools:
            # Make sure required fields exist
            if "input_schema" in tool:
                schema = tool["input_schema"]

                # Ensure required fields
                if "type" not in schema:
                    schema["type"] = "object"

                # Ensure properties exist
                if "properties" not in schema:
                    schema["properties"] = {}

        # Show final tools
        print(f"Registered {len(self.tools)} tools from MCP registry:")
        for i, tool in enumerate(self.tools, 1):
            print(f"  {i}. {tool['name']} - {tool['description'][:60]}...")

    async def _process_tool_calls(self, tool_calls):
        """Process tool calls from Anthropic response.

        Args:
            tool_calls: List of tool call objects from Anthropic response

        Returns:
            List of tool results with standardized format
        """
        tool_results = []
        debug = self.parameters.get('debug_tools', False)

        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.input
            tool_id = tool_call.id

            if debug:
                print(f"\nProcessing tool call: {tool_name}")
                print(f"  Tool ID: {tool_id}")
                print(f"  Args: {json.dumps(tool_args, indent=2)[:200]}...")

            try:
                # Call the tool through the aggregator
                start_time = asyncio.get_event_loop().time()

                # Use the async aggregator call
                result = await self.aggregator.call_tool(tool_name, tool_args)

                end_time = asyncio.get_event_loop().time()

                if debug:
                    print(f"  Tool execution time: {end_time - start_time:.2f}s")

                # Extract content and error status based on result type
                content = None
                is_error = False

                # Check if the result has content attribute
                if hasattr(result, 'content'):
                    content = result.content

                    # Try to determine if it's JSON-serializable
                    try:
                        # This will validate if the content can be serialized to JSON
                        json.dumps(content)
                    except (TypeError, OverflowError, ValueError):
                        # If it can't be serialized, convert to string
                        if debug:
                            print(f"  Content is not JSON-serializable, converting to string")
                        content = str(content)
                else:
                    # If no content attribute, use the whole result
                    content = result

                    # Same serialization check
                    try:
                        json.dumps(content)
                    except (TypeError, OverflowError, ValueError):
                        content = str(content)

                # Check error status
                if hasattr(result, 'isError'):
                    is_error = result.isError
                elif hasattr(result, 'is_error'):
                    is_error = result.is_error
                # For some tools, error might be indicated in the content
                elif isinstance(content, dict) and ('error' in content or 'errors' in content):
                    is_error = True

                tool_result = {
                    "tool_use_id": tool_id,
                    "content": content,
                    "is_error": is_error
                }

                if debug:
                    print(f"  Result: {'ERROR' if is_error else 'SUCCESS'}")
                    if isinstance(content, dict):
                        print(f"  Content (truncated): {json.dumps(content, indent=2)[:200]}...")
                    else:
                        print(f"  Content (truncated): {str(content)[:200]}...")

            except Exception as e:
                # For errors in the tool execution itself, create an error result
                error_message = f"Error calling MCP tool {tool_name}: {str(e)}"

                if debug:
                    print(f"  EXCEPTION: {error_message}")

                tool_result = {
                    "tool_use_id": tool_id,
                    "content": {"error": error_message},
                    "is_error": True
                }

            tool_results.append(tool_result)

        return tool_results

    def reset_state(self, keep_system_prompt: bool = True, keep_preloaded: bool = True) -> None:
        """Reset the conversation state.

        Args:
            keep_system_prompt: Whether to keep the system prompt in the state
            keep_preloaded: Whether to keep preloaded file content in the state
        """
        if keep_system_prompt:
            self.state = [{"role": "system", "content": self.system_prompt}]
        else:
            self.state = []

        # Re-add preloaded content if requested
        if keep_preloaded and hasattr(self, 'preloaded_content') and self.preloaded_content:
            # Rebuild the preload content in the same format
            preload_content = "<preload>\n"
            for file_path, content in self.preloaded_content.items():
                filename = Path(file_path).name
                preload_content += f"<file path=\"{filename}\">\n{content}\n</file>\n"
            preload_content += "</preload>"

            # Add as a single message
            self.state.append({
                "role": "user",
                "content": f"Please read the following preloaded files:\n{preload_content}"
            })
            self.state.append({
                "role": "assistant",
                "content": "I've read all the preloaded files. I'll incorporate this information in our conversation."
            })