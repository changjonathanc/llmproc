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

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """Run the LLM process with user input.
        
        Args:
            user_input: The user message to process
            max_iterations: Maximum number of tool-calling iterations
            
        Returns:
            The model's response as a string
        """
        self.state.append({"role": "user", "content": user_input})
        
        # Extract common parameters
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1000)
        
        # Create provider-specific API calls
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.state,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.choices[0].message.content.strip()
            
            # Update state with assistant response
            self.state.append({"role": "assistant", "content": output})
            return output
            
        elif self.provider == "anthropic":
            # For Anthropic with MCP enabled, handle tool calls
            if self.mcp_enabled and self.tools:
                # Call Anthropic with tool support using async loop
                return asyncio.run(self._run_anthropic_with_tools(max_iterations))
            else:
                # Standard Anthropic call without tools
                # Extract system prompt and user/assistant messages
                system_prompt = None
                messages = []
                
                for msg in self.state:
                    if msg["role"] == "system":
                        system_prompt = msg["content"]
                    else:
                        messages.append(msg)
                
                # Create the response with system prompt separate from messages
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **{k: v for k, v in self.parameters.items() 
                       if k not in ['temperature', 'max_tokens']}
                )
                output = response.content[0].text
                
                # Update state with assistant response
                self.state.append({"role": "assistant", "content": output})
                return output
            
        elif self.provider == "vertex":
            # AnthropicVertex uses the same API signature as Anthropic
            # Extract system prompt and user/assistant messages
            system_prompt = None
            messages = []
            
            for msg in self.state:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    messages.append(msg)
            
            # Create the response with system prompt separate from messages
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in self.parameters.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
            output = response.content[0].text
            
            # Update state with assistant response
            self.state.append({"role": "assistant", "content": output})
            return output
            
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")
    
    async def _run_anthropic_with_tools(self, max_iterations: int = 10) -> str:
        """Run Anthropic with tool support.
        
        Args:
            max_iterations: Maximum number of tool-calling iterations
            
        Returns:
            The final model response as a string
        """
        # Extract common parameters
        temperature = self.parameters.get('temperature', 0.7)
        max_tokens = self.parameters.get('max_tokens', 1000)
        
        # Extract system prompt and messages
        system_prompt = None
        messages = []
        
        for msg in self.state:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append(msg)
        
        # Track iterations to prevent infinite loops
        iterations = 0
        final_response = ""
        
        # Continue the conversation until no more tool calls or max iterations reached
        while iterations < max_iterations:
            iterations += 1
            
            try:
                # Call Claude with current conversation and tools
                response = await self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=self.tools,
                    **{k: v for k, v in self.parameters.items() 
                       if k not in ['temperature', 'max_tokens']}
                )
                
                # Process the response
                has_tool_calls = False
                response_text = ""
                
                # Process text content and tool calls
                for content in response.content:
                    if content.type == "text":
                        response_text = content.text
                        final_response = response_text  # Save the text response
                    elif content.type == "tool_use":
                        has_tool_calls = True
                
                # Add Claude's response to state
                # We need to transform this to the format we use internally
                assistant_response = {"role": "assistant", "content": final_response}
                messages.append(assistant_response)
                
                # Also add to our internal state
                self.state.append(assistant_response)
                
                # If no tool calls were made, we're done with this query
                if not has_tool_calls:
                    break
                
                # Process tool calls and add results to conversation
                tool_calls = [c for c in response.content if c.type == "tool_use"]
                tool_results = await self._process_tool_calls(tool_calls)
                
                # Add tool results to conversation
                for result in tool_results:
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_use_id"],
                                "content": result["content"],
                                "is_error": result["is_error"]
                            }
                        ],
                    }
                    messages.append(tool_result_message)
                    
                    # Also add to our internal state
                    self.state.append(tool_result_message)
                
            except Exception as e:
                # Handle API errors
                error_message = f"Error calling Anthropic API: {str(e)}"
                print(f"API Error: {error_message}")
                final_response = f"I encountered an error: {error_message}"
                
                # Update state with error
                self.state.append({"role": "assistant", "content": final_response})
                break
                
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
        
        # Filter and register tools based on configuration
        server_tools = {}
        for tool in results.tools:
            # Extract server name from tool name (prefix before first '.')
            parts = tool.name.split('.')
            if len(parts) > 1:
                server_name = parts[0]
            else:
                server_name = "unknown"
                
            # Add to server_tools dictionary
            if server_name not in server_tools:
                server_tools[server_name] = []
            server_tools[server_name].append(tool)
        
        # Register tools based on user configuration
        for server_name, tool_config in self.mcp_tools.items():
            if server_name not in server_tools:
                print(f"<warning>Server '{server_name}' not found in MCP registry</warning>")
                continue
                
            # Get list of tool names to enable
            tool_names_to_enable = set()
            if tool_config == "all":
                # Enable all tools from this server
                tool_names_to_enable = {tool.name for tool in server_tools[server_name]}
            elif isinstance(tool_config, list):
                # Enable specific tools
                # Prepend server name if not already present in tool name
                tool_names_to_enable = {
                    name if "." in name else f"{server_name}.{name}" 
                    for name in tool_config
                }
                
            # Register the specified tools
            for tool in server_tools[server_name]:
                if tool.name in tool_names_to_enable:
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    })
        
        print(f"Registered {len(self.tools)} tools from MCP registry")
    
    async def _process_tool_calls(self, tool_calls):
        """Process tool calls from Anthropic response.
        
        Args:
            tool_calls: List of tool call objects from Anthropic response
            
        Returns:
            List of tool results
        """
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.input
            tool_id = tool_call.id
            
            try:
                # Call the tool through the aggregator
                result = await self.aggregator.call_tool(tool_name, tool_args)
                
                # Extract content and error status
                tool_result = {
                    "tool_use_id": tool_id,
                    "content": result.content if hasattr(result, 'content') else result,
                    "is_error": result.isError if hasattr(result, 'isError') else False
                }
            except Exception as e:
                # For errors, create an error result
                tool_result = {
                    "tool_use_id": tool_id,
                    "content": {"error": f"Error calling MCP tool {tool_name}: {str(e)}"},
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