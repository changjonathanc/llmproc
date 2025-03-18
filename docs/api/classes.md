# Class Reference

This document provides detailed information about each class in the LLMProc library, including their methods, parameters, and return values.

## LLMProgram

The `LLMProgram` class represents a program configuration for an LLM, including model, provider, system prompt, and other settings.

### Class Methods

```python
@classmethod
def from_toml(cls, toml_path: Union[str, Path], include_linked: bool = True) -> "LLMProgram":
    """Load and compile a program from a TOML file."""
    
@classmethod
def compile(cls, toml_path: Union[str, Path], include_linked: bool = True, 
            check_linked_files: bool = True, return_all: bool = False) -> Union["LLMProgram", Dict[str, "LLMProgram"]]:
    """Compile a program with advanced options."""
```

### Instance Methods

```python
async def start(self) -> "LLMProcess":
    """Create and fully initialize an LLMProcess from this program."""
    
def get_enriched_system_prompt(self, process_instance=None, include_env: bool = True) -> str:
    """Get the system prompt enriched with preloaded content and environment info."""
```

### Key Properties

- `model_name`: Name of the LLM model
- `provider`: Provider of the model (anthropic, openai, vertex)
- `system_prompt`: Base system prompt
- `api_params`: API parameters for the model
- `linked_programs`: Dictionary of linked programs
- `tools`: Dictionary of tool configurations
- `preload_files`: List of files to preload
- `base_dir`: Base directory for resolving paths

## LLMProcess

The `LLMProcess` class represents a running instance of an LLM program, maintaining state and handling API interactions.

### Class Methods

```python
@classmethod
async def create(cls, program: "LLMProgram", linked_programs_instances: dict[str, "LLMProcess"] = None) -> "LLMProcess":
    """Create and fully initialize an LLMProcess asynchronously."""
```

### Instance Methods

```python
async def run(self, user_input: str, max_iterations: int = 10, callbacks: dict = None) -> "RunResult":
    """Run the LLM process with user input and return metrics."""
    
def get_last_message(self) -> str:
    """Get the most recent assistant message text."""
    
def reset_state(self, keep_system_prompt: bool = True, keep_preloaded: bool = True) -> None:
    """Reset the conversation state."""
    
def get_state(self) -> list[dict[str, str]]:
    """Return the current conversation state."""
    
async def call_tool(self, tool_name: str, args: dict) -> Any:
    """Call a tool by name with the given arguments."""
    
def preload_files(self, file_paths: list[str]) -> None:
    """Preload files and add their content to the preloaded_content dictionary."""
```

### Key Properties

- `program`: Reference to the LLMProgram
- `state`: List of conversation messages
- `tool_registry`: Registry of available tools
- `enriched_system_prompt`: System prompt with preloaded content

## RunResult

The `RunResult` class contains metrics and information about a process run.

### Methods

```python
def add_api_call(self, info: Dict[str, Any]) -> None:
    """Record information about an API call."""
    
def complete(self) -> "RunResult":
    """Mark the run as complete and calculate duration."""
```

### Properties

- `api_call_infos`: List of raw API response data
- `api_calls`: Number of API calls made
- `start_time`: When the run started
- `end_time`: When the run completed
- `duration_ms`: Duration of the run in milliseconds

## ToolRegistry

The `ToolRegistry` class manages tool registration, access, and execution.

### Methods

```python
def register_tool(self, name: str, handler: ToolHandler, definition: ToolSchema) -> ToolSchema:
    """Register a tool with its handler and definition."""
    
def get_handler(self, name: str) -> ToolHandler:
    """Get a handler by tool name."""
    
def list_tools(self) -> List[str]:
    """List all registered tool names."""
    
def get_definitions(self) -> List[ToolSchema]:
    """Get all tool definitions for API calls."""
    
async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
    """Call a tool by name with the given arguments."""
```

### Properties

- `tool_definitions`: List of tool schemas
- `tool_handlers`: Dictionary mapping tool names to handlers

## AnthropicProcessExecutor

The `AnthropicProcessExecutor` class handles Anthropic-specific process execution.

### Methods

```python
async def run(self, process: 'Process', user_prompt: str, max_iterations: int = 10, 
              callbacks: dict = None, run_result = None, is_tool_continuation: bool = False) -> "RunResult":
    """Execute a conversation with the Anthropic API."""
    
async def run_till_text_response(self, process, user_prompt, max_iterations: int = 10):
    """Run the process until a text response is generated."""
```

## System Tools

### spawn_tool

```python
async def spawn_tool(program_name: str, query: str, llm_process = None) -> Dict[str, Any]:
    """Execute a query on a linked program."""
```

### fork_tool

```python
async def fork_tool(prompts: List[str], llm_process = None) -> Dict[str, Any]:
    """Fork the conversation into multiple processes."""
```

## Callback Definitions

```python
# Type definitions for callbacks
on_tool_start_callback = Callable[[str, Dict[str, Any]], None]  # (tool_name, args) -> None
on_tool_end_callback = Callable[[str, Any], None]  # (tool_name, result) -> None
on_response_callback = Callable[[str], None]  # (content) -> None

# Callback dictionary format
callbacks = {
    "on_tool_start": on_tool_start_callback,
    "on_tool_end": on_tool_end_callback,
    "on_response": on_response_callback
}
```