# File Descriptor Integration with Spawn Tool

This document details how the file descriptor system integrates with the spawn tool to enable efficient sharing of large content between processes.

## Current Spawn Tool Interface

The current spawn tool has a simple interface:

```python
spawn(program_name="expert", query="Analyze this data")
```

This basic interface creates a new process from a linked program and sends a query to it, but has no way to share large content between processes other than including it in the query text.

## Enhanced Interface with FD Support

When file descriptors are enabled, the spawn tool interface is enhanced to support sharing FDs:

```python
spawn(
  program_name="expert", 
  query="Analyze this data in fd-12345",
  additional_preload_files=["data/context.txt"],  # Regular files from filesystem (always available)
  additional_preload_fds=["fd-12345"]            # File descriptor content (only when FD enabled)
)
```

### Parameters

1. **program_name**: (Required) The linked program to spawn
2. **query**: (Required) The query to send to the linked program
3. **additional_preload_files**: (Optional) Files from the filesystem to preload into the child's context
4. **additional_preload_fds**: (Optional) File descriptors whose content should be preloaded into the child's context

## Implementation Approach

To avoid showing unavailable parameters when FDs are disabled, we use conditional tool schema registration:

```python
# Define base tool schema (always available)
SPAWN_TOOL_SCHEMA_BASE = {
    "name": "spawn",
    "description": BASE_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "program_name": {"type": "string", "description": "Name of the linked program to call"},
            "query": {"type": "string", "description": "The query to send to the linked program"},
            "additional_preload_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files from filesystem to preload into child's context"
            }
        },
        "required": ["program_name", "query"]
    }
}

# Define enhanced schema with FD support
SPAWN_TOOL_SCHEMA_WITH_FD = {
    "name": "spawn",
    "description": FD_ENHANCED_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "program_name": {"type": "string", "description": "Name of the linked program to call"},
            "query": {"type": "string", "description": "The query to send to the linked program"},
            "additional_preload_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files from filesystem to preload into child's context"
            },
            "additional_preload_fds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File descriptors to preload into child's context"
            }
        },
        "required": ["program_name", "query"]
    }
}
```

### Registration Logic

```python
def register_spawn_tool(registry, process):
    """Register the appropriate spawn tool based on configuration."""
    # Check if FD system is enabled
    fd_enabled = getattr(process, "file_descriptor_enabled", False)
    
    # Choose appropriate schema
    tool_schema = SPAWN_TOOL_SCHEMA_WITH_FD if fd_enabled else SPAWN_TOOL_SCHEMA_BASE
    
    # Update description with available programs
    if hasattr(process, "linked_programs") and process.linked_programs:
        available_programs = ", ".join(process.linked_programs.keys())
        tool_schema["description"] += f"\n\nAvailable programs: {available_programs}"
    
    # Create handler function
    async def spawn_handler(args):
        program_name = args.get("program_name")
        query = args.get("query")
        additional_preload_files = args.get("additional_preload_files", [])
        
        # Only process additional_preload_fds if FD is enabled
        additional_preload_fds = None
        if fd_enabled:
            additional_preload_fds = args.get("additional_preload_fds", [])
        
        return await spawn_tool(
            program_name=program_name,
            query=query,
            additional_preload_files=additional_preload_files,
            additional_preload_fds=additional_preload_fds,
            llm_process=process
        )
    
    # Register with registry
    registry.register_tool("spawn", spawn_handler, tool_schema)
```

### Tool Implementation

```python
async def spawn_tool(
    program_name, 
    query, 
    additional_preload_files=None,
    additional_preload_fds=None, 
    llm_process=None
):
    """Enhanced spawn tool with FD support."""
    if not llm_process or not hasattr(llm_process, "linked_programs"):
        return ToolResult.from_error("Spawn requires a parent process with linked programs")
    
    # Verify linked program exists
    if program_name not in llm_process.linked_programs:
        available = ", ".join(llm_process.linked_programs.keys())
        return ToolResult.from_error(f"Program '{program_name}' not found. Available: {available}")
    
    try:
        # Get the linked program
        linked_program = llm_process.linked_programs[program_name]
        
        # Create process if needed
        if hasattr(linked_program, "run"):
            linked_process = linked_program
        else:
            linked_process = LLMProcess(program=linked_program)
        
        # Preload additional files if provided
        if additional_preload_files:
            linked_process.preload_files(additional_preload_files)
        
        # Process FDs if enabled and provided
        if additional_preload_fds and hasattr(llm_process, "fd_manager"):
            for fd in additional_preload_fds:
                fd_result = llm_process.fd_manager.read_fd(fd, read_all=True)
                if fd_result and "content" in fd_result:
                    # Add FD content as preloaded content in the child process
                    linked_process.preloaded_content[f"fd:{fd}"] = fd_result["content"]
            
            # Reset enriched system prompt to include new preloaded content
            linked_process.enriched_system_prompt = None
        
        # Execute query on the process
        await linked_process.run(query)
        
        # Get response
        response = linked_process.get_last_message()
        return ToolResult.from_success(response)
        
    except Exception as e:
        return ToolResult.from_error(f"Error in spawn: {str(e)}")
```

## Benefits

1. **Cleaner Tool Interface**: Only shows parameters that are actually available
2. **File Sharing**: Enables sharing both filesystem files and FD content
3. **Consistent Experience**: Maintains a single tool name with enhanced functionality when available
4. **Future Extensibility**: Can easily add more parameters as needed