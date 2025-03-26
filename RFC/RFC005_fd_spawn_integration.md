# RFC005: File Descriptor Integration with Spawn Tool

This document details how the file descriptor system integrates with the spawn tool to enable efficient sharing of large content between processes. For the complete system overview, see [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md).

## 1. Background

The spawn tool allows an LLM to delegate tasks to other specialized LLM processes. When working with large content, sending the entire content in the query parameter can be inefficient and may exceed context limits. File descriptors provide an elegant solution for sharing large content between processes.

## 2. Current Spawn Tool Interface

The current spawn tool has a simple interface:

```python
spawn(program_name="expert", query="Analyze this data")
```

This basic interface creates a new process from a linked program and sends a query to it, but has no way to share large content between processes other than including it in the query text.

## 3. Enhanced Interface with FD Support

When file descriptors are enabled, the spawn tool interface is enhanced to support sharing FDs:

```python
spawn(
  program_name="expert", 
  query="Analyze this data in fd-12345",
  additional_preload_files=["data/context.txt"],  # Regular files from filesystem (always available)
  additional_preload_fds=["fd-12345"]            # File descriptor content (only when FD enabled)
)
```

### 3.1 Parameters

1. **program_name**: (Required) The linked program to spawn
2. **query**: (Required) The query to send to the linked program
3. **additional_preload_files**: (Optional) Files from the filesystem to preload into the child's context
4. **additional_preload_fds**: (Optional) File descriptors whose content should be preloaded into the child's context

## 4. Implementation Approach

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

### 4.1 Registration Logic

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

### 4.2 Tool Implementation

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

## 5. Usage Patterns

### 5.1 Basic Usage

```python
# Parent process has a large file descriptor 
fd_id = "fd:12345"  # Contains a large log file

# Spawn a child process with the FD preloaded
spawn(
  program_name="log_analyzer", 
  query="Analyze the log file for errors. The log is available as preloaded content.",
  additional_preload_fds=[fd_id]
)
```

### 5.2 Multiple FDs

```python
# Parent has multiple file descriptors
fd_code = "fd:1001"  # Contains source code
fd_error = "fd:1002"  # Contains error logs

# Spawn a child process with both FDs preloaded
spawn(
  program_name="debug_expert", 
  query="Fix the bug in the code that caused the errors in the logs.",
  additional_preload_fds=[fd_code, fd_error]
)
```

### 5.3 Mixing Files and FDs

```python
# Spawn a child process with both filesystem files and FDs
spawn(
  program_name="requirements_analyzer", 
  query="Compare the requirements.txt with the actual imports in the code.",
  additional_preload_files=["requirements.txt"],
  additional_preload_fds=["fd:3001"]  # Contains source code
)
```

## 6. Benefits

1. **Cleaner Tool Interface**: Only shows parameters that are actually available
2. **File Sharing**: Enables sharing both filesystem files and FD content
3. **Consistent Experience**: Maintains a single tool name with enhanced functionality when available
4. **Future Extensibility**: Can easily add more parameters as needed
5. **Context Efficiency**: Avoids duplicating large content in the query
6. **Cross-Process Communication**: Enables efficient sharing of large content between processes

## 7. Implementation Plan

The integration with spawn will be implemented in phases as outlined in [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md):

1. **Phase 2.2.0**: Basic spawn enhancements
   - Implement additional_preload_files parameter
   - Update tool registration and schema

2. **Phase 2.2.1**: FD-specific enhancements
   - Implement additional_preload_fds parameter
   - Add conditional schema based on FD feature status
   - Update tool handler to process FDs

## 8. References

- [RFC001: File Descriptor System for LLMProc](RFC001_file_descriptor_system.md) - Main specification document
- [RFC003: File Descriptor Implementation Details](RFC003_file_descriptor_implementation.md) - Technical implementation details
- [RFC004: File Descriptor Implementation Phases](RFC004_fd_implementation_phases.md) - Implementation phases and status