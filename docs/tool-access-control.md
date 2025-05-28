# Tool Access Control System

The access control system in LLMProc provides a secure and flexible way to control which tools processes can use, helping prevent race conditions, unauthorized access, and other issues in multi-process environments.

## Access Levels

Tools and processes have three access levels:

| Level | Description | Examples |
|-------|-------------|----------|
| `READ` | Tools that only read state without side effects | `read_file`, `list_dir` |
| `WRITE` | Tools that modify state (default) | Most tools that change state |
| `ADMIN` | Tools that change process topology or have system-level privileges | `fork`, `spawn`, `goto` |

## Process Access Control

Each process has an access level ceiling that restricts which tools it can use. This is controlled at process creation time:

```python
from llmproc.common.access_control import AccessLevel

# Create a process with full admin access (default)
process = await program.start()

# Create a process with write access (can modify state but not fork/spawn)
process = await program.start(access_level=AccessLevel.WRITE)

# Create a read-only process (for safest operation)
process = await program.start(access_level=AccessLevel.READ)
```

### Child Process Inheritance

When a parent process creates children via `fork` or `spawn`:

- Single child with parent waiting: Child inherits `WRITE` access (can modify state but not spawn)
- Multiple children or non-waiting: Children inherit `READ` access only (safer default)

This follows the principle of least privilege from Unix systems.

## Tool Access Level Declaration

### Function-based Tools

```python
from llmproc.common.access_control import AccessLevel
from llmproc.tools.function_tools import register_tool

@register_tool(
    name="read_file",
    description="Reads a file from disk",
    access=AccessLevel.READ,  # Specify access level
)
async def read_file(path: str):
    # ...implementation...
```

### MCP Tools

```python
from llmproc.common.access_control import AccessLevel
from llmproc.tools.mcp import MCPServerTools

# Option 1: All tools with same access
tools = MCPServerTools(server="calculator", names=["add", "subtract"], access=AccessLevel.READ)

# Option 2: Per-tool access levels
tools = MCPServerTools(server="fileserver", names={
    "read_file": AccessLevel.READ,
    "write_file": AccessLevel.WRITE,
})
```

## Runtime Enforcement

When a process attempts to call a tool with a higher access level than its ceiling:

```python
# Process has READ access, but tool requires WRITE
# This will return a ToolResult error:
result = await process.call_tool("write_file", {"path": "/tmp/file.txt", "content": "data"})
# Result: "Access denied: this tool requires write access but process has read"
```

## Future Extensions

The access control system is designed to evolve in future phases:

1. **Resource-based permissions** - Tool access to specific file paths
2. **Disjoint write access** - Child processes with write access to different resources
3. **Mutex/locking system** - Preventing race conditions on shared resources

## Usage in Fork and Spawn

The access control system is particularly useful in fork/spawn operations:

```python
# Parent creates a child process
await process.fork(["Analyze this data"])
# The child gets WRITE access while the parent waits

# Parent spawns a specialized child
await process.spawn("calculator", "What is 42 + 17?")
# The spawned process gets WRITE access
```

This system makes multi-process applications safer and more predictable.
