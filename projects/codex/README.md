# Docker Shell Session

A minimal Python library for asynchronous shell sessions in Docker containers, following a simple timeout-based I/O model.

## Features

- **Persistent Shell State**: Maintains environment, working directory, and variables across commands
- **Timeout-Based Reading**: Non-blocking reads that return available output within specified time
- **Output Continuation**: Read additional output from long-running commands without re-execution
- **Multiple Sessions**: Create independent shell sessions within the same container
- **No Parsing Required**: Raw shell I/O without prompt detection or output interpretation

## Installation

```bash
pip install docker
```

## Quick Start

```python
import asyncio
from shell import APIShell, InteractiveShell, CodexShell

async def main():
    # Create a container
    container = docker_client.containers.run(
        "python:3.9", ["sleep", "infinity"],
        detach=True, tty=True, stdin_open=True
    )

    # Choose your shell type based on needs:

    # For automation and clean output
    shell = await APIShell.create(container)

    # For interactive programs (vim, python REPL)
    # shell = await InteractiveShell.create(container)

    # For code execution with REPL output visibility (NEW)
    # shell = await CodexShell.create(container)

    # Run commands
    output = await shell.run("echo 'Hello, World!'", timeout=0.5)
    print(output)  # => Hello, World!

    # Continue reading from long-running command
    output = await shell.run("find / -name '*.py'", timeout=1.0)
    print(f"First batch: {len(output)} bytes")

    more = await shell.run(timeout=2.0)  # Continue reading
    print(f"Additional: {len(more)} bytes")

    # Cleanup
    await shell.close()
    container.remove(force=True)

asyncio.run(main())
```

## Demo Example

The included `demo.py` shows a practical network isolation use case:

1. **Bootstrap**: Container starts with network to install dependencies
2. **Isolation**: Network is disconnected to create air-gapped environment
3. **Multiple Shells**: Two independent shells in the same container
4. **Local Services**: FastAPI server runs in one shell, accessed from another
5. **Verification**: External network access fails while local services work

```bash
python demo.py
# Output:
# ✓ Packages installed
# ✓ Network disconnected
# shell-2 got: {"message":"Hello FastAPI fans!"}
# ✓ External access blocked
```

## API Reference

### Shell Classes

#### `APIShell` - Clean automation shell
- No TTY mode for clean output without ANSI codes
- Separate stdout/stderr via Docker stream demultiplexing
- No support for interactive programs
- Best for: CI/CD, automation, scripting

#### `InteractiveShell` - Full terminal shell
- TTY mode with complete terminal emulation
- Supports interactive programs (vim, less, Python REPL)
- Can send Ctrl+C to interrupt processes
- Best for: Development, debugging, manual interaction

#### `CodexShell` - Hybrid execution shell (NEW)
- stdin: Regular pipe (no echo, scriptable)
- stdout/stderr: PTY-enabled (shows REPL output)
- Combines benefits of both approaches
- Best for: Code execution environments, Jupyter-like systems

### Common Methods

#### `await APIShell.create(container) -> APIShell`
#### `await InteractiveShell.create(container) -> InteractiveShell`
#### `await CodexShell.create(container) -> CodexShell`
Create a new shell session in the specified Docker container.

- **container**: Docker container object
- **Returns**: Connected Shell instance

#### `await shell.run(cmd: Optional[str], timeout: float) -> str`
Execute a command and/or read output for the specified timeout.

- **cmd**: Command to execute (None to continue reading previous output)
- **timeout**: Maximum seconds to wait for output
- **Returns**: String output captured within timeout

Key behaviors:
- If `cmd` is provided, sends it to the shell before reading
- Always returns after timeout expires (non-blocking)
- Commands continue running beyond timeout
- Empty string returned if no output available

#### `await shell.close()`
Close the shell session gracefully.

## Design Philosophy

This library follows a minimal design philosophy:

1. **No Prompt Detection**: Outputs are not parsed or interpreted
2. **Time-Based Reading**: All reads are bounded by timeout
3. **Stateless Operation**: No tracking of command completion
4. **User Control**: Caller decides when to read more or move on

## Common Patterns

### Long Output Handling
```python
# Start command with short timeout
output = await shell.run("cat large_file.txt", timeout=0.5)

# Continue reading in chunks
while True:
    more = await shell.run(timeout=1.0)
    if not more:
        break
    output += more
```

### Background Processes
```python
# Start server in background
await shell.run("python server.py &", timeout=0.1)

# Continue with other commands
await shell.run("curl localhost:8000", timeout=0.5)
```

### State Preservation
```python
# Commands share environment
await shell.run("export API_KEY=secret", timeout=0.1)
await shell.run("cd /app", timeout=0.1)
output = await shell.run("echo $API_KEY in $(pwd)", timeout=0.1)
# => "secret in /app"
```

### Python REPL with CodexShell
```python
# CodexShell shows Python REPL output
shell = await CodexShell.create(container)

# Start Python - you'll see the prompt
output = await shell.run("python", timeout=0.5)
print(output)  # Shows Python version and >>> prompt

# Send Python code - you'll see the result
output = await shell.run("2 + 2", timeout=0.5)
print(output)  # Shows: 4

# Unlike APIShell where REPL output is invisible
```

## Implementation Details

See [SPEC.md](SPEC.md) for the complete specification and [CLAUDE.md](CLAUDE.md) for technical implementation notes.

## Project Structure

```
codex/
├── shells/                # Shell implementations
│   ├── __init__.py       # Package exports
│   ├── base.py           # Base classes and utilities
│   ├── api_shell.py      # APIShell implementation
│   ├── interactive_shell.py # InteractiveShell implementation
│   ├── codex_shell.py    # CodexShell implementation
│   └── pty_server.py     # PTY server for CodexShell
├── tests/                # Organized test suite
│   ├── test_api_shell.py
│   ├── test_interactive_shell.py
│   ├── test_codex_shell.py
│   ├── test_shell_common.py
│   └── test_integration.py
├── demos/                # Demo applications
│   ├── demo.py          # Network isolation demo
│   ├── demo_codex.py    # CodexShell features demo
│   └── demo_comparison.py # Compare all shells
├── shell.py             # Backward compatibility imports
└── old_tests/           # Archive of old test files
```

## Testing

Run the organized test suite:

```bash
# Run all tests
pytest tests/

# Run specific shell tests
pytest tests/test_api_shell.py -v
pytest tests/test_interactive_shell.py -v
pytest tests/test_codex_shell.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

## Requirements

- Python 3.9+ (3.11 recommended for CodexShell)
- Docker
- Python packages:
  - `docker`
  - `pytest` (for testing)
  - `fastapi` and `uvicorn` (auto-installed by CodexShell)

## License

MIT
