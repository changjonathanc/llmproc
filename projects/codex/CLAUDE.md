# CLAUDE.md - Technical Implementation Notes

## Overview

This project implements a minimal asynchronous shell session interface for Docker containers, following the specification in SPEC.md. The design prioritizes simplicity and flexibility over feature completeness.

## Architecture

### Core Design Principles

1. **Timeout-Based I/O**: All read operations are bounded by time, not by content markers
2. **No State Tracking**: The library doesn't track command completion or parse output
3. **User Control**: Callers decide when to read more output or move to the next command
4. **Minimal Dependencies**: Only requires Docker SDK and Python's asyncio

### Three Shell Types

The library provides three shell implementations, each optimized for different use cases:

1. **APIShell**: Non-TTY mode for clean automation
   - No ANSI escape sequences
   - Separate stdout/stderr streams via Docker demultiplexing
   - No support for interactive programs (Python REPL shows no output)
   - Best for: CI/CD, automation scripts, command execution

2. **InteractiveShell**: Full TTY mode for interactive use
   - Supports interactive programs (vim, Python REPL, etc.)
   - Includes ANSI escape sequences and terminal control
   - Can send Ctrl+C to interrupt processes
   - Best for: Development, debugging, interactive sessions

3. **CodexShell**: Hybrid PTY/pipe mode (NEW)
   - stdin: Regular pipe (no echo, scriptable)
   - stdout/stderr: PTY-enabled (shows interactive program output)
   - Shows Python REPL output while keeping input clean
   - Best for: Code execution environments, Jupyter-like systems

### Implementation Details

#### Socket Unwrapping
Docker-py returns wrapped socket objects that vary by version. The `_unwrap()` function handles this:
```python
def _unwrap(sock_like):
    for attr in ("_sock", "sock", "socket"):
        if hasattr(sock_like, attr):
            return getattr(sock_like, attr)
    return sock_like
```

#### Shell Initialization

**APIShell & InteractiveShell:**
1. Check for bash availability (required)
2. Create Docker exec session with `/bin/bash`
3. Wrap socket with asyncio streams for async I/O
4. Set PS1 to '\w $ ' for working directory visibility
5. InteractiveShell: Disable echo to prevent command duplication
6. Return immediately - no prompt waiting or validation

**CodexShell:**
1. Check for bash, python3, and curl availability (installs curl if needed)
2. Deploy PTY server (FastAPI application) inside container
3. Start PTY server on port 1384 in background
4. Create bash session via HTTP POST to PTY server
5. Use curl for all communication (no persistent socket connection)
6. Each command is sent via HTTP POST, responses read via HTTP

#### Read Loop Design
The `run()` method uses a time-based read loop:
```python
while asyncio.get_event_loop().time() < deadline:
    try:
        chunk = await asyncio.wait_for(
            self.reader.read(1024),
            timeout=min(0.1, remaining)
        )
        output += chunk
    except asyncio.TimeoutError:
        pass
```

Key characteristics:
- Reads in 1KB chunks for balance between memory and syscalls
- 0.1s timeout granularity for responsive partial results
- Accumulates all data received within timeout window
- No parsing or interpretation of output

## Usage Patterns

### Continuous Output Reading
The design enables elegant handling of long-running commands:

```python
# First call starts the command
output1 = await shell.run("docker build .", timeout=2.0)

# Subsequent calls continue reading
output2 = await shell.run(timeout=2.0)  # More build output
output3 = await shell.run(timeout=2.0)  # Even more
```

### Multiplexed Operations
Multiple shells allow concurrent operations:

```python
# Shell 1: Run server
await shell1.run("python app.py", timeout=0.1)

# Shell 2: Run tests against server
results = await shell2.run("pytest", timeout=10.0)
```

## Trade-offs and Limitations

### What We Gain
- **Simplicity**: ~50 lines of code vs hundreds for prompt-based parsing
- **Reliability**: No fragile prompt detection or ANSI escape handling
- **Flexibility**: Works with any shell, any prompt, any locale
- **Performance**: No regex overhead or string searching

### What We Lose
- **Completion Detection**: Can't tell if command finished or is still running
- **Precise Timing**: May return mid-output or mid-line
- **Command Separation**: Output from multiple commands may blend together
- **Error Detection**: No built-in recognition of command failures

## Network Isolation Demo

The demo showcases a practical containerization pattern:

1. **Bootstrap Phase**: Container has network for initial setup
2. **Isolation Phase**: Network disconnected via Docker API
3. **Verification**: Proves local services work without external access

This pattern is useful for:
- Security testing of air-gapped applications
- Validating offline functionality
- Testing distributed systems in isolation
- Simulating network partitions

## Future Considerations

While maintaining the minimal philosophy, potential enhancements could include:

1. **Async Iterator API**: Yield chunks as they arrive
   ```python
   async for chunk in shell.stream("tail -f log.txt"):
       print(chunk)
   ```

2. **Context Manager**: Automatic cleanup
   ```python
   async with Shell.create(container) as shell:
       output = await shell.run("ls")
   ```

3. **Ctrl+C Support**: Send signals to running processes
   ```python
   await shell.interrupt()  # Send SIGINT
   ```

These would be additive - the core timeout-based API remains unchanged.

## Comparison with Traditional Approaches

| Approach | Complexity | Reliability | Use Case |
|----------|------------|-------------|----------|
| Prompt parsing | High | Medium | Interactive CLI tools |
| Process-per-command | Low | High | Stateless operations |
| **Timeout-based** | **Low** | **High** | **Stateful operations** |

Our approach fills a gap: simple stateful shell operations without the complexity of full terminal emulation.

## Shell Type Selection Guide

| Feature | APIShell | InteractiveShell | CodexShell |
|---------|----------|------------------|------------|
| Clean output (no ANSI) | ✓ | ✗ | ✗ |
| Python REPL output | ✗ | ✓ | ✓ |
| No input echo | ✓ | ✓ | ✗ (PTY echoes) |
| Ctrl+C interruption | ✗ | ✓ | ✓ |
| Separate stdout/stderr | ✓ | ✗ | ✗ |
| Interactive programs | ✗ | ✓ | ✓ |
| Best for automation | ✓ | ✗ | ✗ |
| Best for code execution | ✗ | ✗ | ✓ |
| Network isolation resilient | ✗ | ✗ | ✓ |

### When to Use Each Shell

- **APIShell**: Default choice for automation, CI/CD, and clean output needs
- **InteractiveShell**: When you need full terminal features (vim, less, etc.)
- **CodexShell**: For code execution environments where you need to see REPL output but want scriptable input
