# Docker Shell Session Specification

## 1. Purpose and Scope

This specification defines a minimal asynchronous shell session interface for Docker containers that:
- Maintains persistent shell state across commands
- Uses timeout-based reading without blocking
- Allows continuation of output reading from previous commands
- Provides raw access to shell I/O without parsing or interpretation
- Supports both TTY and non-TTY modes for different use cases

## 2. Core Concepts

### 2.1 Shell Session
A **Shell Session** represents a persistent bash process running inside a Docker container with:
- Bidirectional communication via stdin/stdout
- Choice of TTY or non-TTY mode based on use case
- Fixed PS1 prompt showing working directory and readiness
- No prompt detection or removal - output includes everything

### 2.2 Shell Types
Three shell types are provided for different use cases:

**APIShell** (non-TTY mode):
- Clean output without ANSI codes
- Stdout/stderr properly separated by Docker
- Suitable for automation and programmatic use
- No support for interactive programs

**InteractiveShell** (TTY mode):
- Supports interactive programs (vim, python REPL, etc.)
- Terminal control sequences work
- Combined stdout/stderr stream
- Output includes prompts and formatting

**CodexShell** (PTY server mode):
- Uses internal PTY server for hybrid behavior
- stdin: Clean input without echo (via HTTP)
- stdout/stderr: PTY-enabled (shows REPL output)
- Supports Ctrl+C interruption via interrupt() method
- Network isolation resilient
- Best for code execution environments

### 2.3 Timeout-Based Reading
All read operations are timeout-based:
- Reading continues until timeout expires
- Partial output is returned if available
- Commands continue executing beyond timeout
- No distinction between "complete" and "incomplete" output

### 2.4 Output Philosophy
- All output is returned as-is, including prompts
- No parsing, detection, or cleaning of output
- User sees exactly what the shell produces
- Prompts indicate command completion visually

## 3. API Specification

### 3.1 Shell Creation

**Method**: `create(container) -> Shell`
- **Input**: Docker container object
- **Output**: Connected Shell instance
- **Behavior**:
  - Verifies bash is available (fails if not)
  - Creates new bash exec session in container
  - Establishes bidirectional async I/O streams
  - Sets PS1 to show working directory (e.g., `\w $ `)
  - For TTY mode: disables echo with `stty -echo`
  - For CodexShell:
    - Installs dependencies (curl if needed, fastapi, uvicorn)
    - Deploys PTY server (embedded FastAPI app) to container
    - Starts server on localhost:1384 in background
    - Creates bash session via HTTP POST to PTY server
    - All communication via curl HTTP requests (no persistent socket)
  - Returns ready-to-use Shell instance

### 3.2 Command Execution and Reading

**Method**: `run(cmd: Optional[str], timeout: float) -> str`
- **Inputs**:
  - `cmd`: Command string to execute (None to continue reading)
  - `timeout`: Maximum seconds to wait for output
- **Output**: Decoded string of captured output (including prompts)
- **Behavior**:
  - If `cmd` provided: sends command with newline to shell
  - Reads available output for duration of timeout
  - Returns whatever output was captured (may be empty)
  - Output includes prompts, ANSI codes (TTY), everything
  - Decodes bytes with error replacement for invalid UTF-8

### 3.3 Shell Termination

**Method**: `close() -> None`
- **Behavior**:
  - Closes writer stream
  - Waits for writer closure
  - For CodexShell: also kills bash process via PTY server
  - Does not terminate the shell process (handled by Docker)

### 3.4 Process Interruption (CodexShell only)

**Method**: `interrupt() -> None`
- **Behavior**:
  - Sends Ctrl+C (0x03) to PTY via HTTP write
  - PTY line discipline converts to SIGINT
  - Interrupts foreground process group
  - Only affects processes in this shell session

## 4. Behavioral Specifications

### 4.1 Shell Requirements
- Bash must be available in the container
- Creation fails immediately if bash is not found
- No fallback to sh or other shells
- CodexShell additionally requires:
  - Python3 for PTY server
  - netcat (nc) for relay
  - pip for installing fastapi/uvicorn

### 4.2 Prompt Configuration
- A fixed PS1 is set showing working directory
- Default format: `\w $ ` (e.g., `/tmp $ `)
- Prompt appears in output when commands complete
- No prompt detection or removal performed

### 4.3 Reading Behavior
- Read attempts occur in small time slices (e.g., 0.1s chunks)
- Total read duration does not exceed specified timeout
- Empty output is valid when no data available
- Output accumulates across multiple chunk reads within single call

### 4.4 Output Handling

**APIShell (non-TTY)**:
- Docker demultiplexes stdout/stderr streams
- Output is clean without terminal codes
- Binary headers are parsed and removed

**InteractiveShell (TTY)**:
- Raw terminal output including all control sequences
- Prompts appear when commands complete
- Carriage returns and ANSI codes preserved

**CodexShell (PTY server)**:
- PTY-based output with terminal emulation
- Clean input without echo
- Prompts visible like InteractiveShell
- HTTP-based communication resilient to network isolation

### 4.5 State Preservation
- Shell environment persists across commands
- Working directory changes are maintained
- Environment variables remain set
- Background processes continue running
- No isolation between commands in same session

## 5. Usage Patterns

### 5.1 Simple Command (APIShell)
```
1. Call run("ls -la", timeout=0.5)
2. Receive clean directory listing output
```

### 5.2 Interactive Program (InteractiveShell)
```
1. Call run("python", timeout=0.5)
2. Receive Python prompt in output
3. Call run("print('hello')", timeout=0.5)
4. Receive result with prompt
```

### 5.3 Long-Running Command
```
1. Call run("find / -name '*.log'", timeout=1.0)
2. Receive partial output (timeout reached)
3. Call run(None, timeout=2.0)
4. Receive additional output
5. Repeat until prompt appears (command complete)
```

### 5.4 Background Process
```
1. Call run("python server.py &", timeout=0.1)
2. Receive job control output (TTY) or nothing (non-TTY)
3. Call run("curl localhost:8000", timeout=0.5)
4. Receive server response
```

## 6. Error Handling

### 6.1 Bash Availability
- Shell creation fails if bash not found
- Clear error message provided
- No attempt to use alternative shells

### 6.2 Connection Errors
- Creation fails if container doesn't exist
- Creation fails if Docker daemon unavailable
- No automatic reconnection on connection loss

### 6.3 I/O Errors
- Read timeouts are normal behavior (not errors)
- Write errors indicate connection problems
- Decode errors use replacement characters

## 7. Implementation Considerations

### 7.1 TTY vs Non-TTY Trade-offs

**APIShell advantages**:
- Clean output for parsing
- Proper stream separation
- No terminal control codes

**InteractiveShell advantages**:
- Interactive programs work
- Visual prompt feedback
- Terminal features available

### 7.2 Performance
- Timeout granularity affects responsiveness
- APIShell has slight overhead from demultiplexing
- TTY mode has terminal emulation overhead

### 7.3 Security
- No command sanitization performed
- Direct access to shell capabilities
- Container isolation provides security boundary

## 8. Edge Cases and Expected Behaviors

### 8.1 Background Processes
- **APIShell**: Background processes run silently, no job control output
- **InteractiveShell**: Shows job control output (e.g., `[1] 12345`)
- Both shells: Process continues after shell closes

### 8.2 Interactive Commands
- **APIShell**: Interactive programs (python REPL, vim) will hang - avoid these
- **InteractiveShell**: Interactive programs work normally

### 8.3 Partial Output and Timeouts
- Commands exceeding timeout continue running
- Subsequent `run(timeout=X)` calls retrieve more output
- Empty output is normal if no new data available

### 8.4 Large Output
- Output captured is limited by timeout, not size
- Fast commands may return megabytes within timeout
- Slow commands may return little despite long runtime

### 8.5 Command Chaining
- Shell state persists: `cd /tmp` affects subsequent commands
- Environment variables persist across calls
- Use `&&` or `;` for dependent commands

### 8.6 Error Output
- **APIShell**: stderr and stdout combined in output
- **InteractiveShell**: stderr and stdout combined
- Use `2>&1` to ensure error capture

### 8.7 Special Characters
- Newlines preserved in output
- **APIShell**: Clean `\n` line endings
- **InteractiveShell**: May have `\r\n` (carriage return + newline)

### 8.8 Process Interruption
- **APIShell**: Cannot send Ctrl+C or signals
- **InteractiveShell**: Can send `\x03` (Ctrl+C) as command
- **CodexShell**: Use `interrupt()` method to send Ctrl+C via PTY

### 8.9 No Output Commands
- Commands like `true` or `cd` may produce no output
- This is normal - not an error

### 8.10 Prompt Behavior
- Prompts appear when commands complete
- Long-running commands: no prompt until done
- Prompt format: `<working_directory> $ `

## 9. Limitations and Non-Goals

### 9.1 Not Provided
- Prompt detection or removal
- Output parsing or cleaning
- Automatic ANSI code stripping
- Command completion detection (use visual prompts)
- Signal handling beyond TTY capabilities

### 9.2 Design Principles
- Raw output is returned as-is
- Users see full shell state including prompts
- Timeout-based operation removes need for parsing
- Simplicity over clever detection

### 9.3 Requirements
- Container must have bash installed
- TTY allocation must be supported
- User has exec permissions on container
- Network connectivity exists to Docker daemon
