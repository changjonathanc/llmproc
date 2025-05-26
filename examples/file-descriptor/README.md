# File Descriptor Examples

This directory contains examples demonstrating the file descriptor system in llmproc.

## Overview

The file descriptor system enables:
1. Handling content that exceeds context limits
2. Sharing large content between linked processes
3. Advanced content positioning and extraction
4. Managing large user inputs
5. Marking and exporting response content
6. Passing references and content to child processes

## Examples

- `all_features.toml`: Comprehensive example with all FD features enabled
- `main.toml`: Core file descriptor features (read_fd, fd_to_file, read_file)
- `spawn_integration.toml`: Sharing file descriptors between processes
- `analyzer.toml`: Child process for content analysis (used with spawn_integration)
- `user_input.toml`: Handling large user inputs with automatic FD creation
- `references.toml`: Response reference ID system for marking and exporting content

## Running Examples

```bash
# All features combined (comprehensive example)
llmproc-demo ./examples/file_descriptor/all_features.toml

# Basic file descriptor features
llmproc-demo ./examples/file_descriptor/main.toml

# File descriptor with spawn integration
llmproc-demo ./examples/file_descriptor/spawn_integration.toml

# User input handling
llmproc-demo ./examples/file_descriptor/user_input.toml

# Response reference ID system
llmproc-demo ./examples/file_descriptor/references.toml
```

## Key Features Demonstrated

1. **Basic Operations**
   - Reading by page: `read_fd(fd="fd:1", start=0)`
   - Reading all content: `read_fd(fd="fd:1", read_all=true)`
   - Exporting to file: `fd_to_file(fd="fd:1", file_path="output.txt")`

2. **Advanced Positioning**
   - Page-based: `read_fd(fd="fd:1", mode="page", start=2, count=1)`
   - Line-based: `read_fd(fd="fd:1", mode="line", start=10, count=5)`
   - Character-based: `read_fd(fd="fd:1", mode="char", start=100, count=50)`

3. **Content Extraction**
   - Creating new FDs from portions: `read_fd(fd="fd:1", extract_to_new_fd=true)`
   - Extracting specific ranges: `read_fd(fd="fd:1", mode="line", start=10, count=5, extract_to_new_fd=true)`

4. **Process Control and Content Sharing**
   - Spawning child processes: `spawn(program_name="analyzer", prompt="Analyze this content")`
   - Forking the current process: `fork(prompts=["Process this part of the content"])`

5. **User Input Handling**
   - Automatic FD creation for large inputs
   - Preview with metadata: `<fd:1 preview="First few chars..." type="user_input" size="10000">`
   - Reading full input: `read_fd(fd="fd:1", read_all=true)`
   - Reading specific portions: `read_fd(fd="fd:1", mode="line", start=10, count=5)`
   - Delegating large input processing: `spawn(program_name="analyzer", prompt="Process the content in fd:1")` (references are automatically shared)

6. **Response References**
   - Marking content: `<ref id="example">content</ref>`
   - Accessing references: `read_fd(fd="ref:example", read_all=true)`
   - Exporting to files: `fd_to_file(fd="ref:example", file_path="output.txt")`
   - Automatic inheritance by child processes: references created in parent are available to children

## Use Cases

### 1. Working with Large Documents

```
# User sends large document (becomes fd:1)
read_fd(fd="fd:1", mode="line", start=1, count=10)  # Read first 10 lines
read_fd(fd="fd:1", mode="page", start=5, count=2)   # Read pages 5-6
read_fd(fd="fd:1", read_all=true)                  # Read entire document (if reasonable)
```

### 2. Content Extraction and Processing

```
# Extract first 100 lines to a new FD
result = read_fd(fd="fd:1", mode="line", start=1, count=100, extract_to_new_fd=true)
# Result contains new FD ID: fd:2

# Further extract from the new FD
read_fd(fd="fd:2", mode="line", start=50, count=10) # Read lines 50-60 from extracted content
```

### 3. Multi-Agent Processing with Spawn

```
# Analyze document by delegating to a specialized child process
spawn(
  program_name="analyzer",
  prompt="Find all mentions of pricing in the document in fd:1"
)

# Extract specific sections for specialized analysis
spawn(
  program_name="analyzer",
  prompt="Analyze only this section of the document in fd:1"
)
```

### 4. Parallel Processing with Fork

```
# Process different sections of a document in parallel
read_fd(fd="fd:1", mode="page", start=1, count=5, extract_to_new_fd=true)  # Extract first 5 pages to fd:2
read_fd(fd="fd:1", mode="page", start=6, count=5, extract_to_new_fd=true)  # Extract next 5 pages to fd:3

# Fork to process both sections in parallel
fork(
  prompts=[
    "Analyze the first section of the document in fd:2",
    "Analyze the second section of the document in fd:3"
  ]
)
```

### 5. Code Generation with References

```
# Generate multiple code files with references
<ref id="app_main">
console.log("App started");
</ref>

<ref id="app_utils">
function formatDate(date) {
  return date.toISOString();
}
</ref>

# Export references to files
fd_to_file(fd="ref:app_main", file_path="app.js")
fd_to_file(fd="ref:app_utils", file_path="utils.js")
```

### 6. Combining Features for Complex Workflows

```
# User sends large document (becomes fd:1)
# Extract a specific section and create a reference for it
read_fd(fd="fd:1", mode="line", start=100, count=50, extract_to_new_fd=true)  # Creates fd:2

# Generate analysis with references
<ref id="analysis">
## Document Analysis
- Key points found on lines 100-150
- Contains important financial data
</ref>

# Delegate deeper analysis to a specialized process
spawn(
  program_name="financial_analyzer",
  prompt="Analyze the financial section in fd:2 and use the analysis in ref:analysis"
)
```
