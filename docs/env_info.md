# Environment Information Feature

The Environment Information feature allows LLMProc to provide context-aware information about the runtime environment to LLM models. This enables models to better understand the context in which they're running, which can be valuable for tasks that benefit from environment awareness.

> **Note:** This feature is experimental and subject to change. Feedback is welcome as we continue to refine and improve it.

## Overview

### Purpose and Benefits

The Environment Information feature:

1. **Provides Context**: Gives models awareness of their runtime environment, such as operating system, working directory, and date
2. **Improves Relevance**: Helps models provide more relevant responses based on the user's environment
3. **Selective Sharing**: Allows you to choose which system variables to include
4. **Security-Focused**: Opt-in by default, giving you control over what information is shared
5. **Standardized Format**: Uses a consistent XML-tagged format that models can easily recognize and parse

When enabled, environment information is added to the system prompt in a structured `<env>` block:

```
<env>
working_directory: /Users/username/projects/myapp
platform: darwin
date: 2025-03-19
</env>
```

## Configuration

Environment information is configured in the program file (TOML or YAML) using the `[env_info]` section:

```yaml
env_info:
  # Specify which variables to include
  variables:
    - working_directory
    - platform
    - date
```

### Configuration Options

1. **Specifying Variables**:
   - `variables = [...]`: Include specific standard variables from the list
   - `variables = "all"`: Include all standard environment variables
   - `variables = []`: Disable environment information (default)
2. **Dynamic Information**:
   - `env_vars = {region = "MY_ENV"}`: Append value of `MY_ENV` as `region`
3. **Running Commands**:
   - `commands = ["pwd", "git status"]`: Execute shell commands and include their output
4. **file_map Options**:
   - `file_map_root = "path"`: Directory to list (default is current directory)
   - `file_map_max_files = N`: Maximum files to list before truncating (default 50)
   - `file_map_show_size = true|false`: Include file sizes in bytes (default true)

## Available Standard Variables

The following standard environment variables are available:

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `working_directory` | Current working directory | `/Users/username/projects/myapp` |
| `platform` | Operating system | `darwin`, `linux`, `windows` |
| `date` | Current date (YYYY-MM-DD) | `2025-03-19` |
| `python_version` | Python version | `3.12.4` |
| `hostname` | Machine hostname | `macbook-pro.local` |
| `username` | Current user | `username` |
| `file_map` | Recursive listing of `file_map_root` directory | `src/main.py (120 bytes)` |

## Security Considerations

The Environment Information feature is **opt-in by default** - no information is shared unless explicitly configured. When using this feature, consider:

### Information Exposure

- **Be mindful of what you share**: Environment variables can contain sensitive information
- **Usernames and Hostnames**: Consider if exposing these creates privacy concerns
- **Working Directories**: May reveal file paths that could be sensitive

### Recommended Practices

1. **Least Privilege**: Only include variables that are necessary for your use case
2. **Inspect Before Sharing**: Review what information is being included in the environment
3. **Use Different Configurations**: Consider different configurations for development vs. production

## Best Practices

### When to Use Environment Information

Environment information can be particularly useful in these scenarios:

1. **Development Tools**: When building tools that interact with code, files, or version control
2. **System Administration**: For assistants helping with system configuration or troubleshooting
3. **Location or Time-Aware Applications**: When responses should be tailored to the user's locale or timezone

### When to Avoid Environment Information

Consider not using environment information in these scenarios:

1. **Privacy-Sensitive Applications**: Where user identity or location should remain private
2. **Public-Facing Applications**: Where system details should not be exposed to end users
3. **High-Security Contexts**: Where limiting information exposure is a priority

### Integration Patterns

For the best experience:

1. **Reference in System Prompt**: Mention the environment block in your system prompt so the model knows to look for it
2. **Targeted Variables**: Include only variables relevant to your specific use case
3. **Data Validation**: Be aware that environment information is gathered at process startup and won't change during a session
4. **Reset Behavior**: Environment information will be preserved during `reset_state()` calls unless otherwise specified

## Examples

### Basic Environment Information

```yaml
env_info:
  variables:
    - working_directory
    - platform
    - date
```

### All Standard Variables

```yaml
env_info:
  variables: all  # Include all standard environment variables
```

### Development Environment

```yaml
env_info:
  variables:
    - working_directory
    - platform
    - date
    - username
```

### Custom Command Output

```yaml
env_info:
  commands:
    - echo hello
    - uname -s
```

This will produce an environment block like:

```
<env>
> echo hello
hello
> uname -s
Linux
</env>
```

If a command fails, an `error(code)` line is shown after its output.

### Directory File Map

```yaml
env_info:
  variables:
    - file_map
  file_map_root: src
  file_map_max_files: 5
```

### Comprehensive Example (All Features)

This example demonstrates all `env_info` features working together:

```yaml
env_info:
  # Include all standard environment variables
  variables:
    - working_directory  # Current working directory
    - platform           # Operating system
    - date               # Current date
    - python_version     # Python version
    - hostname           # Machine hostname
    - username           # Current user
    - file_map           # Directory listing

  # Configure file_map behavior
  file_map_root: src     # Directory to scan (relative to working directory)
  file_map_max_files: 20   # Maximum files to list
  file_map_show_size: true # Show file sizes

  # Add environment variables
  env_vars:
    region: AWS_REGION       # Value from AWS_REGION env var
    project: PROJECT_NAME     # Value from PROJECT_NAME env var

  # Execute commands and include output
  commands:
    - git rev-parse --short HEAD  # Current git commit
    - date +%H:%M:%S              # Current time
    - pwd                         # Current directory (alternative to working_directory)

  # Custom environment variables (direct values)
  custom_var: This is a custom value
  app_version: '1.0.3'
```

This will produce an environment block similar to:

```
<env>
working_directory: /Users/username/projects/myapp
platform: darwin
date: 2025-03-19
python_version: 3.12.4
hostname: macbook-pro.local
username: username
file_map:
  src/main.py (1250 bytes)
  src/utils.py (750 bytes)
  src/config.py (520 bytes)
  ... (17 more files)
region: us-west-2
project: my-awesome-project
git_rev_parse_--short_HEAD: a1b2c3d
date_+%H:%M:%S: 14:32:45
pwd: /Users/username/projects/myapp
custom_var: This is a custom value
app_version: 1.0.3
</env>
```

## Testing with llmproc-prompt

To test if your environment information configuration works correctly, you can use the `llmproc-prompt` command-line tool. This allows you to see exactly what the model will receive, including the environment information block.

### Steps to Test Environment Information

1. Create a program file with your desired `env_info` configuration (e.g., `my-program.yaml`)
2. Run the `llmproc-prompt` command:

```bash
llmproc-prompt my-program.yaml
```

3. Review the output to see the complete system prompt, including the `<env>` block

### Example Test Workflow

1. Create a test program file (`test-env-info.yaml`):

```yaml
model:
  name: claude-3-5-haiku-20241022
  provider: anthropic

prompt:
  system: You are an assistant with access to environment information.

env_info:
  variables:
    - working_directory
    - platform
    - date
    - file_map
  file_map_root: '.'
  file_map_max_files: 10
  commands:
    - git status --short
```

2. Run the test:

```bash
llmproc-prompt test-env-info.yaml
```

3. Review the output to verify:
   - The `<env>` block appears correctly
   - All requested variables are present
   - Command outputs are included
   - File map shows the expected files

This testing approach helps ensure your environment information is configured correctly before running your actual LLM application.

## Implementation Details

The environment information is implemented in `env_info/builder.py` using the `EnvInfoBuilder` class. The feature:

1. Collects requested environment variables at process initialization time
2. Formats them into an XML-tagged string
3. Adds them to the enriched system prompt
4. Makes them available to the model during conversation

The format used is deliberately simple and consistent to make it easy for models to parse and understand.

## Related Features

- **System Prompts**: Environment information is added to system prompts
- **Preloaded Files**: Similar to file preloading, environment information enhances context
- **Program Compiler**: Handles validation of environment information configuration
