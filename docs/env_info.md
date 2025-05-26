# Environment Information Feature

The Environment Information feature allows LLMProc to provide context-aware information about the runtime environment to LLM models. This enables models to better understand the context in which they're running, which can be valuable for tasks that benefit from environment awareness.

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

```toml
[env_info]
# Specify which variables to include
variables = ["working_directory", "platform", "date"]
```

### Configuration Options

1. **Specifying Variables**:
   - `variables = [...]`: Include specific standard variables from the list
   - `variables = "all"`: Include all standard environment variables
   - `variables = []`: Disable environment information (default)

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

```toml
[env_info]
variables = ["working_directory", "platform", "date"]
```

### All Standard Variables

```toml
[env_info]
variables = "all"  # Include all standard environment variables
```

### Development Environment

```toml
[env_info]
variables = ["working_directory", "platform", "date", "username"]
```

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
