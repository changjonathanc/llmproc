# Release Notes - v0.9.0

## 🚀 New Features

### More GitHub Automation Workflows!
- **`@llmproc /code`**: Automated code implementation workflow for GitHub issues/PRs
- **`@llmproc /ask`**: Question answering workflow for repository context
- **`@llmproc /resolve`**: Automated PR conflict resolution workflow
- Integrated with GitHub Actions for seamless automation

### Improved CLI Experience
- **`llmproc`**: Added flexible `-p/--prompt` flag handling:
  - **Prompt File Support**: New `--prompt-file` option for reading prompts from files
  - **JSON Output**: Added `--json` flag to `llmproc` for structured output, useful for automation
- **`llmproc-demo`**: Added flexible `-p/--prompt` flag handling:
  - `-p "custom prompt"`: Run custom prompt, then continue interactive
  - `-p` (without argument): Skip embedded prompt, go directly to interactive mode
  - Default behavior: Show embedded prompt with confirmation, then interactive
- logging and formatting improvements


### Enhanced Environment Info System (Experimental)
- **Runtime Commands**: The `env_info` configuration now supports a `commands` option to execute shell commands at runtime and include their output in the system prompt
- **Multiple Environment Variables**: Support for specifying multiple environment variables to include in the context
- **File Mapping**: New `file_map` option to map local files to different paths in the environment info
- **Configurable Base Path**: The `preload` feature now supports configurable base paths for file resolution

## 🛠️ Improvements
- Error handling and logging improvements

## Breaking Changes
- Removed deprecated `tool_aliases` parameter in favor of ToolConfig-based aliases
---

For detailed API documentation and more examples, visit the [documentation](https://github.com/cccntu/llmproc/tree/main/docs).
