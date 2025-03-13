# MCP Feature Status: WORK IN PROGRESS

**IMPORTANT: The Model Context Protocol (MCP) feature is currently in development and not ready for production use.**

## Current Status

The MCP integration is being developed on the `feature/mcp` branch and has the following limitations:

1. **Incomplete Async Support**: While tool discovery works, full async execution of tools is incomplete.

2. **Anthropic-Only**: Currently only compatible with Anthropic's Claude models, not OpenAI.

3. **Partial Tool Execution**: The synchronous API can detect when tools are requested but cannot fully execute them yet.

4. **Limited Error Handling**: Error handling and validation need improvement before production use.

## Planned Completion

Before merging to main, the following tasks will be completed:

1. Full asynchronous support with `async_run()` method
2. Complete tool execution in async context
3. Implementation for OpenAI provider
4. Improved error handling and validation
5. Comprehensive documentation
6. Complete test suite

## Experimental Usage

While this feature is in development, you can experiment with it using:

```bash
# Clone the MCP feature branch
git worktree add worktrees/mcp feature/mcp

# Navigate to the worktree
cd worktrees/mcp

# Install dependencies
uv add mcp-registry

# Set required environment variables
export ANTHROPIC_API_KEY=your-api-key
export GITHUB_TOKEN=your-github-token  # For GitHub tools

# Run the demo
python easy_mcp_demo.py
```

## Timeline

Target completion: TBD

## Contributors

This feature is being developed by the LLMProc team.

---

*Last updated: March 2025*