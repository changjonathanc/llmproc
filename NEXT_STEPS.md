# Next Steps for LLMProc Project

## Code Formatting and Quality

- [ ] Run ruff to format all Python files: `ruff format src/ tests/`
- [ ] Run linting to fix common issues: `ruff check --fix src/ tests/`
- [ ] Ensure consistent imports across all files
- [ ] Remove any debug print statements or commented-out code

## OpenAI Implementation Improvements

- [ ] Implement token counting for OpenAI (similar to Anthropic)
- [ ] Add proper support for newer OpenAI models (e.g., gpt-4o)
- [ ] Implement proper error handling for rate limits and other API errors
- [ ] Add streaming response support (optional)

## Documentation Updates

- [ ] Update CLAUDE.md with a session summary documenting our OpenAI executor addition
- [ ] Add/update README section on supported providers
- [ ] Create an example demonstrating differences between OpenAI and Anthropic usage
- [ ] Document the CLI non-interactive mode we implemented

## Testing Enhancements

- [ ] Create thorough tests for all providers
- [ ] Add explicit tests for interactive and non-interactive CLI modes
- [ ] Test edge cases like token limits, rate limits, and error conditions
- [ ] Add CI/CD checks for linting and testing

## Future Improvements

- [ ] Consider implementing tool support for OpenAI models
- [ ] Add caching options for expensive API calls
- [ ] Implement streaming responses for both providers
- [ ] Add proper progress indicators for CLI
- [ ] Evaluate if fork functionality should be extended to OpenAI

## Cleanup

- [x] Remove API_REFACTOR_PLAN.md (completed most items)
- [ ] Review and address all TODOs in the codebase
- [ ] Update type annotations for consistency
- [ ] Consider splitting AnthropicProcessExecutor into smaller functions

## Migration Guide

- [ ] Create a guide for users migrating from direct OpenAI or Anthropic usage
- [ ] Document version compatibility requirements
- [ ] Create a cheat sheet for common operations