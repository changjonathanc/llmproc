# API Refactoring Plan for LLMProc

## Overview

This document outlines the plan for refactoring the LLMProc library API to create a more intuitive, consistent, and user-friendly interface. The refactoring will focus on creating a clear separation between program configuration and process execution, while also adding support for callbacks and improved diagnostics.

## Core Design Principles

1. **Clear Separation of Concerns**
   - Programs (configuration/specification) vs Processes (runtime instances)
   - Configuration/compilation vs execution
   - State management vs result extraction

2. **Async-First Design**
   - All potentially long-running operations should be async
   - Simple mental model for async operations

3. **Single Path to Success**
   - One clear way to accomplish common tasks
   - Consistent patterns throughout the API

4. **Event-Based Programming**
   - Callbacks for monitoring and integration
   - Detailed run results for diagnostics

## Implementation Phases

### Phase 1: Core API Restructuring

1. **Create RunResult Class**
   - Implement in a new file: `src/llmproc/results.py`
   - Include fields for API calls, tools used, duration, etc.
   - Add helper methods for reporting and analysis

2. **Refactor LLMProgram Class**
   - Simplify `from_toml` to focus on loading and validation
   - Add `start()` method that returns an initialized LLMProcess
   - Move all program activation logic to this method

3. **Update LLMProcess Class**
   - Make `__init__` focused on basic initialization
   - Create `_complete_async_init()` method for async setup
   - Implement `get_last_message()` properly
   - Update `run()` to return RunResult and accept callbacks

4. **Callback Framework**
   - Implement callback handling in `_async_run`
   - Define standard callback points (tool start/end, iteration, response)
   - Add documentation and examples for callback usage

### Phase 2: Process Executor Refactoring

1. **Refactor AnthropicProcessExecutor**
   - Update to work with the new callback system
   - Refine fork implementation
   - Ensure proper RunResult creation and returning

2. **Expand Debugging and Diagnostics**
   - Replace all debug flags with proper logging
   - Add execution timing to RunResult
   - Track all tools used during execution

3. **Clean up Legacy Code**
   - Remove `run_anthropic_with_tools` and other deprecated functions
   - Clean up all commented-out code and TODOs
   - Remove debug flags and `LLMPROC_DEBUG` references

### Phase 3: CLI Updates and Documentation

1. **Update CLI**
   - Refactor to use the new API pattern
   - Add progress indicators using callbacks
   - Display run statistics after completion

2. **Update Documentation**
   - Create comprehensive API documentation
   - Add usage examples
   - Update all example programs

3. **Update Tests**
   - Refactor tests to use the new API pattern
   - Add tests for callbacks and RunResult
   - Verify all functionality with updated interfaces

## File-by-File Changes

### `src/llmproc/results.py` (New File)
- Create `RunResult` class
- Add metrics tracking functionality

### `src/llmproc/program.py`
- Update `LLMProgram.from_toml`
- Add `LLMProgram.start()`
- Clean up compilation logic

### `src/llmproc/llm_process.py`
- Refactor `__init__` to be simpler
- Update `run()` to use callbacks and return RunResult
- Improve `get_last_message()`
- Add callback handling

### `src/llmproc/providers/anthropic_process_executor.py`
- Update to collect run metrics
- Implement callback triggers
- Fix fork implementation

### `src/llmproc/cli.py`
- Update to use new API pattern
- Add progress indicators
- Display run statistics

### Example Updates
- Update all examples to use the new API
- Add example with callbacks
- Create demo for diagnostics and metrics

## Testing Strategy

1. **Unit Tests**
   - Test each component in isolation
   - Verify proper callback execution
   - Check RunResult contents

2. **Integration Tests**
   - Verify end-to-end flow
   - Test with complex tool usage
   - Check metrics accuracy

3. **Mocking**
   - Create clear mocks for testing callbacks
   - Mock API responses to test edge cases

## Timeline

1. **Phase 1:** 3-4 days
   - Core API restructuring
   - Basic callback framework

2. **Phase 2:** 2-3 days
   - Process executor refactoring
   - Metrics collection
   - Legacy code cleanup

3. **Phase 3:** 2-3 days
   - CLI updates
   - Documentation
   - Example programs

Total estimated time: 7-10 days

## Migration Guide

Since we're in active development without backward compatibility concerns, we'll focus on:

1. Updating all internal usages to the new pattern
2. Updating example programs
3. Ensuring tests use the new approach

Final Note: This refactoring should be done as a single cohesive project to maintain consistency across the codebase.