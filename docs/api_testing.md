# LLMProc API Testing Guide

This document describes the API tests that verify the functionality of the llmproc library with actual LLM APIs.

## Test Overview

The API tests in `tests/test_example_programs.py` verify that all example programs work correctly with their respective LLM APIs. These tests:

1. Test each example program file
2. Verify both the LLMProcess API and the CLI interface
3. Test special features like file preloading, tools usage, and program linking
4. Check provider-specific functionality
5. Verify error handling and recovery

## Running the Tests

API tests are marked with the `llm_api` marker and are skipped by default to avoid requiring API keys for regular test runs.

To run the API tests:

```bash
# Run all API tests
pytest -m llm_api

# Run specific API tests
pytest -m llm_api tests/test_example_programs.py

# Run with verbose output
pytest -m llm_api -v
```

### Required API Keys

To run the tests, you need to set the following environment variables:

- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Anthropic models
- `ANTHROPIC_VERTEX_PROJECT_ID` - For Anthropic on Vertex AI models
- `CLOUD_ML_REGION` - For Anthropic on Vertex AI models (defaults to us-central1)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account credentials for Google Cloud authentication

## Test Categories

### Basic Program Tests

These tests verify that each example TOML program works with the actual LLM API:

- The parametrized `test_example_program` function tests all example programs
- Uses a simple prompt to verify API connectivity and response
- Checks both standalone examples and program linking examples

### Feature-Specific Tests

#### Minimal Functionality

`test_minimal_functionality`: Tests basic LLMProcess features with the minimal program:
- Verifies basic Q&A functionality
- Tests conversation continuity
- Tests state reset functionality

#### MCP Tool Usage

`test_mcp_tool_functionality`: Tests the Model Context Protocol functionality:
- Verifies tool registration
- Tests tool execution with the codemcp tool
- Confirms the model can use tools to access file content

#### Program Linking

`test_program_linking_functionality`: Tests LLM-to-LLM communication:
- Verifies the spawn tool works correctly
- Tests delegation to specialized LLMs
- Confirms the response maintains formatting guidelines

#### File Preloading

`test_file_preload_functionality`: Tests file preloading for context:
- Verifies preloaded files are accessible in the model's context
- Tests the model can reference content from preloaded files

#### Claude Code Features

`test_claude_code_comprehensive`: Tests comprehensive features in the Claude Code program:
- Tests preloaded file knowledge
- Tests tool execution with dispatch_agent
- Tests combined functionality using both preloaded content and tools

### Provider-Specific Tests

`test_provider_specific_functionality`: Tests each provider with their specific programs:
- OpenAI models
- Anthropic models
- Vertex AI models (if credentials available)

### CLI Interface Tests

These tests verify the command-line interface works correctly:

#### Basic CLI Testing

`test_cli_with_minimal_example`: Tests basic CLI functionality:
- Verifies the CLI displays program information
- Tests that the model responds to prompts
- Confirms the CLI correctly formats and displays responses

#### Program Linking CLI

`test_cli_with_program_linking`: Tests program linking via CLI:
- Verifies the CLI handles tool execution
- Tests LLM-to-LLM communication works in the CLI interface

#### All Programs CLI Test

`test_cli_with_all_programs`: Tests all example programs with the CLI:
- Parametrized test that runs each program
- Verifies consistent behavior across different model providers
- Tests prompt echo functionality to ensure response integrity

#### Error Handling

`test_error_handling_and_recovery`: Tests CLI error handling:
- Tests behavior with invalid program
- Verifies error reporting
- Confirms system recovers with valid program

## Test Implementation

The tests use a combination of:

1. Direct API calls through the LLMProcess class
2. CLI interface testing through subprocess
3. Unique identifier strings to verify response integrity
4. Timeouts to handle slow API responses
5. Exception handling for robustness

## Adding New Tests

When adding new example programs, consider:

1. Adding the program to the parametrized tests
2. Creating specific tests for any unique features
3. Ensuring all tests skip properly when API keys aren't available
4. Adding appropriate timeout values for potentially slow operations
