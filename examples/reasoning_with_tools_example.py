#!/usr/bin/env python3
"""
Claude 4 Interleaved Thinking with Multi-Step Tool Use Example

This example demonstrates Claude 4 Sonnet's advanced interleaved thinking capabilities
with LLMProc's tool system for complex analytical tasks.

Key demonstrations:
- Interleaved thinking: reasoning between every tool call
- Multi-step reasoning and tool use (8-15 iterations)
- Strategic exploration of codebases through systematic tool calling
- How Claude 4 reflects on tool results before deciding next steps
- Proper LLMProc usage patterns for advanced reasoning workflows

The example shows Claude 4 Sonnet systematically analyzing a codebase by:
1. Reasoning about what to explore next after each tool call
2. Building understanding incrementally through strategic tool sequencing
3. Reflecting on tool results to guide subsequent exploration
4. Providing comprehensive analysis through sophisticated reasoning
"""

import asyncio
import os

from llmproc import LLMProgram
from llmproc.tools.builtin.list_dir import list_dir
from llmproc.tools.builtin.read_file import read_file


async def main():
    """Run the codebase analyzer with LLMProc."""
    # Create the LLM program using Claude 4 Sonnet with interleaved thinking
    program = LLMProgram(
        model_name="claude-sonnet-4-20250514",
        provider="anthropic",
        system_prompt="""You are an expert code analyst tasked with understanding and summarizing codebases.

Your goal is to analyze the structure and purpose of a codebase by systematically exploring directories and reading key files. You have access to two tools:

1. list_dir - List contents of directories
2. read_file - Read the contents of specific files

When analyzing a codebase:
1. Think strategically about what to explore next after each tool call
2. Start by exploring the root directory structure
3. Identify key files like README.md, setup.py, pyproject.toml, etc.
4. Explore the main source directories to understand the project structure
5. Read important configuration and documentation files
6. Reflect on each tool result before deciding your next action
7. Provide a comprehensive summary including:
   - Project purpose and description
   - Main components and architecture
   - Key features and capabilities
   - Technology stack and dependencies

Use your interleaved thinking to reason about tool results and plan your exploration strategy.""",
        parameters={
            "max_tokens": 8000,
            "temperature": 1,  # Required for thinking mode
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2000,  # Enable thinking for sophisticated reasoning
            },
            "extra_headers": {
                "anthropic-beta": "interleaved-thinking-2025-05-14"  # Enable interleaved thinking
            },
        },
    )

    # Register the analysis tools
    program.register_tools([list_dir, read_file])

    # Create the process
    process = await program.start()

    # Run the analysis task
    analysis_prompt = """Please analyze this codebase systematically. Start by exploring the directory structure to understand the project layout, then read key files to understand the project's purpose, architecture, and capabilities.

Focus on:
1. What is this project and what does it do?
2. What is the main architecture and key components?
3. What are the primary features and capabilities?
4. What technologies and frameworks are used?
5. How is the project structured and organized?

Use your interleaved thinking to reflect on each tool result before deciding what to explore next. Make multiple strategic tool calls to build a complete understanding."""

    print("🔍 Starting interleaved thinking analysis with Claude 4 Sonnet...")
    print("=" * 70)

    # Execute the analysis
    try:
        result = await process.run(analysis_prompt, max_iterations=15)
    except Exception as e:
        if "model" in str(e).lower() and ("not found" in str(e).lower() or "invalid" in str(e).lower()):
            print(f"❌ Claude 4 Sonnet not available: {e}")
            print("💡 Falling back to Claude 3.7 Sonnet...")

            # Fallback to Claude 3.7 without interleaved thinking
            program = LLMProgram(
                model_name="claude-3-7-sonnet-20250219",
                provider="anthropic",
                system_prompt="""You are an expert code analyst tasked with understanding and summarizing codebases.

Your goal is to analyze the structure and purpose of a codebase by systematically exploring directories and reading key files. You have access to two tools:

1. list_dir - List contents of directories
2. read_file - Read the contents of specific files

When analyzing a codebase:
1. Start by exploring the root directory structure
2. Identify key files like README.md, setup.py, pyproject.toml, etc.
3. Explore the main source directories to understand the project structure
4. Read important configuration and documentation files
5. Provide a comprehensive summary

Work systematically and use multiple tool calls as needed to thoroughly understand the codebase.""",
                parameters={
                    "max_tokens": 8000,
                    "temperature": 0.3,
                },
            )

            # Register the analysis tools
            program.register_tools([list_dir, read_file])

            # Create the process
            process = await program.start()

            print("🔍 Starting analysis with Claude 3.7 Sonnet fallback...")
            print("=" * 70)

            # Execute the analysis with fallback
            result = await process.run(analysis_prompt, max_iterations=15)
        else:
            raise

    print("\n" + "=" * 70)
    print("📊 Analysis Complete!")
    print("=" * 70)

    # Get the final response
    final_response = process.get_last_message()
    print(final_response)

    # Show execution details
    print(f"\n🔍 Result type: {type(result)}")
    print(f"📊 Result: {result}")

    # Show token usage if available
    try:
        token_count = await process.count_tokens()
        print(f"📈 Token Usage: {token_count}")
    except Exception as e:
        print(f"⚠️  Could not retrieve token count: {e}")


if __name__ == "__main__":
    # Check for required environment variable
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable is required")
        print("Please set it with: export ANTHROPIC_API_KEY=your_key_here")
        exit(1)

    print("🚀 LLMProc Claude 4 Interleaved Thinking Example")
    print("Using Claude 4 Sonnet with Interleaved Thinking and Multi-Step Tool Use")
    print(f"📁 Working Directory: {os.getcwd()}")
    print()

    asyncio.run(main())
