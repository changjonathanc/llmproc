"""Comprehensive example of the LLMProc Python SDK.

This example demonstrates the full range of configuration options available
in the Python SDK, including:

1. Basic program configuration
2. Function-based tools
3. Environment information
4. File descriptor system
5. Claude 3.7 thinking models
6. Token-efficient tools
7. MCP configuration
8. Program linking

Usage:
  python comprehensive_python_api.py

Requirements:
  - python -m pip install -e ".[dev,all]"
  - Appropriate API keys in environment variables
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from llmproc import LLMProgram, register_tool

# Load environment variables from .env file
load_dotenv()


# Define custom tools
@register_tool(description="Perform simple arithmetic calculations")
def calculate(expression: str) -> dict[str, Any]:
    """Calculate the result of a simple arithmetic expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "5 * 10"

    Returns:
        A dictionary with the result and the parsed expression
    """
    # Simple and safe evaluation using Python's eval with limited scope
    try:
        # Only allow basic arithmetic operations
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains disallowed characters")

        # Evaluate the expression using a restricted scope
        result = eval(expression, {"__builtins__": {}})

        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@register_tool()
def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize a text to a specified maximum length.

    Args:
        text: The text to summarize
        max_length: Maximum length of the summary in characters

    Returns:
        A summary of the text
    """
    # Simple summarization by truncation with ellipsis
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


async def main():
    """Run the comprehensive Python SDK example."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set the ANTHROPIC_API_KEY environment variable.")
        return

    print("Creating programs...")

    # 1. Create a specialized math expert program
    math_expert = (
        LLMProgram(
            model_name="claude-3-5-haiku-20240307",
            provider="anthropic",
            system_prompt="You are a mathematics expert. Answer all questions with clear, step-by-step explanations.",
            parameters={"temperature": 0.1, "max_tokens": 1024},
        )
        .set_enabled_tools([calculate])
        .configure_env_info([])  # Explicitly disable env info
    )

    # 2. Create a code expert program
    code_expert = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt="You are a coding expert. Provide accurate, efficient code examples with clear explanations.",
            parameters={"temperature": 0.2, "max_tokens": 4096},
        )
        .configure_thinking(budget_tokens=4096)  # Enable thinking capability
        .enable_token_efficient_tools()  # Enable token-efficient tools
        .configure_env_info(["platform", "python_version"])  # Limited env info
    )

    # 3. Create main program with all features
    main_program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt="You are a helpful assistant that can coordinate with specialized experts.",
            parameters={
                "temperature": 0.7,
                "max_tokens": 8192,
                "top_p": 0.95,
                "top_k": 40,
            },
            display_name="Comprehensive Assistant",
        )
        # Add tools
        .set_enabled_tools([calculate, summarize_text])
        # Add specialized programs
        .add_linked_program(
            "math_expert", math_expert, "Expert in mathematics and calculations"
        )
        .add_linked_program(
            "code_expert", code_expert, "Expert in programming and software development"
        )
        # Configure environment info
        .configure_env_info(["working_directory", "platform", "date"])
        # Configure file descriptor system
        .configure_file_descriptor(
            enabled=True,
            max_direct_output_chars=8000,
            default_page_size=4000,
            enable_references=True,
        )
        # Configure thinking capability
        .configure_thinking(enabled=True, budget_tokens=8192)
        # Enable token-efficient tools
        .enable_token_efficient_tools()
        # Optional: Configure MCP if you have it set up
        # .configure_mcp(
        #     config_path="config/mcp_servers.json",
        #     tools={"sequential-thinking": "all", "everything": ["add"]}
        # )
    )

    # Print configuration summary
    print("\nProgram Configuration:")
    print(f"Model: {main_program.model_name}")
    print(f"Provider: {main_program.provider}")
    print(f"Display name: {main_program.display_name}")
    print(f"Linked programs: {list(main_program.linked_programs.keys())}")

    try:
        # Start the process (handles compilation automatically)
        print("\nStarting the process...")
        process = await main_program.start()

        # Run a prompt that will demonstrate the features
        user_prompt = """I have a few questions for you:

1. What's the result of 125 * 48?
2. Can you show me how to calculate the factorial of a number in Python?
3. What operating system are you running on?

Also, here's a very long text that you should summarize:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."""

        print(f"\nRunning with prompt: '{user_prompt[:100]}...'")
        result = await process.run(user_prompt)

        # Print the final response
        print("\n===== FINAL RESPONSE =====")
        print(process.get_last_message())
        print("==========================")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("This might happen if your API keys aren't set up correctly.")


if __name__ == "__main__":
    asyncio.run(main())
