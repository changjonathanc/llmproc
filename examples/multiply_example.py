"""Simple example demonstrating LLMProc with a function-based tool.

This example shows how to register a Python function as a tool.
The @register_tool decorator is optional - function type hints and
docstrings are automatically converted to JSON schema.
"""

import asyncio

from llmproc import LLMProgram


def multiply(a: float, b: float) -> dict:
    """Multiply two numbers and return the result."""
    return {"result": a * b}  # Expected: π * e = 8.539734222677128


async def main():
    program = LLMProgram(
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        system_prompt="You're a helpful assistant.",
        parameters={"max_tokens": 1024},
    )

    # Register the function as a tool
    program.register_tools([multiply])

    process = await program.start()
    await process.run("Can you multiply 3.14159265359 by 2.71828182846?")

    print(process.get_last_message())


if __name__ == "__main__":
    asyncio.run(main())
