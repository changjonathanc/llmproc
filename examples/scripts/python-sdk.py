"""Complete Python SDK demonstration for LLMProc.

This example demonstrates the full range of LLMProc's Python SDK capabilities:

1. Program Configuration
   - Basic model and provider setup
   - Parameter configuration
   - System prompt configuration
   - Program linking with specialized models
   - Environment information configuration
   - File descriptor system setup

2. Function-Based Tools
   - Basic tool registration with @register_tool
   - Automatic schema generation from type hints and docstrings
   - Parameter validation and error handling
   - Custom tool names and descriptions
   - Runtime context tools with context dependency injection
   - Callback functions for monitoring tool execution

3. Advanced Features
   - Claude 3.7 thinking models configuration
   - Token-efficient tool use enabling
   - MCP tool integration
   - Tool aliases for easier tool referencing

Usage:
  python python-sdk.py

Requirements:
  - python -m pip install -e ".[dev,all]"
  - Appropriate API keys in environment variables
"""

import asyncio
import math
import os
from typing import Any, Optional

from dotenv import load_dotenv
from llmproc import LLMProgram, register_tool
from llmproc.callbacks import CallbackEvent
from llmproc.tools.mcp import MCPServerTools

# Load environment variables from .env file
load_dotenv()


# --- SECTION 1: TOOL DEFINITIONS ---


# Basic calculator tool with simple evaluation
@register_tool(name="calculator", description="Perform arithmetic calculations")
def calculate(expression: str) -> dict[str, Any]:
    """Calculate the result of an arithmetic expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "5 * 10"

    Returns:
        A dictionary with the result and the parsed expression
    """
    try:
        # Only allow basic arithmetic operations for safety
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains disallowed characters")

        # Evaluate the expression using a restricted scope
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# Text utility tool with default parameter
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


# Simulated weather lookup tool
@register_tool()
def weather_lookup(location: str, unit: str = "celsius") -> dict[str, Any]:
    """Look up weather information for a location.

    Args:
        location: City name or address
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information for the location
    """
    # Simulate weather lookup
    temps = {
        "New York": {"celsius": 22, "fahrenheit": 72},
        "London": {"celsius": 18, "fahrenheit": 64},
        "Tokyo": {"celsius": 26, "fahrenheit": 79},
        "Sydney": {"celsius": 20, "fahrenheit": 68},
    }

    # Default to a moderate temperature if location not found
    temp = temps.get(location, {"celsius": 21, "fahrenheit": 70})

    return {
        "location": location,
        "temperature": temp[unit.lower()] if unit.lower() in temp else temp["celsius"],
        "unit": unit.lower(),
        "conditions": "Sunny",
        "humidity": "60%",
    }


# Advanced tool with custom parameter descriptions
@register_tool(
    description="Perform statistical calculations on a list of numbers",
    param_descriptions={
        "numbers": "List of numbers to analyze",
        "operations": "Statistical operations to perform (mean, median, sum, min, max, std)",
    },
)
def stats_calculator(numbers: list[float], operations: list[str] = None) -> dict[str, Any]:
    """Calculate statistical measures for a list of numbers.

    Args:
        numbers: List of numbers to analyze
        operations: Statistical operations to perform

    Returns:
        Dictionary with the results of each requested operation
    """
    if operations is None:
        operations = ["mean", "median"]
    if not numbers:
        return {"error": "Empty list provided"}

    results = {}
    for op in operations:
        if op == "mean":
            results["mean"] = sum(numbers) / len(numbers)
        elif op == "median":
            sorted_nums = sorted(numbers)
            mid = len(numbers) // 2
            if len(numbers) % 2 == 0:
                results["median"] = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
            else:
                results["median"] = sorted_nums[mid]
        elif op == "sum":
            results["sum"] = sum(numbers)
        elif op == "min":
            results["min"] = min(numbers)
        elif op == "max":
            results["max"] = max(numbers)
        elif op == "std":
            mean = sum(numbers) / len(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            results["std"] = math.sqrt(variance)

    return results


# Example of a runtime context tool (requires LLMProcess access)
@register_tool(
    name="spawn_example",
    description="Example of a tool requiring runtime context",
    param_descriptions={"program_name": "Name of the program to call", "prompt": "The prompt to send"},
    required=["program_name", "prompt"],
    requires_context=True,
    required_context_keys=["process"],
)
async def spawn_example(
    program_name: str, prompt: str, runtime_context: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Example of a tool that requires runtime context access.

    Args:
        program_name: Name of the program to call
        prompt: The prompt to send
        runtime_context: Runtime context with process information

    Returns:
        Response information
    """
    # This would normally use runtime_context to access the process
    # but for this example we just return a simple response
    return {
        "program": program_name,
        "prompt": prompt,
        "response": "This is a simulated response from a spawned program",
    }


# --- SECTION 2: CALLBACK FUNCTIONS ---


# Class-based callbacks following the new pattern
class SDKCallbacks:
    def tool_start(self, tool_name, tool_args):
        """Callback triggered when a tool starts execution."""
        print(f"\n🛠️ Starting tool: {tool_name}")
        print(f"   Arguments: {tool_args}")

    def tool_end(self, tool_name, result):
        """Callback triggered when a tool completes execution."""
        print(f"✅ Tool completed: {tool_name}")
        print(f"   Result: {result.content}")

    def response(self, message):
        """Callback triggered when a model response is received."""
        if isinstance(message, dict) and "content" in message:
            print(f"\n🤖 Model response received (length: {len(message['content'])})")
        else:
            print("\n🤖 Model response received")


# --- SECTION 3: MAIN EXAMPLE ---


async def main():
    """Run the Python SDK comprehensive example."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set the ANTHROPIC_API_KEY environment variable.")
        return

    print("====== LLMProc Python SDK Demonstration ======")

    # --- CREATING SPECIALIZED PROGRAMS ---
    print("\n1. Creating specialized programs...")

    # Math expert program (faster, simpler model)
    math_expert = (
        LLMProgram(
            model_name="claude-3-5-haiku-20240307",
            provider="anthropic",
            system_prompt="You are a mathematics expert. Answer all questions with clear, step-by-step explanations.",
            parameters={"temperature": 0.1, "max_tokens": 1024},
        )
        .register_tools([calculate, stats_calculator])
        .configure_env_info([])  # Explicitly disable env info
    )
    print("   ✓ Math expert program created (claude-3-5-haiku)")

    # Code expert program (more capable model with thinking)
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
    print("   ✓ Code expert program created (claude-3-7-sonnet with thinking)")

    # --- CREATING MAIN PROGRAM WITH FULL FEATURES ---
    print("\n2. Creating main program with all features...")

    # Main program with all features
    main_program = (
        LLMProgram(
            model_name="claude-3-7-sonnet-20250219",
            provider="anthropic",
            system_prompt=(
                "You are a helpful assistant that can coordinate with specialized experts."
                "Use your tools appropriately when asked for calculations, weather information, or statistics."
                "When a user asks for complex information, consider using your specialized experts."
            ),
            parameters={
                "temperature": 0.7,
                "max_tokens": 8192,
                "top_p": 0.95,
                "top_k": 40,
            },
            display_name="Comprehensive Assistant",
        )
        # Register all our tools
        .register_tools([calculate, summarize_text, weather_lookup, stats_calculator, spawn_example])
        # Set up tool aliases for convenience
        .set_tool_aliases(
            {
                "calc": "calculator",
                "stats": "stats_calculator",
                "weather": "weather_lookup",
                "summarize": "summarize_text",
            }
        )
        # Add specialized programs
        .add_linked_program("math_expert", math_expert, "Expert in mathematics and calculations")
        .add_linked_program("code_expert", code_expert, "Expert in programming and software development")
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
        # .configure_mcp("config/mcp_servers.json")
        # .register_tools([
        #     MCPServerTools(server="sequential-thinking"),  # all tools
        #     MCPServerTools(server="everything", names="add")  # specific tool
        # ])
    )
    print("   ✓ Main program created with all features")

    # Display configuration summary
    print("\n3. Program Configuration Summary:")
    print(f"   Model: {main_program.model_name}")
    print(f"   Provider: {main_program.provider}")
    print(f"   Display name: {main_program.display_name}")
    print(f"   Linked programs: {list(main_program.linked_programs.keys())}")
    print(f"   Tools: {main_program.get_registered_tools()}")

    # Display detailed tool demonstration
    print("\n4. Tool Demonstration Before LLM Integration:")
    print("\n   Direct tool execution:")
    calc_result = calculate("5 * 7 + 3")
    print(f"   calculator('5 * 7 + 3') → {calc_result}")

    weather_result = weather_lookup("New York", "fahrenheit")
    print(f"   weather_lookup('New York', 'fahrenheit') → {weather_result}")

    stats_result = stats_calculator([4, 7, 9, 2, 8, 5], ["mean", "median", "min", "max"])
    print(f"   stats_calculator([4, 7, 9, 2, 8, 5], ['mean', 'median', 'min', 'max']) → {stats_result}")

    try:
        # Start the process
        print("\n5. Starting the process with LLM integration...")
        process = await main_program.start()
        print("   ✓ Process started successfully")

        # Initial simple run to demonstrate basic .run() pattern from README
        print("\n6. Basic run demonstration:")
        intro_result = await process.run("Tell me about the tools you have available and what you can do with them.")
        print("   ✓ Basic prompt execution completed")

        # Get schemas for all tools
        print("\n7. Registered Tools (with schemas):")
        for tool in process.tools:
            print(f"   - {tool['name']}: {tool['description']}")

        # Run a test prompt that will trigger multiple tool calls and features
        user_prompt = """I have a few questions for you:

1. What's the result of 125 * 48?
2. What's the weather like in Tokyo?
3. Can you provide statistics for this dataset: [12, 15, 9, 22, 18, 7, 14]? I'd like the mean, median, min, max, and standard deviation.
4. Can you show me how to calculate the factorial of a number in Python?

Also, here's a very long text that you should summarize:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem."""

        print("\n8. Running with test prompt...")
        print(f"   Prompt (truncated): '{user_prompt[:100]}...'")

        # Register callbacks using the new pattern
        process.add_callback(SDKCallbacks())

        # Run without callbacks parameter
        result = await process.run(user_prompt)

        # Print the final response
        print("\n===== FINAL RESPONSE =====")
        print(process.get_last_message())
        print("==========================")

        # Print stats if available
        if hasattr(result, "duration"):
            print(f"\nExecution time: {result.duration:.2f} seconds")
        if hasattr(result, "iterations"):
            print(f"Total iterations: {result.iterations}")
        if hasattr(result, "tool_calls"):
            print(f"Tool calls: {result.tool_calls}")

        print("\n9. Python SDK Features Demonstrated:")
        print("   ✓ Function-based tools")
        print("   ✓ Tool registration with @register_tool")
        print("   ✓ Parameter validation and type handling")
        print("   ✓ Linked programs")
        print("   ✓ Tool aliases")
        print("   ✓ Runtime callbacks")
        print("   ✓ Environment information")
        print("   ✓ File descriptor system")
        print("   ✓ Claude 3.7 thinking capability")
        print("   ✓ Token-efficient tools")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("This might happen if your API keys aren't set up correctly.")


if __name__ == "__main__":
    asyncio.run(main())
