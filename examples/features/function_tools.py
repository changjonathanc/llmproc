"""Example of function-based tools in LLMProc.

This simple example demonstrates:
1. Registering Python functions as LLM tools 
2. Using the @register_tool decorator
3. Leveraging type hints and docstrings for schema generation
4. Using callbacks to track tool execution
5. Running a single prompt with function tool usage
"""

import asyncio
import os
from typing import Dict, Any

from dotenv import load_dotenv
from llmproc import LLMProgram, register_tool


# Load environment variables from .env file
load_dotenv()


@register_tool(name="calculator", description="Perform arithmetic calculations")
def calculate(expression: str) -> Dict[str, Any]:
    """Calculate the result of an arithmetic expression.
    
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
        
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e)
        }


@register_tool()
def weather_lookup(location: str, unit: str = "celsius") -> Dict[str, Any]:
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
        "Sydney": {"celsius": 20, "fahrenheit": 68}
    }
    
    # Default to a moderate temperature if location not found
    temp = temps.get(location, {"celsius": 21, "fahrenheit": 70})
    
    return {
        "location": location,
        "temperature": temp[unit.lower()] if unit.lower() in temp else temp["celsius"],
        "unit": unit.lower(),
        "conditions": "Sunny",
        "humidity": "60%"
    }


async def main():
    """Run the function tools example."""
    # Set up callbacks to monitor tool usage
    def on_tool_start(tool_name, tool_args):
        print(f"\n🛠️ Starting tool: {tool_name}")
        print(f"   Arguments: {tool_args}")
        
    def on_tool_end(tool_name, result):
        print(f"✅ Tool completed: {tool_name}")
        print(f"   Result: {result.content}")
        
    def on_response(message):
        print(f"\n🤖 Model response received (length: {len(message['content'])})")
        
    callbacks = {
        "on_tool_start": on_tool_start,
        "on_tool_end": on_tool_end,
        "on_response": on_response
    }
    
    # Create a program with function tools
    system_prompt = """You are a helpful assistant with access to the following tools:

1. calculator: Perform arithmetic calculations (add, subtract, multiply, divide)
2. weather_lookup: Get weather information for a location

IMPORTANT: When the user asks you about calculations or weather, ALWAYS use the appropriate tool
rather than generating the answer yourself. Show your work by explaining the results.
"""

    print("Creating program...")
    program = (
        LLMProgram(
            model_name="claude-3-5-sonnet",  # Use a less expensive model
            provider="anthropic",
            system_prompt=system_prompt
        )
        .add_tool(calculate)
        .add_tool(weather_lookup)
        .compile()
    )
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in your environment or create a .env file with this value.")
        print("\nSimulating tool execution instead:")
        
        # Directly call the tools to show how they work
        print("\nDirect tool calls:")
        calc_result = calculate("5 * 7 + 3")
        print(f"calculator('5 * 7 + 3') → {calc_result}")
        
        weather_result = weather_lookup("New York", "fahrenheit")
        print(f"weather_lookup('New York', 'fahrenheit') → {weather_result}")
        return
    
    # Start the LLM process
    print("Starting process...")
    process = await program.start()
    
    # Print available tools
    print("\nRegistered tools:")
    for tool in process.tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    # Run a prompt that will trigger tool usage
    user_prompt = """I have two questions:
1. What's the result of 125 * 48?
2. What's the weather like in Tokyo?"""

    print(f"\nRunning with prompt: '{user_prompt}'")
    result = await process.run(user_prompt, callbacks=callbacks)
    
    # Print the final response
    print("\n===== FINAL RESPONSE =====")
    print(process.get_last_message())
    print("==========================")
    
    # Print execution statistics
    print(f"\nExecution completed in {result.duration:.2f} seconds")
    print(f"Tool calls: {result.tool_calls}")
    print(f"Total iterations: {result.iterations}")


if __name__ == "__main__":
    asyncio.run(main())