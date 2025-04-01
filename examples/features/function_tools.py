"""Example of using function-based tools with LLMProcess.

This example demonstrates how to:
1. Use the register_tool decorator to register functions as tools
2. Register both sync and async functions as tools
3. Use method chaining with the fluent API
4. Use tool schemas generated from function type hints and docstrings
5. Handle LLM string inputs with automatic type conversion
6. Create an interactive CLI for testing function-based tools with a real LLM

Usage:
    python function_tools.py          # Run the non-interactive demo
    python function_tools.py --chat   # Run the interactive chat interface
"""

import asyncio
import argparse
import json
import os
import readline  # Enable command history
import sys
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from llmproc import LLMProgram, register_tool


# Load environment variables from .env file
load_dotenv()


# Simple calculator function with type hints
def add_numbers(x: int, y: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The sum of x and y
    """
    print(f"[Tool Debug] add_numbers called with x={x} (type: {type(x)}), y={y} (type: {type(y)})")
    return x + y


# Function that parses JSON input - great for testing string to object conversion
def parse_and_analyze(json_data: str) -> Dict[str, Any]:
    """Parse JSON string and return analysis.
    
    Args:
        json_data: A JSON string to parse
        
    Returns:
        Analysis of the parsed JSON
    """
    print(f"[Tool Debug] parse_and_analyze called with data: {json_data}")
    
    # Parse the JSON string
    try:
        parsed = json.loads(json_data)
        
        # Perform analysis
        result = {
            "type": type(parsed).__name__,
            "length": len(parsed) if hasattr(parsed, "__len__") else None,
            "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
            "is_array": isinstance(parsed, list),
            "parsed_data": parsed
        }
        
        return result
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}


# Function with complex types
def search_documents(
    query: str,
    limit: int = 5,
    categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Search documents by query.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return
        categories: Optional list of categories to search within
        
    Returns:
        List of document dictionaries matching the query
    """
    print(f"[Tool Debug] search_documents called with query={query}, limit={limit}, categories={categories}")
    
    # Example implementation
    if categories:
        return [{"id": i, "title": f"Result {i} for {query} in {categories[0]}"} for i in range(min(3, limit))]
    else:
        return [{"id": i, "title": f"Result {i} for {query}"} for i in range(min(3, limit))]


# Decorated function with custom name and description
@register_tool(name="weather_info", description="Get weather information for a location")
def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """Get weather for a location.
    
    Args:
        location: City or address
        units: Temperature units (celsius or fahrenheit)
        
    Returns:
        Weather information including temperature and conditions
    """
    print(f"[Tool Debug] get_weather called with location={location}, units={units}")
    
    # Example implementation
    if units == "fahrenheit":
        temp = 72
    else:
        temp = 22
        
    return {
        "location": location,
        "temperature": temp,
        "units": units,
        "conditions": "Sunny"
    }


# Async function
@register_tool()
async def fetch_data(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Fetch data from a URL.
    
    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds
        
    Returns:
        The fetched data
    """
    print(f"[Tool Debug] fetch_data called with url={url}, timeout={timeout}")
    
    # Example implementation - just return fake data
    await asyncio.sleep(0.1)  # Simulate network request
    return {
        "url": url,
        "data": f"Data from {url}",
        "status": 200
    }


def create_program():
    """Create and configure the LLMProgram."""
    # Create a program with function tools using method chaining
    system_prompt = """You are a helpful assistant with access to the following tools:
    
1. add_numbers: Add two numbers together
2. search_documents: Search a database for documents matching a query
3. weather_info: Get weather information for a location
4. fetch_data: Fetch data from a URL
5. parse_and_analyze: Parse a JSON string and return an analysis

When using these tools:
- For add_numbers, provide numeric values for x and y
- For search_documents, you can specify a limit and categories (comma-separated list)
- For weather_info, you can specify units as "celsius" (default) or "fahrenheit"
- For parse_and_analyze, provide a valid JSON string
- For fetch_data, provide a valid URL

IMPORTANT: When the user asks you to perform calculations, search for documents, 
check the weather, or parse JSON, ALWAYS use the appropriate tool rather than trying 
to generate the answer yourself.
"""

    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet",  # Use claude-3.5-sonnet if you prefer
            provider="anthropic",
            system_prompt=system_prompt
        )
        .add_tool(add_numbers)
        .add_tool(search_documents)
        .add_tool(get_weather)
        .add_tool(fetch_data)
        .add_tool(parse_and_analyze)
        .compile()
    )
    
    return program


async def demo_tools(process):
    """Demonstrate direct tool calling."""
    # Print out the available tools
    print("Available tools:")
    for tool in process.tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    # Call the tools directly
    print("\nCalling tools directly:")
    
    # Test add_numbers with integer conversion
    calc_result = await process.call_tool("add_numbers", {"x": 5, "y": 7})
    print(f"\nadd_numbers result: {calc_result.content}")
    
    # Test parse_and_analyze with a string that needs to be parsed as JSON
    json_result = await process.call_tool("parse_and_analyze", {"json_data": '{"name": "John", "items": [1, 2, 3]}'})
    print(f"\nparse_and_analyze result: {json_result.content}")
    
    # Test search_documents with string to list conversion
    search_result = await process.call_tool(
        "search_documents", 
        {"query": "python", "limit": 3, "categories": ["programming", "tutorials"]}
    )
    print(f"\nsearch_documents result: {search_result.content}")
    
    # Test weather_info
    weather_result = await process.call_tool("weather_info", {"location": "New York"})
    print(f"\nweather_info result: {weather_result.content}")
    
    # Test fetch_data
    fetch_result = await process.call_tool("fetch_data", {"url": "https://example.com"})
    print(f"\nfetch_data result: {fetch_result.content}")


async def chat_interface(process):
    """Run an interactive chat interface with the model and function tools."""
    print("\n========== Function-based Tools Chat Interface ==========")
    print("Type 'exit' or 'quit' to end the session")
    print("Example queries to try:")
    print("  - What's 5 + 7?")
    print("  - Search for documents about machine learning")
    print("  - What's the weather in San Francisco?")
    print("  - Can you parse this JSON: {\"name\": \"John\", \"age\": 30}")
    print("========================================================\n")
    
    # Simple chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n> ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break
                
            # Process the input with the LLM
            print("\nProcessing...")
            await process.run(user_input)
            
            # Get and print the response
            response = process.get_last_message()
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\nExiting due to keyboard interrupt...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


async def main():
    """Run the example, either in demo or interactive mode."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Function-based tools example")
    parser.add_argument("--chat", action="store_true", help="Run in interactive chat mode")
    args = parser.parse_args()
    
    # Create and start the program
    program = create_program()
    process = await program.start()
    
    # Check for API key in interactive mode
    if args.chat and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in your environment or create a .env file with this value.")
        return
    
    if args.chat:
        # Run interactive chat mode
        await chat_interface(process)
    else:
        # Run demo mode
        await demo_tools(process)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to interrupt...")
        sys.exit(0)