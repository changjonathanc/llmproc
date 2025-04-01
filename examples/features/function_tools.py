"""Example of using function-based tools with LLMProcess.

This example demonstrates how to:
1. Use the register_tool decorator to register functions as tools
2. Register both sync and async functions as tools
3. Use method chaining with the fluent API
4. Use tool schemas generated from function type hints and docstrings
"""

import asyncio
from typing import Dict, List, Any, Optional

from llmproc import LLMProgram, register_tool


# Simple calculator function with type hints
def add_numbers(x: int, y: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        The sum of x and y
    """
    return x + y


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
    # Example implementation
    await asyncio.sleep(0.1)  # Simulate network request
    return {
        "url": url,
        "data": f"Data from {url}",
        "status": 200
    }


async def main():
    """Run a simple example demonstrating function tools."""
    # Create a program with function tools using method chaining
    program = (
        LLMProgram(
            model_name="claude-3-7-sonnet",
            provider="anthropic",
            system_prompt="You are a helpful assistant with various tools."
        )
        .add_tool(add_numbers)
        .add_tool(search_documents)
        .add_tool(get_weather)
        .add_tool(fetch_data)
        .compile()
    )
    
    # Start the LLM process
    process = await program.start()
    
    # Print out the available tools
    print("Available tools:")
    for tool in process.tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    # Call the tools directly
    print("\nCalling tools directly:")
    
    calc_result = await process.call_tool("add_numbers", {"x": 5, "y": 7})
    print(f"add_numbers result: {calc_result.content}")
    
    search_result = await process.call_tool("search_documents", {"query": "python"})
    print(f"search_documents result: {search_result.content}")
    
    weather_result = await process.call_tool("weather_info", {"location": "New York"})
    print(f"weather_info result: {weather_result.content}")
    
    fetch_result = await process.call_tool("fetch_data", {"url": "https://example.com"})
    print(f"fetch_data result: {fetch_result.content}")
    
    # Run the LLM process with a query (uncomment if you have API keys set up)
    # result = await process.run("What's the weather in New York?")
    # print(f"\nModel response: {process.get_last_message()}")


if __name__ == "__main__":
    asyncio.run(main())