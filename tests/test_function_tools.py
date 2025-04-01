"""Tests for function-based tool registration."""

import pytest
import asyncio
from typing import List, Dict, Any, Optional

from llmproc import LLMProgram, register_tool
from llmproc.tools import ToolResult
from llmproc.tools.function_tools import (
    function_to_tool_schema,
    extract_docstring_params,
    type_to_json_schema,
    prepare_tool_handler,
    create_tool_from_function
)


# Simple function with type hints
def get_calculator(x: int, y: int) -> int:
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
    # Dummy implementation
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
    # Dummy implementation
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
    # Dummy implementation
    await asyncio.sleep(0.1)  # Simulate network request
    return {
        "url": url,
        "data": f"Data from {url}",
        "status": 200
    }


def test_extract_docstring_params():
    """Test extracting parameter information from docstrings."""
    # Extract params from the calculator function
    params = extract_docstring_params(get_calculator)
    
    # Check that we extracted the expected parameters
    assert "x" in params
    assert "y" in params
    assert "return" in params
    
    # Check parameter descriptions
    assert "First number" in params["x"]["description"]
    assert "Second number" in params["y"]["description"]
    assert "sum of x and y" in params["return"]["description"]


def test_type_to_json_schema():
    """Test converting Python types to JSON schema."""
    # Check basic types
    int_schema = type_to_json_schema(int, "x", {})
    assert int_schema["type"] == "integer"
    
    str_schema = type_to_json_schema(str, "x", {})
    assert str_schema["type"] == "string"
    
    # Check complex types
    list_schema = type_to_json_schema(List[str], "items", {})
    assert list_schema["type"] == "array"
    assert "items" in list_schema
    assert list_schema["items"]["type"] == "string"
    
    # Check optional types
    optional_schema = type_to_json_schema(Optional[int], "count", {})
    assert optional_schema["type"] == "integer"


def test_function_to_tool_schema():
    """Test converting a function to a tool schema."""
    # Convert the calculator function to a tool schema
    schema = function_to_tool_schema(get_calculator)
    
    # Check the schema structure
    assert schema["name"] == "get_calculator"
    assert "Calculate the sum" in schema["description"]
    assert "properties" in schema["input_schema"]
    assert "x" in schema["input_schema"]["properties"]
    assert "y" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["x"]["type"] == "integer"
    assert schema["input_schema"]["properties"]["y"]["type"] == "integer"
    assert "required" in schema["input_schema"]
    assert "x" in schema["input_schema"]["required"]
    assert "y" in schema["input_schema"]["required"]


def test_function_with_custom_name():
    """Test a function with custom name and description via decorator."""
    # Convert the weather function to a tool schema
    schema = function_to_tool_schema(get_weather)
    
    # Check that the decorator attributes were properly applied
    assert schema["name"] == "weather_info"  # Custom name from decorator
    assert "Get weather information" in schema["description"]  # Custom description
    assert "location" in schema["input_schema"]["properties"]
    assert "units" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["units"]["type"] == "string"
    assert "location" in schema["input_schema"]["required"]
    assert "units" not in schema["input_schema"]["required"]  # Has default value


@pytest.mark.asyncio
async def test_prepare_tool_handler():
    """Test the tool handler preparation for both sync and async functions."""
    # Test synchronous function
    calc_handler = prepare_tool_handler(get_calculator)
    calc_result = await calc_handler({"x": 5, "y": 7})
    assert isinstance(calc_result, ToolResult)
    assert calc_result.is_error is False
    assert calc_result.content == 12
    
    # Test asynchronous function
    async_handler = prepare_tool_handler(fetch_data)
    async_result = await async_handler({"url": "https://example.com"})
    assert isinstance(async_result, ToolResult)
    assert async_result.is_error is False
    assert async_result.content["url"] == "https://example.com"
    assert async_result.content["status"] == 200
    
    # Test error handling - missing required parameter
    error_result = await calc_handler({"x": 5})  # Missing y
    assert error_result.is_error is True
    assert "Missing required parameter" in error_result.content


def test_create_tool_from_function():
    """Test creating a complete tool from a function."""
    # Create a tool from the search function
    handler, schema = create_tool_from_function(search_documents)
    
    # Check the schema
    assert schema["name"] == "search_documents"
    assert "Search documents by query" in schema["description"]
    assert "query" in schema["input_schema"]["properties"]
    assert "limit" in schema["input_schema"]["properties"]
    assert "categories" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["properties"]["limit"]["type"] == "integer"
    
    # Check required parameters
    assert "query" in schema["input_schema"]["required"]
    assert "limit" not in schema["input_schema"]["required"]  # Has default
    assert "categories" not in schema["input_schema"]["required"]  # Has default


def test_program_with_function_tools():
    """Test adding function-based tools to a program."""
    # Create a program with function tools
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant with tools.",
        tools=[get_calculator, search_documents]
    )
    
    # Check that tools were added
    assert hasattr(program, "_function_tools")
    assert len(program._function_tools) == 2
    
    # Compile the program to process the function tools
    program.compile()
    
    # Check that function tools were processed
    assert hasattr(program.tool_manager, "tool_handlers")
    assert len(program.tool_manager.tool_handlers) >= 2
    assert "get_calculator" in program.tool_manager.tool_handlers
    assert "search_documents" in program.tool_manager.tool_handlers
    
    # Check that tools were added to enabled tools
    assert "enabled" in program.tools
    assert "get_calculator" in program.tools["enabled"]
    assert "search_documents" in program.tools["enabled"]


def test_add_tool_method():
    """Test adding tools using the add_tool method."""
    # Create a program with no tools
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    
    # Add a function tool
    program.add_tool(get_weather)
    
    # Add a function tool with method chaining
    program.add_tool(get_calculator).add_tool(search_documents)
    
    # Compile the program
    program.compile()
    
    # Check that all tools were processed
    assert len(program.tool_manager.tool_handlers) >= 3
    assert "weather_info" in program.tool_manager.tool_handlers  # Custom name from decorator
    assert "get_calculator" in program.tool_manager.tool_handlers
    assert "search_documents" in program.tool_manager.tool_handlers
    
    # Check enabled tools list
    assert len(program.tools["enabled"]) == 3
    assert "weather_info" in program.tools["enabled"]
    assert "get_calculator" in program.tools["enabled"]
    assert "search_documents" in program.tools["enabled"]
    
    
def test_set_enabled_tools_with_function_tools():
    """Test interaction between set_enabled_tools and function tools."""
    # Create a program with function tools
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant."
    )
    
    # Add function tools
    program.add_tool(get_weather)
    program.add_tool(get_calculator)
    
    # Should have function tools enabled
    program.compile()
    assert "weather_info" in program.tools["enabled"]
    assert "get_calculator" in program.tools["enabled"]
    
    # Set built-in tools - should replace function tools
    program.set_enabled_tools(["calculator", "read_file"])
    
    # Check that built-in tools replaced function tools
    assert "calculator" in program.tools["enabled"]
    assert "read_file" in program.tools["enabled"]
    assert "weather_info" not in program.tools["enabled"]
    assert "get_calculator" not in program.tools["enabled"]
    
    # Function tools should still be registered but not enabled
    assert "weather_info" in program.tool_manager.tool_schemas
    assert "get_calculator" in program.tool_manager.tool_schemas


@pytest.mark.asyncio
async def test_function_tool_execution():
    """Test executing a function-based tool through a process."""
    # Create a program with function tools
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant with tools.",
        tools=[get_calculator]
    )
    
    # Compile and start the process
    process = await program.compile().start()
    
    # Check that the tool is registered in the process
    tool_defs = process.tools
    assert any(tool["name"] == "get_calculator" for tool in tool_defs)
    
    # Call the tool directly through the process
    result = await process.call_tool("get_calculator", {"x": 10, "y": 15})
    
    # Check result
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == 25