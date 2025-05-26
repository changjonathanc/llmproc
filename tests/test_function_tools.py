"""Tests for function-based tool registration."""

import asyncio
import inspect
import logging
from typing import Any, Optional

import pytest
from llmproc import LLMProgram, register_tool
from llmproc.tools import ToolResult
from llmproc.tools.function_tools import (
    create_tool_from_function,
    extract_docstring_params,
    function_to_tool_schema,
    prepare_tool_handler,
    type_to_json_schema,
)

# Set up debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up logging for tests
logger.info("Running function tools tests")


@pytest.fixture(autouse=True)
def patch_process_for_tests():
    """Patch LLMProcess.call_tool for more stable testing.

    This ensures tests can still use both styles of parameters during the transition.
    """
    import llmproc.llm_process

    original_call_tool = llmproc.llm_process.LLMProcess.call_tool

    # Define patched version that supports both styles consistently
    async def patched_call_tool(self, tool_name, args=None, **kwargs):
        if args is not None and isinstance(args, dict):
            # Using original style: process.call_tool("tool", {"param": "value"})
            handler = self.tool_handlers.get(tool_name)
            if not handler:
                return ToolResult.from_error(
                    f"Tool '{tool_name}' not found. Available: {list(self.tool_handlers.keys())}"
                )
            try:
                # Extract params from the dict and pass as kwargs
                return await handler(**args)
            except Exception as e:
                return ToolResult.from_error(f"Error in tool '{tool_name}': {str(e)}")
        else:
            # Using new style: process.call_tool("tool", param="value")
            handler = self.tool_handlers.get(tool_name)
            if not handler:
                return ToolResult.from_error(
                    f"Tool '{tool_name}' not found. Available: {list(self.tool_handlers.keys())}"
                )
            try:
                return await handler(**kwargs)
            except Exception as e:
                return ToolResult.from_error(f"Error in tool '{tool_name}': {str(e)}")

    # Apply the patch just for this test
    llmproc.llm_process.LLMProcess.call_tool = patched_call_tool

    # Run the test, then restore original implementation
    yield
    llmproc.llm_process.LLMProcess.call_tool = original_call_tool


@pytest.fixture(autouse=True)
def patch_initialize_client():
    """Patch client initialization to avoid API key requirements."""
    from unittest.mock import MagicMock, patch

    with patch("llmproc.program_exec.initialize_client", return_value=MagicMock()):
        yield


@pytest.fixture
def basic_program():
    """Create a basic program for testing."""
    program = LLMProgram(
        model_name="claude-3-7-sonnet",
        provider="anthropic",
        system_prompt="You are a helpful assistant.",
    )
    return program


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
def search_documents(query: str, limit: int = 5, categories: list[str] | None = None) -> list[dict[str, Any]]:
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
def get_weather(location: str, units: str = "celsius") -> dict[str, Any]:
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
        "conditions": "Sunny",
    }


# Async function
@register_tool()
async def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch data from a URL.

    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds

    Returns:
        The fetched data
    """
    # Dummy implementation
    await asyncio.sleep(0.1)  # Simulate network request
    return {"url": url, "data": f"Data from {url}", "status": 200}


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
    list_schema = type_to_json_schema(list[str], "items", {})
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
    calc_result = await calc_handler(x=5, y=7)
    assert isinstance(calc_result, ToolResult)
    assert calc_result.is_error is False
    assert calc_result.content == 12

    # Test asynchronous function
    async_handler = prepare_tool_handler(fetch_data)
    async_result = await async_handler(url="https://example.com")
    assert isinstance(async_result, ToolResult)
    assert async_result.is_error is False
    assert async_result.content["url"] == "https://example.com"
    assert async_result.content["status"] == 200

    # Test error handling - missing required parameter
    error_result = await calc_handler(x=5)  # Missing y
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
    """Test adding and using function-based tools in a program."""
    # Create a program
    program = LLMProgram(
        model_name="claude-3-5-haiku-latest",
        provider="anthropic",
        system_prompt="You are a helpful assistant with tools.",
    )

    # Set enabled tools with function tools
    program.register_tools([get_calculator, search_documents])

    # Check that tools were added to the function_tools list in the tool_manager
    assert len(program.tool_manager.function_tools) == 2

    # Verify the tools are the ones we specified
    function_tools = program.tool_manager.function_tools
    assert any(func is get_calculator for func in function_tools)
    assert any(func is search_documents for func in function_tools)

    # Function tools are processed when the process is started

    # Process function tools to register handlers and schemas
    program.tool_manager.process_function_tools()

    # Verify tools appear in the API-ready schema
    tool_schemas = program.tool_manager.get_tool_schemas()
    tool_names = [schema["name"] for schema in tool_schemas]
    assert "get_calculator" in tool_names
    assert "search_documents" in tool_names

    # Verify tools are registered (using the proper method for single source of truth)
    registered_tools = program.get_registered_tools()
    assert "get_calculator" in registered_tools
    assert "search_documents" in registered_tools

    # At this point, the tools are properly registered and can be called
    # This is tested in test_function_tool_execution


@pytest.mark.asyncio
async def test_tool_enabling_methods(basic_program):
    """Test registering tools and verifying they work correctly."""
    # Use the basic program fixture
    program = basic_program

    # Register tools
    program.register_tools([get_weather])
    # Use only function references for tools - directly use the functions
    # This ensures we never pass string literals to register_tools
    from llmproc.tools.builtin import BUILTIN_TOOLS

    # Set directly with function references
    program.register_tools([get_weather, get_calculator, search_documents])

    # Verify expected tools are in the function_tools list (converted from callables)
    function_names = [func.__name__ for func in program.tool_manager.function_tools]

    # The decorated function get_weather is renamed to "weather_info" but its __name__ is still "get_weather"
    assert "get_weather" in function_names
    assert "get_calculator" in function_names
    assert "search_documents" in function_names

    # Process function tools to register handlers and schemas
    program.tool_manager.process_function_tools()

    # Verify custom name from decorator works
    tool_schemas = program.tool_manager.get_tool_schemas()
    has_weather_tool = any(schema["name"] == "weather_info" for schema in tool_schemas)
    assert has_weather_tool

    # Verify tool schemas have correct structure
    calculator_schema = None
    for schema in tool_schemas:
        if schema["name"] == "get_calculator":
            calculator_schema = schema
            break

    assert calculator_schema is not None
    assert "input_schema" in calculator_schema
    assert "properties" in calculator_schema["input_schema"]
    assert "x" in calculator_schema["input_schema"]["properties"]
    assert "y" in calculator_schema["input_schema"]["properties"]

    # Verify weather tool schema has custom name from decorator
    weather_schema = None
    for schema in tool_schemas:
        if schema["name"] == "weather_info":
            weather_schema = schema
            break

    assert weather_schema is not None
    assert "Get weather information" in weather_schema["description"]

    # Create a process to test actual tool execution
    process = await program.start()

    # Check if the tool is registered as expected
    assert "weather_info" in process.tool_handlers, "weather_info tool is not registered in handlers"

    # Test weather tool with explicit parameters
    weather_result = await process.call_tool("weather_info", {"location": "New York"})
    assert isinstance(weather_result, ToolResult)
    assert not weather_result.is_error, f"Error in weather tool: {weather_result.content}"
    assert "location" in weather_result.content
    assert "New York" in str(weather_result.content)

    # Test calculator tool
    calc_result = await process.call_tool("get_calculator", {"x": 5, "y": 9})
    assert not calc_result.is_error
    assert calc_result.content == 14  # Note: this is x + y, not expression

    # Test search tool
    search_result = await process.call_tool("search_documents", {"query": "example", "limit": 2})
    assert not search_result.is_error
    assert len(search_result.content) == 2  # Should return 2 results based on limit


@pytest.mark.asyncio
async def test_register_tools_with_function_tools(basic_program, create_program):
    """Test interaction between register_tools and function tools by verifying behavior."""
    # Use the basic program fixture
    program = basic_program

    # Register function tools
    program.register_tools([get_weather, get_calculator])

    # Process function tools to register them in the registry
    program.tool_manager.process_function_tools()

    # Verify function tools are registered in function_tools list
    # Function tools are processed during program.compile(), so we need to
    # match the weather_info tool name against the metadata
    tool_names = []
    for func in program.tool_manager.function_tools:
        from llmproc.common.metadata import get_tool_meta

        meta = get_tool_meta(func)
        if meta.name:
            tool_names.append(meta.name)
        else:
            tool_names.append(func.__name__)

    # Make sure 'weather_info' is present in the list of tool names
    assert "weather_info" in tool_names
    # Check the actual function name is also present as a fallback
    function_names = [func.__name__ for func in program.tool_manager.function_tools]
    assert "get_calculator" in function_names

    # Start the process to test actual tool execution
    process = await program.start()

    # Verify function tools work by calling them
    weather_result = await process.call_tool("weather_info", {"location": "London"})
    assert not weather_result.is_error
    assert "London" in str(weather_result.content)

    # Create a new program without adding function tools first
    program2 = create_program()

    # Register specific tools using function references
    from llmproc.tools.builtin import calculator, read_file

    program2.register_tools([calculator, read_file])

    # Process the tools to register them
    program2.tool_manager.process_function_tools()

    # Verify that built-in tools are registered in configuration
    registered_tools = program2.get_registered_tools()
    assert "calculator" in registered_tools
    assert "read_file" in registered_tools

    # Note: Function tools remain in the enabled_tools list due to how function tools are processed
    # But their behavior should be correctly updated based on the API's schema list

    # Start the process to test tool execution
    process2 = await program2.start()

    # Check what tools are actually available to the LLM API
    process_tools = process2.tools
    tool_names = [tool["name"] for tool in process_tools]

    # Verify that only the enabled built-in tools are included in the API schema
    assert "calculator" in tool_names
    assert "read_file" in tool_names

    # Verify that disabled tools return error results when called
    weather_result = await process2.call_tool("weather_info", location="Paris")
    assert weather_result.is_error
    # With our patched version, the error message is different
    assert "not found" in weather_result.content.lower() or "not enabled" in weather_result.content.lower()

    # Verify that enabled built-in calculator tool works
    calc_result = await process2.call_tool("calculator", expression="7+3")
    assert not calc_result.is_error
    assert calc_result.content == "10"

    # Test a completely different approach - start with weather tool only
    weather_program = create_program()
    weather_program.register_tools([get_weather])

    # Create a process with just the weather tool
    weather_process = await weather_program.start()

    # Verify weather tool works
    weather_result = await weather_process.call_tool("weather_info", {"location": "Tokyo"})
    assert not weather_result.is_error
    assert "Tokyo" in str(weather_result.content)

    # Verify calculator tool doesn't work because it was never added
    calc_result = await weather_process.call_tool("get_calculator", {"x": 1, "y": 2})
    assert calc_result.is_error
    # With our patched version, the error message is different
    assert "not found" in calc_result.content.lower() or "not enabled" in calc_result.content.lower()

    # Verify the tool schemas that are actually available to the LLM API
    process_tools = weather_process.tools
    tool_names = [tool["name"] for tool in process_tools]
    assert "weather_info" in tool_names
    assert "get_calculator" not in tool_names


@pytest.mark.asyncio
async def test_function_tool_execution(create_program):
    """Test executing a function-based tool through a process."""
    # Create a program
    program = create_program(
        system_prompt="You are a helpful assistant with tools.",
    )

    # Set enabled tools
    program.register_tools([get_calculator])

    # Start the process
    process = await program.start()

    # Check that the tool is registered in the process
    tool_defs = process.tools
    assert any(tool["name"] == "get_calculator" for tool in tool_defs)

    # Call the tool directly through the process
    result = await process.call_tool("get_calculator", {"x": 10, "y": 15})

    # Check result
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == 25
