"""Utility functions for the Gemini provider."""

try:
    from google import genai
except ImportError:
    genai = None


def convert_tools_to_gemini_format(tools):
    """Converts tools to the Gemini format.

    Args:
        tools: List of tool definitions in internal format

    Returns:
        List of function declarations in Gemini format, or None if no tools
    """
    if not tools:
        return None

    function_declarations = []
    for tool in tools:
        # Convert tool schema to Gemini FunctionDeclaration format
        function_declaration = {
            "name": tool["name"],
            "description": tool["description"],
        }

        # Add parameters if they exist
        if "parameters" in tool and tool["parameters"]:
            function_declaration["parameters"] = tool["parameters"]

        function_declarations.append(function_declaration)

    return [{"function_declarations": function_declarations}]


def format_tool_result_for_gemini(result, tool_call_name):
    """Formats a tool result for the Gemini API.

    Args:
        result: ToolResult object containing the execution result
        tool_call_name: Name of the tool that was called

    Returns:
        Dictionary in Gemini FunctionResponse format
    """
    if genai is None:
        # Fallback format when SDK not available
        return {
            "name": tool_call_name,
            "response": result.content if not result.is_error else f"ERROR: {result.content}",
        }

    # Use Gemini SDK's FunctionResponse format
    return genai.types.FunctionResponse(
        name=tool_call_name,
        response=result.content if not result.is_error else f"ERROR: {result.content}",
    )
