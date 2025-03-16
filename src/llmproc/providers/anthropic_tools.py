"""Anthropic provider tools implementation for LLMProc."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


async def run_anthropic_with_tools(
    llm_process,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    max_iterations: int = 10,
) -> str:
    """Run Anthropic with tool support.

    Handles multiple iterations of tool calls and responses in an asynchronous context,
    processing each tool and providing the result back to the model.

    Args:
        llm_process: The LLMProcess instance
        system_prompt: The system prompt to use
        messages: The conversation messages
        max_iterations: Maximum number of tool-calling iterations

    Returns:
        The final model response as a string
    """
    # Extract needed values from the LLMProcess instance
    client = llm_process.client
    model_name = llm_process.model_name
    tools = llm_process.tools
    debug = getattr(llm_process, 'debug_tools', False)
    api_params = getattr(llm_process, 'api_params', {})
    tool_handlers = getattr(llm_process, 'tool_handlers', {})
    aggregator = getattr(llm_process, 'aggregator', None)

    # Track iterations to prevent infinite loops
    iterations = 0
    final_response = ""


    if debug:
        print("\n=== Starting Anthropic Tool Execution ===")
        print(f"Tools available: {len(tools)}")

    # Continue the conversation until no more tool calls or max iterations reached
    while iterations < max_iterations:
        iterations += 1
        if debug:
            print(f"\n--- Iteration {iterations}/{max_iterations} ---")

        try:
            # Prepare API parameters
            api_call_params = prepare_api_params(
                model_name, system_prompt, messages, tools, api_params
            )
            #print(f"API call params: {api_call_params}")
            import rich
            rich.print(f"{messages=}")
            rich.print(f"{tools=}")


            # Call Claude with current conversation
            try:
                if debug:
                    print(f"Calling Anthropic API with {len(messages)} messages...")
                response = await client.messages.create(**api_call_params)
                print(f"Response: {response}")
            except Exception as e:
                dump_file = dump_api_error(
                    e,
                    "anthropic",
                    model_name,
                    system_prompt,
                    messages,
                    api_params.get("temperature"),
                    api_params.get("max_tokens"),
                    tools,
                    iterations,
                )
                raise RuntimeError(
                    f"Error calling Anthropic API: {str(e)} (see {dump_file} for details)"
                )
            messages.append({"role": "assistant", "content": response.content})
            print(f"Response content: {response.content}")

            if response.stop_reason == "end_turn":
#                print("Model completed naturally")
                if response.content and len(response.content) > 0 and hasattr(response.content[0], "text"):
                    final_response = response.content[0].text
                else:
                    final_response = "model completed naturally but no response content text, try to ask again"
                return final_response
            elif response.stop_reason == "max_tokens":
                return response.content[0].text
            elif response.stop_reason == "stop_sequence":
                return response.content[0].text
            elif response.stop_reason == "tool_use":
                tool_results = await process_response_content(
                    response.content, aggregator, tool_handlers, debug
                )
                # Add tool results to conversation
                add_tool_results_to_conversation(messages, tool_results, debug)

        except Exception as e:
            if debug:
                print(f"ERROR in tool processing loop: {str(e)}")
            # If we have a final response, use it; otherwise return the error
            if not final_response:
                final_response = f"Error during tool execution: {str(e)}"
            break

    # If we've reached max iterations without a final text response, note that
    if iterations >= max_iterations and not final_response:
        final_response = "Maximum tool iterations reached without final response."

    if debug:
        print(
            f"=== Anthropic Tool Execution Complete ({iterations}/{max_iterations} iterations) ===\n"
        )

    return final_response


def prepare_api_params(
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    api_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare API parameters for Anthropic request.

    Args:
        model_name: The name of the model to use
        system_prompt: The system prompt
        messages: The conversation messages
        tools: The tools configuration
        api_params: Core API parameters (temperature, max_tokens, etc.)

    Returns:
        A dictionary of API parameters
    """
    return {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
        "tools": tools,
        **api_params,
    }


async def process_response_content(content_list, aggregator, tool_handlers: Dict[str, callable] = None, debug: bool = False):
    """Process the content from an Anthropic response.

    Args:
        content_list: The content from the response
        aggregator: The MCP aggregator
        tool_handlers: Dictionary mapping tool names to handler functions
        debug: Whether to enable debug output

    Returns:
        List of tool results
    """
    tool_results = []
    tool_calls = []

    # Initialize handlers if not provided
    tool_handlers = tool_handlers or {}

    # First, collect all content items
    for content in content_list:
        if content.type == "tool_use":
            # Store the tool call data for later processing
            tool_calls.append({
                "name": content.name,
                "args": content.input,
                "id": content.id
            })

    # If no tool calls, return early
    if not tool_calls:
        return tool_results

    # Now process all tool calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if debug:
            print(f"Calling tool {tool_name} with args {tool_args}")

        try:
            # Check if we have a custom handler for this tool
            print(f"Tool handlers: {tool_handlers}, tool_name: {tool_name}")
            if tool_name in tool_handlers:
                if debug:
                    print(f"Using custom handler for tool {tool_name}")
                # Call the custom handler with the arguments
                result = await tool_handlers[tool_name](tool_args)
            # Otherwise try to use the MCP aggregator
            elif aggregator is not None:
                if debug:
                    print(f"Using MCP aggregator for tool {tool_name}")
                # Execute the tool through the aggregator
                result = await aggregator.call_tool(tool_name, tool_args)
            else:
                # No handler found for this tool
                raise ValueError(f"No handler found for tool {tool_name}")

            # Process and format the result
            formatted_result = format_tool_result(result)

            if debug:
                print(f"Tool result: {str(formatted_result)[:150]}...")

            tool_results.append({
                "tool_use_id": tool_id,
                "content": str(formatted_result),
                "is_error": False,
            })
        except Exception as e:
            import sys
            import traceback

            error_msg = f"Error processing tool {tool_name}: {str(e)}"
            if debug:
                print(f"ERROR: {error_msg}", file=sys.stderr)
                print("Traceback:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            tool_results.append({
                "tool_use_id": tool_id,
                "content": error_msg,
                "is_error": True,
            })

    return tool_results


def add_tool_results_to_conversation(
    messages: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    debug: bool = False
):
    """Add tool results to the conversation as User messages.

    Args:
        messages: The conversation messages
        tool_results: The tool results to add
        debug: Whether to enable debug output
    """
    for result in tool_results:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_use_id"],
                        "content": result["content"],
                        **(
                            {"is_error": True} if result.get("is_error") else {}
                        ),
                    }
                ],
            }
        )

        if debug:
            print(f"Added result for tool_id: {result['tool_use_id']}")


def format_tool_result(result: Any) -> str:
    """Format tool result for Anthropic API consumption.

    Args:
        result: The raw result from the tool

    Returns:
        Formatted result as a string
    """
    import sys
    debug = False  # Only enable for deep debugging

    # Extract content based on result type
    content = None

    # Check if the result has specific attributes for errors
    is_error = False
    if hasattr(result, "is_error"):
        is_error = result.is_error
    elif isinstance(result, dict) and "is_error" in result:
        is_error = result["is_error"]

    # Check if the result has error message
    error_message = None
    if hasattr(result, "error"):
        error_message = result.error
    elif isinstance(result, dict) and "error" in result:
        error_message = result["error"]

    # Handle error case first
    if is_error and error_message:
        if debug:
            print(f"TOOL ERROR: {error_message}", file=sys.stderr)
        return f"ERROR: {error_message}"

    # Extract regular content
    if hasattr(result, "response"):
        content = result.response
    elif hasattr(result, "content"):
        content = result.content
    elif isinstance(result, dict) and "response" in result:
        content = result["response"]
    elif isinstance(result, dict) and "content" in result:
        content = result["content"]
    else:
        # If no specific content attribute, use the whole result
        content = result

    # Try to make it JSON serializable
    try:
        # This will validate if the content can be serialized to JSON
        json.dumps(content)
        return content
    except (TypeError, OverflowError, ValueError):
        # If it can't be serialized, convert to string
        return str(content)


def dump_api_error(
    error: Exception,
    provider: str,
    model_name: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    iteration: int = 0,
) -> str:
    """Dump API error details to a file for debugging.

    Args:
        error: The exception that occurred
        provider: The provider name
        model_name: The model name
        system_prompt: The system prompt
        messages: The messages sent to the API
        temperature: The temperature parameter
        max_tokens: The max_tokens parameter
        tools: Optional tools configuration
        iteration: Iteration number for tool calls

    Returns:
        Path to the dump file
    """
    # Create debug dump directory if it doesn't exist
    dump_dir = Path("debug_dumps")
    dump_dir.mkdir(exist_ok=True)

    # Dump message content to file for debugging
    dump_file = dump_dir / f"{provider}_api_error_{id(messages)}_{iteration}.json"

    # Truncate message content for readability
    truncated_messages = []
    for m in messages:
        if isinstance(m.get("content"), str):
            content = (
                m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"]
            )
        else:
            content = m["content"]  # Keep structured content as is
        truncated_messages.append({"role": m["role"], "content": content})

    with open(dump_file, "w") as f:
        json.dump(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "api_params": {
                    "model": model_name,
                    "system": system_prompt,
                    "messages": truncated_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools_count": len(tools) if tools else 0,
                    "tools": [{"name": tool["name"], "description": tool["description"]} for tool in tools] if tools else None,
                },
            },
            f,
            indent=2,
        )

    return str(dump_file)


