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

            # Call Claude with current conversation
            try:
                if debug:
                    print(f"Calling Anthropic API with {len(messages)} messages...")
                response = await client.messages.create(**api_call_params)
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

            # Process the response
            has_tool_calls, tool_results, response_text = await process_response_content(
                response.content, aggregator, tool_handlers, debug
            )
            
            # Store any text response
            if response_text:
                final_response = response_text

            # If no tool calls were made, we're done
            if not has_tool_calls:
                if debug:
                    print(f"Final response (no tool calls): {final_response[:150]}...")
                return final_response

            # Add the assistant's response with tool calls to messages
            messages.append({"role": "assistant", "content": response.content})

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
        Tuple of (has_tool_calls, tool_results, response_text)
    """
    has_tool_calls = False
    tool_results = []
    response_text = ""
    tool_calls = []
    
    # Initialize handlers if not provided
    tool_handlers = tool_handlers or {}
    
    # First, collect all content items
    for content in content_list:
        if content.type == "text":
            response_text = content.text
        elif content.type == "tool_use":
            has_tool_calls = True
            # Store the tool call data for later processing
            tool_calls.append({
                "name": content.name,
                "args": content.input,
                "id": content.id
            })
    
    # If no tool calls, return early
    if not has_tool_calls:
        return has_tool_calls, tool_results, response_text
    
    # Now process all tool calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if debug:
            print(f"Calling tool {tool_name} with args {tool_args}")
        
        try:
            # Check if we have a custom handler for this tool
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
            error_msg = f"Error processing tool {tool_name}: {str(e)}"
            if debug:
                print(f"ERROR: {error_msg}")
                
            tool_results.append({
                "tool_use_id": tool_id,
                "content": error_msg,
                "is_error": True,
            })
            
    return has_tool_calls, tool_results, response_text


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
    # Extract content based on result type
    content = None

    # Check if the result has content attribute
    if hasattr(result, "content"):
        content = result.content
    else:
        # If no content attribute, use the whole result
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
                },
            },
            f,
            indent=2,
        )

    return str(dump_file)


def filter_empty_text_blocks(
    messages: List[Dict[str, Any]], debug: bool = False
) -> List[Dict[str, Any]]:
    """Filter out empty text blocks from structured content.

    Args:
        messages: The messages to filter
        debug: Whether to print debug information

    Returns:
        Filtered messages
    """
    filtered_messages = []

    for msg in messages:
        if msg["role"] == "system":
            filtered_messages.append(msg)
            continue

        # Handle potential structured content (for tools)
        if isinstance(msg["content"], list):
            # Filter out empty text blocks
            empty_blocks = [
                i
                for i, block in enumerate(msg["content"])
                if block.get("type") == "text" and not block.get("text")
            ]

            if empty_blocks:
                # Create a safe copy without empty text blocks
                filtered_content = [
                    block
                    for block in msg["content"]
                    if not (block.get("type") == "text" and not block.get("text"))
                ]
                msg_copy = msg.copy()
                msg_copy["content"] = filtered_content
                filtered_messages.append(msg_copy)

                if debug:
                    print(
                        f"WARNING: Filtered out {len(empty_blocks)} empty text blocks from message with role {msg['role']}"
                    )
            else:
                filtered_messages.append(msg)
        else:
            filtered_messages.append(msg)

    return filtered_messages
