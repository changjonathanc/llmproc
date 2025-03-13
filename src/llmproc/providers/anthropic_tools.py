"""Anthropic provider tools implementation for LLMProc."""

import json
from pathlib import Path
from typing import Any


async def run_anthropic_with_tools(
    client,
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    aggregator,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    max_iterations: int = 10,
    debug: bool = False,
    **kwargs,
) -> str:
    """Run Anthropic with tool support.

    Handles multiple iterations of tool calls and responses in an asynchronous context,
    processing each tool and providing the result back to the model.

    Args:
        client: The Anthropic client
        model_name: The name of the Anthropic model to use
        system_prompt: The system prompt to use
        messages: The conversation messages
        tools: The list of available tools
        aggregator: The MCP aggregator for tool execution
        temperature: The temperature for generation
        max_tokens: The maximum number of tokens to generate
        max_iterations: Maximum number of tool-calling iterations
        debug: Whether to enable debug output
        **kwargs: Additional parameters to pass to the Anthropic API

    Returns:
        The final model response as a string
    """
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
            api_params = {
                "model": model_name,
                "system": system_prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_tokens", "debug_tools"]
                },
            }

            # Call Claude with current conversation
            try:
                if debug:
                    print(f"Calling Anthropic API with {len(messages)} messages...")
                response = await client.messages.create(**api_params)
            except Exception as e:
                dump_file = dump_api_error(
                    e,
                    "anthropic",
                    model_name,
                    system_prompt,
                    messages,
                    temperature,
                    max_tokens,
                    tools,
                    iterations,
                )
                raise RuntimeError(
                    f"Error calling Anthropic API: {str(e)} (see {dump_file} for details)"
                )

            # Process the response
            has_tool_calls = False
            tool_results = []
            response_text = ""

            # Process text content and tool calls
            for content in response.content:
                if content.type == "text":
                    response_text = content.text
                    final_response = response_text  # Save the text response
                elif content.type == "tool_use":
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_id = content.id

                    # Execute the tool
                    if debug:
                        print(f"Calling tool {tool_name} with args {tool_args}")
                    try:
                        result = await aggregator.call_tool(tool_name, tool_args)

                        # Process and format the result
                        formatted_result = format_tool_result(result)

                        if debug:
                            print(f"Tool result: {str(formatted_result)[:150]}...")
                        tool_results.append(
                            {
                                "tool_use_id": tool_id,
                                "content": str(formatted_result),
                                "is_error": False,
                            }
                        )
                    except Exception as e:
                        error_msg = f"Error processing tool {tool_name}: {str(e)}"
                        if debug:
                            print(f"ERROR: {error_msg}")
                        tool_results.append(
                            {
                                "tool_use_id": tool_id,
                                "content": error_msg,
                                "is_error": True,
                            }
                        )

            # If no tool calls were made, we're done
            if not has_tool_calls:
                if debug:
                    print(f"Final response (no tool calls): {final_response[:150]}...")
                return final_response

            # Add the assistant's response with tool calls to messages
            messages.append({"role": "assistant", "content": response.content})

            # Add tool results to conversation as User message
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
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
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
    messages: list[dict[str, Any]], debug: bool = False
) -> list[dict[str, Any]]:
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
