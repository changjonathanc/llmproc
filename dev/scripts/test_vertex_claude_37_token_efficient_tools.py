#!/usr/bin/env python
"""
Test script to determine if token-efficient tools beta feature works on Vertex AI with Claude 3.7.

This test sends requests to the Vertex AI API with the token-efficient-tools beta header
and measures token usage to see if the feature is supported.
"""

import os
import json
import time
import asyncio

try:
    from anthropic import AsyncAnthropicVertex
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

async def test_token_efficient_tools():
    """Test if token-efficient tools header works with Vertex AI."""
    print("\n=== Testing token-efficient tools with Claude 3.7 Sonnet on Vertex AI ===")
    
    if not VERTEX_AVAILABLE:
        print("Anthropic Vertex SDK not installed. Skipping test.")
        return
    
    try:
        # Get environment variables
        project_id = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
        region = os.environ.get("CLOUD_ML_REGION", "us-central1")
        
        # Initialize the Vertex AI client
        client = AsyncAnthropicVertex(
            project_id=project_id,
            region=region
        )
    
        print(f"Testing token-efficient tools on Vertex AI Claude 3.7 (project: {project_id}, region: {region})")
        
        # Define a simple calculator tool
        calculator_tool = {
            "name": "calculator",
            "description": "Use this tool to perform calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
        
        # First test WITHOUT token-efficient tools header
        print("\n--- Test WITHOUT token-efficient tools header ---")
        response_standard = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the square root of 256? Please use the calculator tool."}
            ],
            tools=[calculator_tool],
            system="You are a helpful AI assistant that uses tools when appropriate."
        )
        
        print("Standard response tokens:")
        print(f"Input tokens: {response_standard.usage.input_tokens}")
        print(f"Output tokens: {response_standard.usage.output_tokens}")
        print(f"Content types: {[c.type for c in response_standard.content]}")
        
        # Tool response for later
        tool_response = None
        if any(c.type == "tool_use" for c in response_standard.content):
            for content in response_standard.content:
                if content.type == "tool_use":
                    tool_response = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": "16.0"
                            }
                        ]
                    }
        
        # Wait a bit to avoid rate limits
        time.sleep(2)
        
        # Now test WITH token-efficient tools header
        print("\n--- Test WITH token-efficient tools header ---")
        response_efficient = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the square root of 256? Please use the calculator tool."}
            ],
            tools=[calculator_tool],
            system="You are a helpful AI assistant that uses tools when appropriate.",
            extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"}
        )
        
        print("With token-efficient tools header tokens:")
        print(f"Input tokens: {response_efficient.usage.input_tokens}")
        print(f"Output tokens: {response_efficient.usage.output_tokens}")
        print(f"Content types: {[c.type for c in response_efficient.content]}")
        
        # Compare token usage
        output_tokens_standard = response_standard.usage.output_tokens
        output_tokens_efficient = response_efficient.usage.output_tokens
        
        difference = output_tokens_standard - output_tokens_efficient
        percent_reduction = (difference / output_tokens_standard) * 100 if output_tokens_standard > 0 else 0
        
        print("\n--- Results ---")
        print(f"Standard output tokens: {output_tokens_standard}")
        print(f"With header output tokens: {output_tokens_efficient}")
        print(f"Difference: {difference} tokens ({percent_reduction:.2f}% reduction)")
        
        # Complete the conversation to see token usage in full exchange
        if tool_response:
            # Complete standard conversation
            print("\n--- Completing standard conversation ---")
            response_standard_complete = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "What is the square root of 256? Please use the calculator tool."},
                    {"role": "assistant", "content": response_standard.content},
                    tool_response
                ],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate."
            )
            
            # Complete token-efficient conversation
            print("\n--- Completing token-efficient conversation ---")
            response_efficient_complete = await client.messages.create(
                model="claude-3-7-sonnet@20250219",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "What is the square root of 256? Please use the calculator tool."},
                    {"role": "assistant", "content": response_efficient.content},
                    tool_response
                ],
                tools=[calculator_tool],
                system="You are a helpful AI assistant that uses tools when appropriate.",
                extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"}
            )
            
            # Compare complete conversation token usage
            print("\n--- Final Results (Complete Conversation) ---")
            print(f"Standard complete output tokens: {response_standard_complete.usage.output_tokens}")
            print(f"With header complete output tokens: {response_efficient_complete.usage.output_tokens}")
            
            difference_complete = response_standard_complete.usage.output_tokens - response_efficient_complete.usage.output_tokens
            percent_reduction_complete = (difference_complete / response_standard_complete.usage.output_tokens) * 100 if response_standard_complete.usage.output_tokens > 0 else 0
            
            print(f"Difference: {difference_complete} tokens ({percent_reduction_complete:.2f}% reduction)")
            
            # If there's a significant reduction, it suggests the feature is supported
            if percent_reduction_complete > 5:
                print("\nCONCLUSION: Token-efficient tools appears to be SUPPORTED on Vertex AI")
            else:
                print("\nCONCLUSION: Token-efficient tools appears to be NOT SUPPORTED on Vertex AI")
    except Exception as e:
        print(f"Error testing Vertex AI: {e}")

if __name__ == "__main__":
    asyncio.run(test_token_efficient_tools())