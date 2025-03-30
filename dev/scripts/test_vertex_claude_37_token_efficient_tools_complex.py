#!/usr/bin/env python
"""
Test script with more complex tools to determine if token-efficient tools beta feature 
works on Vertex AI with Claude 3.7.

This test uses multiple complex tools which would typically see larger benefits from
token-efficient tools optimization.
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

async def test_token_efficient_tools_complex():
    """Test if token-efficient tools header works with Vertex AI using complex tools."""
    print("\n=== Testing token-efficient tools with complex tools on Vertex AI Claude 3.7 ===")
    
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
    
        print(f"Testing token-efficient tools with complex tools on Vertex AI Claude 3.7")
        print(f"Project: {project_id}, Region: {region}")
        
        # Define multiple complex tools
        tools = [
            {
                "name": "calculator",
                "description": "Use this tool to perform complex calculations",
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
            },
            {
                "name": "weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for (city, state, country)"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        },
                        "include_forecast": {
                            "type": "boolean",
                            "description": "Whether to include a 5-day forecast"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "database_query",
                "description": "Query a customer database for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["customer_lookup", "order_history", "product_inventory", "sales_analytics"],
                            "description": "The type of query to perform"
                        },
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID for lookups"
                        },
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "description": "Start date in YYYY-MM-DD format"
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "End date in YYYY-MM-DD format"
                                }
                            }
                        },
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "operator": {"type": "string"},
                                    "value": {"type": "string"}
                                }
                            },
                            "description": "Additional filters to apply to the query"
                        }
                    },
                    "required": ["query_type"]
                }
            }
        ]
        
        # Complex prompt that should trigger more verbose tool use
        prompt = """
        I need help with several tasks:
        
        1. Calculate the compound interest on an investment of $10,000 with an annual interest rate of 7% compounded monthly for 10 years.
        
        2. What's the weather like in New York City? Please include a forecast if possible, and show temperatures in both Celsius and Fahrenheit.
        
        3. I need to query our database to find all premium customers (customer_id starting with 'P') who have made purchases over $1000 in the last quarter (2025-01-01 to 2025-03-31). Please construct the appropriate database query.
        
        Please use the appropriate tools for each of these tasks.
        """
        
        # First test WITHOUT token-efficient tools header
        print("\n--- Test WITHOUT token-efficient tools header ---")
        response_standard = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            system="You are a helpful AI assistant that uses tools effectively when appropriate."
        )
        
        print("Standard response tokens:")
        print(f"Input tokens: {response_standard.usage.input_tokens}")
        print(f"Output tokens: {response_standard.usage.output_tokens}")
        print(f"Content types: {[c.type for c in response_standard.content]}")
        
        # Extract tool calls for logging
        tool_calls_standard = []
        for content in response_standard.content:
            if hasattr(content, 'type') and content.type == "tool_use":
                tool_calls_standard.append({
                    "name": content.name,
                    "input": content.input
                })
        
        print(f"Tool calls (standard): {json.dumps(tool_calls_standard, indent=2)}")
        
        # Wait a bit to avoid rate limits
        time.sleep(2)
        
        # Now test WITH token-efficient tools header
        print("\n--- Test WITH token-efficient tools header ---")
        response_efficient = await client.messages.create(
            model="claude-3-7-sonnet@20250219",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            system="You are a helpful AI assistant that uses tools effectively when appropriate.",
            extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"}
        )
        
        print("With token-efficient tools header tokens:")
        print(f"Input tokens: {response_efficient.usage.input_tokens}")
        print(f"Output tokens: {response_efficient.usage.output_tokens}")
        print(f"Content types: {[c.type for c in response_efficient.content]}")
        
        # Extract tool calls for logging
        tool_calls_efficient = []
        for content in response_efficient.content:
            if hasattr(content, 'type') and content.type == "tool_use":
                tool_calls_efficient.append({
                    "name": content.name,
                    "input": content.input
                })
        
        print(f"Tool calls (efficient): {json.dumps(tool_calls_efficient, indent=2)}")
        
        # Compare token usage
        output_tokens_standard = response_standard.usage.output_tokens
        output_tokens_efficient = response_efficient.usage.output_tokens
        
        difference = output_tokens_standard - output_tokens_efficient
        percent_reduction = (difference / output_tokens_standard) * 100 if output_tokens_standard > 0 else 0
        
        print("\n--- Results ---")
        print(f"Standard output tokens: {output_tokens_standard}")
        print(f"With header output tokens: {output_tokens_efficient}")
        print(f"Difference: {difference} tokens ({percent_reduction:.2f}% reduction)")
        
        # Determine if token-efficient tools is likely supported
        if percent_reduction > 5:
            print("\nCONCLUSION: Token-efficient tools appears to be SUPPORTED on Vertex AI")
            print(f"Observed a {percent_reduction:.2f}% reduction in output tokens with the header")
        else:
            print("\nCONCLUSION: Token-efficient tools appears to be NOT SUPPORTED on Vertex AI")
            print(f"Only observed a {percent_reduction:.2f}% reduction, which is negligible")
    except Exception as e:
        print(f"Error testing Vertex AI: {e}")

if __name__ == "__main__":
    asyncio.run(test_token_efficient_tools_complex())