"""Tests for instance method support in function-based tools."""

import asyncio
import unittest
from typing import Any

from llmproc.common.metadata import get_tool_meta
from llmproc.tools.function_tools import register_tool, wrap_instance_method


class TestInstanceMethodTools(unittest.TestCase):
    """Test that instance methods can be registered as tools."""

    def setUp(self):
        """Set up a test class with instance methods."""
        self.provider = DataProvider()

    def test_wrap_instance_method(self):
        """Test that wrap_instance_method correctly wraps an instance method."""
        # Get the bound method
        method = self.provider.get_data

        # Wrap it
        wrapped = wrap_instance_method(method)

        # Check that it's properly wrapped
        self.assertTrue(hasattr(wrapped, "__wrapped_instance_method__"))
        self.assertEqual(wrapped.__original_instance__, self.provider)
        self.assertEqual(wrapped.__original_method_name__, "get_data")

        # Check that it still works correctly
        result = wrapped("users")
        self.assertEqual(result["key"], "users")
        self.assertEqual(result["value"], ["Alice", "Bob"])
        self.assertEqual(result["access_count"], 1)

        # Call it again and verify state is maintained
        result = wrapped("users")
        self.assertEqual(result["access_count"], 2)

    def test_wrap_async_instance_method(self):
        """Test that wrap_instance_method works with async methods."""
        # Get the bound async method
        method = self.provider.fetch_remote

        # Wrap it
        wrapped = wrap_instance_method(method)

        # Check that it's properly wrapped
        self.assertTrue(hasattr(wrapped, "__wrapped_instance_method__"))

        # Check that it still works correctly
        result = asyncio.run(wrapped("resource1"))
        self.assertEqual(result["id"], "resource1")
        self.assertEqual(result["access_count"], 1)

        # Call it again and verify state is maintained
        result = asyncio.run(wrapped("resource2"))
        self.assertEqual(result["access_count"], 2)

    def test_register_tool_on_instance_method(self):
        """Test that register_tool works on instance methods."""
        # Apply register_tool to an instance method
        decorated = register_tool(
            name="get_provider_data", description="Get data from the provider with access tracking"
        )(self.provider.get_data)

        # Check that metadata is properly attached
        meta = get_tool_meta(decorated)
        self.assertEqual(meta.name, "get_provider_data")
        self.assertEqual(meta.description, "Get data from the provider with access tracking")

        # Verify it's a wrapped instance method
        self.assertTrue(hasattr(decorated, "__wrapped_instance_method__"))

        # Check that it still works correctly
        result = decorated("users")
        self.assertEqual(result["key"], "users")
        self.assertEqual(result["access_count"], 1)


class DataProvider:
    """Test class with instance methods to register as tools."""

    def __init__(self):
        """Initialize with some state."""
        self.counter = 0
        self.data = {"users": ["Alice", "Bob"]}

    def get_data(self, key: str) -> dict[str, Any]:
        """Get data for the specified key.

        Args:
            key: The data key to retrieve

        Returns:
            Data associated with the key
        """
        self.counter += 1
        return {"key": key, "value": self.data.get(key, None), "access_count": self.counter}

    async def fetch_remote(self, resource_id: str) -> dict[str, Any]:
        """Fetch data from remote resource.

        Args:
            resource_id: ID of the resource to fetch

        Returns:
            The fetched resource data
        """
        # Async implementation
        self.counter += 1
        return {"id": resource_id, "access_count": self.counter}


if __name__ == "__main__":
    unittest.main()
