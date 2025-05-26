"""Tests for the ID utilities module."""

import importlib

import llmproc.utils.id_utils
import pytest
from llmproc.utils.id_utils import render_id


def test_render_id_default_style():
    """Test default bracketed ID rendering."""
    # Save original style to restore after test
    original_style = llmproc.utils.id_utils.DEFAULT_ID_STYLE

    try:
        # Ensure default style is brackets
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = "brackets"

        # Test with an integer message ID
        result = render_id(0)
        assert result == "[msg_0] ", "Default formatting should add prefix, brackets and a space"

        # Test with a larger integer
        result = render_id(42)
        assert result == "[msg_42] ", "Should work with any integer message ID"

        # Test with a custom string ID
        result = render_id("custom_id")
        assert result == "[custom_id] ", "Should work with any message ID string"
    finally:
        # Restore original style in case other tests depend on it
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = original_style


def test_render_id_xml_style():
    """Test XML-style ID rendering."""
    # Save original style to restore after test
    original_style = llmproc.utils.id_utils.DEFAULT_ID_STYLE

    try:
        # Change to XML style
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = "xml"

        # Test with an integer message ID
        result = render_id(0)
        assert result == "<message_id>msg_0</message_id> ", "XML formatting should use <message_id> tags"

        # Test with a larger integer
        result = render_id(42)
        assert result == "<message_id>msg_42</message_id> ", "Should work with any integer message ID"

        # Test with a custom string ID
        result = render_id("custom_id")
        assert result == "<message_id>custom_id</message_id> ", "Should work with any message ID string"
    finally:
        # Restore original style
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = original_style


def test_render_id_invalid_style():
    """Test error handling for invalid styles."""
    # Save original style to restore after test
    original_style = llmproc.utils.id_utils.DEFAULT_ID_STYLE

    try:
        # Set an invalid style
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = "invalid_style"

        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            render_id(0)

        # Check error message
        assert "Unknown id style: invalid_style" in str(excinfo.value)
    finally:
        # Restore original style
        llmproc.utils.id_utils.DEFAULT_ID_STYLE = original_style
