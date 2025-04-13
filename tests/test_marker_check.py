"""Simple test to verify marker detection."""

import pytest


@pytest.mark.llm_api
@pytest.mark.essential_api
def test_marker_is_recognized():
    """Test that the llm_api marker is recognized."""
    assert True
