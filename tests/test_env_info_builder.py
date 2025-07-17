"""Tests for the environment info builder component."""

import pytest

from llmproc.config import EnvInfoConfig
from llmproc.plugins.env_info.builder import EnvInfoBuilder


def test_build_env_info_with_variables():
    """Test that build_env_info correctly formats environment variables."""
    # Test with specific variables
    env_config = EnvInfoConfig(variables=["working_directory", "platform"])
    env_info = EnvInfoBuilder.build_env_info(env_config)

    # Verify the output format contains the requested variables
    assert "<env>" in env_info
    assert "working_directory:" in env_info
    assert "platform:" in env_info
    assert "</env>" in env_info

    # Verify it doesn't contain unrequested variables
    assert "date:" not in env_info
    assert "python_version:" not in env_info


def test_build_env_info_with_all_variables():
    """Test that build_env_info handles 'all' variables correctly."""
    env_config = EnvInfoConfig(variables="all")
    env_info = EnvInfoBuilder.build_env_info(env_config)

    # Verify all standard variables are included
    assert "<env>" in env_info
    assert "working_directory:" in env_info
    assert "platform:" in env_info
    assert "date:" in env_info
    assert "python_version:" in env_info
    assert "hostname:" in env_info
    assert "username:" in env_info
    assert "</env>" in env_info


def test_build_env_info_with_custom_variables():
    """Test that build_env_info correctly includes custom variables."""
    env_config = EnvInfoConfig(variables=["working_directory"], custom_var="custom value")
    env_info = EnvInfoBuilder.build_env_info(env_config)

    # Verify standard and custom variables
    assert "<env>" in env_info
    assert "working_directory:" in env_info
    assert "custom_var: custom value" in env_info
    assert "</env>" in env_info


def test_build_env_info_disabled():
    """Test that build_env_info returns empty string when disabled."""
    # Test with include_env=False
    env_config = EnvInfoConfig(variables=["working_directory"])
    env_info = EnvInfoBuilder.build_env_info(env_config, include_env=False)
    assert env_info == ""

    # Test with empty variables list
    env_config = EnvInfoConfig(variables=[])
    env_info = EnvInfoBuilder.build_env_info(env_config)
    assert env_info == ""
