"""Tests for the environment info builder component."""

import os
import tempfile
from pathlib import Path

import pytest

from llmproc.env_info.builder import EnvInfoBuilder


def test_build_env_info_with_variables():
    """Test that build_env_info correctly formats environment variables."""
    # Test with specific variables
    env_config = {"variables": ["working_directory", "platform"]}
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
    env_config = {"variables": "all"}
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
    env_config = {"variables": ["working_directory"], "custom_var": "custom value"}
    env_info = EnvInfoBuilder.build_env_info(env_config)

    # Verify standard and custom variables
    assert "<env>" in env_info
    assert "working_directory:" in env_info
    assert "custom_var: custom value" in env_info
    assert "</env>" in env_info


def test_build_env_info_disabled():
    """Test that build_env_info returns empty string when disabled."""
    # Test with include_env=False
    env_config = {"variables": ["working_directory"]}
    env_info = EnvInfoBuilder.build_env_info(env_config, include_env=False)
    assert env_info == ""

    # Test with empty variables list
    env_config = {"variables": []}
    env_info = EnvInfoBuilder.build_env_info(env_config)
    assert env_info == ""


def test_load_files():
    """Test that load_files correctly loads file content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        file1_path = Path(temp_dir) / "file1.txt"
        file2_path = Path(temp_dir) / "file2.txt"

        with open(file1_path, "w") as f:
            f.write("Content of file1")
        with open(file2_path, "w") as f:
            f.write("Content of file2")

        # Test loading files
        files = [str(file1_path), str(file2_path)]
        content_dict = EnvInfoBuilder.load_files(files)

        # Verify content was loaded correctly
        assert len(content_dict) == 2
        assert str(file1_path.resolve()) in content_dict
        assert str(file2_path.resolve()) in content_dict
        assert content_dict[str(file1_path.resolve())] == "Content of file1"
        assert content_dict[str(file2_path.resolve())] == "Content of file2"


def test_build_preload_content():
    """Test that build_preload_content formats content correctly."""
    # Test with preloaded content
    preloaded_content = {
        "/path/to/file1.txt": "Content of file1",
        "/path/to/file2.txt": "Content of file2",
    }
    preload_content = EnvInfoBuilder.build_preload_content(preloaded_content)

    # Verify format
    assert "<preload>" in preload_content
    assert '<file path="file1.txt">' in preload_content
    assert '<file path="file2.txt">' in preload_content
    assert "Content of file1" in preload_content
    assert "Content of file2" in preload_content
    assert "</file>" in preload_content
    assert "</preload>" in preload_content

    # Test with empty content
    assert EnvInfoBuilder.build_preload_content({}) == ""


def test_get_enriched_system_prompt_with_env_info():
    """Test get_enriched_system_prompt with environment info."""
    base_prompt = "Test system prompt"
    env_config = {"variables": ["working_directory"]}

    # Get enriched prompt with environment info
    enriched_prompt = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt, env_config=env_config, include_env=True
    )

    # Verify environment info is included
    assert base_prompt in enriched_prompt
    assert "<env>" in enriched_prompt
    assert "working_directory:" in enriched_prompt
    assert "</env>" in enriched_prompt

    # Test with include_env=False
    enriched_prompt_no_env = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt, env_config=env_config, include_env=False
    )

    # Verify environment info is not included
    assert base_prompt in enriched_prompt_no_env
    assert "<env>" not in enriched_prompt_no_env


def test_get_enriched_system_prompt_with_preloaded_content():
    """Test get_enriched_system_prompt with preloaded content."""
    base_prompt = "Test system prompt"
    env_config = {}
    preloaded_content = {"/path/to/file1.txt": "Content of file1"}

    # Get enriched prompt with preloaded content
    enriched_prompt = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt,
        env_config=env_config,
        preloaded_content=preloaded_content,
    )

    # Verify preloaded content is included
    assert base_prompt in enriched_prompt
    assert "<preload>" in enriched_prompt
    assert '<file path="file1.txt">' in enriched_prompt
    assert "Content of file1" in enriched_prompt
    assert "</file>" in enriched_prompt
    assert "</preload>" in enriched_prompt


def test_get_enriched_system_prompt_with_fd_features():
    """Test get_enriched_system_prompt with file descriptor features."""
    base_prompt = "Test system prompt"
    env_config = {}

    # Get enriched prompt with file descriptor enabled
    enriched_prompt = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt, env_config=env_config, file_descriptor_enabled=True
    )

    # Verify file descriptor instructions are included
    assert base_prompt in enriched_prompt
    assert "<file_descriptor_instructions>" in enriched_prompt

    # Test with user input paging
    enriched_prompt_with_paging = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt,
        env_config=env_config,
        file_descriptor_enabled=True,
        page_user_input=True,
    )

    # Verify user input paging instructions are included
    assert "<fd_user_input_instructions>" in enriched_prompt_with_paging

    # Test with references
    enriched_prompt_with_refs = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt,
        env_config=env_config,
        file_descriptor_enabled=True,
        references_enabled=True,
    )

    # Verify reference instructions are included
    assert "<reference_instructions>" in enriched_prompt_with_refs


def test_get_enriched_system_prompt_with_everything():
    """Test get_enriched_system_prompt with all features enabled."""
    base_prompt = "Test system prompt"
    env_config = {"variables": ["working_directory"], "custom_var": "custom value"}
    preloaded_content = {"/path/to/file1.txt": "Content of file1"}

    # Get enriched prompt with all features
    enriched_prompt = EnvInfoBuilder.get_enriched_system_prompt(
        base_prompt=base_prompt,
        env_config=env_config,
        preloaded_content=preloaded_content,
        file_descriptor_enabled=True,
        references_enabled=True,
        page_user_input=True,
    )

    # Verify all parts are included
    assert base_prompt in enriched_prompt
    assert "<env>" in enriched_prompt
    assert "working_directory:" in enriched_prompt
    assert "custom_var: custom value" in enriched_prompt
    assert "</env>" in enriched_prompt
    assert "<preload>" in enriched_prompt
    assert '<file path="file1.txt">' in enriched_prompt
    assert "Content of file1" in enriched_prompt
    assert "</file>" in enriched_prompt
    assert "</preload>" in enriched_prompt
    assert "<file_descriptor_instructions>" in enriched_prompt
    assert "<fd_user_input_instructions>" in enriched_prompt
    assert "<reference_instructions>" in enriched_prompt
