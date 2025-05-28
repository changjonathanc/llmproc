#!/usr/bin/env python3
"""Tests for the CLI module."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from click.testing import CliRunner

from llmproc.cli.demo import main
from llmproc.cli.run import main as run_main

# No fixtures needed - all mocks are created in the test functions


def test_interactive_cli_session():
    """Test the CLI in interactive mode."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"
    mock_process.tools = []
    mock_process.enriched_system_prompt = ""
    mock_process.api_params = {}
    mock_process.model_name = "m"
    mock_process.user_prompt = ""

    # Configure mock program to use start_sync
    mock_program.start_sync = MagicMock(return_value=mock_process)

    # Configure RunResult mock for run (new API)
    run_result = MagicMock()
    run_result.api_calls = 1
    mock_process.run.return_value = run_result

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.demo.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.demo.Path.exists") as mock_exists,
        patch("llmproc.cli.demo.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.demo.Path.absolute") as mock_absolute,
        patch("click.prompt") as mock_prompt,
        patch("llmproc.cli.demo.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_file.return_value = mock_program
        mock_prompt.side_effect = ["Hello", "exit"]
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path("/fake/path/test.toml")  # Mock Path.absolute()

        # Create a temporary example file
        with runner.isolated_filesystem():
            Path("test.toml").write_text("[model]\nname='test'\nprovider='x'")
            result = runner.invoke(main, ["test.toml"])

        # Verify that the code ran as expected
        assert mock_llm_program.from_file.called
        assert mock_program.start_sync.called
        assert mock_process.run.called  # Changed to run instead of run_sync


def test_cli_prompt_option():
    """Test the CLI in non-interactive mode with --prompt."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"
    mock_process.tools = []
    mock_process.enriched_system_prompt = ""
    mock_process.api_params = {}
    mock_process.model_name = "m"

    # Configure mock program to use start (async)
    mock_program.start = AsyncMock(return_value=mock_process)

    # Configure RunResult mock for run (new API)
    run_result = MagicMock()
    run_result.api_calls = 1
    mock_process.run = AsyncMock(return_value=run_result)

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.run.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.run.Path.exists") as mock_exists,
        patch("llmproc.cli.run.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.run.Path.absolute") as mock_absolute,
        patch("llmproc.cli.run.click.echo") as mock_echo,
        patch("llmproc.cli.run.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_file.return_value = mock_program
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path("/fake/path/test.toml")  # Mock Path.absolute()

        # Create a temporary example file
        with runner.isolated_filesystem():
            Path("test.toml").write_text("[model]\nname='test'\nprovider='x'")
            result = runner.invoke(run_main, ["test.toml", "--prompt", "Hello world"])

        # Verify that the code ran as expected
        assert mock_llm_program.from_file.called
        assert mock_program.start.called
        assert mock_process.run.called  # Changed to async run


def test_cli_prompt_file_option():
    """Test the CLI with --prompt-file option."""
    runner = CliRunner()

    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"
    mock_process.tools = []
    mock_process.enriched_system_prompt = ""
    mock_process.api_params = {}
    mock_process.model_name = "m"

    mock_program.start = AsyncMock(return_value=mock_process)

    run_result = MagicMock()
    run_result.api_calls = 1
    mock_process.run = AsyncMock(return_value=run_result)

    with (
        patch("llmproc.cli.run.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.run.Path.exists") as mock_exists,
        patch("llmproc.cli.run.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.run.Path.absolute") as mock_absolute,
        patch("llmproc.cli.run.click.echo") as mock_echo,
        patch("llmproc.cli.run.sys.exit") as mock_exit,
    ):
        mock_llm_program.from_file.return_value = mock_program
        mock_exists.return_value = True
        mock_suffix.return_value = ".toml"
        mock_absolute.return_value = Path("/fake/path/test.toml")

        with runner.isolated_filesystem():
            Path("test.toml").write_text("[model]\nname='test'\nprovider='x'")
            Path("prompt.txt").write_text("Hello from file")
            result = runner.invoke(
                run_main,
                ["test.toml", "--prompt-file", "prompt.txt"],
            )

        assert mock_llm_program.from_file.called
        assert mock_program.start.called
        assert mock_process.run.called


def test_cli_stdin_input_non_interactive():
    """Test the CLI in non-interactive mode with stdin."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"

    mock_process.tools = []
    mock_process.enriched_system_prompt = ""
    mock_process.api_params = {}
    mock_process.model_name = "m"
    # Configure mock program to use start (async)
    mock_program.start = AsyncMock(return_value=mock_process)

    # Configure RunResult mock for run (new API)
    run_result = MagicMock()
    run_result.api_calls = 1
    mock_process.run = AsyncMock(return_value=run_result)

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.run.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.run.Path.exists") as mock_exists,
        patch("llmproc.cli.run.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.run.Path.absolute") as mock_absolute,
        patch("llmproc.cli.run.sys.stdin.isatty") as mock_isatty,
        patch("llmproc.cli.run.click.echo") as mock_echo,
        patch("llmproc.cli.run.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_file.return_value = mock_program
        mock_isatty.return_value = False  # Simulate stdin having data
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path("/fake/path/test.toml")  # Mock Path.absolute()

        # Create a temporary example file
        with runner.isolated_filesystem():
            Path("test.toml").write_text("[model]\nname='test'\nprovider='x'")
            result = runner.invoke(run_main, ["test.toml"], input="Hello from stdin")

        # Verify that the code ran as expected
        assert mock_llm_program.from_file.called
        assert mock_program.start.called
        assert mock_process.run.called  # Changed to async run


def test_cli_json_output():
    """Test JSON output flag for the non-interactive CLI."""
    runner = CliRunner()

    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"
    mock_process.get_stderr_log.return_value = []
    mock_process.tools = []
    mock_process.enriched_system_prompt = ""
    mock_process.api_params = {}
    mock_process.model_name = "m"
    mock_program.start = AsyncMock(return_value=mock_process)

    run_result = MagicMock()
    run_result.api_calls = 2
    mock_process.run = AsyncMock(return_value=run_result)

    with (
        patch("llmproc.cli.run.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.run.Path.exists") as mock_exists,
        patch("llmproc.cli.run.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.run.Path.absolute") as mock_absolute,
        patch("llmproc.cli.run.sys.exit") as mock_exit,
        patch("llmproc.cli.run.setup_logger") as mock_setup_logger,
    ):
        mock_llm_program.from_file.return_value = mock_program
        mock_exists.return_value = True
        mock_suffix.return_value = ".toml"
        mock_absolute.return_value = Path("/fake/path/test.toml")
        mock_setup_logger.return_value = MagicMock()

        with runner.isolated_filesystem():
            Path("test.toml").write_text("[model]\nname='test'\nprovider='x'")
            result = runner.invoke(run_main, ["test.toml", "--prompt", "hello", "--json"])

        data = json.loads(result.output)
        assert data["api_calls"] == 2
        assert data["last_message"] == "Test response"
        assert data["stderr"] == []
