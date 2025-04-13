#!/usr/bin/env python3
"""Tests for the CLI module."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from click.testing import CliRunner

from llmproc.cli import main

# No fixtures needed - all mocks are created in the test functions


def test_cli_interactive_mode():
    """Test the CLI in interactive mode."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"

    # Use a regular MagicMock instead of AsyncMock to avoid coroutine warnings
    mock_program.start = MagicMock()

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.Path.exists") as mock_exists,
        patch("llmproc.cli.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.Path.absolute") as mock_absolute,
        patch("llmproc.cli.asyncio.run") as mock_run,
        patch("click.prompt") as mock_prompt,
        patch("llmproc.cli.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_toml.return_value = mock_program
        mock_prompt.side_effect = ["Hello", "exit"]
        mock_run.return_value = MagicMock()  # Mock RunResult
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path(
            "/fake/path/test.toml"
        )  # Mock Path.absolute()

        # Set a simple side effect that always returns the mock_process for start() calls
        # and a MagicMock for other calls
        mock_run.side_effect = lambda coro: mock_process

        # Create a temporary example file
        with runner.isolated_filesystem():
            # Run the CLI with the test file
            result = runner.invoke(main, ["test.toml"])

        # Verify that the code ran as expected
        assert mock_llm_program.from_toml.called
        assert mock_program.start.called
        assert mock_run.called


def test_cli_non_interactive_mode():
    """Test the CLI in non-interactive mode with --prompt."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"

    # Use a regular MagicMock instead of AsyncMock to avoid coroutine warnings
    mock_program.start = MagicMock()

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.Path.exists") as mock_exists,
        patch("llmproc.cli.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.Path.absolute") as mock_absolute,
        patch("llmproc.cli.asyncio.run") as mock_run,
        patch("llmproc.cli.click.echo") as mock_echo,
        patch("llmproc.cli.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_toml.return_value = mock_program
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path(
            "/fake/path/test.toml"
        )  # Mock Path.absolute()

        # Set a simple side effect that always returns the mock_process for start() calls
        # and a MagicMock for other calls
        mock_run.side_effect = lambda coro: mock_process

        # Create a temporary example file
        with runner.isolated_filesystem():
            # Run the CLI with the test file and --prompt
            result = runner.invoke(main, ["test.toml", "--prompt", "Hello world"])

        # Verify that the code ran as expected
        assert mock_llm_program.from_toml.called
        assert mock_program.start.called
        assert mock_run.called


def test_cli_stdin_non_interactive_mode():
    """Test the CLI in non-interactive mode with stdin."""
    runner = CliRunner()

    # Mocks and patches
    mock_process = MagicMock()
    mock_program = MagicMock()
    mock_process.display_name = "TestModel"
    mock_process.get_last_message.return_value = "Test response"

    # Use a regular MagicMock instead of AsyncMock to avoid coroutine warnings
    mock_program.start = MagicMock()

    # Patch various functions and classes to avoid actual API calls
    with (
        patch("llmproc.cli.LLMProgram") as mock_llm_program,
        patch("llmproc.cli.Path.exists") as mock_exists,
        patch("llmproc.cli.Path.suffix", new_callable=PropertyMock) as mock_suffix,
        patch("llmproc.cli.Path.absolute") as mock_absolute,
        patch("llmproc.cli.asyncio.run") as mock_run,
        patch("llmproc.cli.sys.stdin.isatty") as mock_isatty,
        patch("llmproc.cli.click.echo") as mock_echo,
        patch("llmproc.cli.sys.exit") as mock_exit,
    ):
        # Set up the mocks
        mock_llm_program.from_toml.return_value = mock_program
        mock_isatty.return_value = False  # Simulate stdin having data
        mock_exists.return_value = True  # Make Path.exists() return True
        mock_suffix.return_value = ".toml"  # Set the suffix to .toml
        mock_absolute.return_value = Path(
            "/fake/path/test.toml"
        )  # Mock Path.absolute()

        # Set a simple side effect that always returns the mock_process for start() calls
        # and a MagicMock for other calls
        mock_run.side_effect = lambda coro: mock_process

        # Create a temporary example file
        with runner.isolated_filesystem():
            # Run the CLI with the test file and --non-interactive
            result = runner.invoke(
                main, ["test.toml", "--non-interactive"], input="Hello from stdin"
            )

        # Verify that the code ran as expected
        assert mock_llm_program.from_toml.called
        assert mock_program.start.called
        assert mock_run.called
