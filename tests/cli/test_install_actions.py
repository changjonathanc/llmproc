#!/usr/bin/env python3
"""Tests for the install_actions CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llmproc.cli.install_actions import main


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.content = data
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - not used
        pass


class _FakeClient:
    def __init__(self, file_map: dict[str, bytes]) -> None:
        self.file_map = file_map

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - not used
        return None

    async def get(self, url: str) -> _FakeResponse:  # noqa: D401 - simple
        """Return fake response for *url*."""
        # URLs are of the form https://raw.githubusercontent.com/<repo>/<ref>/<path>
        path = "/".join(url.split("/", 7)[6:])
        return _FakeResponse(self.file_map[path])


async def _run_cli(runner: CliRunner) -> tuple[int, str]:
    result = await asyncio.to_thread(runner.invoke, main, ["--yes"])
    return result.exit_code, result.output


def test_install_actions_cli_not_git_repo() -> None:
    """Verify warning when not in a git repository."""
    runner = CliRunner()

    with (
        patch("llmproc.cli.install_actions._is_git_repo", return_value=False),
    ):
        with runner.isolated_filesystem():
            result = runner.invoke(main, [], input="n\n")
            assert result.exit_code == 0

            # Verify git warning is shown
            assert "Warning: You don't appear to be in a git repository" in result.output
            assert "Please run this command from the root of your GitHub repository" in result.output

            # Verify no files were created
            assert not Path(".github").exists()


def test_install_actions_cli_no_response() -> None:
    """Verify that files are listed even when user says no."""
    runner = CliRunner()

    with (
        patch("llmproc.cli.install_actions.WORKFLOW_FILES", [
            ".github/workflows/test.yml",
        ]),
        patch("llmproc.cli.install_actions.CONFIG_FILES", [
            ".github/config/example.yaml",
        ]),
        patch("llmproc.cli.install_actions._is_git_repo", return_value=True),
    ):
        with runner.isolated_filesystem():
            result = runner.invoke(main, [], input="n\n")
            assert result.exit_code == 0

            # Verify that files are listed in output before the prompt
            assert "Workflow files:" in result.output
            assert ".github/workflows/test.yml" in result.output
            assert "Configuration files:" in result.output
            assert ".github/config/example.yaml" in result.output
            assert "Total: 2 files" in result.output

            # Verify no files were created
            assert not Path(".github").exists()


def test_install_actions_cli_with_existing_files() -> None:
    """Verify warning when files already exist."""
    runner = CliRunner()
    file_map = {
        ".github/workflows/test.yml": b"name: Test",
        ".github/config/example.yaml": b"config: true",
    }

    def fake_client(*args, **kwargs) -> _FakeClient:
        return _FakeClient(file_map)

    with (
        patch("httpx.AsyncClient", fake_client),
        patch("llmproc.cli.install_actions.WORKFLOW_FILES", [
            ".github/workflows/test.yml",
        ]),
        patch("llmproc.cli.install_actions.CONFIG_FILES", [
            ".github/config/example.yaml",
        ]),
        patch("llmproc.cli.install_actions._is_git_repo", return_value=True),
    ):
        with runner.isolated_filesystem():
            # Create existing files
            Path(".github/workflows").mkdir(parents=True)
            Path(".github/workflows/test.yml").write_text("old content")

            exit_code, output = asyncio.run(_run_cli(runner))
            assert exit_code == 0

            # Verify warning about existing files
            assert "Warning: The following files already exist and will be overwritten:" in output
            assert ".github/workflows/test.yml (will overwrite)" in output

            # Verify file was overwritten
            assert Path(".github/workflows/test.yml").read_bytes() == b"name: Test"


def test_install_actions_cli() -> None:
    """Verify that files are downloaded and hints printed."""
    runner = CliRunner()
    file_map = {
        ".github/workflows/test.yml": b"name: Test",
        ".github/config/example.yaml": b"config: true",
    }

    def fake_client(*args, **kwargs) -> _FakeClient:
        return _FakeClient(file_map)

    with (
        patch("httpx.AsyncClient", fake_client),
        patch("llmproc.cli.install_actions.WORKFLOW_FILES", [
            ".github/workflows/test.yml",
        ]),
        patch("llmproc.cli.install_actions.CONFIG_FILES", [
            ".github/config/example.yaml",
        ]),
        patch("llmproc.cli.install_actions._infer_repo_slug", return_value="test/repo"),
        patch("llmproc.cli.install_actions._is_git_repo", return_value=True),
    ):
        with runner.isolated_filesystem():
            exit_code, output = asyncio.run(_run_cli(runner))
            assert exit_code == 0
            assert Path(".github/workflows/test.yml").is_file()
            assert Path(".github/config/example.yaml").is_file()

            # Verify that files are listed in output
            assert "Workflow files:" in output
            assert ".github/workflows/test.yml" in output
            assert "Configuration files:" in output
            assert ".github/config/example.yaml" in output
            assert "Total: 2 files" in output

            # Verify git instructions
            assert "git add" in output
            assert "git commit" in output
            assert "git push" in output

            # Verify help links
            assert "https://github.com/settings/tokens" in output
            assert "https://console.anthropic.com/settings/keys" in output
            assert "https://github.com/test/repo/settings/secrets/actions" in output
            assert "LLMPROC_WRITE_TOKEN" in output
            assert "ANTHROPIC_API_KEY" in output
