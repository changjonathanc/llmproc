#!/usr/bin/env python3
"""CLI to install GitHub Actions workflows.

This command downloads workflow and configuration files from a specified GitHub
repository and installs them into the current project's ``.github`` directory.

Instead of fetching the entire repository archive, each file is downloaded
individually using ``httpx`` for efficiency. After installation, helpful links
are printed so users can easily generate tokens and add them as repository
secrets. Use ``--yes`` to skip all confirmation prompts.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import click
import httpx

DEFAULT_REPO = "cccntu/llmproc"
DEFAULT_REF = "main"

WORKFLOW_FILES = [
    ".github/workflows/llmproc-resolve.yml",
    ".github/workflows/llmproc-ask.yml",
    ".github/workflows/llmproc-code.yml",
]

CONFIG_FILES = [
    ".github/config/llmproc-resolve-claude.yaml",
    ".github/config/llmproc-ask-claude.yaml",
    ".github/config/llmproc-code-claude.yaml",
]


async def _download_file(client: httpx.AsyncClient, url: str) -> bytes:
    """Return the contents at *url* using the provided client."""
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.content


async def _install(repo: str, ref: str, dest: Path) -> None:
    """Download workflow/config files and write them under *dest*."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else {}

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = []
        for rel in WORKFLOW_FILES + CONFIG_FILES:
            url = f"https://raw.githubusercontent.com/{repo}/{ref}/{rel}"
            tasks.append(_download_file(client, url))
        contents = await asyncio.gather(*tasks)

    for rel, data in zip(WORKFLOW_FILES + CONFIG_FILES, contents, strict=True):
        target = dest / rel
        target.write_bytes(data)


def _is_git_repo() -> bool:
    """Check if the current directory is a git repository."""
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _infer_repo_slug() -> str | None:
    """Return the current repository slug if available."""
    try:
        url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
    except Exception:
        return None

    if url.startswith("git@github.com:"):
        slug = url.split("git@github.com:", 1)[1]
    elif url.startswith("https://github.com/"):
        slug = url.split("https://github.com/", 1)[1]
    else:
        return None

    if slug.endswith(".git"):
        slug = slug[:-4]
    return slug or None


@click.command()
@click.option("--repo", default=DEFAULT_REPO, show_default=True, help="Repository in owner/repo format.")
@click.option("--ref", default=DEFAULT_REF, show_default=True, help="Branch or tag to fetch.")
@click.option(
    "--dest",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    show_default=True,
    help="Destination directory for installed files.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Automatically answer yes to all prompts for non-interactive use.",
)
def main(repo: str, ref: str, dest: Path, yes: bool) -> None:
    """Download and install GitHub Actions workflows."""
    # Check if we're in a git repository
    if not _is_git_repo():
        click.echo("⚠️  Warning: You don't appear to be in a git repository.")
        click.echo("Please run this command from the root of your GitHub repository.")
        if not yes and not click.confirm("Continue anyway?"):
            raise SystemExit(0)

    # Check for existing files
    all_files = WORKFLOW_FILES + CONFIG_FILES
    existing_files = [f for f in all_files if (dest / f).exists()]

    if existing_files:
        click.echo("⚠️  Warning: The following files already exist and will be overwritten:")
        for file in existing_files:
            click.echo(f"  {file}")
        click.echo("")

    click.echo(f"This will download the following files from {repo}@{ref}:")
    click.echo("\nWorkflow files:")
    for file in WORKFLOW_FILES:
        status = " (will overwrite)" if file in existing_files else ""
        click.echo(f"  {file}{status}")
    click.echo("\nConfiguration files:")
    for file in CONFIG_FILES:
        status = " (will overwrite)" if file in existing_files else ""
        click.echo(f"  {file}{status}")
    click.echo(f"\nTotal: {len(WORKFLOW_FILES) + len(CONFIG_FILES)} files")
    if not yes and not click.confirm("\nContinue?"):
        raise SystemExit(0)

    missing_dirs = sorted(
        {(dest / Path(rel)).parent for rel in WORKFLOW_FILES + CONFIG_FILES if not (dest / Path(rel)).parent.exists()}
    )
    if missing_dirs:
        click.echo("The following directories will be created:")
        for d in missing_dirs:
            click.echo(f"  {d}")
        if not yes and not click.confirm("Create directories and copy files?"):
            raise SystemExit(0)
        for d in missing_dirs:
            d.mkdir(parents=True, exist_ok=True)

    asyncio.run(_install(repo, ref, dest))
    click.echo(f"\n✅ Successfully installed GitHub Actions from {repo}@{ref}")

    # Provide git instructions
    click.echo("\n📝 Next steps:")
    click.echo("\n1. Add the files to git:")
    all_files = WORKFLOW_FILES + CONFIG_FILES
    if len(all_files) <= 6:
        for file in all_files:
            click.echo(f"   git add {file}")
    else:
        click.echo("   git add .github/")

    click.echo("\n2. Commit the changes:")
    click.echo('   git commit -m "Add llmproc GitHub Actions workflows"')

    click.echo("\n3. Push to GitHub:")
    click.echo("   git push")

    # Provide token setup instructions
    repo_slug = _infer_repo_slug() or "<user>/<repo>"
    click.echo("\n🔑 Set up required secrets:")
    click.echo("\n1. Generate a GitHub personal access token:")
    click.echo("   https://github.com/settings/tokens")
    click.echo("   (Select 'repo' scope for private repositories)")

    click.echo("\n2. Generate an Anthropic API key:")
    click.echo("   https://console.anthropic.com/settings/keys")

    click.echo("\n3. Add both tokens as repository secrets:")
    click.echo(f"   https://github.com/{repo_slug}/settings/secrets/actions")
    click.echo("   - Add LLMPROC_WRITE_TOKEN (your GitHub personal access token)")
    click.echo("   - Add ANTHROPIC_API_KEY")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
