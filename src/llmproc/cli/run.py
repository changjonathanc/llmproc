#!/usr/bin/env python3
"""Simplified non-interactive CLI for LLMProc.

This command executes a single prompt using a program configuration defined in
either TOML or YAML format.
"""

import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from llmproc import LLMProgram
from llmproc.cli.log_utils import (
    CliCallbackHandler,
    CostLimitExceededError,
    get_logger,
    log_program_info,
    setup_logger,
)
from llmproc.common.results import RunResult


async def run_with_prompt(
    process: Any,
    user_prompt: str,
    source: str,
    logger: logging.Logger,
    callback_handler: Any,
    quiet: bool,
    json_output: bool = False,
) -> RunResult:
    """Run a single prompt with an async process.

    Args:
        process: The LLMProcess to run the prompt with.
        user_prompt: The prompt text to run.
        source: Description of where the prompt came from.
        logger: Logger for diagnostic messages.
        callback_handler: Callback instance registered with the process.
        quiet: Whether to run in quiet mode.
        json_output: Suppress stdout/stderr output and defer printing to caller.

    Returns:
        RunResult with the execution results.
    """
    logger.info(f"Running with {source} prompt")
    start_time = asyncio.get_event_loop().time()
    run_result = await process.run(user_prompt, max_iterations=process.max_iterations)
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"Used {run_result.api_calls} API calls in {elapsed:.2f}s")
    if not json_output:
        stderr_log = process.get_stderr_log()
        print("\n".join(stderr_log), file=sys.stderr)
        response = process.get_last_message()
        click.echo(response)
    return run_result


def _get_provided_prompt(prompt: str | None, prompt_file: str | None, logger: logging.Logger) -> str | None:
    """Retrieve prompt from CLI argument, file, or stdin."""
    if prompt is not None and prompt_file is not None:
        click.echo("Error: --prompt and --prompt-file are mutually exclusive", err=True)
        sys.exit(1)

    provided_prompt = None
    if prompt is not None:
        provided_prompt = prompt
        logger.info("Using prompt from command line argument")
    elif prompt_file is not None:
        provided_prompt = Path(prompt_file).read_text()
        logger.info(f"Using prompt from file {prompt_file}")
    else:
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                provided_prompt = stdin_content
                logger.info("Using input from stdin")
            else:
                logger.info("Stdin was empty")
    return provided_prompt


def _resolve_prompt(
    provided_prompt: str | None,
    embedded_prompt: str,
    append: bool,
    logger: logging.Logger,
) -> str:
    """Combine provided and embedded prompts according to append flag."""
    if append:
        logger.info("Appending provided prompt to embedded prompt")
        parts = []
        if embedded_prompt and embedded_prompt.strip():
            parts.append(embedded_prompt.rstrip())
        if provided_prompt:
            parts.append(provided_prompt.lstrip())
        prompt = "\n".join(parts)
    else:
        if provided_prompt is not None:
            prompt = provided_prompt
        elif embedded_prompt and embedded_prompt.strip():
            prompt = embedded_prompt
            logger.info("Using embedded user prompt from configuration")
        else:
            click.echo(
                "Error: No prompt provided via command line, stdin, or configuration",
                err=True,
            )
            sys.exit(1)

    if not prompt.strip():
        click.echo("Error: Empty prompt", err=True)
        sys.exit(1)

    return prompt


@click.command()
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--prompt", "-p", help="Prompt text. If omitted, read from stdin")
@click.option(
    "--prompt-file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    help="Read prompt from file",
)
@click.option("--append", "-a", is_flag=True, help="Append provided prompt to embedded prompt")
@click.option(
    "--log-level",
    "-l",
    default="INFO",
    show_default=True,
    help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress most output while retaining chosen log level",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON for automation",
)
@click.option(
    "--cost-limit",
    type=float,
    metavar="USD",
    help="Stop execution when cost exceeds this limit in USD",
)
def main(
    program_path: str,
    prompt: str | None = None,
    prompt_file: str | None = None,
    log_level: str = "INFO",
    quiet: bool = False,
    append: bool = False,
    json_output: bool = False,
    cost_limit: float | None = None,
) -> None:
    """Run a single prompt using the given PROGRAM_PATH."""
    asyncio.run(
        _async_main(
            program_path,
            prompt,
            prompt_file,
            log_level,
            quiet,
            append,
            json_output,
            cost_limit,
        )
    )


async def _async_main(
    program_path: str,
    prompt: str | None = None,
    prompt_file: str | None = None,
    log_level: str = "INFO",
    quiet: bool = False,
    append: bool = False,
    json_output: bool = False,
    cost_limit: float | None = None,
) -> None:
    """Async implementation for running a single prompt."""
    logger = get_logger(log_level)
    level_num = getattr(logging, log_level.upper(), logging.INFO)
    quiet_mode = quiet or level_num >= logging.ERROR

    path = Path(program_path)

    try:
        program = LLMProgram.from_file(path)
    except Exception as e:  # pragma: no cover - pass through to user
        click.echo(f"Error loading program file: {e}", err=True)
        sys.exit(1)

    try:
        process = await program.start()
    except RuntimeError as e:
        if "Global timeout fetching tools from MCP servers" in str(e):
            # Extract the server names and timeout from the error message for cleaner display
            error_lines = str(e).strip().split("\n")
            click.echo(f"ERROR: {error_lines[0]}", err=True)
            click.echo("\nThis error occurs when MCP tool servers fail to initialize.", err=True)
            click.echo("Possible solutions:", err=True)
            click.echo("1. Increase the timeout: export LLMPROC_TOOL_FETCH_TIMEOUT=300", err=True)
            click.echo("2. Check if the MCP server is running properly", err=True)
            click.echo(
                "3. If you're using npx to run an MCP server, make sure the package exists and is accessible", err=True
            )
            click.echo("4. To run without requiring MCP tools: export LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT=false", err=True)
            sys.exit(2)
        else:
            click.echo(f"Error initializing process: {e}", err=True)
            sys.exit(1)

    # Priority for prompt sources:
    # 1. Command-line argument (-p/--prompt)
    # 2. Non-empty stdin
    # 3. Embedded user prompt in configuration

    logger.info(f"Process user_prompt exists: {hasattr(process, 'user_prompt')}")
    if hasattr(process, "user_prompt"):
        logger.info(f"Process user_prompt value: {process.user_prompt!r}")

    provided_prompt = _get_provided_prompt(prompt, prompt_file, logger)
    embedded_prompt = getattr(process, "user_prompt", "")
    prompt = _resolve_prompt(provided_prompt, embedded_prompt, append, logger)

    # Create callback handler and register it with the process
    callback_handler = CliCallbackHandler(logger, cost_limit=cost_limit)
    process.add_callback(callback_handler)

    log_program_info(process, prompt, logger)

    try:
        run_result = await run_with_prompt(
            process,
            prompt,
            "command line",
            logger,
            callback_handler,
            quiet_mode,
            json_output=json_output,
        )

        if json_output:
            output = {
                "api_calls": run_result.api_calls,
                "usd_cost": run_result.usd_cost,
                "last_message": process.get_last_message(),
                "stderr": process.get_stderr_log(),
                "stop_reason": run_result.stop_reason,
            }
            click.echo(json.dumps(output))

    except CostLimitExceededError as e:
        # Handle cost limit exceeded gracefully
        if json_output:
            # Create a minimal result with cost limit information
            current_result = RunResult()
            current_result.set_stop_reason("cost_limit_exceeded")

            # Try to get actual run state if available
            try:
                # Get the actual accumulated costs from the process if possible
                if hasattr(process, "_run_result") and process._run_result:
                    current_result = process._run_result
                    current_result.set_stop_reason("cost_limit_exceeded")
            except Exception:
                pass

            output = {
                "api_calls": current_result.api_calls,
                "usd_cost": e.actual_cost,  # Use the actual cost from exception
                "last_message": process.get_last_message(),
                "stderr": process.get_stderr_log(),
                "stop_reason": current_result.stop_reason,
                "cost_limit": e.cost_limit,
            }
            click.echo(json.dumps(output))
        else:
            click.echo(f"⚠️  Execution stopped: {e}", err=True)

        # Exit with success code - this is user-requested behavior
        sys.exit(0)

    # Ensure resources are cleaned up with strict timeout. We create a task so
    # the aclose coroutine is always awaited even if wait_for is mocked.
    close_task = asyncio.create_task(process.aclose())
    try:
        await asyncio.wait_for(close_task, timeout=2.0)
    except TimeoutError:
        logger.warning("Process cleanup timed out after 2.0 seconds - forcing exit")
        close_task.cancel()
        with contextlib.suppress(BaseException):
            await close_task
    except Exception as e:
        logger.warning(f"Error during process cleanup: {e}")


if __name__ == "__main__":
    main()
