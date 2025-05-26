#!/usr/bin/env python3
"""Simplified non-interactive CLI for LLMProc.

This command executes a single prompt using a program configuration defined in
either TOML or YAML format.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from llmproc import LLMProgram
from llmproc.cli.demo import get_logger
from llmproc.cli.log_utils import log_program_info
from llmproc.common.results import RunResult


async def run_with_prompt(
    process: Any,
    user_prompt: str,
    source: str,
    logger: logging.Logger,
    callback_handler: Any,
    quiet: bool,
) -> RunResult:
    """Run a single prompt with an async process.

    Args:
        process: The LLMProcess to run the prompt with.
        user_prompt: The prompt text to run.
        source: Description of where the prompt came from.
        logger: Logger for diagnostic messages.
        callback_handler: Callback instance registered with the process.
        quiet: Whether to run in quiet mode.

    Returns:
        RunResult with the execution results.
    """
    logger.info(f"Running with {source} prompt")
    start_time = asyncio.get_event_loop().time()
    run_result = await process.run(user_prompt, max_iterations=process.max_iterations)
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"Used {run_result.api_calls} API calls in {elapsed:.2f}s")
    stderr_log = process.get_stderr_log()
    print("\n".join(stderr_log), file=sys.stderr)
    response = process.get_last_message()
    click.echo(response)
    return run_result


@click.command()
@click.argument("program_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--prompt", "-p", help="Prompt text. If omitted, read from stdin")
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
def main(
    program_path: str,
    prompt: str | None = None,
    log_level: str = "INFO",
    quiet: bool = False,
    append: bool = False,
) -> None:
    """Run a single prompt using the given PROGRAM_PATH."""
    asyncio.run(_async_main(program_path, prompt, log_level, quiet, append))


async def _async_main(
    program_path: str,
    prompt: str | None = None,
    log_level: str = "INFO",
    quiet: bool = False,
    append: bool = False,
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
            click.echo("3. If you're using npx to run an MCP server, make sure the package exists and is accessible", err=True)
            click.echo("4. To run without requiring MCP tools: export LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT=false", err=True)
            sys.exit(2)
        else:
            click.echo(f"Error initializing process: {e}", err=True)
            sys.exit(1)

    # Priority for prompt sources:
    # 1. Command-line argument (-p/--prompt)
    # 2. Non-empty stdin
    # 3. Embedded user prompt in configuration

    # Debug info about the user prompt from the process
    logger.info(f"Process user_prompt exists: {hasattr(process, 'user_prompt')}")
    if hasattr(process, "user_prompt"):
        logger.info(f"Process user_prompt value: {process.user_prompt!r}")

    provided_prompt = prompt

    if provided_prompt is not None:
        logger.info("Using prompt from command line argument")
    else:
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read().strip()
            if stdin_content:
                provided_prompt = stdin_content
                logger.info("Using input from stdin")
            else:
                logger.info("Stdin was empty")

    embedded_prompt = getattr(process, "user_prompt", "")

    if append:
        logger.info("Appending provided prompt to embedded prompt")
        parts = []
        if embedded_prompt and embedded_prompt.strip():
            parts.append(embedded_prompt.rstrip())
        if provided_prompt:
            parts.append(provided_prompt.lstrip())
        prompt = "\n".join(parts)
        if not prompt:
            click.echo("Error: No prompt provided via command line, stdin, or configuration", err=True)
            sys.exit(1)
    else:
        if provided_prompt is not None:
            prompt = provided_prompt
        elif embedded_prompt and embedded_prompt.strip():
            prompt = embedded_prompt
            logger.info("Using embedded user prompt from configuration")
        else:
            click.echo("Error: No prompt provided via command line, stdin, or configuration", err=True)
            sys.exit(1)

    # Final validation that we have a non-empty prompt
    if not prompt.strip():
        click.echo("Error: Empty prompt", err=True)
        sys.exit(1)

    # Create callback class for real-time updates (similar to demo.py)

    class CliCallbackHandler:
        def __init__(self) -> None:
            self.turn = 0

        def tool_start(self, tool_name, args):
            logger.info(json.dumps({"tool_start": {"tool_name": tool_name, "args": args}}, indent=2))

        def tool_end(self, tool_name, result):
            logger.info(json.dumps({"tool_end": {"tool_name": tool_name, "result": result.to_dict()}}, indent=2))

        def response(self, content):
            logger.info(json.dumps({"text response": content}, indent=2))

        def api_response(self, response):
            logger.info(json.dumps({"api response usage": response.usage.model_dump()}, indent=2))

        def stderr_write(self, text):
            logger.warning(json.dumps({"STDERR": text}, indent=2))

        async def turn_start(self, process):
            self.turn += 1  # Increment turn counter at the start of a turn
            info = await process.count_tokens()
            logger.warning(f"--------- TURN {self.turn} start, token count {info['input_tokens']} --------")

        def turn_end(self, process, response, tool_results):
            count = len(tool_results) if tool_results is not None else 0
            logger.warning(f"--------- TURN {self.turn} end, {count} tools used in this turn ----")

    # Create callback handler and register it with the process
    callback_handler = CliCallbackHandler()
    process.add_callback(callback_handler)

    log_program_info(process, prompt, logger)
    await run_with_prompt(process, prompt, "command line", logger, callback_handler, quiet_mode)

    # Ensure resources are cleaned up with strict timeout
    try:
        await asyncio.wait_for(process.aclose(), timeout=2.0)
    except TimeoutError:
        logger.warning("Process cleanup timed out after 2.0 seconds - forcing exit")
    except Exception as e:
        logger.warning(f"Error during process cleanup: {e}")


if __name__ == "__main__":
    main()
