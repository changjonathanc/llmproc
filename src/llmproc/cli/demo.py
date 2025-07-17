#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version.

This module provides the main CLI functionality for the llmproc-demo command.
"""

import logging
import sys
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

from llmproc import LLMProgram
from llmproc.cli.log_utils import (
    CliCallbackHandler,
    CostLimitExceededError,
    get_logger,
    log_program_info,
)
from llmproc.common.results import RunResult


def run_demo_mode(
    prompts: list[str],
    run_prompt_func: callable,
    quiet: bool,
    logger: logging.Logger,
    *,
    pause: bool = True,
) -> bool:
    """Run a series of prompts in demo mode."""
    logger.info(f"Starting demo mode with {len(prompts)} prompts")

    if not quiet:
        click.echo("\n===== Demo Mode =====")
        click.echo(f"Running {len(prompts)} prompts in sequence")
        if pause:
            click.echo("Press Enter after each response to continue to the next prompt")
        click.echo("====================\n")

    for i, prompt in enumerate(prompts):
        if not quiet:
            click.echo(f"\n----- Prompt {i + 1}/{len(prompts)} -----")
            click.echo(f"User: {prompt}")

        result = run_prompt_func(prompt, source=f"demo prompt {i + 1}/{len(prompts)}")

        if getattr(result, "cost_limit_exceeded", False):
            if not quiet:
                click.echo("\n⚠️  Demo stopped due to cost limit")
            return True

        if pause and i < len(prompts) - 1:
            click.echo("\nPress Enter to continue to the next prompt...", nl=False)
            input()

    if not quiet:
        click.echo("\n===== Demo Complete =====")

    return True


def run_with_prompt(
    process: Any, user_prompt: str, source: str, logger: logging.Logger, callback_handler: Any, quiet: bool
) -> RunResult:
    """Run a single prompt with the given process.

    Args:
        process: The LLMProcess to run the prompt with
        user_prompt: The prompt text to run
        source: Description of where the prompt came from
        logger: Logger for diagnostic messages
        callback_handler: Callback instance to register with the process
        quiet: Whether to run in quiet mode

    Returns:
        RunResult with the execution results
    """
    # Report mode
    logger.info(f"Running with {source} prompt")

    # Track time for this run
    start_time = time.time()

    # Execute the process with the given prompt
    try:
        run_result = process.run(user_prompt, max_iterations=process.max_iterations)
    except CostLimitExceededError as e:
        if not quiet:
            click.echo(f"\n⚠️  Session stopped: {e}", err=True)
        # Return a minimal result to indicate cost limit was exceeded
        run_result = RunResult()
        run_result.cost_limit_exceeded = True
        return run_result

    # Get the elapsed time
    elapsed = time.time() - start_time

    # Log run result information
    logger.info(f"Used {run_result.api_call_count} in {elapsed:.2f}s, cost ${run_result.usd_cost:.4f}")

    # Get the last assistant message and print the response
    response = process.get_last_message()
    click.echo(response)

    return run_result


def _show_initial_tokens(process: Any, logger: logging.Logger, quiet_mode: bool) -> None:
    """Display the initial token count if possible."""
    if quiet_mode:
        return
    try:
        token_info = process.count_tokens()
        if token_info and "input_tokens" in token_info:
            click.echo(
                f"Initial context size: {token_info['input_tokens']:,} tokens ({token_info['percentage']:.1f}% of {token_info['context_window']:,} token context window)"
            )
    except Exception as exc:  # pragma: no cover - best effort display
        logger.warning(f"Failed to count initial tokens: {str(exc)}")


def _get_token_display(process: Any, logger: logging.Logger, quiet_mode: bool) -> str:
    """Return a short token usage string for the prompt."""
    if quiet_mode:
        return ""
    try:
        token_info = process.count_tokens()
        if token_info and "input_tokens" in token_info:
            return f" [Tokens: {token_info['input_tokens']:,}/{token_info['context_window']:,}]"
    except Exception as exc:  # pragma: no cover - best effort display
        logger.warning(f"Failed to count tokens for prompt: {str(exc)}")
    return ""


def _get_user_input(token_display: str) -> str:
    """Prompt the user for input."""
    return click.prompt(f"\nYou{token_display}", prompt_suffix=">")


def _display_response(process: Any, run_result: RunResult, quiet_mode: bool) -> None:
    """Print the assistant response and token summary."""
    response = process.get_last_message()
    if quiet_mode:
        click.echo(f"\n{response}")
    else:
        click.echo(f"\n{process.model_name}> {response}")
        if getattr(run_result, "total_tokens", 0) > 0:
            click.echo(
                f"[API calls: {run_result.api_call_count}, Tool calls: {run_result.tool_call_count}, Tokens: {run_result.input_tokens}/{run_result.output_tokens}/{run_result.total_tokens} (in/out/total)]"
            )


def interactive_chat(
    process: Any,
    run_prompt_func: callable,
    quiet_mode: bool,
    logger: logging.Logger,
) -> None:
    """Run the interactive chat loop."""
    if not quiet_mode:
        click.echo("\nStarting interactive chat session. Type 'exit' or 'quit' to end.")

    _show_initial_tokens(process, logger, quiet_mode)

    while True:
        token_display = _get_token_display(process, logger, quiet_mode)
        user_input = _get_user_input(token_display)

        if user_input.lower() in ("exit", "quit"):
            if not quiet_mode:
                click.echo("Ending session.")
            break

        start_time = time.time()
        if not quiet_mode:
            click.echo("Thinking...", nl=False)

        if not getattr(process, "state", []):
            log_program_info(process, user_input, logger)

        run_result = run_prompt_func(user_input, source="interactive")

        elapsed = time.time() - start_time
        if not quiet_mode:
            click.echo("\r" + " " * 12 + "\r", nl=False)

        if run_result.api_call_count > 0:
            logger.info(
                f"Used {run_result.api_call_count} API calls and {run_result.tool_call_count} tool calls in {elapsed:.2f}s, cost ${run_result.usd_cost:.4f}"
            )

        _display_response(process, run_result, quiet_mode)


def check_and_run_demo_mode(
    program: LLMProgram, run_prompt_func: callable, quiet: bool, logger: logging.Logger
) -> bool:
    """Check if program has demo mode configured and run it if available.

    Args:
        program: The LLMProgram to check for demo configuration
        run_prompt_func: Function to run individual prompts
        quiet: Whether to run in quiet mode
        logger: Logger for diagnostic messages

    Returns:
        True if demo mode was found and executed, False otherwise
    """
    if not hasattr(program, "source_path") or not program.source_path:
        return False

    # Read the original TOML file to check for demo section
    try:
        with open(program.source_path, "rb") as f:
            config_data = tomllib.load(f)

        if "demo" in config_data and "prompts" in config_data["demo"] and config_data["demo"]["prompts"]:
            demo_prompts = config_data["demo"]["prompts"]
            pause_between = config_data["demo"].get("pause_between_prompts", True)
            return run_demo_mode(
                demo_prompts,
                run_prompt_func,
                quiet,
                logger,
                pause=pause_between,
            )

    except Exception as e:
        logger.warning(f"Failed to check for demo configuration: {str(e)}")

    return False


def display_program_summary(program: LLMProgram, quiet_mode: bool) -> None:
    """Display a summary of the loaded program."""
    if quiet_mode:
        return

    click.echo("\nProgram Summary:")
    click.echo(f"  Model: {program.model_name}")
    click.echo(f"  Provider: {program.provider}")
    click.echo(f"  Display Name: {program.model_name}")

    if hasattr(program, "system_prompt") and program.system_prompt:
        system_prompt = program.system_prompt
        if len(system_prompt) > 60:
            system_prompt = system_prompt[:57] + "..."
        click.echo(f"  System Prompt: {system_prompt}")

    if hasattr(program, "api_params") and program.api_params:
        params = program.api_params
        if "temperature" in params:
            click.echo(f"  Temperature: {params['temperature']}")
        if "max_tokens" in params:
            click.echo(f"  Max Tokens: {params['max_tokens']}")


def create_process(
    program: LLMProgram,
    logger: logging.Logger,
    cost_limit: float | None,
    quiet_mode: bool,
) -> tuple[Any, callable]:
    """Initialize the process and return it with a prompt runner."""
    logger.info("Starting process initialization...")
    start_time = time.time()

    try:
        process = program.start_sync()
        init_time = time.time() - start_time
        logger.info(f"Process initialized in {init_time:.2f} seconds")
    except RuntimeError as e:
        if "Global timeout fetching tools from MCP servers" in str(e):
            error_lines = str(e).strip().split("\n")
            click.echo(f"ERROR: {error_lines[0]}", err=True)
            click.echo(
                "\nThis error occurs when MCP tool servers fail to initialize.",
                err=True,
            )
            click.echo("Possible solutions:", err=True)
            click.echo(
                "1. Increase the timeout: export LLMPROC_TOOL_FETCH_TIMEOUT=300",
                err=True,
            )
            click.echo("2. Check if the MCP server is running properly", err=True)
            click.echo(
                "3. If you're using npx to run an MCP server, make sure the package exists and is accessible",
                err=True,
            )
            click.echo(
                "4. To run without requiring MCP tools: export LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT=false",
                err=True,
            )
            sys.exit(2)
        raise

    callback_handler = CliCallbackHandler(logger, cost_limit=cost_limit)
    process.add_plugins(callback_handler)

    def run_prompt_func(user_prompt: str, source: str = "command line") -> RunResult:
        if not getattr(process, "state", []):
            log_program_info(process, user_prompt, logger)
        return run_with_prompt(process, user_prompt, source, logger, callback_handler, quiet_mode)

    return process, run_prompt_func


def handle_prompt_flow(
    process: Any,
    program: LLMProgram,
    run_prompt_func: callable,
    prompt: str | None,
    quiet_mode: bool,
    logger: logging.Logger,
) -> None:
    """Run demo, embedded prompt, and interactive chat."""
    if check_and_run_demo_mode(program, run_prompt_func, quiet_mode, logger):
        return

    if prompt is not None:
        if prompt:
            run_prompt_func(prompt, source="command line")
    else:
        embedded_prompt = getattr(process, "user_prompt", "")
        if isinstance(embedded_prompt, str) and embedded_prompt:
            if not quiet_mode:
                click.echo("\nFound embedded user prompt:")
                prompt_preview = embedded_prompt
                if len(prompt_preview) > 200:
                    prompt_preview = prompt_preview[:197] + "..."
                click.echo(f'  "{prompt_preview}"')
                click.echo()

                if click.confirm("Run this embedded prompt?"):
                    run_prompt_func(embedded_prompt, source="embedded")
            else:
                run_prompt_func(embedded_prompt, source="embedded")

    interactive_chat(process, run_prompt_func, quiet_mode, logger)


@click.command()
@click.argument("program_path", required=True)
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
    help="Suppress most CLI output while retaining chosen log level",
)
@click.option(
    "--prompt",
    "-p",
    default=None,
    flag_value="",  # When -p is used without value, use empty string
    help="Override embedded prompt with custom prompt. Use -p alone to skip embedded prompt.",
)
@click.option(
    "--cost-limit",
    type=float,
    metavar="USD",
    help="Stop execution when cost exceeds this limit in USD",
)
def main(
    program_path, log_level: str = "INFO", quiet: bool = False, prompt: str = None, cost_limit: float | None = None
) -> None:
    """Run an interactive CLI for LLMProc.

    PROGRAM_PATH is the path to a TOML or YAML program file. The command always
    starts in interactive mode, optionally running a demo sequence if configured
    in the program file.
    """
    load_dotenv()
    level_num = getattr(logging, log_level.upper(), logging.INFO)
    quiet_mode = quiet or level_num >= logging.ERROR

    if not quiet_mode:
        click.echo("LLMProc CLI Demo")
        click.echo("----------------")

    program_file = Path(program_path)
    if not program_file.exists():
        click.echo(f"Error: Program file not found: {program_path}")
        sys.exit(1)

    try:
        cli_logger = get_logger(log_level)
        program = LLMProgram.from_file(program_file.absolute())
        display_program_summary(program, quiet_mode)
        process, run_prompt_func = create_process(program, cli_logger, cost_limit, quiet_mode)
        handle_prompt_flow(process, program, run_prompt_func, prompt, quiet_mode, cli_logger)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if not quiet_mode:
            click.echo("\nFull traceback:", err=True)
            traceback.print_exc()
        sys.exit(1)
    finally:
        if "process" in locals() and hasattr(process, "close"):
            try:
                process.close()
                cli_logger.debug("Process resources cleaned up")
            except Exception as e:
                cli_logger.warning(f"Error during process cleanup: {e}")


if __name__ == "__main__":
    main()
