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

from llmproc import LLMProgram
from llmproc.cli.log_utils import (
    CliCallbackHandler,
    get_logger,
    log_program_info,
)
from llmproc.common.results import RunResult


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
    run_result = process.run(user_prompt, max_iterations=process.max_iterations)

    # Get the elapsed time
    elapsed = time.time() - start_time

    # Log run result information
    logger.info(f"Used {run_result.api_calls} in {elapsed:.2f}s")

    # Get the last assistant message and print the response
    response = process.get_last_message()
    click.echo(response)

    return run_result


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
            # Get demo configuration
            demo_prompts = config_data["demo"]["prompts"]
            pause_between = config_data["demo"].get("pause_between_prompts", True)

            # Define demo mode function
            def run_demo_mode(prompts: list[str], pause: bool = True) -> bool:
                """Run a series of prompts in demo mode."""
                logger.info(f"Starting demo mode with {len(prompts)} prompts")

                # Prepare the demo
                if not quiet:
                    click.echo("\n===== Demo Mode =====")
                    click.echo(f"Running {len(prompts)} prompts in sequence")
                    if pause:
                        click.echo("Press Enter after each response to continue to the next prompt")
                    click.echo("====================\n")

                # Run each prompt in sequence
                for i, prompt in enumerate(prompts):
                    # Show prompt number and content
                    if not quiet:
                        click.echo(f"\n----- Prompt {i + 1}/{len(prompts)} -----")
                        click.echo(f"User: {prompt}")

                    # Run the prompt
                    run_prompt_func(prompt, source=f"demo prompt {i + 1}/{len(prompts)}")

                    # Pause for user input if configured, except after the last prompt
                    if pause and i < len(prompts) - 1:
                        click.echo("\nPress Enter to continue to the next prompt...", nl=False)
                        input()

                if not quiet:
                    click.echo("\n===== Demo Complete =====")

                return True

            # Run in demo mode
            return run_demo_mode(demo_prompts, pause=pause_between)

    except Exception as e:
        logger.warning(f"Failed to check for demo configuration: {str(e)}")

    return False


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
def main(program_path, log_level: str = "INFO", quiet: bool = False, prompt: str = None) -> None:
    """Run an interactive CLI for LLMProc.

    PROGRAM_PATH is the path to a TOML or YAML program file. The command always
    starts in interactive mode, optionally running a demo sequence if configured
    in the program file.
    """
    level_num = getattr(logging, log_level.upper(), logging.INFO)
    quiet_mode = quiet or level_num >= logging.ERROR

    # Display a simple header unless in quiet mode
    if not quiet_mode:
        click.echo("LLMProc CLI Demo")
        click.echo("----------------")

    # Validate program path
    program_file = Path(program_path)
    if not program_file.exists():
        click.echo(f"Error: Program file not found: {program_path}")
        sys.exit(1)
    selected_program = program_file

    # Load the selected program
    try:
        selected_program_abs = selected_program.absolute()

        # Get logger with appropriate level
        cli_logger = get_logger(log_level)

        # Use the new API: load program first, then start it asynchronously
        cli_logger.info(f"Loading program from: {selected_program_abs}")
        program = LLMProgram.from_file(selected_program_abs)

        # Display program summary unless in quiet mode
        if not quiet_mode:
            click.echo("\nProgram Summary:")
            click.echo(f"  Model: {program.model_name}")
            click.echo(f"  Provider: {program.provider}")
            click.echo(f"  Display Name: {program.display_name or program.model_name}")

        # Initial token count will be displayed after initialization

        # Show brief system prompt summary unless in quiet mode
        if not quiet_mode and hasattr(program, "system_prompt") and program.system_prompt:
            system_prompt = program.system_prompt
            # Truncate if too long
            if len(system_prompt) > 60:
                system_prompt = system_prompt[:57] + "..."
            click.echo(f"  System Prompt: {system_prompt}")

        # Show parameter summary if available and not in quiet mode
        if not quiet_mode and hasattr(program, "api_params") and program.api_params:
            params = program.api_params
            if "temperature" in params:
                click.echo(f"  Temperature: {params['temperature']}")
            if "max_tokens" in params:
                click.echo(f"  Max Tokens: {params['max_tokens']}")

        # Initialize the process asynchronously
        cli_logger.info("Starting process initialization...")
        start_time = time.time()

        try:
            # Use start_sync() to create a process with a persistent event loop
            process = program.start_sync()

            init_time = time.time() - start_time
            cli_logger.info(f"Process initialized in {init_time:.2f} seconds")
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
                    "3. If you're using npx to run an MCP server, make sure the package exists and is accessible",
                    err=True,
                )
                click.echo(
                    "4. To run without requiring MCP tools: export LLMPROC_FAIL_ON_MCP_INIT_TIMEOUT=false", err=True
                )
                sys.exit(2)
            else:
                raise

        # Create callback handler and register it with the process
        callback_handler = CliCallbackHandler(cli_logger)
        process.add_callback(callback_handler)

        # Use the helper function to run prompts
        def run_prompt_func(user_prompt: str, source: str = "command line") -> RunResult:
            if not getattr(process, "state", []):
                log_program_info(process, user_prompt, cli_logger)
            return run_with_prompt(process, user_prompt, source, cli_logger, callback_handler, quiet_mode)

        # Priority order:
        # 1. Demo mode (if configured)
        # 2. User prompt from TOML
        # 3. Interactive mode

        # Check for demo mode configuration in the program's TOML
        if check_and_run_demo_mode(program, run_prompt_func, quiet_mode, cli_logger):
            return  # Exit after demo is complete

        # Handle prompt logic: custom prompt (-p), embedded prompt, or skip to interactive
        if prompt is not None:
            # Custom prompt provided via -p flag
            if prompt:  # Non-empty string
                run_prompt_func(prompt, source="command line")
            # else: empty string means skip to interactive mode
        else:
            embedded_prompt = getattr(process, "user_prompt", "")
            if isinstance(embedded_prompt, str) and embedded_prompt:
                # Show embedded user prompt and ask user if they want to run it
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
                    # In quiet mode, just run the embedded prompt
                    run_prompt_func(embedded_prompt, source="embedded")

        # Interactive mode (always reached now unless demo mode exited)
        if not quiet_mode:
            click.echo("\nStarting interactive chat session. Type 'exit' or 'quit' to end.")

        # Show initial token count if supported (unless in quiet mode)
        if not quiet_mode:
            try:
                # Count tokens in the current context - SyncLLMProcess.count_tokens is already synchronous
                token_info = process.count_tokens()
                if token_info and "input_tokens" in token_info:
                    click.echo(
                        f"Initial context size: {token_info['input_tokens']:,} tokens ({token_info['percentage']:.1f}% of {token_info['context_window']:,} token context window)"
                    )
            except Exception as e:
                cli_logger.warning(f"Failed to count initial tokens: {str(e)}")

        while True:
            # Display token usage if available from the count_tokens method (unless in quiet mode)
            token_display = ""
            if not quiet_mode:
                try:
                    # Count tokens for display - SyncLLMProcess.count_tokens is already synchronous
                    token_info = process.count_tokens()
                    if token_info and "input_tokens" in token_info:
                        token_display = f" [Tokens: {token_info['input_tokens']:,}/{token_info['context_window']:,}]"
                except Exception as e:
                    cli_logger.warning(f"Failed to count tokens for prompt: {str(e)}")

            user_input = click.prompt(f"\nYou{token_display}", prompt_suffix="> ")

            if user_input.lower() in ("exit", "quit"):
                if not quiet_mode:
                    click.echo("Ending session.")
                break

            # Track time for this run
            start_time = time.time()

            # Show a spinner while running (unless in quiet mode)
            if not quiet_mode:
                click.echo("Thinking...", nl=False)

            if not getattr(process, "state", []):
                log_program_info(process, user_input, cli_logger)
            # Run the process without passing callbacks (already registered above)
            run_result = process.run(user_input)

            # Get the elapsed time
            elapsed = time.time() - start_time

            # Clear the "Thinking..." text (unless in quiet mode)
            if not quiet_mode:
                click.echo("\r" + " " * 12 + "\r", nl=False)

            # Token counting is now handled before each prompt using the count_tokens method

            # Log basic info
            if run_result.api_calls > 0:
                cli_logger.info(
                    f"Used {run_result.api_calls} API calls and {run_result.tool_calls} tool calls in {elapsed:.2f}s"
                )

            # Get the last assistant message
            response = process.get_last_message()

            # Display the response (with or without model name based on quiet mode)
            if quiet_mode:
                click.echo(f"\n{response}")
            else:
                display_name = process.display_name or process.model_name
                click.echo(f"\n{display_name}> {response}")

            # Display token usage stats if available (unless in quiet mode)
            if not quiet_mode and hasattr(run_result, "total_tokens") and run_result.total_tokens > 0:
                click.echo(
                    f"[API calls: {run_result.api_calls}, "
                    f"Tool calls: {run_result.tool_calls}, "
                    f"Tokens: {run_result.input_tokens}/{run_result.output_tokens}/{run_result.total_tokens} (in/out/total)]"
                )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        # Only show full traceback if not in quiet mode
        if not quiet_mode:
            click.echo("\nFull traceback:", err=True)
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure we properly clean up resources with SyncLLMProcess.close()
        if "process" in locals() and hasattr(process, "close"):
            try:
                process.close()
                cli_logger.debug("Process resources cleaned up")
            except Exception as e:
                cli_logger.warning(f"Error during process cleanup: {e}")


if __name__ == "__main__":
    main()
