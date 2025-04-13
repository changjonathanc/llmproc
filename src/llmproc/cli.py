#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version."""

import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path

import click

from llmproc import LLMProgram
from llmproc.providers.constants import ANTHROPIC_PROVIDERS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llmproc.cli")


def get_logger(quiet=False):
    """Get a logger with the appropriate level based on quiet mode."""
    if quiet:
        # Set llmproc.cli logger to ERROR level in quiet mode
        logger.setLevel(logging.ERROR)

        # Also reduce logging from httpx and llmproc modules
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("llmproc.llm_process").setLevel(logging.ERROR)

        # Suppress Pydantic warnings
        import warnings

        warnings.filterwarnings("ignore", module="pydantic")
    return logger


@click.command()
@click.argument("program_path", required=True)
@click.option(
    "--prompt", "-p", help="Run in non-interactive mode with the given prompt"
)
@click.option(
    "--non-interactive",
    "-n",
    is_flag=True,
    help="Run in non-interactive mode (reads from stdin if no prompt provided)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Reduce logging output and callbacks",
)
def main(program_path, prompt=None, non_interactive=False, quiet=False) -> None:
    """Run a simple CLI for LLMProc.

    PROGRAM_PATH is the path to a TOML program file.

    Supports three modes:
    1. Interactive mode (default): Chat continuously with the model
    2. Non-interactive with prompt: Use --prompt/-p "your prompt here"
    3. Non-interactive with stdin: Use --non-interactive/-n and pipe input
    """
    # Only show header in interactive mode or if verbose logging is enabled
    if (
        not (prompt or non_interactive)
        or logging.getLogger("llmproc").level == logging.DEBUG
    ):
        click.echo("LLMProc CLI Demo")
        click.echo("----------------")

    # Validate program path
    program_file = Path(program_path)
    if not program_file.exists():
        click.echo(f"Error: Program file not found: {program_path}")
        sys.exit(1)
    if program_file.suffix != ".toml":
        click.echo(f"Error: Program file must be a TOML file: {program_path}")
        sys.exit(1)
    selected_program = program_file

    # Load the selected program
    try:
        selected_program_abs = selected_program.absolute()

        # Get logger with appropriate level
        cli_logger = get_logger(quiet)

        # Use the new API: load program first, then start it asynchronously
        cli_logger.info(f"Loading program from: {selected_program_abs}")
        program = LLMProgram.from_toml(selected_program_abs)

        # Display program summary unless in quiet mode
        if not quiet:
            click.echo("\nProgram Summary:")
            click.echo(f"  Model: {program.model_name}")
            click.echo(f"  Provider: {program.provider}")
            click.echo(f"  Display Name: {program.display_name or program.model_name}")

        # Initial token count will be displayed after initialization

        # Show brief system prompt summary unless in quiet mode
        if not quiet and hasattr(program, "system_prompt") and program.system_prompt:
            system_prompt = program.system_prompt
            # Truncate if too long
            if len(system_prompt) > 60:
                system_prompt = system_prompt[:57] + "..."
            click.echo(f"  System Prompt: {system_prompt}")

        # Show parameter summary if available and not in quiet mode
        if not quiet and hasattr(program, "api_params") and program.api_params:
            params = program.api_params
            if "temperature" in params:
                click.echo(f"  Temperature: {params['temperature']}")
            if "max_tokens" in params:
                click.echo(f"  Max Tokens: {params['max_tokens']}")

        # Initialize the process asynchronously
        cli_logger.info("Starting process initialization...")
        start_time = time.time()
        process = asyncio.run(program.start())
        init_time = time.time() - start_time
        cli_logger.info(f"Process initialized in {init_time:.2f} seconds")

        # Set up callbacks for real-time updates - minimal in quiet mode
        if quiet:
            # Minimal callbacks in quiet mode
            callbacks = {}
        else:
            callbacks = {
                "on_tool_start": lambda tool_name, args: cli_logger.info(
                    f"Using tool: {tool_name}"
                ),
                "on_tool_end": lambda tool_name, result: cli_logger.info(
                    f"Tool {tool_name} completed"
                ),
                "on_response": lambda content: cli_logger.info(
                    f"Received response: {content[:50]}..."
                ),
            }

        # Check if we're in non-interactive mode
        # Important: prompt='' is considered non-interactive with empty prompt
        if prompt is not None or non_interactive:
            # Non-interactive mode with single prompt (empty prompt is valid)
            user_prompt = prompt if prompt is not None else ""

            # If no prompt is provided and non-interactive flag is set, read from stdin
            if prompt is None and non_interactive:
                if not sys.stdin.isatty():  # Check if input is being piped in
                    user_prompt = sys.stdin.read().strip()
                else:
                    click.echo(
                        "Error: No prompt provided for non-interactive mode. Use --prompt or pipe input.",
                        err=True,
                    )
                    sys.exit(1)

            # Report mode
            cli_logger.info("Running in non-interactive mode with single prompt")

            # Track time for this run
            start_time = time.time()

            # Exit with error on empty prompts
            if user_prompt.strip() == "":
                click.echo(
                    "Error: Empty prompt provided. Please provide a non-empty prompt with --prompt.",
                    err=True,
                )
                sys.exit(1)

            # Run the process with the processed prompt
            run_result = asyncio.run(process.run(user_prompt, callbacks=callbacks))

            # Get the elapsed time
            elapsed = time.time() - start_time

            # Log run result information
            cli_logger.info(
                f"Used {run_result.api_calls} API calls and {run_result.tool_calls} tool calls in {elapsed:.2f}s"
            )

            # Get the last assistant message and just print the raw response
            response = process.get_last_message()
            click.echo(response)

        else:
            # Interactive mode
            if not quiet:
                click.echo(
                    "\nStarting interactive chat session. Type 'exit' or 'quit' to end."
                )

            # Show initial token count for Anthropic models (unless in quiet mode)
            if not quiet and process.provider in ANTHROPIC_PROVIDERS:
                try:
                    token_info = asyncio.run(process.count_tokens())
                    if token_info and "input_tokens" in token_info:
                        click.echo(
                            f"Initial context size: {token_info['input_tokens']:,} tokens ({token_info['percentage']:.1f}% of {token_info['context_window']:,} token context window)"
                        )
                except Exception as e:
                    cli_logger.warning(f"Failed to count initial tokens: {str(e)}")

            while True:
                # Display token usage if available from the count_tokens method (unless in quiet mode)
                token_display = ""
                if not quiet and process.provider in ANTHROPIC_PROVIDERS:
                    try:
                        token_info = asyncio.run(process.count_tokens())
                        if token_info and "input_tokens" in token_info:
                            token_display = f" [Tokens: {token_info['input_tokens']:,}/{token_info['context_window']:,}]"
                    except Exception as e:
                        cli_logger.warning(
                            f"Failed to count tokens for prompt: {str(e)}"
                        )

                user_input = click.prompt(f"\nYou{token_display}", prompt_suffix="> ")

                if user_input.lower() in ("exit", "quit"):
                    if not quiet:
                        click.echo("Ending session.")
                    break

                # Track time for this run
                start_time = time.time()

                # Show a spinner while running (unless in quiet mode)
                if not quiet:
                    click.echo("Thinking...", nl=False)

                # Run the process with the new API
                run_result = asyncio.run(process.run(user_input, callbacks=callbacks))

                # Get the elapsed time
                elapsed = time.time() - start_time

                # Clear the "Thinking..." text (unless in quiet mode)
                if not quiet:
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
                if quiet:
                    click.echo(f"\n{response}")
                else:
                    click.echo(f"\n{process.display_name}> {response}")

                # Display token usage stats if available (unless in quiet mode)
                if (
                    not quiet
                    and hasattr(run_result, "total_tokens")
                    and run_result.total_tokens > 0
                ):
                    click.echo(
                        f"[API calls: {run_result.api_calls}, "
                        f"Tool calls: {run_result.tool_calls}, "
                        f"Tokens: {run_result.input_tokens}/{run_result.output_tokens}/{run_result.total_tokens} (in/out/total)]"
                    )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        # Only show full traceback if not in quiet mode
        if not quiet:
            click.echo("\nFull traceback:", err=True)
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
