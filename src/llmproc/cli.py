#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version."""

import asyncio
import logging
import sys
import time
from pathlib import Path

import click

from llmproc import LLMProgram

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llmproc.cli")


@click.command()
@click.argument("program_path", required=False)
@click.option(
    "--prompt", "-p", help="Run in non-interactive mode with the given prompt"
)
@click.option(
    "--non-interactive",
    "-n",
    is_flag=True,
    help="Run in non-interactive mode (reads from stdin if no prompt provided)",
)
def main(program_path=None, prompt=None, non_interactive=False) -> None:
    """Run a simple CLI for LLMProc.

    PROGRAM_PATH is an optional path to a TOML program file.
    If not provided, you'll be prompted to select from available examples.

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

    # If program path is provided, use it
    if program_path:
        program_file = Path(program_path)
        if not program_file.exists():
            click.echo(f"Error: Program file not found: {program_path}")
            sys.exit(1)
        if program_file.suffix != ".toml":
            click.echo(f"Error: Program file must be a TOML file: {program_path}")
            sys.exit(1)
        selected_program = program_file
    # Otherwise, prompt user to select from examples
    else:
        # Find available program files
        program_dir = Path("./examples")
        if not program_dir.exists():
            click.echo("Error: examples directory not found.")
            sys.exit(1)

        program_files = list(program_dir.glob("*.toml"))
        if not program_files:
            click.echo("Error: No TOML program files found in examples directory.")
            sys.exit(1)

        # Display available programs
        click.echo("\nAvailable programs:")
        for i, program_file in enumerate(sorted(program_files), 1):
            click.echo(f"{i}. {program_file.name}")

        # Let user select a program
        selection = click.prompt("\nSelect a program (number)", type=int)

        if selection < 1 or selection > len(program_files):
            click.echo("Invalid selection.")
            sys.exit(1)

        selected_program = sorted(program_files)[selection - 1]

    # Load the selected program
    try:
        selected_program_abs = selected_program.absolute()

        # Use the new API: load program first, then start it asynchronously
        logger.info(f"Loading program from: {selected_program_abs}")
        program = LLMProgram.from_toml(selected_program_abs)

        # Display program summary
        click.echo("\nProgram Summary:")
        click.echo(f"  Model: {program.model_name}")
        click.echo(f"  Provider: {program.provider}")
        click.echo(f"  Display Name: {program.display_name or program.model_name}")

        # Show brief system prompt summary
        if hasattr(program, "system_prompt") and program.system_prompt:
            system_prompt = program.system_prompt
            # Truncate if too long
            if len(system_prompt) > 60:
                system_prompt = system_prompt[:57] + "..."
            click.echo(f"  System Prompt: {system_prompt}")

        # Show parameter summary if available
        if hasattr(program, "api_params") and program.api_params:
            params = program.api_params
            if "temperature" in params:
                click.echo(f"  Temperature: {params['temperature']}")
            if "max_tokens" in params:
                click.echo(f"  Max Tokens: {params['max_tokens']}")

        # Initialize the process asynchronously
        logger.info("Starting process initialization...")
        start_time = time.time()
        process = asyncio.run(program.start())
        init_time = time.time() - start_time
        logger.info(f"Process initialized in {init_time:.2f} seconds")

        # Set up callbacks for real-time updates
        callbacks = {
            "on_tool_start": lambda tool_name, args: logger.info(
                f"Using tool: {tool_name}"
            ),
            "on_tool_end": lambda tool_name, result: logger.info(
                f"Tool {tool_name} completed"
            ),
            "on_response": lambda content: logger.info(
                f"Received response: {content[:50]}..."
            ),
        }

        # Check if we're in non-interactive mode
        if prompt or non_interactive:
            # Non-interactive mode with single prompt
            user_prompt = prompt

            # If no prompt is provided but non-interactive flag is set, read from stdin
            if not user_prompt and non_interactive:
                if not sys.stdin.isatty():  # Check if input is being piped in
                    user_prompt = sys.stdin.read().strip()
                else:
                    click.echo(
                        "Error: No prompt provided for non-interactive mode. Use --prompt or pipe input.",
                        err=True,
                    )
                    sys.exit(1)

            logger.info("Running in non-interactive mode with single prompt")

            # Track time for this run
            start_time = time.time()

            # Run the process with the provided prompt
            run_result = asyncio.run(process.run(user_prompt, callbacks=callbacks))

            # Get the elapsed time
            elapsed = time.time() - start_time

            # Log run result information
            logger.info(f"Used {run_result.api_calls} API calls and {run_result.tool_calls} tool calls in {elapsed:.2f}s")

            # Get the last assistant message and just print the raw response
            response = process.get_last_message()
            click.echo(response)

        else:
            # Interactive mode
            click.echo(
                "\nStarting interactive chat session. Type 'exit' or 'quit' to end."
            )
            click.echo("Type 'reset' to reset the conversation state.")
            click.echo("Type 'verbose' to toggle verbose logging.")

            # Toggle for verbose logging
            verbose = False

            while True:
                user_input = click.prompt("\nYou", prompt_suffix="> ")

                if user_input.lower() in ("exit", "quit"):
                    click.echo("Ending session.")
                    break

                if user_input.lower() == "reset":
                    process.reset_state()
                    click.echo("Conversation state has been reset.")
                    continue

                if user_input.lower() == "verbose":
                    verbose = not verbose
                    level = logging.DEBUG if verbose else logging.INFO
                    logging.getLogger("llmproc").setLevel(level)
                    click.echo(
                        f"Verbose logging {'enabled' if verbose else 'disabled'}"
                    )
                    continue

                # Track time for this run
                start_time = time.time()

                # Show a spinner while running (to be implemented)
                click.echo("Thinking...", nl=False)

                # Run the process with the new API
                run_result = asyncio.run(process.run(user_input, callbacks=callbacks))

                # Get the elapsed time
                elapsed = time.time() - start_time

                # Clear the "Thinking..." text
                click.echo("\r" + " " * 12 + "\r", nl=False)

                # Log run result information
                if verbose:
                    logger.debug(
                        f"Run completed in {elapsed:.2f}s with {run_result.api_calls} API calls and {run_result.tool_calls} tool calls"
                    )
                    # Log API calls
                    for i, api_info in enumerate(run_result.api_call_infos):
                        if "model" in api_info:
                            logger.debug(f"API call {i + 1}: model={api_info['model']}")
                    
                    # Log tool calls
                    for i, tool_info in enumerate(run_result.tool_call_infos):
                        logger.debug(
                            f"Tool call {i + 1}: {tool_info.get('tool_name', 'unknown')}"
                        )
                else:
                    if run_result.api_calls > 0:
                        logger.info(
                            f"Used {run_result.api_calls} API calls and {run_result.tool_calls} tool calls in {elapsed:.2f}s"
                        )

                # Get the last assistant message
                response = process.get_last_message()

                # Display the response
                click.echo(f"\n{process.display_name}> {response}")

    except Exception as e:
        import traceback

        click.echo(f"Error: {str(e)}", err=True)
        click.echo("\nFull traceback:", err=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
