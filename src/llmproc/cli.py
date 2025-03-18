#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version."""

import asyncio
import sys
from pathlib import Path

import click

from llmproc import LLMProcess


@click.command()
@click.argument("program_path", required=False)
def main(program_path=None) -> None:
    """Run a simple interactive CLI for LLMProc.

    PROGRAM_PATH is an optional path to a TOML program file.
    If not provided, you'll be prompted to select from available examples.
    """
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
            click.echo(
                "Error: No TOML program files found in examples directory."
            )
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
        process = LLMProcess.from_toml(selected_program_abs)
        click.echo(f"\nLoaded program from: {selected_program_abs}")

        # Display program summary
        click.echo("\nProgram Summary:")
        # Try to extract and show key info from the program
        try:
            import tomli

            with open(selected_program_abs, "rb") as f:
                program_config = tomli.load(f)

            # Show model info
            if "model" in program_config:
                model_info = program_config["model"]
                # Show display name if available
                if "display_name" in model_info:
                    click.echo(f"  Display Name: {model_info['display_name']}")
                click.echo(f"  Model: {model_info.get('name', 'Not specified')}")
                click.echo(f"  Provider: {model_info.get('provider', 'Not specified')}")

            # Show brief system prompt summary
            if "prompt" in program_config and "system_prompt" in program_config["prompt"]:
                system_prompt = program_config["prompt"]["system_prompt"]
                # Truncate if too long
                if len(system_prompt) > 60:
                    system_prompt = system_prompt[:57] + "..."
                click.echo(f"  System Prompt: {system_prompt}")

            # Show a few key parameters if present
            if "parameters" in program_config:
                params = program_config["parameters"]
                if "temperature" in params:
                    click.echo(f"  Temperature: {params['temperature']}")
                if "max_tokens" in params:
                    click.echo(f"  Max Tokens: {params['max_tokens']}")
        except Exception:
            click.echo("  (Could not parse program details)")

        # Start interactive session
        click.echo("\nStarting interactive chat session. Type 'exit' or 'quit' to end.")
        click.echo("Type 'reset' to reset the conversation state.")

        while True:
            user_input = click.prompt("\nYou", prompt_suffix="> ")

            if user_input.lower() in ("exit", "quit"):
                click.echo("Ending session.")
                break

            if user_input.lower() == "reset":
                process.reset_state()
                click.echo("Conversation state has been reset.")
                continue

            # Run the process with async support
            api_calls = asyncio.run(process.run(user_input))
            
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
