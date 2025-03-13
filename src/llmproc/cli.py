#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version."""

import os
import sys
from pathlib import Path

import click

from llmproc import LLMProcess


def main() -> None:
    """Run a simple interactive CLI for LLMProc."""
    click.echo("LLMProc CLI Demo")
    click.echo("----------------")
    
    # Find available config files
    config_dir = Path("./examples")
    if not config_dir.exists():
        click.echo("Error: examples directory not found.")
        sys.exit(1)
        
    config_files = list(config_dir.glob("*.toml"))
    if not config_files:
        click.echo("Error: No TOML configuration files found in examples directory.")
        sys.exit(1)
    
    # Display available configs
    click.echo("\nAvailable configurations:")
    for i, config_file in enumerate(sorted(config_files), 1):
        click.echo(f"{i}. {config_file.name}")
    
    # Let user select a config
    selection = click.prompt("\nSelect a configuration (number)", type=int)
    
    if selection < 1 or selection > len(config_files):
        click.echo("Invalid selection.")
        sys.exit(1)
    
    selected_config = sorted(config_files)[selection-1]
    
    # Load the selected configuration
    try:
        process = LLMProcess.from_toml(selected_config)
        click.echo(f"\nLoaded configuration from {selected_config}")
        
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
                
            response = process.run(user_input)
            click.echo(f"\nAI> {response}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()