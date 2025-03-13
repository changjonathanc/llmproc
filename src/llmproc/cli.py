#!/usr/bin/env python3
"""Command-line interface for LLMProc - Demo version."""

import os
import sys
from pathlib import Path

import click

from llmproc import LLMProcess


@click.command()
@click.argument("config_path", required=False)
def main(config_path=None) -> None:
    """Run a simple interactive CLI for LLMProc.
    
    CONFIG_PATH is an optional path to a TOML configuration file.
    If not provided, you'll be prompted to select from available examples.
    """
    click.echo("LLMProc CLI Demo")
    click.echo("----------------")
    
    # If config path is provided, use it
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            click.echo(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        if config_file.suffix != ".toml":
            click.echo(f"Error: Config file must be a TOML file: {config_path}")
            sys.exit(1)
        selected_config = config_file
    # Otherwise, prompt user to select from examples
    else:
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
        selected_config_abs = selected_config.absolute()
        process = LLMProcess.from_toml(selected_config_abs)
        click.echo(f"\nLoaded configuration from: {selected_config_abs}")
        
        # Display config summary
        click.echo("\nConfiguration Summary:")
        # Try to extract and show key info from the config
        try:
            import tomli
            with open(selected_config_abs, "rb") as f:
                config = tomli.load(f)
            
            # Show model info
            if "model" in config:
                model_info = config["model"]
                click.echo(f"  Model: {model_info.get('name', 'Not specified')}")
                click.echo(f"  Provider: {model_info.get('provider', 'Not specified')}")
            
            # Show brief system prompt summary
            if "prompt" in config and "system_prompt" in config["prompt"]:
                system_prompt = config["prompt"]["system_prompt"]
                # Truncate if too long
                if len(system_prompt) > 60:
                    system_prompt = system_prompt[:57] + "..."
                click.echo(f"  System Prompt: {system_prompt}")
                
            # Show a few key parameters if present
            if "parameters" in config:
                params = config["parameters"]
                if "temperature" in params:
                    click.echo(f"  Temperature: {params['temperature']}")
                if "max_tokens" in params:
                    click.echo(f"  Max Tokens: {params['max_tokens']}")
        except:
            click.echo("  (Could not parse configuration details)")
            
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