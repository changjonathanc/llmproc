"""Print System Prompt CLI Tool.

This tool allows users to print the enriched system prompt that would be sent to the LLM
based on their program configuration, without actually making an API call.
"""

import argparse
import sys
from pathlib import Path
from typing import TextIO

from llmproc.program import LLMProgram


def print_system_prompt(
    program_path: str,
    output_file: TextIO | None = None,
    include_env: bool = True,
    color: bool = True,
) -> None:
    """Print the enriched system prompt for a program.

    Args:
        program_path: Path to the program TOML file
        output_file: File to write output to (default: stdout)
        include_env: Whether to include environment info
        color: Whether to colorize the output

    Raises:
        FileNotFoundError: If the program file doesn't exist
        ValueError: If the program file is invalid
    """
    # If output_file is None, use stdout
    output_file = output_file or sys.stdout

    try:
        # Load the program file
        program = LLMProgram.from_toml(program_path)

        # Create a simple process-like object to pass the necessary flags
        process_info = type("ProcessInfo", (), {})()

        # Set file descriptor and reference flags based on program configuration
        fd_enabled = False
        ref_enabled = False
        preloaded_content = {}

        # Check if file descriptor is enabled either explicitly or by having read_fd in tools
        if hasattr(program, "file_descriptor") and program.file_descriptor:
            fd_config = program.file_descriptor
            # Use dictionary get method instead of getattr
            fd_enabled = fd_config.get("enabled", False)
            ref_enabled = fd_config.get("enable_references", False)
            page_user_input = fd_config.get("page_user_input", False)

        if not fd_enabled and hasattr(program, "tools") and program.tools:
            tools = program.tools.get("enabled", [])
            if "read_fd" in tools:
                fd_enabled = True

        # Handle preloaded files
        if hasattr(program, "preload_files") and program.preload_files:
            # Simple mock preloading - just use file names
            for file_path in program.preload_files:
                path = Path(file_path)
                if path.exists():
                    try:
                        preloaded_content[str(path)] = path.read_text()
                    except Exception:
                        # Just ignore errors for the preview tool
                        pass

        # Set the flags and content on the process info object
        process_info.file_descriptor_enabled = fd_enabled
        process_info.references_enabled = ref_enabled
        process_info.preloaded_content = preloaded_content

        # Create a mock fd_manager to enable user input paging
        if fd_enabled:
            fd_manager = type("MockFDManager", (), {})()
            fd_manager.page_user_input = page_user_input
            process_info.fd_manager = fd_manager

        # Get the enriched system prompt with process info
        enriched_prompt = program.get_enriched_system_prompt(process_instance=process_info, include_env=include_env)

        # Print header
        output_file.write("\n===== ENRICHED SYSTEM PROMPT =====\n\n")

        # Print the enriched system prompt, optionally with color
        if color and output_file.isatty():
            # Colorize different sections
            lines = enriched_prompt.split("\n")
            in_section = False
            section_name = ""

            for line in lines:
                # Detect start of XML section
                if line.startswith("<") and ">" in line and not line.startswith("</"):
                    tag = line.split(">")[0] + ">"
                    in_section = True
                    section_name = tag
                    # Print section header in cyan
                    output_file.write(f"\033[36m{line}\033[0m\n")
                # Detect end of XML section
                elif line.startswith("</") and ">" in line:
                    in_section = False
                    # Print section footer in cyan
                    output_file.write(f"\033[36m{line}\033[0m\n")
                # Print section content with appropriate color
                elif in_section:
                    if section_name == "<env>":
                        # Environment info in green
                        output_file.write(f"\033[32m{line}\033[0m\n")
                    elif section_name == "<preload>":
                        # Preloaded content in yellow
                        output_file.write(f"\033[33m{line}\033[0m\n")
                    elif section_name == "<file_descriptor_instructions>":
                        # File descriptor instructions in magenta
                        output_file.write(f"\033[35m{line}\033[0m\n")
                    elif section_name == "<fd_user_input_instructions>":
                        # User input instructions in bright magenta
                        output_file.write(f"\033[95m{line}\033[0m\n")
                    elif section_name == "<reference_instructions>":
                        # Reference instructions in blue
                        output_file.write(f"\033[34m{line}\033[0m\n")
                    else:
                        # Other sections in normal color
                        output_file.write(f"{line}\n")
                else:
                    # Base system prompt in normal color
                    output_file.write(f"{line}\n")
        else:
            # Print without color
            output_file.write(enriched_prompt + "\n")

        # Print a summary of the sections found
        output_file.write("\n===== SECTIONS SUMMARY =====\n\n")

        summary = []

        # Base system prompt is always there
        summary.append("- Base System Prompt ✅")

        # Check for environment information
        if "<env>" in enriched_prompt:
            summary.append("- Environment Information ✅")
        else:
            summary.append("- Environment Information ❌")

        # Check for preloaded files
        if "<preload>" in enriched_prompt:
            summary.append("- Preloaded Files ✅")
        else:
            summary.append("- Preloaded Files ❌")

        # Check for file descriptor instructions
        if "<file_descriptor_instructions>" in enriched_prompt:
            summary.append("- File Descriptor Instructions ✅")
        else:
            summary.append("- File Descriptor Instructions ❌")

        # Check for FD user input instructions
        if "<fd_user_input_instructions>" in enriched_prompt:
            summary.append("- FD User Input Paging Instructions ✅")
        else:
            summary.append("- FD User Input Paging Instructions ❌")

        # Check for reference instructions
        if "<reference_instructions>" in enriched_prompt:
            summary.append("- Reference ID Instructions ✅")
        else:
            summary.append("- Reference ID Instructions ❌")

        # Print the summary
        for item in summary:
            output_file.write(item + "\n")

        # Print program configuration
        output_file.write("\n===== PROGRAM CONFIGURATION =====\n\n")

        # Print some key program attributes
        output_file.write(f"Model: {program.model_name}\n")
        output_file.write(f"Provider: {program.provider}\n")

        if hasattr(program, "display_name") and program.display_name:
            output_file.write(f"Display Name: {program.display_name}\n")

        # Print file descriptor configuration if available
        if hasattr(program, "file_descriptor") and program.file_descriptor:
            fd_config = program.file_descriptor
            output_file.write("\nFile Descriptor Configuration:\n")
            output_file.write(f"  Enabled: {fd_config.get('enabled', False)}\n")
            output_file.write(f"  Max Direct Output Chars: {fd_config.get('max_direct_output_chars', 8000)}\n")
            output_file.write(f"  Default Page Size: {fd_config.get('default_page_size', 4000)}\n")
            output_file.write(f"  Max Input Chars: {fd_config.get('max_input_chars', 8000)}\n")
            output_file.write(f"  Page User Input: {fd_config.get('page_user_input', True)}\n")
            output_file.write(f"  Enable References: {fd_config.get('enable_references', False)}\n")

        # Print tools configuration if available
        if hasattr(program, "tools") and program.tools:
            tools = program.tools.get("enabled", [])
            if tools:
                output_file.write("\nEnabled Tools:\n")
                for tool in tools:
                    output_file.write(f"  - {tool}\n")

        # Print a footer
        output_file.write("\n============================\n")

    except FileNotFoundError:
        output_file.write(f"Error: Program file {program_path} not found\n")
        sys.exit(1)
    except ValueError as e:
        output_file.write(f"Error: {str(e)}\n")
        sys.exit(1)
    except Exception as e:
        output_file.write(f"Unexpected error: {str(e)}\n")
        sys.exit(1)


def main():
    """Main entry point for the CLI tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Print the enriched system prompt for a program")
    parser.add_argument(
        "program_path",
        help="Path to the program TOML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="File to write output to (default: stdout)",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )
    parser.add_argument(
        "--no-env",
        "-E",
        help="Don't include environment information",
        action="store_true",
    )
    parser.add_argument(
        "--no-color",
        "-C",
        help="Don't colorize the output",
        action="store_true",
    )

    args = parser.parse_args()

    # Call the main function
    print_system_prompt(
        args.program_path,
        args.output,
        include_env=not args.no_env,
        color=not args.no_color,
    )


if __name__ == "__main__":
    main()
