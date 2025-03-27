"""Test script to verify that print_system_prompt correctly shows all file descriptor features."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from llmproc.tools.print_system_prompt import print_system_prompt


def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test print_system_prompt tool")
    parser.add_argument(
        "program_path",
        nargs="?",
        default="./examples/file_descriptor/all_features.toml",
        help="Path to the program TOML file",
    )
    args = parser.parse_args()
    
    # Run the tool with no color for cleaner output
    print_system_prompt(args.program_path, color=False)


if __name__ == "__main__":
    main()