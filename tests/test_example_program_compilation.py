"""Tests compilation of all example programs."""

import os
from pathlib import Path

import pytest

from llmproc.program import LLMProgram


def test_compile_all_example_programs():
    """Test that all example programs compile successfully."""
    # Get the examples directory
    examples_dir = Path(__file__).parent.parent / "examples"

    # Function to collect all TOML files
    def collect_toml_files(directory):
        toml_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".toml"):
                    toml_files.append(Path(root) / file)
        return toml_files

    # Collect all TOML files in the examples directory
    toml_files = collect_toml_files(examples_dir)

    # Make sure we found some files
    assert len(toml_files) > 0, f"No TOML files found in {examples_dir}"

    # Keep track of files that fail to compile
    failed_files = []
    success_count = 0

    # Known files that still require external resources or have special syntax
    skip_files = [
        # "claude-code.toml",  # Uses a complex linked_programs syntax
        # "main.toml",  # Uses a complex linked_programs syntax in program-linking folder
    ]

    # Try to compile each file
    for toml_file in toml_files:
        # Skip files known to require external resources
        if toml_file.name in skip_files:
            print(f"Skipping {toml_file.name} (requires external resources)")
            continue

        try:
            # Load without requiring real linked program files
            # and don't include linked programs to avoid API calls
            program = LLMProgram.from_toml(toml_file, include_linked=False)

            # Verify it's a valid program
            assert program.model_name, f"No model name in {toml_file}"
            assert program.provider, f"No provider in {toml_file}"

            # Track success
            success_count += 1

            # Check for linked program descriptions if applicable
            if hasattr(program, "linked_programs") and program.linked_programs:
                assert hasattr(program, "linked_program_descriptions"), (
                    f"Missing linked_program_descriptions in {toml_file}"
                )

                # Print out programs with descriptions for debugging
                descriptions_count = sum(
                    1 for desc in program.linked_program_descriptions.values() if desc
                )
                if descriptions_count > 0:
                    print(f"Found {descriptions_count} descriptions in {toml_file}")

        except Exception as e:
            failed_files.append((toml_file, str(e)))

    # Report results
    if failed_files:
        for file, error in failed_files:
            print(f"Failed to compile {file}: {error}")

        pytest.fail(
            f"Failed to compile {len(failed_files)} out of {len(toml_files)} files"
        )
    else:
        print(f"Successfully compiled all {success_count} TOML files")


if __name__ == "__main__":
    test_compile_all_example_programs()
