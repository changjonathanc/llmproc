#!/usr/bin/env python
"""Script to run API tests with parallelism and optimizations.

This script runs API tests using the tiered testing strategy:
- essential_api: Essential tests for CI/CD and daily development (fastest)
- extended_api: Extended coverage for regular validation
- release_api: Comprehensive coverage for pre-release testing

Usage:
  python run_api_tests.py --tier essential  # Run essential tests
  python run_api_tests.py --tier extended   # Run extended tests
  python run_api_tests.py --tier release    # Run release tests
  python run_api_tests.py --tier all        # Run all tests

Options:
  --tier TIER       Test tier to run (essential, extended, release, all)
  --workers N       Number of parallel workers (default: 2)
  --verbose         Enable verbose output
  --provider PROV   Only run tests for specific provider (anthropic, openai, vertex)
  --coverage        Generate coverage report
"""

import argparse
import os
import subprocess
import sys
import time


def run_api_tests(args):
    """Run API tests for the specified tier."""
    tier = args.tier
    print(f"Running {tier} API tests with optimizations...")

    # Base command - both tier marker and llm_api are required
    cmd = [
        "python",
        "-m",
        "pytest",
    ]

    # Add test markers based on tier
    if tier == "all":
        cmd.extend(["-m", "llm_api", "--run-api-tests"])
    else:
        cmd.extend(["-m", f"llm_api and {tier}_api", "--run-api-tests"])

    # Provider-specific filtering
    if args.provider:
        cmd[3] = f"{cmd[3]} and {args.provider}_api"

    # Parallelism
    cmd.extend([f"-n{args.workers}", "--no-cov"])

    # Show test name and error message on failure, but still keep output minimal
    cmd.extend(["-v", "--showlocals"])

    # Enhanced verbosity if requested
    if args.verbose:
        cmd.extend(["-vv", "--tb=native"])

    # Coverage report if requested
    if args.coverage:
        cmd.extend(["--cov=src/llmproc", "--cov-report=term"])

    # Print command for debugging
    print(f"Running command: {' '.join(cmd)}")

    # Record start time
    start_time = time.time()

    # Execute the command
    result = subprocess.run(cmd)

    # Report elapsed time
    elapsed = time.time() - start_time
    print(f"Tests completed in {elapsed:.2f} seconds with exit code {result.returncode}")

    return result.returncode


def check_api_keys():
    """Check if API keys are available in the environment."""
    api_keys = {
        "Anthropic": os.environ.get("ANTHROPIC_API_KEY"),
        "OpenAI": os.environ.get("OPENAI_API_KEY"),
        "Vertex AI": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    }

    print("API key availability:")
    for provider, key in api_keys.items():
        status = "✓ Available" if key else "✗ Missing"
        print(f"  {provider}: {status}")

    if not any(api_keys.values()):
        print("\nNo API keys found. Set at least one of these environment variables:")
        print("  - ANTHROPIC_API_KEY (for Anthropic tests)")
        print("  - OPENAI_API_KEY (for OpenAI tests)")
        print("  - GOOGLE_APPLICATION_CREDENTIALS (for Vertex AI tests)")
        return False

    return True


def main():
    """Parse arguments and run API tests."""
    parser = argparse.ArgumentParser(description="Run API tests with optimizations.")
    parser.add_argument(
        "--tier",
        choices=["essential", "extended", "release", "all"],
        default="essential",
        help="Test tier to run (essential, extended, release, all)",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vertex"],
        help="Only run tests for specific provider",
    )
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    args = parser.parse_args()

    # Check for API keys
    if not check_api_keys():
        return 1

    # Run tests
    return run_api_tests(args)


if __name__ == "__main__":
    sys.exit(main())
