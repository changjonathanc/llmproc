"""Entry point for CLI execution as a module."""

import sys

if __name__ == "__main__":
    # Import and run main function directly, without going through __init__.py
    from llmproc.cli.demo import main

    # Run the main function
    main()
