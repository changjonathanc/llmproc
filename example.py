#!/usr/bin/env python3
"""Example usage of the LLMProc package."""

from llmproc import LLMProcess


def main() -> None:
    """Run examples of LLMProc usage."""
    # Example with minimal configuration
    print("Running minimal example...")
    process_minimal = LLMProcess.from_toml("examples/minimal.toml")
    
    output = process_minimal.run("Hello!")
    print(f"Response: {output}\n")
    
    output = process_minimal.run("Explain quantum computing simply.")
    print(f"Response: {output}\n")
    
    print("-" * 50)
    
    # Example with complex configuration
    print("\nRunning complex example...")
    process_complex = LLMProcess.from_toml("examples/complex.toml")
    
    output = process_complex.run("What is machine learning?")
    print(f"Response: {output}\n")
    
    print("Conversation state:")
    for message in process_complex.get_state():
        print(f"[{message['role']}]: {message['content'][:50]}...")
    
    # Reset state
    print("\nResetting conversation state...")
    process_complex.reset_state()
    
    output = process_complex.run("How does a neural network work?")
    print(f"Response after reset: {output}")


if __name__ == "__main__":
    main()