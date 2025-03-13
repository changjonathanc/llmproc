#!/usr/bin/env python3
"""Example usage of the LLMProc package."""

from llmproc import LLMProcess


def main() -> None:
    """Run examples of LLMProc usage."""
    # Example with minimal configuration
    print("Running minimal example...")
    process_minimal = LLMProcess.from_toml("examples/minimal.toml")
    
    output = process_minimal.run("Hello! Keep your response under 10 words.")
    print(f"Response: {output}\n")
    
    output = process_minimal.run("Explain quantum computing in 1-2 sentences only.")
    print(f"Response: {output}\n")
    
    print("-" * 50)
    
    # Example with complex configuration
    print("\nRunning complex example...")
    process_complex = LLMProcess.from_toml("examples/complex.toml")
    
    output = process_complex.run("Define machine learning in exactly one sentence.")
    print(f"Response: {output}\n")
    
    print("Conversation state:")
    for message in process_complex.get_state():
        print(f"[{message['role']}]: {message['content'][:50]}...")
    
    # Reset state
    print("\nResetting conversation state...")
    process_complex.reset_state()
    
    output = process_complex.run("Explain neural networks in 2-3 bullet points only.")
    print(f"Response after reset: {output}")
    
    print("-" * 50)
    
    # Example with preloaded files from TOML
    print("\nRunning preload example (TOML configuration)...")
    try:
        process_preload = LLMProcess.from_toml("examples/preload.toml")
        
        # The model already has context from preloaded files
        print("Initial conversation state contains preloaded files:")
        for message in process_preload.get_state():
            print(f"[{message['role']}]: {message['content'][:50]}...")
        
        output = process_preload.run("What are the key features of LLMProc?")
        print(f"\nResponse with preloaded context: {output}\n")
        
        # Reset but keep preloaded files
        print("Resetting with preloaded files...")
        process_preload.reset_state(keep_preloaded=True)
        
        # Check that preloaded files are still in the state
        print("State after reset still contains preloaded files:")
        for message in process_preload.get_state():
            print(f"[{message['role']}]: {message['content'][:50]}...")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Note: Make sure preloaded files exist relative to the examples directory.")
        
    print("-" * 50)
    
    # Example with runtime preloading
    print("\nRunning runtime preload example...")
    try:
        # Start with minimal configuration
        process_runtime = LLMProcess.from_toml("examples/minimal.toml")
        
        print("Initial state (before preloading):")
        for message in process_runtime.get_state():
            print(f"[{message['role']}]: {message['content'][:50]}...")
        
        # Add files at runtime
        process_runtime.preload_files(["README.md"])
        
        print("\nState after runtime preloading:")
        for message in process_runtime.get_state():
            print(f"[{message['role']}]: {message['content'][:50]}...")
        
        output = process_runtime.run("Summarize the key points of this project.")
        print(f"\nResponse with runtime preloaded context: {output}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Note: Make sure README.md exists in the project root.")


if __name__ == "__main__":
    main()