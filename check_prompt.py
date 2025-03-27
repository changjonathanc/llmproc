"""Check the enriched system prompt for a file descriptor configuration."""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llmproc.program import LLMProgram


async def main():
    """Check the enriched system prompt."""
    program_path = "./examples/file_descriptor/all_features.toml"
    print(f"Loading program from {program_path}")
    
    program = LLMProgram.from_toml(program_path)
    print("Program loaded successfully")
    
    print(f"File descriptor config: {program.file_descriptor}")
    print(f"References enabled: {program.file_descriptor.get('enable_references')}")
    
    process = await program.start()
    print("Process started successfully")
    
    print(f"File descriptor enabled: {process.file_descriptor_enabled}")
    print(f"References enabled: {process.references_enabled}")
    
    if process.fd_manager:
        print(f"User input paging: {process.fd_manager.page_user_input}")
    
    enriched_prompt = program.get_enriched_system_prompt(process)
    
    # Check for instruction sections
    fd_base = "<file_descriptor_base_instructions>" in enriched_prompt
    user_input = "<fd_user_input_instructions>" in enriched_prompt
    references = "<reference_instructions>" in enriched_prompt
    
    print(f"File descriptor base instructions: {fd_base}")
    print(f"User input paging instructions: {user_input}")
    print(f"Reference instructions: {references}")
    
    # Print the full prompt
    print("\n--- ENRICHED SYSTEM PROMPT ---\n")
    print(enriched_prompt)


if __name__ == "__main__":
    asyncio.run(main())