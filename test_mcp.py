import asyncio
from llmproc.program import LLMProgram

async def main():
    print("Loading MCP program...")
    program = LLMProgram.from_toml("./examples/mcp.toml")
    print("Program loaded successfully!")
    print("Creating process...")
    process = await program.start()
    print("Process created successfully!")
    print(f"MCP initialized: {process._mcp_initialized}")
    print(f"Number of registered tools: {len(process.tools)}")
    
    # Print tool names
    tool_names = [tool["name"] for tool in process.tools]
    print(f"Tool names: {tool_names}")

if __name__ == "__main__":
    asyncio.run(main())