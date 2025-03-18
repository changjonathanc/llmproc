import asyncio
from llmproc.program import LLMProgram

async def main():
    print("Loading MCP program...")
    program = LLMProgram.from_toml("./examples/mcp.toml")
    print("Program loaded successfully!")
    
    print("Creating process...")
    process = await program.start()
    print("Process created successfully!")
    
    # Show initially registered tools
    print(f"Initial MCP initialized: {process._mcp_initialized}")
    print(f"Initial number of tools: {len(process.tools)}")
    tool_names = [tool["name"] for tool in process.tools]
    print(f"Initial tool names: {tool_names}")
    
    # Now try sending a message
    print("\nSending message...")
    result = await process.run("hello")
    
    # Check if any duplication occurred
    print(f"After message MCP initialized: {process._mcp_initialized}")
    print(f"After message number of tools: {len(process.tools)}")
    tool_names = [tool["name"] for tool in process.tools]
    print(f"After message tool names: {tool_names}")
    
    # Check for duplicates
    name_counts = {}
    for tool in process.tools:
        name = tool["name"]
        name_counts[name] = name_counts.get(name, 0) + 1
    
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    if duplicates:
        print(f"WARNING: Found duplicate tools: {duplicates}")
    else:
        print("SUCCESS: No duplicate tools found")

if __name__ == "__main__":
    asyncio.run(main())