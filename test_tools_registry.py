"""Test tool registration to ensure no duplicate tools are registered."""

import sys
import json
from pathlib import Path
from llmproc import LLMProcess
from unittest.mock import patch, MagicMock

# Mock the client to avoid API calls
with patch("llmproc.providers.providers.get_provider_client") as mock_get_client:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # Load a program with linked programs
    main_path = Path("./examples/program_linking/main.toml")
    print(f"Loading program from {main_path}")
    process = LLMProcess.from_toml(main_path)
    
    # Print the tools 
    tools = getattr(process, "tools", [])
    print(f"Registered {len(tools)} tools:")
    
    tool_names = set()
    for i, tool in enumerate(tools, 1):
        name = tool.get("name", "unknown")
        if name in tool_names:
            print(f"ERROR: Duplicate tool name found: {name}")
        tool_names.add(name)
        print(f"  {i}. {name} - {tool.get('description', '')[:60]}...")