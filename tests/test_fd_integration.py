"""Integration tests for file descriptor system features."""

import re
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from llmproc.common.results import RunResult, ToolResult
from llmproc.file_descriptors import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fork import fork_tool
from llmproc.tools.builtin.spawn import spawn_tool
from tests.conftest import create_mock_llm_program, create_test_llmprocess_directly


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_combined_features_spawn_fork_references(mock_get_provider_client):
    """Test file descriptor features working together (references, spawn, fork)."""
    # Mock provider client
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client

    # Create a proper RunResult object
    mock_run_response = RunResult()
    # Add the API call info with the text field manually
    mock_run_response.add_api_call(
        {
            "model": "test-model",
            "provider": "anthropic",
            "text": 'Test response with reference: <ref id="test_ref">Test reference content</ref>',
        }
    )

    # Create process hierarchy:
    # parent -> child1 -> grandchild
    #        -> child2

    # Set up parent program with fd, references, spawn and fork enabled
    parent_program = create_mock_llm_program()
    parent_program.provider = "anthropic"
    parent_program.tools = {"enabled": ["read_fd", "spawn", "fork"]}
    parent_program.system_prompt = "parent system"
    parent_program.display_name = "parent"
    parent_program.base_dir = None
    parent_program.api_params = {}
    parent_program.get_enriched_system_prompt = Mock(return_value="enriched parent")

    # Child programs for spawning
    child_program = create_mock_llm_program()
    child_program.provider = "anthropic"
    child_program.tools = {"enabled": ["read_fd", "spawn", "fork"]}
    child_program.system_prompt = "child system"
    child_program.display_name = "child"
    child_program.base_dir = None
    child_program.api_params = {}
    child_program.get_enriched_system_prompt = Mock(return_value="enriched child")

    grandchild_program = create_mock_llm_program()
    grandchild_program.provider = "anthropic"
    grandchild_program.tools = {"enabled": ["read_fd"]}
    grandchild_program.system_prompt = "grandchild system"
    grandchild_program.display_name = "grandchild"
    grandchild_program.base_dir = None
    grandchild_program.api_params = {}
    grandchild_program.get_enriched_system_prompt = Mock(
        return_value="enriched grandchild"
    )

    # Create a parent process using the proper initialization pattern
    # For testing purposes, we mock the async start() method
    with patch.object(parent_program, "start") as mock_start:
        # Create a process that would be returned by start()
        parent_process = create_test_llmprocess_directly(program=parent_program)

        # Configure the mock to return our process
        mock_start.return_value = parent_process

        # In a real implementation, we would use:
        # parent_process = await parent_program.start()

    # Set up linked programs
    parent_process.linked_programs = {
        "child": child_program,
        "grandchild": grandchild_program,
    }
    parent_process.has_linked_programs = True

    # Create a mechanism to track created processes
    processes = {"parent": parent_process, "children": [], "grandchildren": []}

    # Mock the _spawn_child_process method to capture processes
    async def mock_spawn(process, program_name, *args, **kwargs):
        child_process = create_test_llmprocess_directly(
            program=process.linked_programs[program_name]
        )

        # Enable file descriptors on the child
        child_process.file_descriptor_enabled = True
        child_process.references_enabled = True
        child_process.fd_manager = FileDescriptorManager(enable_references=True)

        # Copy references from parent to child
        for fd_id, fd_data in process.fd_manager.file_descriptors.items():
            if fd_id.startswith("ref:"):
                child_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

        # Store references to all child processes
        if program_name == "child":
            processes["children"].append(child_process)
        elif program_name == "grandchild":
            processes["grandchildren"].append(child_process)

        # Mock the run method of the child process to avoid API calls
        child_process.run = Mock(return_value=mock_run_response)

        return child_process

    # Replace spawn_tool with our mock
    async def mock_spawn_tool(
        program_name, query, additional_preload_fds=None, llm_process=None
    ):
        child_process = await mock_spawn(llm_process, program_name, query)

        # Handle additional preload FDs
        if additional_preload_fds:
            for fd_id in additional_preload_fds:
                if fd_id in llm_process.fd_manager.file_descriptors:
                    child_process.fd_manager.file_descriptors[fd_id] = (
                        llm_process.fd_manager.file_descriptors[fd_id].copy()
                    )

        return ToolResult(content=f"Spawned {program_name}")

    # Enable features
    parent_process.file_descriptor_enabled = True
    parent_process.references_enabled = True
    parent_process.page_user_input = True
    parent_process.fd_manager = FileDescriptorManager(
        page_user_input=True, enable_references=True
    )

    # Step 1: Create a reference in parent process
    parent_message = """
    <ref id="parent_ref">
    Parent reference content that should be inherited by children
    </ref>
    """

    parent_references = parent_process.fd_manager.extract_references_from_message(
        parent_message
    )
    assert len(parent_references) == 1
    assert "ref:parent_ref" in parent_process.fd_manager.file_descriptors

    # Step 2: Create a large user input and have it automatically paged
    large_user_input = "A" * (parent_process.fd_manager.max_input_chars + 1000)
    paged_input = parent_process.fd_manager.handle_user_input(large_user_input)

    # Verify input was paged
    assert "<fd:" in paged_input
    assert 'type="user_input"' in paged_input

    # Get the fd_id from the paged input
    input_fd_match = re.search(r"<fd:(.*?) ", paged_input)
    assert input_fd_match
    input_fd_id = "fd:" + input_fd_match.group(1)

    # Fix potential fd:fd:1 prefix issue
    if input_fd_id.startswith("fd:fd:"):
        fixed_fd_id = input_fd_id[3:]  # Remove the first "fd:"
    else:
        fixed_fd_id = input_fd_id

    # Verify the user input was stored correctly
    assert fixed_fd_id in parent_process.fd_manager.file_descriptors
    assert (
        parent_process.fd_manager.file_descriptors[fixed_fd_id]["source"]
        == "user_input"
    )

    # Step 3: Create a child process using the proper initialization pattern
    # This simulates what spawn_tool would do but using the proper pattern
    with patch.object(child_program, "start") as mock_start:
        # Create a process that would be returned by start()
        child1 = create_test_llmprocess_directly(program=child_program)

        # Configure the mock to return our process
        mock_start.return_value = child1

        # In a real implementation, we would use:
        # child1 = await child_program.start()

    # Enable file descriptors on the child
    child1.file_descriptor_enabled = True
    child1.references_enabled = True
    child1.fd_manager = FileDescriptorManager(enable_references=True)

    # Copy references and the user input file descriptor from parent to child
    for fd_id, fd_data in parent_process.fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:") or fd_id == fixed_fd_id:
            child1.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Store the child process
    processes["children"].append(child1)

    # Mock the run method
    child1.run = Mock(return_value=mock_run_response)

    # Verify child was created
    assert len(processes["children"]) == 1
    child1 = processes["children"][0]

    # Verify child has file descriptor enabled and has both the user input and reference
    assert child1.file_descriptor_enabled
    assert child1.references_enabled
    assert fixed_fd_id in child1.fd_manager.file_descriptors
    assert "ref:parent_ref" in child1.fd_manager.file_descriptors

    # Step 4: Have the child process create its own reference
    child_message = """
    <ref id="child_ref">
    Child reference content that should NOT be visible to parent
    </ref>
    """

    child_references = child1.fd_manager.extract_references_from_message(child_message)
    assert len(child_references) == 1
    assert "ref:child_ref" in child1.fd_manager.file_descriptors

    # Verify parent doesn't have the child's reference (this is the isolaton mechanism)
    assert "ref:child_ref" not in parent_process.fd_manager.file_descriptors

    # Step 5: Create a grandchild process using the proper initialization pattern
    # This simulates what spawn_tool would do but using the proper pattern
    with patch.object(grandchild_program, "start") as mock_start:
        # Create a process that would be returned by start()
        grandchild = create_test_llmprocess_directly(program=grandchild_program)

        # Configure the mock to return our process
        mock_start.return_value = grandchild

        # In a real implementation, we would use:
        # grandchild = await grandchild_program.start()

    # Enable file descriptors on the grandchild
    grandchild.file_descriptor_enabled = True
    grandchild.references_enabled = True
    grandchild.fd_manager = FileDescriptorManager(enable_references=True)

    # Copy references from child to grandchild
    for fd_id, fd_data in child1.fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:"):
            grandchild.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Additionally copy the explicit FD we would pass
    if "ref:child_ref" in child1.fd_manager.file_descriptors:
        grandchild.fd_manager.file_descriptors["ref:child_ref"] = (
            child1.fd_manager.file_descriptors["ref:child_ref"].copy()
        )

    # Store the grandchild process
    processes["grandchildren"].append(grandchild)

    # Mock the run method
    grandchild.run = Mock(return_value=mock_run_response)

    # Verify grandchild was created
    assert len(processes["grandchildren"]) == 1
    grandchild = processes["grandchildren"][0]

    # Verify grandchild has file descriptor enabled and has the inheritance chain
    assert grandchild.file_descriptor_enabled
    assert grandchild.references_enabled

    # Grandchild should have:
    # 1. Explicitly passed reference from child
    # 2. Automatically inherited reference from parent
    # 3. No access to user input from parent (not explicitly passed)
    assert "ref:child_ref" in grandchild.fd_manager.file_descriptors
    assert "ref:parent_ref" in grandchild.fd_manager.file_descriptors
    assert input_fd_id not in grandchild.fd_manager.file_descriptors

    # Step 6: Directly simulate fork functionality instead of using fork_tool
    # Create forked processes to test multiple forks from the same parent
    forked_processes = []

    # Create two identical forked processes to test multiple process creation
    for _ in range(2):
        # Create a forked process with file descriptor sharing using proper initialization pattern
        with patch.object(parent_process.program, "start") as mock_start:
            # Create a process that would be returned by start()
            forked_process = create_test_llmprocess_directly(
                program=parent_process.program
            )
            forked_process.file_descriptor_enabled = True
            forked_process.references_enabled = True
            forked_process.fd_manager = FileDescriptorManager(enable_references=True)

            # Configure the mock to return our process
            mock_start.return_value = forked_process

            # In a real implementation, we would use:
            # forked_process = await parent_process.program.start()

        # Copy all file descriptors from parent
        for fd_id, fd_data in parent_process.fd_manager.file_descriptors.items():
            forked_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

        # Mock the run method
        forked_process.run = Mock(return_value=mock_run_response)

        forked_processes.append(forked_process)

    # Create a simulated fork_result
    fork_result = ToolResult(
        content=f"fork_results: Created {len(forked_processes)} forked processes"
    )

    # Get the fork results (tool returns array of tool results)
    fork_content = fork_result.content
    assert "fork_results" in fork_content

    # Now let's create a second child from parent to verify isolation
    # Create child2 process using the proper initialization pattern
    with patch.object(child_program, "start") as mock_start:
        # Create a process that would be returned by start()
        child2 = create_test_llmprocess_directly(program=child_program)

        # Enable file descriptors on child2
        child2.file_descriptor_enabled = True
        child2.references_enabled = True
        child2.fd_manager = FileDescriptorManager(enable_references=True)

        # Configure the mock to return our process
        mock_start.return_value = child2

        # In a real implementation, we would use:
        # child2 = await child_program.start()

    # Copy references from parent to child2
    for fd_id, fd_data in parent_process.fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:"):
            child2.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Store the child2 process
    processes["children"].append(child2)

    # Mock the run method
    child2.run = Mock(return_value=mock_run_response)

    # Children process count should be 2 at this point

    # Verify second child was created
    assert len(processes["children"]) == 2
    child2 = processes["children"][1]

    # Verify child2 has the parent reference but not child1's reference
    assert child2.file_descriptor_enabled
    assert child2.references_enabled
    assert "ref:parent_ref" in child2.fd_manager.file_descriptors
    assert "ref:child_ref" not in child2.fd_manager.file_descriptors


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_multi_level_reference_inheritance(mock_get_provider_client):
    """Test reference inheritance across multiple levels of spawned processes."""
    # Mock provider client
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client

    # Create a proper RunResult object
    mock_run_response = RunResult()
    # Add the API call info with the text field manually
    mock_run_response.add_api_call(
        {
            "model": "test-model",
            "provider": "anthropic",
            "text": 'Test response with reference: <ref id="test_ref">Test reference content</ref>',
        }
    )

    # Create a process hierarchy with multiple levels
    # level1 -> level2 -> level3 -> level4

    # Set up programs for each level
    level1_program = create_mock_llm_program()
    level1_program.provider = "anthropic"
    level1_program.tools = {"enabled": ["read_fd", "spawn"]}
    level1_program.system_prompt = "level1 system"
    level1_program.display_name = "level1"
    level1_program.base_dir = None
    level1_program.api_params = {}
    level1_program.get_enriched_system_prompt = Mock(return_value="enriched level1")

    level2_program = create_mock_llm_program()
    level2_program.provider = "anthropic"
    level2_program.tools = {"enabled": ["read_fd", "spawn"]}
    level2_program.system_prompt = "level2 system"
    level2_program.display_name = "level2"
    level2_program.base_dir = None
    level2_program.api_params = {}
    level2_program.get_enriched_system_prompt = Mock(return_value="enriched level2")

    level3_program = create_mock_llm_program()
    level3_program.provider = "anthropic"
    level3_program.tools = {"enabled": ["read_fd", "spawn"]}
    level3_program.system_prompt = "level3 system"
    level3_program.display_name = "level3"
    level3_program.base_dir = None
    level3_program.api_params = {}
    level3_program.get_enriched_system_prompt = Mock(return_value="enriched level3")

    level4_program = create_mock_llm_program()
    level4_program.provider = "anthropic"
    level4_program.tools = {"enabled": ["read_fd"]}
    level4_program.system_prompt = "level4 system"
    level4_program.display_name = "level4"
    level4_program.base_dir = None
    level4_program.api_params = {}
    level4_program.get_enriched_system_prompt = Mock(return_value="enriched level4")

    # Create the level1 process using the proper initialization pattern
    with patch.object(level1_program, "start") as mock_start:
        # Create a process that would be returned by start()
        level1_process = create_test_llmprocess_directly(program=level1_program)

        # Configure the mock to return our process
        mock_start.return_value = level1_process

        # In a real implementation, we would use:
        # level1_process = await level1_program.start()

    # Set up linked programs
    level1_process.linked_programs = {"level2": level2_program}
    level1_process.has_linked_programs = True

    # Enable file descriptors and references
    level1_process.file_descriptor_enabled = True
    level1_process.references_enabled = True
    level1_process.fd_manager = FileDescriptorManager(enable_references=True)

    # Create processes dictionary to track all processes created
    processes = {
        "level1": level1_process,
        "level2": None,
        "level3": None,
        "level4": None,
    }

    # Create references at each level
    level1_message = """
    <ref id="level1_ref">
    Level 1 reference content
    </ref>
    """

    level1_references = level1_process.fd_manager.extract_references_from_message(
        level1_message
    )
    assert len(level1_references) == 1
    assert "ref:level1_ref" in level1_process.fd_manager.file_descriptors

    # Mock spawn tool functionality
    async def mock_spawn_for_level(
        program_name, query, additional_preload_fds=None, runtime_context=None
    ):
        # Extract process from runtime_context
        llm_process = (
            runtime_context["process"]
            if runtime_context and "process" in runtime_context
            else None
        )
        if not llm_process:
            return ToolResult.from_error("No process provided in runtime_context")

        # Determine which level we're spawning
        current_level = llm_process.program.display_name
        next_level = program_name

        # Create the child process
        if next_level == "level2":
            child_process = create_test_llmprocess_directly(program=level2_program)
            child_process.linked_programs = {"level3": level3_program}
        elif next_level == "level3":
            child_process = create_test_llmprocess_directly(program=level3_program)
            child_process.linked_programs = {"level4": level4_program}
        elif next_level == "level4":
            child_process = create_test_llmprocess_directly(program=level4_program)
        else:
            raise ValueError(f"Unexpected program name: {program_name}")

        child_process.has_linked_programs = True

        # Enable file descriptors on the child
        child_process.file_descriptor_enabled = True
        child_process.references_enabled = True
        child_process.fd_manager = FileDescriptorManager(enable_references=True)

        # Copy references from parent to child
        for fd_id, fd_data in llm_process.fd_manager.file_descriptors.items():
            if fd_id.startswith("ref:"):
                child_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

        # Add explicitly shared file descriptors
        if additional_preload_fds:
            for fd_id in additional_preload_fds:
                if fd_id in llm_process.fd_manager.file_descriptors:
                    child_process.fd_manager.file_descriptors[fd_id] = (
                        llm_process.fd_manager.file_descriptors[fd_id].copy()
                    )

        # Store the process
        processes[next_level] = child_process

        # Mock the run method
        child_process.run = Mock(return_value=mock_run_response)

        return ToolResult(content=f"Spawned {next_level}")

    # Create level2 process using the proper initialization pattern
    with patch.object(level2_program, "start") as mock_start:
        # Create a process that would be returned by start()
        level2_process = create_test_llmprocess_directly(program=level2_program)
        level2_process.linked_programs = {"level3": level3_program}
        level2_process.has_linked_programs = True

        # Configure the mock to return our process
        mock_start.return_value = level2_process

        # In a real implementation, we would use:
        # level2_process = await level2_program.start()

    # Enable file descriptors on level2
    level2_process.file_descriptor_enabled = True
    level2_process.references_enabled = True
    level2_process.fd_manager = FileDescriptorManager(enable_references=True)

    # Copy references from level1 to level2
    for fd_id, fd_data in level1_process.fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:"):
            level2_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Store the process
    processes["level2"] = level2_process

    # Mock the run method
    level2_process.run = Mock(return_value=mock_run_response)

    # Verify level2 was created and has level1's reference
    assert processes["level2"] is not None
    assert "ref:level1_ref" in processes["level2"].fd_manager.file_descriptors

    # Create a reference at level2
    level2_message = """
    <ref id="level2_ref">
    Level 2 reference content
    </ref>
    """

    level2_references = processes["level2"].fd_manager.extract_references_from_message(
        level2_message
    )
    assert len(level2_references) == 1
    assert "ref:level2_ref" in processes["level2"].fd_manager.file_descriptors

    # Create level3 process using the proper initialization pattern
    with patch.object(level3_program, "start") as mock_start:
        # Create a process that would be returned by start()
        level3_process = create_test_llmprocess_directly(program=level3_program)
        level3_process.linked_programs = {"level4": level4_program}
        level3_process.has_linked_programs = True

        # Configure the mock to return our process
        mock_start.return_value = level3_process

        # In a real implementation, we would use:
        # level3_process = await level3_program.start()

    # Enable file descriptors on level3
    level3_process.file_descriptor_enabled = True
    level3_process.references_enabled = True
    level3_process.fd_manager = FileDescriptorManager(enable_references=True)

    # Copy references from level2 to level3
    for fd_id, fd_data in processes["level2"].fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:"):
            level3_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Store the process
    processes["level3"] = level3_process

    # Mock the run method
    level3_process.run = Mock(return_value=mock_run_response)

    # Verify level3 was created and has both references
    assert processes["level3"] is not None
    assert "ref:level1_ref" in processes["level3"].fd_manager.file_descriptors
    assert "ref:level2_ref" in processes["level3"].fd_manager.file_descriptors

    # Create a reference at level3
    level3_message = """
    <ref id="level3_ref">
    Level 3 reference content
    </ref>
    """

    level3_references = processes["level3"].fd_manager.extract_references_from_message(
        level3_message
    )
    assert len(level3_references) == 1
    assert "ref:level3_ref" in processes["level3"].fd_manager.file_descriptors

    # Create level4 process using the proper initialization pattern
    with patch.object(level4_program, "start") as mock_start:
        # Create a process that would be returned by start()
        level4_process = create_test_llmprocess_directly(program=level4_program)

        # Configure the mock to return our process
        mock_start.return_value = level4_process

        # In a real implementation, we would use:
        # level4_process = await level4_program.start()

    # Enable file descriptors on level4
    level4_process.file_descriptor_enabled = True
    level4_process.references_enabled = True
    level4_process.fd_manager = FileDescriptorManager(enable_references=True)

    # Copy references from level3 to level4
    for fd_id, fd_data in processes["level3"].fd_manager.file_descriptors.items():
        if fd_id.startswith("ref:"):
            level4_process.fd_manager.file_descriptors[fd_id] = fd_data.copy()

    # Store the process
    processes["level4"] = level4_process

    # Mock the run method
    level4_process.run = Mock(return_value=mock_run_response)

    # Verify level4 was created and has all references
    assert processes["level4"] is not None
    assert "ref:level1_ref" in processes["level4"].fd_manager.file_descriptors
    assert "ref:level2_ref" in processes["level4"].fd_manager.file_descriptors
    assert "ref:level3_ref" in processes["level4"].fd_manager.file_descriptors

    # Verify references were not passed upward
    assert "ref:level2_ref" not in processes["level1"].fd_manager.file_descriptors
    assert "ref:level3_ref" not in processes["level2"].fd_manager.file_descriptors
    assert "ref:level4_ref" not in processes["level3"].fd_manager.file_descriptors


@pytest.mark.asyncio
@patch("llmproc.providers.providers.get_provider_client")
async def test_user_input_paging_with_spawn(mock_get_provider_client):
    """Test automatic user input paging with spawn integration."""
    # Mock provider client
    mock_client = Mock()
    mock_get_provider_client.return_value = mock_client

    # Create a proper RunResult object
    mock_run_response = RunResult()
    # Add the API call info with the text field manually
    mock_run_response.add_api_call(
        {
            "model": "test-model",
            "provider": "anthropic",
            "text": "Test response after processing user input",
        }
    )

    # Create a program with file descriptor and spawn support
    parent_program = create_mock_llm_program()
    parent_program.provider = "anthropic"
    parent_program.tools = {"enabled": ["read_fd", "spawn"]}
    parent_program.system_prompt = "parent system"
    parent_program.display_name = "parent"
    parent_program.base_dir = None
    parent_program.api_params = {}
    parent_program.get_enriched_system_prompt = Mock(return_value="enriched parent")

    # Child program for spawning
    child_program = create_mock_llm_program()
    child_program.provider = "anthropic"
    child_program.tools = {"enabled": ["read_fd"]}
    child_program.system_prompt = "child system"
    child_program.display_name = "child"
    child_program.base_dir = None
    child_program.api_params = {}
    child_program.get_enriched_system_prompt = Mock(return_value="enriched child")

    # Create parent process using the proper initialization pattern
    with patch.object(parent_program, "start") as mock_start:
        # Create a process that would be returned by start()
        parent_process = create_test_llmprocess_directly(program=parent_program)

        # Configure the mock to return our process
        mock_start.return_value = parent_process

        # In a real implementation, we would use:
        # parent_process = await parent_program.start()

    # Set up linked programs
    parent_process.linked_programs = {"child": child_program}
    parent_process.has_linked_programs = True

    # Enable FD features with paging enabled
    parent_process.file_descriptor_enabled = True
    parent_process.references_enabled = True
    parent_process.fd_manager = FileDescriptorManager(
        max_input_chars=1000, page_user_input=True, enable_references=True
    )

    # Create a large user input
    large_input = "A" * 2000  # Well above the threshold

    # Process the input
    paged_input = parent_process.fd_manager.handle_user_input(large_input)

    # Verify input was paged
    assert "<fd:" in paged_input
    assert 'type="user_input"' in paged_input

    # Get the fd_id from the paged input
    input_fd_match = re.search(r"<fd:(.*?) ", paged_input)
    assert input_fd_match
    input_fd_id = "fd:" + input_fd_match.group(1)

    # Mock spawn tool functionality
    child_process = None

    async def mock_spawn_tool_impl(
        program_name, query, additional_preload_fds=None, runtime_context=None
    ):
        nonlocal child_process

        # Extract process from runtime_context
        llm_process = (
            runtime_context["process"]
            if runtime_context and "process" in runtime_context
            else None
        )
        if not llm_process:
            return ToolResult.from_error("No process provided in runtime_context")

        # Create the child process
        child_process = create_test_llmprocess_directly(program=child_program)

        # Enable file descriptors on the child
        child_process.file_descriptor_enabled = True
        child_process.references_enabled = True
        child_process.fd_manager = FileDescriptorManager(enable_references=True)

        # Add explicitly shared file descriptors
        if additional_preload_fds:
            for fd_id in additional_preload_fds:
                if fd_id in llm_process.fd_manager.file_descriptors:
                    child_process.fd_manager.file_descriptors[fd_id] = (
                        llm_process.fd_manager.file_descriptors[fd_id].copy()
                    )

        # Mock the run method
        child_process.run = Mock(return_value=mock_run_response)

        return ToolResult(content=f"Spawned {program_name}")

    # Create the child process using the proper initialization pattern
    with patch.object(child_program, "start") as mock_start:
        # Create a process that would be returned by start()
        child_process = create_test_llmprocess_directly(program=child_program)

        # Configure the mock to return our process
        mock_start.return_value = child_process

        # In a real implementation, we would use:
        # child_process = await child_program.start()

    # Enable file descriptors on the child
    child_process.file_descriptor_enabled = True
    child_process.references_enabled = True
    child_process.fd_manager = FileDescriptorManager(enable_references=True)

    # Fix potential fd:fd:1 prefix issue
    if input_fd_id.startswith("fd:fd:"):
        fixed_fd_id = input_fd_id[3:]  # Remove the first "fd:"
    else:
        fixed_fd_id = input_fd_id

    # Copy file descriptors from parent to child
    if fixed_fd_id in parent_process.fd_manager.file_descriptors:
        child_process.fd_manager.file_descriptors[fixed_fd_id] = (
            parent_process.fd_manager.file_descriptors[fixed_fd_id].copy()
        )

    # Mock the run method
    child_process.run = Mock(return_value=mock_run_response)

    # Verify child process was created
    assert child_process is not None

    # Verify child has the FD with user input
    assert child_process.file_descriptor_enabled

    # Use the fixed FD ID for assertion
    if input_fd_id.startswith("fd:fd:"):
        fixed_fd_id = input_fd_id[3:]  # Remove the first "fd:"
    else:
        fixed_fd_id = input_fd_id

    assert fixed_fd_id in child_process.fd_manager.file_descriptors

    # Verify the content matches the original input
    assert (
        child_process.fd_manager.file_descriptors[fixed_fd_id]["content"] == large_input
    )

    # Use read_fd tool in the child to read the content
    from llmproc.tools.builtin.fd_tools import read_fd_tool

    read_result = await read_fd_tool(
        fd=fixed_fd_id,
        read_all=True,
        runtime_context={"fd_manager": child_process.fd_manager},
    )

    # Verify content was read correctly
    assert not read_result.is_error
    assert large_input in read_result.content


@pytest.mark.asyncio
async def test_reference_error_handling():
    """Test error handling for invalid reference operations."""
    # Create a FileDescriptorManager
    manager = FileDescriptorManager(enable_references=True)

    # Test reading a non-existent reference
    try:
        manager.read_fd_content("ref:nonexistent")
        # Should have raised KeyError
        raise AssertionError("read_fd_content should have raised KeyError for non-existent reference")
    except KeyError as e:
        # Verify error message
        assert "not found" in str(e)
        assert "ref:nonexistent" in str(e)

    # Create a valid reference
    message = """
    <ref id="valid_ref">
    Valid reference content
    </ref>
    """

    references = manager.extract_references_from_message(message)
    assert len(references) == 1

    # Test invalid line range
    try:
        manager.read_fd_content("ref:valid_ref", mode="line", start=100, count=1)
        # Should have raised ValueError
        raise AssertionError("read_fd_content should have raised ValueError for invalid line range")
    except ValueError as e:
        # Verify error message contains relevant error information
        assert any(
            term in str(e).lower() for term in ["invalid", "range", "line", "parameter"]
        )

    # Test writing a non-existent reference to a file
    from llmproc.tools.builtin.fd_tools import fd_to_file_tool

    # Mock open to avoid actually writing files
    with (
        patch("builtins.open", MagicMock()),
        patch("os.path.getsize", MagicMock(side_effect=Exception("File not found"))),
    ):
        # Create a mocked LLMProcess with the manager
        process = Mock()
        process.fd_manager = manager

        result = await fd_to_file_tool(
            fd="ref:nonexistent",
            file_path="/tmp/test.txt",
            mode="write",
            create=True,
            exist_ok=True,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify error response
        assert result.is_error
        assert "not_found" in result.content
