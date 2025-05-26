"""Tests for the enhanced file descriptor API."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from llmproc.common.results import ToolResult
from llmproc.file_descriptors.manager import FileDescriptorManager
from llmproc.llm_process import LLMProcess
from llmproc.program import LLMProgram
from llmproc.tools.builtin.fd_tools import fd_to_file_tool, read_fd_tool


class TestEnhancedFileDescriptorAPI:
    """Tests for the enhanced file descriptor API."""

    def test_advanced_positioning_modes(self):
        """Test different advanced positioning modes (line and character-based)."""
        manager = FileDescriptorManager()

        # Test 1: Line-based positioning
        # Create a multi-line content file
        line_content = "\n".join([f"Line {i + 1}: This is test content line {i + 1}" for i in range(20)])

        # Create file descriptor
        line_fd_xml = manager.create_fd_content(line_content)
        line_fd_id = line_fd_xml.split('fd="')[1].split('"')[0]

        # Read specific lines using line mode
        line_result = ToolResult(content=manager.read_fd_content(line_fd_id, mode="line", start=5, count=3))

        # Extract and verify content
        line_text = line_result.content.split(">\n")[1].split("\n</fd_content")[0]
        assert "Line 5:" in line_text
        assert "Line 6:" in line_text
        assert "Line 7:" in line_text
        assert "Line 4:" not in line_text
        assert "Line 8:" not in line_text

        # Check metadata
        assert 'mode="line"' in line_result.content
        assert 'start="5"' in line_result.content
        assert 'count="3"' in line_result.content
        assert 'lines="5-7"' in line_result.content

        # Test 2: Character-based positioning
        # Create content with a known structure
        char_content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4  # 104 characters

        # Create file descriptor
        char_fd_xml = manager.create_fd_content(char_content)
        char_fd_id = char_fd_xml.split('fd="')[1].split('"')[0]

        # Read specific characters
        char_result = ToolResult(content=manager.read_fd_content(char_fd_id, mode="char", start=10, count=15))

        # Extract and verify content
        char_text = char_result.content.split(">\n")[1].split("\n</fd_content")[0]
        assert char_text == "KLMNOPQRSTUVWXY"
        assert len(char_text) == 15

        # Check metadata
        assert 'mode="char"' in char_result.content
        assert 'start="10"' in char_result.content
        assert 'count="15"' in char_result.content

    def test_extraction_operations(self):
        """Test various content extraction operations with file descriptors."""
        manager = FileDescriptorManager()

        # Test 1: Extract section from structured document
        # Create a multi-line content file with structured data
        doc_content = "# Document Title\n\n"
        doc_content += "## Section 1\n\n"
        doc_content += "\n".join([f"Data point 1.{i}: Value {i * 10}" for i in range(1, 6)])
        doc_content += "\n\n## Section 2\n\n"
        doc_content += "\n".join([f"Data point 2.{i}: Value {i * 100}" for i in range(1, 6)])
        doc_content += "\n\n## Section 3\n\n"
        doc_content += "\n".join([f"Data point 3.{i}: Value {i * 1000}" for i in range(1, 6)])

        # Create file descriptor
        doc_fd_xml = manager.create_fd_content(doc_content)
        doc_fd_id = doc_fd_xml.split('fd="')[1].split('"')[0]

        # Read entire content to find section boundaries
        full_content = ToolResult(content=manager.read_fd_content(doc_fd_id, read_all=True))
        content_all = full_content.content.split(">\n")[1].split("\n</fd_content")[0]
        lines = content_all.split("\n")

        # Find Section 2 start and end
        section2_start = next((i + 1 for i, line in enumerate(lines) if line.strip() == "## Section 2"), 0)
        section3_start = next((i for i, line in enumerate(lines) if line.strip() == "## Section 3"), len(lines))

        # Extract Section 2 to new FD
        section_extract = ToolResult(
            content=manager.read_fd_content(
                doc_fd_id,
                mode="line",
                start=section2_start,
                count=section3_start - section2_start,
                extract_to_new_fd=True,
            )
        )

        # Get and validate extracted content
        new_section_fd_id = section_extract.content.split('new_fd="')[1].split('"')[0]
        section_content = ToolResult(content=manager.read_fd_content(new_section_fd_id, read_all=True))
        extracted_section = section_content.content.split(">\n")[1].split("\n</fd_content")[0]

        # Verify correct section extraction
        assert "## Section 2" in extracted_section
        assert "Data point 2.1:" in extracted_section
        assert "## Section 1" not in extracted_section
        assert "## Section 3" not in extracted_section

        # Test 2: Extract page from paginated content
        # Create multi-line content with pagination
        paged_content = "\n".join([f"Line {i}" for i in range(1, 101)])
        manager.default_page_size = 100  # Force pagination

        # Create file descriptor
        paged_fd_xml = manager.create_fd_content(paged_content)
        paged_fd_id = paged_fd_xml.split('fd="')[1].split('"')[0]

        # Extract page 2 to new FD
        page_extract = ToolResult(
            content=manager.read_fd_content(paged_fd_id, mode="page", start=2, extract_to_new_fd=True)
        )

        # Verify extraction metadata
        assert "<fd_extraction " in page_extract.content
        assert "new_fd" in page_extract.content

        # Get new FD and verify content matches original page 2
        new_page_fd_id = page_extract.content.split('new_fd="')[1].split('"')[0]
        assert new_page_fd_id in manager.file_descriptors

        # Compare extracted content with original page 2
        new_page_content = ToolResult(content=manager.read_fd_content(new_page_fd_id, read_all=True))
        page2_content = ToolResult(content=manager.read_fd_content(paged_fd_id, mode="page", start=2))

        new_page_text = new_page_content.content.split(">\n")[1].split("\n</fd_content")[0]
        page2_text = page2_content.content.split(">\n")[1].split("\n</fd_content")[0]
        assert new_page_text == page2_text

        # Test 3: Extract entire content at once
        simple_content = "This is test content that will be extracted to a new FD"
        simple_fd_xml = manager.create_fd_content(simple_content)
        simple_fd_id = simple_fd_xml.split('fd="')[1].split('"')[0]

        # Extract all content
        full_extract = ToolResult(content=manager.read_fd_content(simple_fd_id, read_all=True, extract_to_new_fd=True))

        # Verify extracted content
        new_fd_id = full_extract.content.split('new_fd="')[1].split('"')[0]
        full_content = ToolResult(content=manager.read_fd_content(new_fd_id, read_all=True))
        full_text = full_content.content.split(">\n")[1].split("\n</fd_content")[0]
        assert full_text == simple_content


@pytest.mark.asyncio
async def test_read_fd_tool_with_extraction():
    """Test the read_fd tool function with extraction to new FD."""
    # Mock LLMProcess with fd_manager
    process = Mock()
    process.fd_manager = Mock()

    # Mock FD manager response for extraction
    # Mock FileDescriptorManager.read_fd_content to return proper content string
    process.fd_manager.read_fd_content.return_value = '<fd_extraction source_fd="fd:1" new_fd="fd:2" page="1" content_size="100">\n  <message>Content from fd:1 has been extracted to fd:2</message>\n</fd_extraction>'

    # Call the tool with extract_to_new_fd=True
    result = await read_fd_tool(
        fd="fd:1",
        start=2,
        extract_to_new_fd=True,
        runtime_context={"fd_manager": process.fd_manager},
    )

    # Verify fd_manager.read_fd_content was called with correct args
    process.fd_manager.read_fd_content.assert_called_once_with(
        fd_id="fd:1",
        read_all=False,
        extract_to_new_fd=True,
        mode="page",
        start=2,
        count=1,
    )

    # Check result
    assert "Content from fd:1 has been extracted to fd:2" in result.content


@pytest.mark.asyncio
async def test_fd_to_file_operations():
    """Test various fd_to_file operations including modes and creation parameters."""
    import os
    import tempfile

    # Mock process with FD manager
    process = Mock()
    process.fd_manager = FileDescriptorManager()
    process.file_descriptor_enabled = True

    # Create test content
    test_content = "This is test content for fd_to_file operations"

    # Create a file descriptor
    fd_xml = process.fd_manager.create_fd_content(test_content)
    fd_id = fd_xml.split('fd="')[1].split('"')[0]

    # Use a temporary directory for all test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Basic write mode (default)
        file_path_write = os.path.join(temp_dir, "test_write.txt")
        write_result = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_write,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify content
        assert os.path.exists(file_path_write)
        with open(file_path_write) as f:
            assert f.read() == test_content

        # Test 2: Append mode
        file_path_append = os.path.join(temp_dir, "test_append.txt")

        # First write to create file
        await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_append,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Then append to it
        append_result = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_append,
            mode="append",
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify appended content
        with open(file_path_append) as f:
            assert f.read() == test_content + test_content
        assert 'mode="append"' in append_result.content

        # Test 3: File existence and creation parameters
        # 3.1: Default behavior - create=True, exist_ok=True (overwrite existing)
        file_path_default = os.path.join(temp_dir, "test_default.txt")
        result_default = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_default,
            runtime_context={"fd_manager": process.fd_manager},
        )

        assert os.path.exists(file_path_default)
        assert 'success="true"' in result_default.content
        assert 'create="true"' in result_default.content
        assert 'exist_ok="true"' in result_default.content

        # 3.2: Overwrite existing with default params
        result_overwrite = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_default,  # Same file
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert 'success="true"' in result_overwrite.content

        # 3.3: Create only if doesn't exist - should fail if file exists
        result_fail = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_default,  # Same file (exists)
            exist_ok=False,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert "<fd_error type=" in result_fail.content
        assert "already exists and exist_ok=False" in result_fail.content

        # 3.4: Create only if doesn't exist - should succeed with new file
        file_path_new = os.path.join(temp_dir, "test_new_only.txt")
        result_new = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_new,
            exist_ok=False,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert os.path.exists(file_path_new)
        assert 'success="true"' in result_new.content

        # 3.5: Update existing only (create=False) - should succeed with existing
        result_update = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_default,  # Existing file
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert 'success="true"' in result_update.content
        assert 'create="false"' in result_update.content

        # 3.6: Update existing only (create=False) - should fail with non-existent
        file_path_nonexistent = os.path.join(temp_dir, "nonexistent.txt")
        result_nonexistent = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_nonexistent,
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert "<fd_error type=" in result_nonexistent.content
        assert "doesn't exist and create=False" in result_nonexistent.content

        # 3.7: Append with create=True - should work even on non-existent file
        file_path_append_create = os.path.join(temp_dir, "append_create.txt")
        result_append_create = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_append_create,
            mode="append",
            create=True,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert os.path.exists(file_path_append_create)
        assert 'success="true"' in result_append_create.content
        assert 'mode="append"' in result_append_create.content

        # 3.8: Append with create=False - should fail on non-existent file
        file_path_append_fail = os.path.join(temp_dir, "append_fail.txt")
        result_append_fail = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_append_fail,
            mode="append",
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )
        assert "<fd_error type=" in result_append_fail.content
        assert "doesn't exist and create=False" in result_append_fail.content


@pytest.mark.asyncio
async def test_fd_integration_workflows(mocked_llm_process):
    """Test comprehensive file descriptor integration workflows.

    Args:
        mocked_llm_process: Fixture providing a properly mocked LLMProcess instance
    """
    import os
    import tempfile

    from llmproc.tools import ToolRegistry

    # Test 1: End-to-end integration with tool registry
    # Set up process with file descriptor support
    process = mocked_llm_process
    process.file_descriptor_enabled = True
    process.fd_manager = FileDescriptorManager()

    # Create test content and file descriptor
    test_content = "This is test content for fd operations\n" * 10
    fd_xml = process.fd_manager.create_fd_content(test_content)
    fd_id = fd_xml.split('fd="')[1].split('"')[0]

    # Create registry and handlers that use runtime_context
    registry = ToolRegistry()

    # Register read_fd handler
    async def read_fd_handler(args):
        return await read_fd_tool(
            fd=args.get("fd"),
            start=args.get("start", 1),
            count=args.get("count", 1),
            read_all=args.get("read_all", False),
            extract_to_new_fd=args.get("extract_to_new_fd", False),
            mode=args.get("mode", "page"),
            runtime_context={"fd_manager": process.fd_manager},
        )

    registry.register_tool(
        "read_fd",
        read_fd_handler,
        {"name": "read_fd", "description": "Read file descriptor", "parameters": {}},
    )

    # Extract content to new FD
    handler = registry.get_handler("read_fd")
    extract_result = await handler({"fd": fd_id, "start": 1, "extract_to_new_fd": True})

    # Verify extraction success
    assert "<fd_extraction" in extract_result.content
    assert "new_fd" in extract_result.content
    new_fd_id = extract_result.content.split('new_fd="')[1].split('"')[0]
    assert new_fd_id in process.fd_manager.file_descriptors

    # Test 2: Complete workflow with file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up process and tools
        workflow_process = Mock()
        workflow_process.fd_manager = FileDescriptorManager(default_page_size=1000)
        workflow_process.file_descriptor_enabled = True

        # Create fresh registry for the workflow
        workflow_registry = ToolRegistry()

        # Register both tools needed for the workflow
        async def read_fd_workflow(args):
            return await read_fd_tool(
                fd=args.get("fd"),
                start=args.get("start", 1),
                count=args.get("count", 1),
                read_all=args.get("read_all", False),
                extract_to_new_fd=args.get("extract_to_new_fd", False),
                mode=args.get("mode", "page"),
                runtime_context={"fd_manager": workflow_process.fd_manager},
            )

        async def fd_to_file_workflow(args):
            return await fd_to_file_tool(
                fd=args.get("fd"),
                file_path=args.get("file_path"),
                mode=args.get("mode", "write"),
                create=args.get("create", True),
                exist_ok=args.get("exist_ok", True),
                runtime_context={"fd_manager": workflow_process.fd_manager},
            )

        # Register both handlers
        workflow_registry.register_tool("read_fd", read_fd_workflow, {"name": "read_fd"})
        workflow_registry.register_tool("fd_to_file", fd_to_file_workflow, {"name": "fd_to_file"})

        # Get handlers
        read_handler = workflow_registry.get_handler("read_fd")
        write_handler = workflow_registry.get_handler("fd_to_file")

        # Create content and file descriptor
        workflow_content = "Content for workflow test\n" * 10
        workflow_fd_xml = workflow_process.fd_manager.create_fd_content(workflow_content)
        workflow_fd_id = workflow_fd_xml.split('fd="')[1].split('"')[0]

        # Execute workflow steps
        # Step 1: Read content
        read_result = await read_handler({"fd": workflow_fd_id, "start": 1})
        assert "<fd_content" in read_result.content

        # Step 2: Extract to new FD
        extract_result = await read_handler({"fd": workflow_fd_id, "extract_to_new_fd": True})
        extracted_fd_id = extract_result.content.split('new_fd="')[1].split('"')[0]
        assert extracted_fd_id in workflow_process.fd_manager.file_descriptors

        # Step 3: Write to file
        output_file = os.path.join(temp_dir, "output.txt")
        write_result = await write_handler({"fd": extracted_fd_id, "file_path": output_file})
        assert os.path.exists(output_file)
        assert 'success="true"' in write_result.content

        # Step 4: Append to same file
        append_result = await write_handler({"fd": extracted_fd_id, "file_path": output_file, "mode": "append"})
        assert 'mode="append"' in append_result.content

        # Verify content was duplicated
        with open(output_file) as f:
            content = f.read()
            original_size = len(workflow_process.fd_manager.file_descriptors[extracted_fd_id]["content"])
            assert len(content) >= original_size * 2

        # Step 5: Try with exist_ok=False (should fail on existing file)
        fail_result = await write_handler({"fd": extracted_fd_id, "file_path": output_file, "exist_ok": False})
        assert "<fd_error" in fail_result.content

        # Step 6: Create new file with exist_ok=False (should succeed)
        new_output = os.path.join(temp_dir, "new_output.txt")
        new_file_result = await write_handler({"fd": extracted_fd_id, "file_path": new_output, "exist_ok": False})
        assert os.path.exists(new_output)
        assert 'success="true"' in new_file_result.content
