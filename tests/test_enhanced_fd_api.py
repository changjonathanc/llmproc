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

    def test_advanced_positioning_line_mode(self):
        """Test line-based positioning with mode, start, and count parameters."""
        manager = FileDescriptorManager()

        # Create a multi-line content file
        content = "\n".join(
            [f"Line {i + 1}: This is test content line {i + 1}" for i in range(20)]
        )

        # Create the file descriptor
        fd_xml = manager.create_fd_content(content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Read specific lines using line mode
        content_xml = manager.read_fd_content(fd_id, mode="line", start=5, count=3)
        # For testing compatibility, wrap in ToolResult
        result = ToolResult(content=content_xml, is_error=False)

        # Extract the content
        content_text = result.content.split(">\n")[1].split("\n</fd_content")[0]

        # Verify the content
        assert "Line 5:" in content_text
        assert "Line 6:" in content_text
        assert "Line 7:" in content_text
        assert "Line 4:" not in content_text
        assert "Line 8:" not in content_text

        # Check metadata
        assert 'mode="line"' in result.content
        assert 'start="5"' in result.content
        assert 'count="3"' in result.content
        assert 'lines="5-7"' in result.content

    def test_advanced_positioning_char_mode(self):
        """Test character-based positioning with mode, start, and count parameters."""
        manager = FileDescriptorManager()

        # Create content with a known structure
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4  # 104 characters

        # Create the file descriptor
        fd_xml = manager.create_fd_content(content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Read specific characters using char mode
        content_xml = manager.read_fd_content(fd_id, mode="char", start=10, count=15)
        # For testing compatibility, wrap in ToolResult
        result = ToolResult(content=content_xml, is_error=False)

        # Extract the content
        content_text = result.content.split(">\n")[1].split("\n</fd_content")[0]

        # Verify the content - should be "KLMNOPQRSTUVWXY"
        assert content_text == "KLMNOPQRSTUVWXY"
        assert len(content_text) == 15

        # Check metadata
        assert 'mode="char"' in result.content
        assert 'start="10"' in result.content
        assert 'count="15"' in result.content

    def test_advanced_positioning_with_extraction(self):
        """Test combining advanced positioning with extraction to a new FD."""
        manager = FileDescriptorManager()

        # Create a multi-line content file with structured data
        content = "# Document Title\n\n"
        content += "## Section 1\n\n"
        content += "\n".join([f"Data point 1.{i}: Value {i * 10}" for i in range(1, 6)])
        content += "\n\n## Section 2\n\n"
        content += "\n".join(
            [f"Data point 2.{i}: Value {i * 100}" for i in range(1, 6)]
        )
        content += "\n\n## Section 3\n\n"
        content += "\n".join(
            [f"Data point 3.{i}: Value {i * 1000}" for i in range(1, 6)]
        )

        # Create the file descriptor
        fd_xml = manager.create_fd_content(content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # First locate the section boundaries using line mode
        content_xml = manager.read_fd_content(fd_id, read_all=True)
        # For testing compatibility, wrap in ToolResult
        result = ToolResult(content=content_xml, is_error=False)
        content_all = result.content.split(">\n")[1].split("\n</fd_content")[0]
        lines = content_all.split("\n")

        # Find the Section 2 start and end
        section2_start = next(
            (i + 1 for i, line in enumerate(lines) if line.strip() == "## Section 2"), 0
        )
        section3_start = next(
            (i for i, line in enumerate(lines) if line.strip() == "## Section 3"),
            len(lines),
        )

        # Now extract just Section 2 using line mode
        content_xml = manager.read_fd_content(
            fd_id,
            mode="line",
            start=section2_start,
            count=section3_start - section2_start,
            extract_to_new_fd=True,
        )
        # For testing compatibility, wrap in ToolResult
        result = ToolResult(content=content_xml, is_error=False)

        # Get the new FD ID
        new_fd_id = result.content.split('new_fd="')[1].split('"')[0]

        # Read the extracted content
        extracted_content_xml = manager.read_fd_content(new_fd_id, read_all=True)
        # For testing compatibility, wrap in ToolResult
        extracted_result = ToolResult(content=extracted_content_xml, is_error=False)
        extracted_content = extracted_result.content.split(">\n")[1].split(
            "\n</fd_content"
        )[0]

        # Verify section 2 was extracted
        assert "## Section 2" in extracted_content
        assert "Data point 2.1:" in extracted_content
        assert "Data point 2.5:" in extracted_content
        assert "## Section 1" not in extracted_content
        assert "## Section 3" not in extracted_content

    def test_extract_to_new_fd(self):
        """Test extracting content to a new file descriptor."""
        manager = FileDescriptorManager()

        # Create multi-line content that spans multiple pages
        content = "\n".join([f"Line {i}" for i in range(1, 101)])

        # Set small page size to force pagination
        manager.default_page_size = 100

        # Create first FD
        fd_xml = manager.create_fd_content(content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Extract page 2 to a new file descriptor
        extract_content_xml = manager.read_fd_content(
            fd_id, mode="page", start=2, extract_to_new_fd=True
        )
        # For testing compatibility, wrap in ToolResult
        extract_result = ToolResult(content=extract_content_xml, is_error=False)

        # Check that the result has the right format
        assert "<fd_extraction " in extract_result.content
        assert "new_fd" in extract_result.content

        # Extract the new FD ID
        new_fd_id = extract_result.content.split('new_fd="')[1].split('"')[0]

        # Check the new FD exists
        assert new_fd_id in manager.file_descriptors

        # Read from the new FD to verify content
        new_fd_xml = manager.read_fd_content(new_fd_id, read_all=True)
        # For testing compatibility, wrap in ToolResult
        new_fd_content = ToolResult(content=new_fd_xml, is_error=False)

        # Verify content from new FD is similar to what we'd get from reading page 2
        page2_xml = manager.read_fd_content(fd_id, mode="page", start=2)
        # For testing compatibility, wrap in ToolResult
        page2_content = ToolResult(content=page2_xml, is_error=False)

        # Extract actual content from XML result
        page2_text = page2_content.content.split(">\n")[1].split("\n</fd_content")[0]
        new_fd_text = new_fd_content.content.split(">\n")[1].split("\n</fd_content")[0]

        # They should contain similar content
        assert len(new_fd_text) > 0
        assert new_fd_text == page2_text

    def test_extract_entire_content(self):
        """Test extracting entire content to a new file descriptor."""
        manager = FileDescriptorManager()

        # Create content
        content = "This is test content that will be extracted to a new FD"

        # Create first FD
        fd_xml = manager.create_fd_content(content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Extract all content to a new file descriptor
        extract_xml = manager.read_fd_content(
            fd_id, read_all=True, extract_to_new_fd=True
        )
        # For testing compatibility, wrap in ToolResult
        extract_result = ToolResult(content=extract_xml, is_error=False)

        # Extract the new FD ID
        new_fd_id = extract_result.content.split('new_fd="')[1].split('"')[0]

        # Read from the new FD to verify content
        new_fd_xml = manager.read_fd_content(new_fd_id, read_all=True)
        # For testing compatibility, wrap in ToolResult
        new_fd_content = ToolResult(content=new_fd_xml, is_error=False)
        new_fd_text = new_fd_content.content.split(">\n")[1].split("\n</fd_content")[0]

        # Should contain the entire original content
        assert new_fd_text == content


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
async def test_fd_to_file_modes():
    """Test fd_to_file with different modes."""
    # Create temporary files for testing
    import os
    import tempfile

    # Mock process with FD manager
    process = Mock()
    process.fd_manager = FileDescriptorManager()
    process.file_descriptor_enabled = True

    # Create test content
    test_content = "This is test content for fd_to_file modes"

    # Create a file descriptor
    fd_xml = process.fd_manager.create_fd_content(test_content)
    fd_id = fd_xml.split('fd="')[1].split('"')[0]

    # Use a temporary directory for the test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test write mode (default)
        file_path_write = os.path.join(temp_dir, "test_write.txt")
        write_result = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_write,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the file was created with the content
        assert os.path.exists(file_path_write)
        with open(file_path_write) as f:
            content = f.read()
            assert content == test_content

        # Test append mode
        file_path_append = os.path.join(temp_dir, "test_append.txt")

        # First write to create the file
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

        # Verify content was appended
        with open(file_path_append) as f:
            content = f.read()
            assert content == test_content + test_content

        # Check that mode is reported in the result
        assert 'mode="append"' in append_result.content


@pytest.mark.asyncio
async def test_fd_to_file_create_and_exist_ok():
    """Test fd_to_file with create and exist_ok parameters."""
    import os
    import tempfile

    # Mock process with FD manager
    process = Mock()
    process.fd_manager = FileDescriptorManager()
    process.file_descriptor_enabled = True

    # Create test content
    test_content = "This is test content for fd_to_file parameters"

    # Create a file descriptor
    fd_xml = process.fd_manager.create_fd_content(test_content)
    fd_id = fd_xml.split('fd="')[1].split('"')[0]

    # Use a temporary directory for the test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Default behavior (create=True, exist_ok=True)
        # Should create a new file
        file_path_1 = os.path.join(temp_dir, "test_default.txt")
        result_1 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_1,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the file was created
        assert os.path.exists(file_path_1)
        assert 'success="true"' in result_1.content
        assert 'create="true"' in result_1.content
        assert 'exist_ok="true"' in result_1.content

        # Test 2: Overwrite existing (create=True, exist_ok=True)
        # Should overwrite the file
        result_2 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_1,  # Same file
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the file was overwritten
        assert 'success="true"' in result_2.content

        # Test 3: Create only if doesn't exist (create=True, exist_ok=False)
        # Should fail because file exists
        file_path_3 = file_path_1  # Same file, should exist
        result_3 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_3,
            exist_ok=False,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the operation failed
        assert "<fd_error type=" in result_3.content
        assert "already exists and exist_ok=False" in result_3.content

        # Test 4: Create only if doesn't exist (create=True, exist_ok=False)
        # Should succeed with new file
        file_path_4 = os.path.join(temp_dir, "test_new_only.txt")
        result_4 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_4,
            exist_ok=False,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the file was created
        assert os.path.exists(file_path_4)
        assert 'success="true"' in result_4.content

        # Test 5: Update existing only (create=False, exist_ok=True)
        # Should succeed with existing file
        file_path_5 = file_path_1  # Use existing file
        result_5 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_5,
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the operation succeeded
        assert 'success="true"' in result_5.content
        assert 'create="false"' in result_5.content

        # Test 6: Update existing only (create=False, exist_ok=True)
        # Should fail with non-existing file
        file_path_6 = os.path.join(temp_dir, "test_nonexistent.txt")
        result_6 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_6,
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the operation failed
        assert "<fd_error type=" in result_6.content
        assert "doesn't exist and create=False" in result_6.content

        # Test 7: Append mode with create=True (append and create if needed)
        file_path_7 = os.path.join(temp_dir, "test_append_create.txt")
        result_7 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_7,
            mode="append",
            create=True,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the file was created
        assert os.path.exists(file_path_7)
        assert 'success="true"' in result_7.content
        assert 'mode="append"' in result_7.content

        # Test 8: Append with create=False (only append to existing)
        # Should fail with non-existing file
        file_path_8 = os.path.join(temp_dir, "test_append_fail.txt")
        result_8 = await fd_to_file_tool(
            fd=fd_id,
            file_path=file_path_8,
            mode="append",
            create=False,
            runtime_context={"fd_manager": process.fd_manager},
        )

        # Verify the operation failed
        assert "<fd_error type=" in result_8.content
        assert "doesn't exist and create=False" in result_8.content


@pytest.mark.asyncio
async def test_fd_integration_end_to_end():
    """Test integration of the enhanced file descriptor API with the system."""
    # Mock the provider client to avoid actual API calls
    with patch("llmproc.providers.providers.get_provider_client") as mock_get_provider:
        mock_client = Mock()
        mock_get_provider.return_value = mock_client

        # Create a program with file descriptor support
        from unittest.mock import MagicMock

        from tests.conftest import create_mock_llm_program

        program = create_mock_llm_program(enabled_tools=["read_fd"])

        # Create a process with minimal required parameters
        process = LLMProcess(
            program=program,
            model_name="test-model",
            provider="test-provider",
            original_system_prompt="Test prompt",
            system_prompt="Test prompt",
            tool_manager=MagicMock(),
        )

        # Manually enable file descriptors
        process.file_descriptor_enabled = True
        process.fd_manager = FileDescriptorManager()

        # Create a file descriptor with test content
        test_content = "This is test content for fd extraction\n" * 10
        fd_xml = process.fd_manager.create_fd_content(test_content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Create registry and tool handlers
        from llmproc.tools import ToolRegistry

        registry = ToolRegistry()

        # Create a handler that uses runtime_context
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

        # Register our handler
        registry.register_tool(
            "read_fd",
            read_fd_handler,
            {
                "name": "read_fd",
                "description": "Read file descriptor",
                "parameters": {},
            },
        )

        # Get the read_fd handler
        handler = registry.get_handler("read_fd")

        # Call the handler with extract_to_new_fd=True
        result = await handler({"fd": fd_id, "start": 1, "extract_to_new_fd": True})

        # Check result format
        assert "<fd_extraction" in result.content
        assert "new_fd" in result.content

        # Extract the new FD ID from the result
        new_fd_id = result.content.split('new_fd="')[1].split('"')[0]

        # Verify the new FD exists
        assert new_fd_id in process.fd_manager.file_descriptors


@pytest.mark.asyncio
async def test_enhanced_fd_workflow():
    """Test a complete enhanced FD workflow with typical operations sequence."""
    import os
    import tempfile

    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup a process with FD manager
        process = Mock()
        process.fd_manager = FileDescriptorManager(
            default_page_size=1000
        )  # Larger page size
        process.file_descriptor_enabled = True
        process.enabled_tools = ["read_fd", "fd_to_file"]

        # Create registry and register tools
        from llmproc.tools import ToolRegistry

        registry = ToolRegistry()

        # Create handlers that use runtime_context instead of process
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

        async def fd_to_file_handler(args):
            return await fd_to_file_tool(
                fd=args.get("fd"),
                file_path=args.get("file_path"),
                mode=args.get("mode", "write"),
                create=args.get("create", True),
                exist_ok=args.get("exist_ok", True),
                runtime_context={"fd_manager": process.fd_manager},
            )

        # Register our context-aware handlers
        registry.register_tool(
            "read_fd",
            read_fd_handler,
            {
                "name": "read_fd",
                "description": "Read file descriptor",
                "parameters": {},
            },
        )

        registry.register_tool(
            "fd_to_file",
            fd_to_file_handler,
            {"name": "fd_to_file", "description": "Write FD to file", "parameters": {}},
        )

        # Create test content (simple to avoid pagination issues)
        test_content = "Test content for file descriptor workflow\n" * 10

        # Create a file descriptor with the test content
        fd_xml = process.fd_manager.create_fd_content(test_content)
        fd_id = fd_xml.split('fd="')[1].split('"')[0]

        # Get handlers for the tools
        read_handler = registry.get_handler("read_fd")
        fd_to_file_handler = registry.get_handler("fd_to_file")

        # Step 1: Read the content
        read_result = await read_handler({"fd": fd_id, "start": 1})

        # Verify we got some content
        assert "<fd_content" in read_result.content

        # Step 2: Extract content to new FD
        extract_result = await read_handler(
            {"fd": fd_id, "start": 1, "extract_to_new_fd": True}
        )

        # Verify extraction was successful
        assert "<fd_extraction" in extract_result.content
        new_fd_id = extract_result.content.split('new_fd="')[1].split('"')[0]
        assert new_fd_id in process.fd_manager.file_descriptors

        # Step 3: Create a file with the new FD
        output_file_1 = os.path.join(temp_dir, "output1.txt")
        create_result = await fd_to_file_handler(
            {"fd": new_fd_id, "file_path": output_file_1}
        )

        # Verify create operation
        assert os.path.exists(output_file_1)
        assert 'success="true"' in create_result.content

        # Step 4: Test exist_ok=False on existing file
        fail_result = await fd_to_file_handler(
            {"fd": new_fd_id, "file_path": output_file_1, "exist_ok": False}
        )

        # Verify operation failed with right error
        assert "<fd_error type=" in fail_result.content
        assert "already exists and exist_ok=False" in fail_result.content

        # Step 5: Append to existing file
        append_result = await fd_to_file_handler(
            {"fd": new_fd_id, "file_path": output_file_1, "mode": "append"}
        )

        # Verify append operation
        assert 'mode="append"' in append_result.content

        # Check file size (should be doubled from append)
        with open(output_file_1) as f:
            content = f.read()
            # Content should be duplicated because we appended
            original_size = len(
                process.fd_manager.file_descriptors[new_fd_id]["content"]
            )
            assert len(content) >= original_size * 2

        # Step 6: Test create=False on non-existing file
        new_file = os.path.join(temp_dir, "nonexistent.txt")
        update_only_result = await fd_to_file_handler(
            {"fd": new_fd_id, "file_path": new_file, "create": False}
        )

        # Verify operation failed with right error
        assert "<fd_error type=" in update_only_result.content
        assert "doesn't exist and create=False" in update_only_result.content

        # Step 7: Create a new file with "create only" mode
        new_file_2 = os.path.join(temp_dir, "newonly.txt")
        create_only_result = await fd_to_file_handler(
            {"fd": new_fd_id, "file_path": new_file_2, "exist_ok": False}
        )

        # Verify create_only operation
        assert os.path.exists(new_file_2)
        assert 'success="true"' in create_only_result.content
        assert 'exist_ok="false"' in create_only_result.content
