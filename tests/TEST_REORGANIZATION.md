# Test Suite Reorganization Plan

Now that the Unix-inspired Program/Process refactoring is complete and the architecture is stable, we should refactor the tests for clarity and long-term maintainability. Below are the recommended changes to our test organization based on expert review:

## Organizing Principle
1. **Unit Tests (`tests/unit/`):** Test individual functions/methods in isolation, mocking dependencies
2. **Integration Tests (`tests/`):** Test interactions between components (e.g., `program.start()` flow)
3. **Behavioral Tests:** Test runtime behavior and public API of components

## Specific Recommendations

### 1. `tests/test_program_start.py`
- **Action:** Remove
- **Rationale:** Redundant with `test_program_exec_integration.py` and other tests that use `program.start()`

### 2. `tests/test_program_exec_integration.py`
- **Action:** Keep and enhance
- **Rename to:** `tests/test_process_creation_integration.py`
- **Enhancements:**
  - Add tests for different program configurations (with/without FDs, linked programs, MCP)
  - Ensure it tests the interactions between all initialization steps

### 3. `tests/unit/program_exec/test_phase5d_fork_process.py`
- **Action:** Review and potentially remove/merge
- **Rationale:** Phase-specific tests may be redundant now that `test_fork_process_refactored.py` tests the final behavior
- **Recommendation:** Move any unique tests to `test_fork_process_refactored.py`

### 4. `tests/test_fork_process_refactored.py`
- **Action:** Keep and refine
- **Rename to:** `tests/test_fork_process.py`
- **Refinements:**
  - Explicitly verify deep copying of all state elements
  - Test `allow_fork=False` on child process
  - Test `RuntimeError` when `allow_fork` is already `False`
  - Use `create_test_process` fixture for setup

### 5. `tests/unit/program_exec/test_phase5e_initialization.py`
- **Action:** Keep but rename
- **Rename to:** `tests/unit/test_llmprocess_init_contract.py`
- **Rationale:** Tests the final contract of `LLMProcess.__init__` following the refactoring

### 6. `tests/test_llm_process.py`
- **Action:** Refactor and refocus
- **Changes:**
  - Remove initialization tests (`test_initialization`)
  - Remove tool initialization tests (`test_programexec_initializes_tools`, `test_mcp_tools_initialization`)
  - Refocus `test_tool_calling_works` on runtime behavior
  - Rename `test_async_initialize_tools` to `test_run_handles_deferred_tool_initialization`
  - Keep runtime tests for: `run`, `reset_state`, `get_state`, etc.
  - Use `create_test_process` fixture for all setups

### 7. `tests/unit/program_exec/test_create_process.py`
- **Action:** Keep
- **Refinements:** 
  - Mock all functions called by `create_process` 
  - Verify they're called in the correct order with expected arguments

### 8. `tests/unit/program_exec/test_initialization_functions.py`
- **Action:** Keep
- **Refinements:**
  - Ensure comprehensive coverage for each initialization function
  - Test edge cases (file not found, different configurations)

### 9. `tests/unit/program_exec/test_instantiate_process.py`
- **Action:** Keep
- **Refinements:**
  - Test extra parameters being filtered out
  - Test missing required parameters being caught

## Proposed Target Structure

```
tests/ (Integration & Behavioral Tests)
├── test_process_creation_integration.py
├── test_llm_process.py
├── test_fork_process.py
├── test_spawn_tool.py
├── test_file_descriptor_integration.py
└── ... (Other integration/feature tests)

tests/unit/
├── test_llmprocess_init_contract.py
├── program_exec/
│   ├── test_create_process.py
│   ├── test_initialization_functions.py
│   ├── test_instantiate_process.py
│   ├── test_setup_runtime_context.py
│   ├── test_validate_process.py
│   └── ... (Other unit tests)
├── tools/
│   ├── test_tool_manager.py
│   └── ... (Unit tests for tools)
├── file_descriptors/
│   └── test_manager.py
├── providers/
│   ├── test_anthropic_executor.py
│   └── ... (Other provider tests)
└── ... (Other unit test modules)
```

This reorganization will reduce redundancy, clarify the purpose of each test file, and make the suite easier to navigate and maintain as the project evolves.