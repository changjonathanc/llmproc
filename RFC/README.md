# LLMProc RFCs (Request for Comments)

This directory contains Request for Comments (RFC) documents for LLMProc features. These RFCs describe proposed features, implementation details, and design decisions.

## RFC Status

| RFC | Title | Status | Implementation Date |
|-----|-------|--------|---------------------|
| [RFC001](RFC001_file_descriptor_system.md) | File Descriptor System for LLMProc | ✅ Implemented | - |
| [RFC003](RFC003_file_descriptor_implementation.md) | File Descriptor Implementation Details | ✅ Implemented | - |
| [RFC004](RFC004_fd_implementation_phases.md) | File Descriptor System Implementation Phases | ✅ Implemented | - |
| [RFC005](RFC005_fd_spawn_integration.md) | File Descriptor Integration with Spawn Tool | ✅ Implemented | - |
| [RFC006](RFC006_response_reference_id.md) | Response Reference ID System | ✅ Implemented | - |
| [RFC007](RFC007_fd_enhanced_api_design.md) | Enhanced File Descriptor API Design | ✅ Implemented | - |
| [RFC008](RFC008_openai_reasoning_model_support.md) | OpenAI Reasoning Model Support | ✅ Implemented | March 27, 2025 |
| [RFC009](RFC009_program_linking_descriptions.md) | Enhancing Program Linking with Descriptions | ✅ Implemented | March 28, 2025 |
| [RFC010](RFC010_claude_thinking_model_support.md) | Claude 3.7 Thinking Model Support | ✅ Implemented | March 27, 2025 |
| [RFC011](RFC011_token_efficient_tool_use.md) | Token-Efficient Tool Use for Claude 3.7 | ❌ Planned | - |
| [RFC013](RFC013_prompt_caching_implementation.md) | Automatic Prompt Caching Implementation for Anthropic API | ✅ Implemented | March 27, 2025 |

## RFC Process

The LLMProc RFC process follows these steps:

1. **Proposal**: Create an RFC document with a unique number
2. **Discussion**: Review and discuss the RFC with the team
3. **Refinement**: Update the RFC based on feedback
4. **Implementation**: Implement the feature as described
5. **Verification**: Test and verify the implementation
6. **Completion**: Mark the RFC as implemented

## Creating a New RFC

To create a new RFC:

1. Create a new file named `RFCXXX_title_with_underscores.md` in this directory
2. Use the template below for the RFC content
3. Fill in the relevant sections
4. Submit a PR for review

## RFC Template

```markdown
# RFCXXX: Title of the RFC

## Summary
Brief summary of the proposal.

## Motivation
Why this feature is needed.

## Approach
High-level description of the approach.

## Implementation Details
Technical details of the implementation.

## Example Usage
Example code or configuration showing how to use the feature.

## Backward Compatibility
Impact on existing code and migration strategy.

## Testing Plan
How the feature will be tested.

## Next Steps
Action items for implementation.
```