# RFC019: Program Serialization and Export

## Status
Proposed

## Summary
This RFC proposes a comprehensive approach to serializing and exporting LLMProgram configurations. It introduces a standardized directory structure for program exports and defines how linked programs, prompts, and preloaded files should be organized to enable easy sharing and reloading.

## Motivation
As LLMProc applications become more complex, developers need ways to:

1. Save and reload program configurations for deployment and sharing
2. Export/import complex linked program structures
3. Create versioned program snapshots for reproducibility
4. Share program configurations with predictable directory structures

Currently, while programs can be configured via TOML files, there's no standardized way to export a programmatically created configuration with all its dependencies in a portable format.

## Detailed Design

### 1. Program Export Structure

Programs will be serialized to a directory structure with the following organization:

```
exported_project/
├── main.toml            # Main program configuration
├── prompts/             # System prompts
│   ├── main_prompt.md
│   ├── expert1_prompt.md
│   └── expert2_prompt.md
├── preloaded/           # Preloaded content files
│   ├── context1.md
│   └── reference2.json
└── linked/              # Linked program configurations
    ├── expert1.toml
    └── expert2.toml
```

The main TOML file will use relative paths to reference prompts, preloaded files, and linked programs.

Example of exported `main.toml`:

```toml
[model]
name = "claude-3-7-sonnet"
provider = "anthropic"

[prompt]
system_prompt_file = "./prompts/main_prompt.md"

[preload]
files = [
    "./preloaded/context1.md",
    "./preloaded/reference2.json"
]

[tools]
enabled = ["spawn", "calculator"]

[linked_programs]
expert1 = "./linked/expert1.toml"
expert2 = "./linked/expert2.toml"
```

### 2. Export API

The export feature will be implemented as methods on the `LLMProgram` class:

```python
# First ensure the program is fully compiled
compiled_program = LLMProgram.compile(main_program, include_linked=True)

# Export a program to a specified directory
compiled_program.export_to_directory(
    path="./my_project/",
    include_linked=True,        # Export linked programs recursively
    include_prompts=True,       # Export system prompts as files
    include_preloaded=True,     # Export preloaded files
    create_relative_paths=True  # Convert paths to relative in the exported TOML
)

# Export with specific options
compiled_program.export_to_directory(
    path="./minimal_export/", 
    include_linked=False,       # Only export the main program
    include_prompts=False,      # Keep prompts inline
    include_preloaded=False     # Keep preload references to original locations
)
```

### 3. Program Metadata

Programs will include metadata to track version information:

```python
# The export will include a metadata.json file
# ./exported_project/metadata.json

{
    "version": "1.0",
    "export_date": "2025-04-01T10:00:00",
    "llmproc_version": "0.5.2",
    "description": "Main program with billing and tech experts",
    "programs": {
        "main": {
            "model": "claude-3-7-sonnet",
            "linked_programs": ["expert1", "expert2"]
        },
        "expert1": {
            "model": "claude-3-7-sonnet",
            "preloaded_files": 2
        },
        "expert2": {
            "model": "claude-3-7-sonnet",
            "preloaded_files": 3
        }
    }
}
```

### 4. Program Import

Importing will automatically resolve and load all related files:

```python
# Import from an exported project
program = LLMProgram.load_from_directory("./my_project/")

# Equivalent to:
# LLMProgram.from_toml("./my_project/main.toml") 
# but with better handling of included files
```

### 5. Verification and Validation

The export process will include verification steps:

```python
# Validate an exported program structure
validation_result = LLMProgram.validate_export("./my_project/")
if validation_result.is_valid:
    print("Export is valid")
else:
    print(f"Export validation failed: {validation_result.errors}")
```

## Implementation Plan

1. **Phase 1: Basic Export Structure**
   - Implement `export_to_directory` method
   - Create directory structure with subdirectories
   - Export main program configuration
   - Handle paths and references properly

2. **Phase 2: Linked Program Export**
   - Implement recursive export of linked programs
   - Ensure proper path references between configurations
   - Validate linked program relationships

3. **Phase 3: Import and Verification**
   - Implement `load_from_directory` functionality
   - Add validation for exported structures
   - Create helpers for loading with custom overrides

4. **Phase 4: Metadata and Documentation**
   - Add metadata file generation 
   - Generate export documentation
   - Create export integrity checks

## Benefits

1. **Reproducibility**: Exported programs can be reliably reconstructed
2. **Shareability**: Complete program structures can be shared as a single directory
3. **Versioning**: Program configurations can be versioned in source control
4. **Deployment**: Consistent structure for deployment packaging
5. **Documentation**: Self-documenting structure for program relationships

## Compatibility

The serialization format is designed to maintain compatibility with existing TOML-based configurations but adds structure and consistency. All paths in exported TOMLs will be made relative to enable portability.

## Open Questions

1. Should we include a manifest file to track versions and dependencies?
2. How should we handle sensitive information in exported programs (API keys, etc.)?
3. Should we support different levels of export verbosity?
4. How should we handle conflicts when importing (e.g., same file names in different locations)?
5. Should we implement a specific compression format for exports?

## References

- [Program Linking Documentation](../docs/program-linking.md)
- [File Descriptor System Documentation](../docs/file-descriptor-system.md)
- RFC018: SDK Developer Experience