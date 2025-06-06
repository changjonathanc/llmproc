# Release Notes - v0.9.3

## 🚀 New Features

### Cost Control
- **Cost Limit Feature**: New `--cost-limit` option for both `llmproc` and `llmproc-demo` CLIs to set budget limits in USD
- **Proactive Cost Control**: Early termination when costs exceed limits
- **Enhanced Cost Tracking**: USD cost included in JSON output

### Enhanced Callbacks
- **Smart Parameter Passing**: Callbacks support intelligent parameter filtering
- **RunResult Support**: TURN_START callbacks can receive `run_result` for budget checking
- **Full Backward Compatibility**: Existing callbacks continue to work unchanged

### Improved Tool Registration
- **Class Method Support**: Enhanced `@register_tool` decorator for class methods
- **Instance Method Tools**: Better support for tools from class instances

## 🛠️ Improvements
- **GitHub Actions Security**: Fixed script injection vulnerabilities
- **GitHub Workflows**: Support empty `/code` queries when issue body is not empty
- **CLI**: Added `--upgrade` option and GitHub CLI secret setup commands to `llmproc-install-actions`
- **Documentation**: Simplified tool registration examples
- **Bug Fixes**: Claude pricing, tool registration edge cases

## Breaking Changes
- **Stop Reason API**: `process.run_stop_reason` deprecated in favor of `run_result.stop_reason`

---

For detailed documentation, visit the [docs](https://github.com/cccntu/llmproc/tree/main/docs).
