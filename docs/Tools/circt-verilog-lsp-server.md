# circt-verilog-lsp-server

The `circt-verilog-lsp-server` is a Language Server Protocol (LSP) implementation for Verilog and SystemVerilog that provides rich IDE features. It leverages the slang frontend for accurate parsing and semantic analysis.

## Features

### Core Language Features

- **Go to Definition**: Navigate to the definition of modules, signals, parameters, functions, and tasks
- **Find References**: Find all usages of a symbol across the workspace
- **Hover Information**: View detailed information about symbols including types, dimensions, and documentation
- **Document Symbols**: Navigate the structure of a file with outline view
- **Workspace Symbols**: Search for symbols across the entire workspace

### Advanced Features

- **Auto-completion**: Context-aware completions for:
  - Module names and instantiations
  - Port names and connections
  - Signal and variable names
  - Keywords and snippets
  - Preprocessor macros

- **Code Actions**: Quick fixes for common issues:
  - Add missing module ports
  - Fix signal width mismatches
  - Apply naming conventions

- **Rename Symbol**: Safely rename signals, modules, and other identifiers across files

- **Document Links**: Clickable links for include directives

- **Semantic Highlighting**: Enhanced syntax highlighting with semantic information:
  - Distinguishes between signals, parameters, ports
  - Highlights module instantiations
  - Marks preprocessor directives

- **Inlay Hints**: Inline hints showing:
  - Parameter values at instantiation sites
  - Inferred signal widths
  - Port direction indicators

### Diagnostics

- **Real-time Error Detection**: Parse errors and semantic issues as you type
- **Rich Diagnostic Information**:
  - Source locations with line/column numbers
  - Related information showing connected issues
  - Suggested fixes where applicable
- **Configurable Linting**: Enable/disable specific lint rules

### Project Configuration

- **Multi-root Workspace Support**: Work with multiple project roots
- **Automatic Configuration Discovery**: Finds `circt-project.yaml` automatically
- **Include Path Resolution**: Configurable include directories
- **Preprocessor Defines**: Support for define flags

## Installation

### Building from Source

```bash
# Build CIRCT with the LSP server enabled
cmake -G Ninja ../llvm/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=circt \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON

ninja circt-verilog-lsp-server
```

### Pre-built Binaries

Download from the [CIRCT releases page](https://github.com/llvm/circt/releases).

## Usage

### Command Line

```bash
circt-verilog-lsp-server [options]
```

### Options

| Option | Description |
|--------|-------------|
| `-y <dir>`, `--libdir=<dir>` | Library search paths for missing modules |
| `-C <file>`, `--command-file=<file>` | Command file with project dependencies |
| `--source-location-include-dir=<dir>` | Additional source location directories |
| `--log=<level>` | Log level: error, info, verbose |
| `--no-debounce` | Disable debouncing (rebuild synchronously) |
| `--debounce-min-ms=<ms>` | Minimum idle time before rebuild (default: 150) |
| `--debounce-max-ms=<ms>` | Maximum wait while edits continue (default: 500) |
| `--lit-test` | Testing mode with delimited input |

### VS Code Integration

1. Install the "CIRCT Verilog" extension (coming soon)
2. Or manually configure in `.vscode/settings.json`:

```json
{
  "verilog.languageServer.path": "/path/to/circt-verilog-lsp-server",
  "verilog.languageServer.arguments": [
    "-y", "${workspaceFolder}/lib",
    "--source-location-include-dir=${workspaceFolder}/rtl"
  ]
}
```

### Neovim Integration

Using [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig):

```lua
local lspconfig = require('lspconfig')

lspconfig.circt_verilog.setup{
  cmd = { "circt-verilog-lsp-server" },
  filetypes = { "verilog", "systemverilog" },
  root_dir = lspconfig.util.root_pattern("circt-project.yaml", ".git"),
}
```

### Emacs Integration

Using [eglot](https://github.com/joaotavora/eglot):

```elisp
(add-to-list 'eglot-server-programs
  '((verilog-mode systemverilog-mode) . ("circt-verilog-lsp-server")))
```

## Project Configuration

Create `circt-project.yaml` in your project root:

```yaml
project:
  name: "my_design"
  top: "top_module"
  version: "1.0.0"

sources:
  include_dirs:
    - "rtl/"
    - "includes/"
  defines:
    - "SYNTHESIS"
    - "DEBUG_LEVEL=2"
  files:
    - "rtl/**/*.sv"
    - "tb/**/*.sv"
  file_lists:
    - "project.f"

lint:
  enabled: true
  config: "lint.yaml"

simulation:
  timescale: "1ns/1ps"

targets:
  synthesis:
    top: "chip_top"
    defines: ["SYNTHESIS"]
  simulation:
    top: "tb_top"
    defines: ["SIMULATION"]
```

## File List Support

The server supports standard file list (`.f`) format:

```
// Comment
# Also a comment
+incdir+rtl/includes
+define+SYNTHESIS
+define+DEBUG_LEVEL=2
-y lib/cells
-f dependencies.f
rtl/module1.sv
rtl/module2.sv
```

## LSP Protocol Support

### Implemented Methods

| Method | Status |
|--------|--------|
| `textDocument/didOpen` | Supported |
| `textDocument/didChange` | Supported |
| `textDocument/didClose` | Supported |
| `textDocument/didSave` | Supported |
| `textDocument/definition` | Supported |
| `textDocument/references` | Supported |
| `textDocument/hover` | Supported |
| `textDocument/documentSymbol` | Supported |
| `textDocument/completion` | Supported |
| `textDocument/codeAction` | Supported |
| `textDocument/rename` | Supported |
| `textDocument/prepareRename` | Supported |
| `textDocument/documentLink` | Supported |
| `textDocument/semanticTokens/full` | Supported |
| `textDocument/inlayHint` | Supported |
| `workspace/workspaceFolders` | Supported |

### Semantic Token Types

| Token Type | Description |
|------------|-------------|
| `namespace` | Module names |
| `type` | Type names |
| `class` | Class definitions |
| `enum` | Enum types |
| `interface` | Interface definitions |
| `struct` | Struct types |
| `parameter` | Parameters and localparams |
| `variable` | Signals, wires, regs |
| `property` | Properties |
| `function` | Functions |
| `method` | Methods |
| `macro` | Preprocessor macros |
| `comment` | Comments |
| `string` | String literals |
| `number` | Numeric literals |
| `keyword` | Keywords |
| `operator` | Operators |

## Performance Considerations

### Debouncing

The server uses intelligent debouncing to balance responsiveness with resource usage:

- **Minimum delay**: Waits at least 150ms after the last edit before rebuilding
- **Maximum delay**: Ensures rebuilds happen within 500ms even during continuous typing
- **Disable**: Use `--no-debounce` for testing or when low latency is critical

### Large Projects

For large projects:

1. Use `file_lists` to specify exactly which files to include
2. Limit `include_dirs` to necessary paths
3. Consider using target-specific configurations to reduce scope
4. Monitor memory usage and adjust workspace folder scope

## Troubleshooting

### Common Issues

#### "Module not found" errors

1. Check that `include_dirs` in `circt-project.yaml` includes all necessary paths
2. Use `-y` flag for library directories
3. Verify file extensions are correct (`.sv` for SystemVerilog)

#### Slow response times

1. Enable debouncing (remove `--no-debounce`)
2. Reduce the scope of `files` patterns
3. Use more specific include directories

#### Missing completions

1. Ensure the file parses successfully (check for errors first)
2. Verify include files are found
3. Check that defines are correctly specified

### Logging

Enable verbose logging to diagnose issues:

```bash
circt-verilog-lsp-server --log=verbose 2>&1 | tee lsp.log
```

## Development

### Testing

Run the LSP server test suite:

```bash
ninja check-circt-verilog-lsp-server
```

Use lit-test mode for integration tests:

```bash
circt-verilog-lsp-server --lit-test < test-input.txt
```

### Protocol Tracing

Set the log level to verbose to see all LSP messages:

```bash
circt-verilog-lsp-server --log=verbose
```

## Related Documentation

- [circt-verilog](circt-verilog.md) - Verilog/SystemVerilog compiler frontend
- [Project Configuration](../ci-integration/README.md) - CI/CD integration guide
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) - LSP specification
