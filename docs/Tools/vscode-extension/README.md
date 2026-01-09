# CIRCT Verilog VS Code Extension

This directory contains the configuration for the CIRCT Verilog VS Code extension, which provides Verilog/SystemVerilog language support powered by the CIRCT project.

## Features

- **Language Server Integration**: Connects to `circt-verilog-lsp-server` for:
  - Go to Definition
  - Find References
  - Hover Information
  - Auto-completion
  - Code Actions
  - Rename Symbol
  - Document Outline
  - Semantic Highlighting
  - Inlay Hints

- **Syntax Highlighting**: TextMate grammars for Verilog and SystemVerilog

- **Code Snippets**: Common code patterns for:
  - Module declarations
  - Always blocks (ff, comb, latch)
  - FSM templates
  - Functions and tasks
  - Assertions and coverage

- **Problem Matchers**: Parse compiler output for quick navigation

## Installation

### From VSIX

1. Download the `.vsix` file from releases
2. In VS Code, press `Ctrl+Shift+P` and select "Extensions: Install from VSIX..."
3. Select the downloaded file

### From Source

```bash
cd docs/Tools/vscode-extension
npm install
npm run compile
vsce package
```

## Configuration

### Basic Setup

1. Install the `circt-verilog-lsp-server` binary
2. Configure the path in VS Code settings:

```json
{
  "circtVerilog.server.path": "/path/to/circt-verilog-lsp-server"
}
```

### Project Configuration

Create a `circt-project.yaml` in your workspace root. The extension will automatically detect it.

### Include Directories

Configure include directories via:
- `circt-project.yaml` (recommended)
- VS Code settings:

```json
{
  "circtVerilog.includeDirs": [
    "${workspaceFolder}/rtl",
    "${workspaceFolder}/includes"
  ]
}
```

### Preprocessor Defines

```json
{
  "circtVerilog.defines": [
    "SYNTHESIS",
    "DEBUG_LEVEL=2"
  ]
}
```

## Commands

| Command | Description |
|---------|-------------|
| `CIRCT Verilog: Restart Language Server` | Restart the LSP server |
| `CIRCT Verilog: Show Output` | Show server output log |
| `CIRCT Verilog: Generate Project Configuration` | Create a `circt-project.yaml` template |

## Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `circtVerilog.server.path` | string | `circt-verilog-lsp-server` | Path to server binary |
| `circtVerilog.server.arguments` | array | `[]` | Additional server arguments |
| `circtVerilog.server.enabled` | boolean | `true` | Enable language server |
| `circtVerilog.server.trace` | string | `off` | Trace level (off/messages/verbose) |
| `circtVerilog.lint.enabled` | boolean | `true` | Enable linting |
| `circtVerilog.lint.configFile` | string | `""` | Lint configuration file |
| `circtVerilog.includeDirs` | array | `[]` | Include directories |
| `circtVerilog.defines` | array | `[]` | Preprocessor defines |
| `circtVerilog.libraryDirs` | array | `[]` | Library directories |
| `circtVerilog.semanticHighlighting.enabled` | boolean | `true` | Enable semantic highlighting |
| `circtVerilog.inlayHints.enabled` | boolean | `true` | Enable inlay hints |

## Semantic Token Types

The extension provides semantic highlighting for:

| Token Type | Description |
|------------|-------------|
| `namespace` | Module names |
| `type` | Type names |
| `class` | Class definitions |
| `enum` | Enum types |
| `interface` | Interface definitions |
| `parameter` | Parameters |
| `variable` | Signals and variables |
| `function` | Functions |
| `method` | Methods |
| `macro` | Preprocessor macros |
| `comment` | Comments |
| `keyword` | Keywords |
| `operator` | Operators |

## Snippets

Type the prefix and press `Tab` to expand:

| Prefix | Description |
|--------|-------------|
| `module` | Module declaration |
| `modulep` | Module with parameters |
| `interface` | Interface declaration |
| `always_ff` | Flip-flop always block |
| `always_comb` | Combinational always block |
| `case` | Case statement |
| `fsm` | FSM template |
| `function` | Function declaration |
| `task` | Task declaration |
| `assert` | Assertion |
| `property` | Property with assertion |
| `covergroup` | Coverage group |
| `class` | Class declaration |
| `package` | Package declaration |

## Troubleshooting

### Server not starting

1. Check the server path is correct
2. Verify the server is executable
3. Check the Output panel for errors (`View > Output > CIRCT Verilog`)

### Missing completions

1. Ensure the file parses without errors
2. Check include directories are configured
3. Verify defines are set correctly

### Performance issues

1. Limit workspace folders to necessary directories
2. Use specific file patterns in `circt-project.yaml`
3. Consider disabling inlay hints for large files

## Development

### Building

```bash
npm install
npm run compile
```

### Testing

```bash
npm run test
```

### Packaging

```bash
vsce package
```

## Contributing

Contributions are welcome! Please see the [CIRCT Contributing Guide](https://circt.llvm.org/docs/contributing/).

## License

Apache-2.0 with LLVM Exceptions
