# LSP Diagnostics Publishing - Feature Demonstration

## Overview

The CIRCT Verilog LSP server provides comprehensive diagnostic publishing that reports errors, warnings, and other issues to LSP clients in real-time.

## Feature Capabilities

### 1. Error Detection

**Input Code:**
```systemverilog
module test;
  wire x
  // missing semicolon
endmodule
```

**Published Diagnostic:**
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "diagnostics": [
      {
        "message": "expected ';'",
        "range": {
          "start": {"line": 1, "character": 8},
          "end": {"line": 1, "character": 8}
        },
        "severity": 1,
        "source": "slang"
      }
    ],
    "uri": "test:///test.sv",
    "version": 1
  }
}
```

### 2. Warning Detection

**Input Code:**
```systemverilog
module test;
  logic [7:0] a;
  logic [15:0] b;
  initial b = a;  // width mismatch
endmodule
```

**Published Diagnostic:**
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "diagnostics": [
      {
        "message": "implicit conversion expands from 8 to 16 bits",
        "range": {
          "start": {"line": 3, "character": 14},
          "end": {"line": 3, "character": 15}
        },
        "severity": 2,
        "source": "slang",
        "relatedInformation": [
          {
            "location": {
              "uri": "test:///test.sv",
              "range": {
                "start": {"line": 3, "character": 12},
                "end": {"line": 3, "character": 13}
              }
            },
            "message": "related location"
          }
        ]
      }
    ],
    "uri": "test:///test.sv",
    "version": 1
  }
}
```

### 3. Real-time Updates

When the user fixes the code, diagnostics are updated immediately:

**Fixed Code:**
```systemverilog
module test;
  logic a;
endmodule
```

**Published Diagnostic (Empty):**
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "diagnostics": [],
    "uri": "test:///test.sv",
    "version": 2
  }
}
```

### 4. Diagnostic Clearing on Close

When a document is closed, diagnostics are cleared:

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "diagnostics": [],
    "uri": "test:///test.sv",
    "version": 1
  }
}
```

## Severity Levels

The LSP server maps slang diagnostic severities to LSP severity levels:

| Slang Severity | LSP Severity | Value | Description |
|----------------|--------------|-------|-------------|
| Fatal          | Error        | 1     | Fatal compilation errors |
| Error          | Error        | 1     | Syntax and semantic errors |
| Warning        | Warning      | 2     | Potential issues and warnings |
| Note           | Information  | 3     | Informational messages |
| Ignored        | Information  | 3     | Suppressed diagnostics |

## Diagnostic Information

Each diagnostic includes:

- **message**: Human-readable error/warning message
- **range**: Source code location (line and column range)
- **severity**: Error level (1=Error, 2=Warning, 3=Info)
- **source**: Always "slang" to indicate the diagnostic source
- **relatedInformation** (optional): Additional locations related to the diagnostic
- **uri**: File URI
- **version**: Document version number

## Testing

Run the diagnostic tests:

```bash
# Build the LSP server
cd ~/circt/build
ninja circt-verilog-lsp-server

# Run diagnostic tests (when lit is properly configured)
lit test/Tools/circt-verilog-lsp-server/diagnostic.test
lit test/Tools/circt-verilog-lsp-server/diagnostics.test
lit test/Tools/circt-verilog-lsp-server/diagnostics-comprehensive.test
```

## Integration with Editors

### VS Code
Diagnostics appear automatically in:
- The "Problems" panel
- Inline squiggly underlines in the editor
- Hover tooltips

### Neovim
With nvim-lspconfig, diagnostics are shown via:
- Virtual text at the end of lines
- Sign column indicators
- Floating windows on hover

### Emacs
With eglot, diagnostics appear as:
- Flycheck/Flymake overlays
- Mode line indicators
- Echo area messages

## Performance

The LSP server uses debouncing to optimize diagnostic publishing:
- **Minimum delay**: 150ms after last edit
- **Maximum delay**: 500ms even during continuous typing
- **Configurable**: Can be disabled with `--no-debounce` flag

## Thread Safety

Diagnostic publishing is thread-safe:
- Mutex-protected publication queue
- Serialized diagnostic messages
- Safe concurrent document updates
