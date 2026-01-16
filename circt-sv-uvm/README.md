# circt-sv-uvm

A Claude Code plugin for SystemVerilog and UVM development with CIRCT LSP integration.

## Features

### MCP Integration
- **circt-verilog-lsp-server**: Language server for SystemVerilog/Verilog (`.sv`, `.svh`, `.v`, `.vh`)
  - Go to definition
  - Find references
  - Hover information
  - Document symbols
  - Diagnostics
  - Code completion
  - Rename symbol
  - Semantic tokens
  - Inlay hints

- **circt-lsp-server**: Language server for MLIR/CIRCT dialects (FIRRTL, HW, Comb, SV)
  - Go to definition
  - Find references
  - Hover information
  - Document symbols
  - Diagnostics
  - Code completion

### Commands

| Command | Description |
|---------|-------------|
| `/lint-sv <path>` | Run SystemVerilog linting and static analysis |
| `/analyze-coverage <file>` | Analyze functional coverage results |
| `/generate-uvm-component <type> <name>` | Generate UVM component boilerplate |
| `/generate-sva <description>` | Generate SystemVerilog Assertions |

### Skills

- **UVM Methodology**: Testbench architecture, agents, sequences, factory, phases, RAL, coverage
- **SystemVerilog Patterns**: Interfaces, clocking blocks, constraints, classes, assertions

## Installation

### Option 1: Local Plugin
```bash
claude --plugin-dir ./circt-sv-uvm
```

### Option 2: Project Plugin
Copy to your project's `.claude-plugin/` directory.

## Prerequisites

- **CIRCT build**: The LSP servers require a built CIRCT with Slang enabled:
  ```bash
  cmake -DCIRCT_SLANG_FRONTEND_ENABLED=ON ...
  ninja circt-verilog-lsp-server circt-lsp-server
  ```

  Set the build directory if not using `./build`:
  ```bash
  export CIRCT_BUILD_DIR=/path/to/circt/build
  ```

- **Optional linting tools** (for `/lint-sv`):
  - `slang` (preferred)
  - `verilator`
  - `iverilog`

## Usage Examples

### Linting
```
/lint-sv src/rtl/
/lint-sv my_module.sv
```

### Generate UVM Components
```
/generate-uvm-component agent apb
/generate-uvm-component sequence axi_write
/generate-uvm-component scoreboard my_checker
```

### Generate Assertions
```
/generate-sva valid-ready handshake
/generate-sva request acknowledged within 10 cycles
```

### Coverage Analysis
```
/analyze-coverage coverage_report.txt
```

## LSP Status

| Server | Status | File Types |
|--------|--------|------------|
| `circt-verilog-lsp-server` | Active | `.sv`, `.svh`, `.v`, `.vh` |
| `circt-lsp-server` | Active | `.mlir` |

Both LSP servers are fully functional. The Verilog LSP requires building CIRCT with `CIRCT_SLANG_FRONTEND_ENABLED=ON`.

## Project Structure

```
circt-sv-uvm/
├── .claude-plugin/
│   └── plugin.json
├── .mcp.json
├── commands/
│   ├── lint-sv.md
│   ├── analyze-coverage.md
│   ├── generate-uvm-component.md
│   └── generate-sva.md
├── skills/
│   ├── uvm-methodology/
│   │   └── SKILL.md
│   └── systemverilog-patterns/
│       └── SKILL.md
└── README.md
```

## License

Same as CIRCT project.
