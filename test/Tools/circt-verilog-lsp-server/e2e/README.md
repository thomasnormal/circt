# CIRCT Verilog LSP Server - End-to-End Stress Tests

This directory contains comprehensive end-to-end stress tests for the CIRCT Verilog LSP server, designed to exercise all LSP features against real-world SystemVerilog/UVM codebases.

## Quick Start

```bash
# Run all tests
./run_stress_tests.sh

# Run specific phase
./run_stress_tests.sh phase1

# Run with verbose output
./run_stress_tests.sh -v

# Run benchmarks only
./run_stress_tests.sh --benchmark
```

## Prerequisites

1. **CIRCT Build**: The LSP server must be built:
   ```bash
   cd ~/circt/build
   ninja circt-verilog-lsp-server
   ```

2. **OpenTitan** (optional but recommended): Clone OpenTitan for real-world testing:
   ```bash
   cd ~
   git clone https://github.com/lowRISC/opentitan.git
   ```

3. **Python 3.10+**: Required for pytest-lsp

## Test Phases

### Phase 1: Basic Feature Verification (`test_phase1_basic_features.py`)
Tests basic LSP features on small, well-structured files:
- Diagnostics (no false positives)
- Hover information
- Go-to-definition
- Find references
- Document symbols
- Code completion
- Response time targets

**Target file**: `~/opentitan/hw/ip/uart/rtl/uart_tx.sv`

### Phase 2: Large File Stress (`test_phase2_large_files.py`)
Tests performance on large auto-generated files:
- Opening 40K+ LOC files
- Hover/completion in large files
- Document symbols for complex files
- Multiple large files open simultaneously

**Target files**:
- `pinmux_reg_top.sv` (40K LOC)
- `alert_handler_reg_top.sv` (20K LOC)
- `top_earlgrey.sv` (100+ module instantiations)

### Phase 3: Cross-File Navigation (`test_phase3_cross_file.py`)
Tests navigation across the module hierarchy:
- Deep hierarchy drilling
- Package dependency navigation
- Interface/modport handling
- Find references across files

### Phase 4: UVM Testbench Features (`test_phase4_uvm.py`)
Tests UVM/DV infrastructure:
- UVM class hierarchy navigation
- UVM macro handling
- Call hierarchy (if supported)
- Type hierarchy (if supported)

**Target files**: `~/opentitan/hw/dv/sv/cip_lib/`

### Phase 5: Rapid Editing Simulation (`test_phase5_rapid_editing.py`)
Tests LSP behavior under rapid document changes:
- Character-by-character typing simulation
- Rapid line edits
- Edit-undo-redo cycles
- Completion during typing
- Diagnostics update behavior
- Concurrent request handling

### Phase 6: Feature Gap Identification (`test_phase6_feature_gaps.py`)
Documents known limitations and evaluates wishlist features:
- Macro expansion handling
- Cross-file analysis
- Incremental indexing
- Semantic tokens
- Code actions
- Rename support
- Folding ranges

### Phase 7: Performance Benchmarking (`test_phase7_benchmarks.py`)
Comprehensive performance measurements:
- Response time targets (hover <100ms, completion <300ms, etc.)
- Scaling behavior across file sizes
- Throughput testing
- Latency distribution (p50, p90, p99)

## Mini-Project: LED Controller (`test_mini_project_led_ctrl.py`)

Tests LSP features while developing a new peripheral:
- Files: `~/opentitan/hw/ip/led_ctrl/`
- Tests completion, navigation, and diagnostics during development

## Performance Targets

| Operation | Target |
|-----------|--------|
| Cold start indexing | < 30s for OpenTitan |
| Hover response | < 100ms |
| Go-to-definition | < 200ms |
| Completion | < 300ms |
| Find all references | < 1s |
| Memory usage | < 2GB for OpenTitan |

## Running Individual Tests

```bash
# Run a specific test
./run_stress_tests.sh -k "test_hover_target"

# Run tests matching a pattern
./run_stress_tests.sh -k "large_file"

# Run with maximum verbosity
./run_stress_tests.sh -vv -s
```

## Test Output

The benchmark collector automatically generates a summary at the end of each test run:

```
============================================================
BENCHMARK SUMMARY
============================================================

hover:
  Count: 15/15
  Avg: 45.2ms, Min: 12.1ms, Max: 89.3ms

completion:
  Count: 10/10
  Avg: 156.8ms, Min: 78.4ms, Max: 234.5ms

...
============================================================
```

## Verification Checklist

After completing the stress test, verify:

- [ ] All basic LSP features work on small files
- [ ] Large files (40K+ LOC) don't crash/hang the server
- [ ] Cross-file navigation works for module instances
- [ ] Package type resolution works
- [ ] UVM class hierarchy navigation works
- [ ] Rapid editing doesn't cause race conditions
- [ ] Memory usage stays reasonable (< 2GB for OpenTitan)
- [ ] No false positive diagnostics on valid code
- [ ] Completion suggestions are relevant and fast

## Troubleshooting

### Tests skip with "OpenTitan file not found"
Clone OpenTitan to `~/opentitan` or update paths in `conftest.py`.

### LSP server not found
Build the server:
```bash
cd ~/circt/build && ninja circt-verilog-lsp-server
```

### Import errors
The virtual environment may be missing dependencies:
```bash
cd ~/circt/test/Tools/circt-verilog-lsp-server/e2e
source .venv/bin/activate
pip install pytest pytest-asyncio pytest-lsp lsprotocol pygls cattrs
```

### Server crashes during tests
Check server logs and consider running with a debugger:
```bash
lldb ~/circt/build/bin/circt-verilog-lsp-server
```

## Adding New Tests

1. Create a new test file: `test_phase_N_description.py`
2. Import fixtures from `conftest.py`
3. Use `@skip_if_missing("FILE_ATTR")` for optional files
4. Use `benchmark.start()/stop()` to collect metrics
5. Add to `run_stress_tests.sh` if needed
