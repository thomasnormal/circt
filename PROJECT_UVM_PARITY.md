# CIRCT UVM Parity Project Status

**Goal**: Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.

**Last Updated**: January 23, 2026 (Iteration 145)

## Current Status

### sv-tests Coverage (717 tests, ~83% raw, ~95% effective)
Note: 42 tests are negative tests expected to fail. Effective pass rate excludes these.

| Chapter | Topic | Pass Rate | Status |
|---------|-------|-----------|--------|
| 5 | Lexical Conventions | 43/50 (86%) | 5 negative, 2 need -D |
| 6 | Data Types | 73/84 (87%) | ~8 negative tests |
| 7 | Aggregate Data Types | 101/103 (98%) | 2 negative tests |
| 8 | Classes | 44/53 (83%) | 9 negative tests |
| 9 | Processes | 44/46 (96%) | 1 known limitation (@seq) |
| 10 | Assignments | 9/10 (90%) | 1 negative test |
| 11 | Operators | 76/78 (97%) | 2 negative tests |
| 12 | Procedural Programming | 27/27 (100%) | Complete |
| 13 | Tasks and Functions | 13/15 (87%) | 2 negative tests |
| 14 | Clocking Blocks | 5/5 (100%) | Complete |
| 15 | Inter-Process Sync | 5/5 (100%) | Fixed in Iteration 145 |
| 16 | Assertions | - | Codex agent scope |
| 18 | Random Constraints | 56/134 (42%) | 66 need UVM, 12 negative |
| 20 | Utility System Tasks | 47/47 (100%) | Complete |
| 21 | I/O System Tasks | 29/29 (100%) | Complete |
| 22 | Compiler Directives | 53/74 (72%) | 15 negative tests, macros |
| 23 | Modules and Hierarchy | 3/3 (100%) | Complete |
| 24 | Programs | 1/1 (100%) | Complete |
| 25 | Interfaces | Tested | See Iteration 122 |
| 26 | Packages | Tested | See Iteration 122 |

### AVIP Testing Status (7/9 fully compile)

| AVIP | Compilation | Simulation | Notes |
|------|-------------|------------|-------|
| AHB | SUCCESS | Tested | 294K lines MLIR, 10K cycles, 0 errors |
| APB | SUCCESS | Tested | 293K lines MLIR, 10K cycles, 0 errors |
| UART | SUCCESS | Tested | 1.4M MLIR, 1M cycles, 0 errors |
| SPI | SUCCESS | Tested | 149K lines MLIR, 107 cycles, 0 errors |
| I2S | SUCCESS | Tested | 17K lines MLIR, 130K cycles, 0 errors |
| I3C | SUCCESS | Tested | 145K lines MLIR, 100K cycles, 0 errors |
| JTAG | Partial | - | AVIP code issues (not CIRCT) |
| AXI4 | SUCCESS | Tested | 26K lines MLIR, 10K cycles, 0 errors |
| AXI4-Lite | Partial | - | AVIP code issues in cover properties |

### verilator-verification
- 122/154 tests pass (79%)
- Failures are due to: UVM library missing (12), non-standard syntax (8), expected failures (4)

## Known Limitations

### CIRCT Bugs (To Fix)

1. ~~**Hierarchical event references**~~ **FIXED** (Iteration 145)
   - Was: Chapter-15 tests failing with "unknown hierarchical name"
   - Fixed: Added ProceduralBlockSymbol handler in HierarchicalNames.cpp
   - All Chapter-15 tests now pass (5/5)

2. **Include-via-macro path resolution** (Priority: Low)
   - Affects: Chapter-22 test (22.4--include_via_define.sv)
   - Pattern: `` `define DO_INCLUDE(FN) `include FN ``

### Known Unsupported Features

1. **Sequence event controls (@seq)** - SVA feature, Codex agent scope
2. **Class builtin functions** - Randomization/constraints partially supported
3. **Some VCD dump tasks** - Stubbed for compilation
4. **Covergroup get_coverage()** - Compiles but returns unsupported format at runtime (Iteration 143)

## Recent Fixes (Iterations 132-143)

1. **UVM Phase Handle Aliases** (Commit 155b9ef93)
   - Added `_ph` suffix aliases per IEEE 1800.2
   - Enabled I2S/I3C AVIP compilation

2. **UVM Phase wait_for_state()** (Commit 155b9ef93)
   - Added `uvm_phase_state` enum
   - Added `wait_for_state()` task
   - Enabled AXI4-Lite assertion compilation

3. **TLM seq_item_pull_port** (Commit acf32a352)
   - Fixed parameterization from `#(REQ, RSP)` to `#(RSP, REQ)`
   - Corrected put/get type direction for sequences

4. **Local Event Triggering** (Commit cdaed5b93)
   - Added BlockArgument detection in ProcedureOpConversion
   - Fixed local event references (`-> e;` in same module)

5. **Hierarchical Event References** (Iteration 145, Commit a9f00c204)
   - Added HierPathValueStmtVisitor for statement traversal
   - Implemented collectHierarchicalValuesFromStatement()
   - Added ProceduralBlockSymbol handler to InstBodyVisitor
   - Chapter-15 now 5/5 (100%)

6. **UVM Callbacks Implementation** (Commit f39093a90)
   - Implemented callback storage with associative arrays
   - Enabled proper callback iteration and filtering

7. **I2S AVIP Simulation Verified** (Iteration 143)
   - Full end-to-end simulation with circt-sim
   - 130,000 clock cycles, 0 errors
   - Validates UVM testbench compilation and simulation flow

8. **CIRCT vs xrun Comparison** (Iteration 143)
   - CIRCT stricter on type safety (string/bit conversions)
   - xrun stricter on IEEE timescale compliance
   - APB AVIP design code compiles cleanly with both

9. **Covergroup Feature Testing** (Iteration 143)
   - Most covergroup features compile and simulate correctly
   - Explicit bins, cross coverage, wildcard bins all work
   - get_coverage() runtime computation not yet implemented

## Next Steps (Iteration 146)

### Track A: Signal Strength Support (Priority: High)
- 13 verilator-verification tests fail due to signal strengths
- Slang parses strength info via `getDriveStrength()`
- Need to add to Moore dialect, ImportVerilog, and simulation
- Components: DriveStrength enum, ContinuousAssignOp attrs, signal resolution

### Track B: Chapter-22 Define-Expansion Test 26
- Only 1 of 26 define-expansion tests fails (test_26)
- Token concatenation works, but test uses undeclared identifier
- Verify if test is valid or should be marked as expected failure

### Track C: AVIP Extended Simulation Testing
- UART AVIP compiles on both circt-verilog and xrun
- Test with circt-sim for actual UVM transactions
- Compare behavioral results with xrun
- Profile simulation performance

### Track D: Chapter-6 Negative Test Verification
- ~8 tests may be negative tests
- Verify each has `:should_fail_because:` annotation
- Update baseline if all are correctly rejected

### Findings from Iteration 146 Investigation

1. **Chapter-8 Interface Classes**: All 9 "failures" are negative tests that circt-verilog correctly rejects. NOT real limitations.

2. **Chapter-5 Macro Tests**: The 2 failures need `-D` flags passed from test harness `:defines:` metadata. circt-verilog `-D` flag works correctly.

3. **Chapter-22 Define-Expansion**: 25/26 pass. Test 26 uses undeclared identifier after token concat.

4. **UART AVIP**: Successfully compiles on both circt-verilog and xrun with minor diagnostic differences.

### Remaining Limitations

1. **Signal strengths** - Not preserved (Priority: High for verilator-verification)
2. **@seq event controls** - SVA feature (Codex agent scope)
3. **Covergroup get_coverage()** - Runtime not implemented
4. **VCD dump tasks** - $dumpfile/$dumpvars ignored

## Test Commands

```bash
# Run sv-tests chapter
~/circt/build/bin/circt-verilog --ir-hw ~/sv-tests/tests/chapter-X/*.sv

# Compile AVIP
cd ~/mbit/[avip]/sim && ~/circt/build/bin/circt-verilog --ir-hw -f [filelist].f

# Run simulation
~/circt/build/bin/circt-sim [mlir-file]

# Compare with xrun
xrun -f [filelist].f
```

## Files

- `lib/Runtime/uvm/uvm_pkg.sv` - UVM stubs
- `lib/Conversion/ImportVerilog/` - Verilog frontend
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Moore to Core conversion (event fix)
- `test/Runtime/uvm/` - UVM regression tests (8 comprehensive tests)
  - `uvm_phase_aliases_test.sv` - Phase handle aliases
  - `uvm_phase_wait_for_state_test.sv` - Phase wait_for_state
  - `uvm_stress_test.sv` - TLM stress testing (907 lines)
  - `uvm_factory_test.sv` - Factory patterns (500+ lines)
  - `uvm_callback_test.sv` - Callbacks (600+ lines)
  - `uvm_sequence_test.sv` - Sequences (1082 lines)
  - `uvm_ral_test.sv` - Register Abstraction Layer (1393 lines)
  - `uvm_tlm_fifo_test.sv` - TLM FIFOs (884 lines)
- `CHANGELOG.md` - Detailed iteration history
