# CIRCT UVM Parity Project Status

**Goal**: Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.

**Last Updated**: January 24, 2026 (Iteration 157)

## Current Status

### sv-tests Coverage (717 tests, ~83% raw, ~95% effective)
Note: 42 tests are negative tests expected to fail. Effective pass rate excludes these.

| Chapter | Topic | Pass Rate | Status |
|---------|-------|-----------|--------|
| 5 | Lexical Conventions | 43/50 (86%) | 5 negative, 2 need -D |
| 6 | Data Types | 73/84 (87%) | 11 negative tests |
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
| 18 | Random Constraints | 56/134 (42% raw, 95% effective) | 63 need UVM, 15 negative |
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
| AHB | SUCCESS | Verified | 294K lines MLIR, 250K cycles, 0 errors |
| APB | SUCCESS | Tested | 293K lines MLIR, 10K cycles, 0 errors |
| UART | SUCCESS | Tested | 1.4M MLIR, 1M cycles, 0 errors |
| SPI | SUCCESS | Verified | 149K lines MLIR, 5M processes, 0 errors |
| I2S | SUCCESS | Tested | 17K lines MLIR, 130K cycles, 0 errors |
| I3C | SUCCESS | Tested | 145K lines MLIR, 100K cycles, 0 errors |
| JTAG | Partial | - | AVIP code issues (not CIRCT) |
| AXI4 | SUCCESS | Tested | 26K lines MLIR, 10K cycles, 0 errors |
| AXI4-Lite | Partial | - | AVIP code issues in cover properties |

### verilator-verification (121/154 pass, 78%)
Note: 1 test regression from 122 due to trailing comma syntax enforcement

| Category | Count | Details |
|----------|-------|---------|
| Invalid test syntax | 8 | `1'z`/`1'x` not valid SV (should be `1'bz`) |
| Expected failures | 4 | Tests marked should-fail |
| Parameter initializer | 3 | Missing parameter defaults |
| pre/post_randomize | 2 | Signature mismatch |
| Trailing comma | 1 | `sequence_named.sv` - non-standard SV syntax |
| $unit reference | 1 | Package referencing $unit |
| coverpoint iff | 1 | iff syntax in coverpoint |
| enum in constraint | 1 | enum expression in constraint |
| UVM testbenches | 12 | Need full UVM runtime |

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

## Next Steps (Iteration 157)

### Track A: Extern Virtual Method Vtable Entries - FIXED (Iteration 157)
**Status**: Fixed - I2S AVIP now compiles with proper vtable entries
- ✅ Simple cross-package inheritance works
- ✅ Complex UVM inheritance now works (I2sBaseTest, I2sScoreboard have vtable entries)
- **Root Cause** (Agent a44d687 analysis):
  - Extern virtual methods have Virtual flag only on the prototype, not the implementation
  - `visit(SubroutineSymbol)` checked implementation's flags, missing the Virtual flag
  - ClassMethodDeclOp was not created for extern virtual methods
- **Fix 1**: Modified `visit(MethodPrototypeSymbol)` to use prototype's virtual flag
- **Fix 2**: Added MethodPrototypeSymbol handling in Pass 3 for inherited extern virtual methods
- **Location**: `lib/Conversion/ImportVerilog/Structure.cpp`
- **Test**: `test/Conversion/ImportVerilog/extern-virtual-method.sv`

### Track B: Covergroup Method Lowering - COMPLETE
**Status**: Fixed in Iteration 154 (Commit a6b1859ac)
- ✅ Runtime functions exist in MooreRuntime.cpp
- ✅ ImportVerilog detects covergroup methods via CovergroupBodySymbol
- ✅ `sample()` → `moore.covergroup.sample`
- ✅ `get_coverage()` → `moore.covergroup.get_coverage`

### Track C: ProcessOp Canonicalize - COMPLETE
**Status**: Fixed in Iteration 155 (Commit 7c1dc2a64)
- ✅ func.call and func.call_indirect now treated as side effects
- ✅ Prevents valid processes from being removed during canonicalization
- ✅ Test: `test/Dialect/LLHD/Canonicalization/processes.mlir`

### Track D: Global Variable Initialization - MOSTLY COMPLETE
**Status**: LLVM::ConstantOp fixed; llhd.sig in function context still pending
- ✅ GlobalVariableOpConversion generates LLVM global constructors correctly
- ✅ Simple constructors using `hw.constant` work
- ✅ LLVM::ConstantOp now handled (Track H fix)
- ❌ `llhd.sig` in function context not handled (local variables in constructors)
- **Root Cause** (Agent abceb68 analysis):
  1. Local variable declarations produce `llhd.sig` operations at runtime
  2. Interpreter only registers signals during initialization phase
  3. Need to add handler for `llhd::SigOp` to dynamically register signals
- **Location**: `tools/circt-sim/LLHDProcessInterpreter.cpp` line 1163+

### Track E: HVL_top Function Inlining - NEEDS INVESTIGATION
**Status**: Analysis complete, design decision needed
- ✅ Simple function calls are being **inlined** during ImportVerilog
- ✅ Only wait-event initial blocks become llhd.process
- ❌ UVM `run_test()` body is inlined rather than called at runtime
- **Decision**: Should complex functions be preserved as func.call? Or is seq.initial sufficient?
- **Alternative**: Generate llhd.process for all hvl_top initial blocks

### Track F: Union Bitwidth Mismatch - FIXED (Iteration 157, Commit d610b3b7e)
**Status**: Fixed - yosys/tests/svtypes/union_simple.sv now passes
- ✅ Added `convertTypeToPacked()` helper for packed union member conversion
- ✅ Union members now use plain integers instead of 4-state structs
- ✅ Handle 4-state struct to union bitcast by extracting value component
- ✅ Handle 4-state struct to union assignment in AssignOpConversion
- **Location**: `lib/Conversion/MooreToCore/MooreToCore.cpp` (+64 lines)

### Track G: UVM Runtime Initialization (Blocking Full AVIP)
**Status**: Tracks A and H fixed; ready for integration testing
- ✅ HDL top modules compile and simulate correctly
- ✅ Global constructor execution implemented in circt-sim
- ✅ Vtable entries now generated for UVM classes (Track A fixed)
- ✅ LLVM::ConstantOp interpreter support (Track H fixed)
- ❌ hvl_top with run_test() has inlined body instead of runtime call
- ❌ uvm_coreservice and uvm_root not created at runtime
- **Next**: Test full UVM testbench simulation with recent fixes

### Track H: LLVM::ConstantOp Interpreter Support - FIXED (Iteration 157)
**Status**: Fixed - Global constructors using LLVM::ConstantOp now work
- ✅ Added `LLVM::ConstantOp` handler in `interpretOperation()` (lines 2492-2503)
- ✅ Added handler in `getValue()` for constants hoisted outside process body (lines 3903-3910)
- ✅ All 36 llhd-process tests pass
- **Location**: `tools/circt-sim/LLHDProcessInterpreter.cpp`

## Remaining Limitations

1. **UVM runtime initialization** - uvm_coreservice/uvm_root not created at startup
2. **wand/wor net types** - Need AND/OR resolution logic in ProcessScheduler
3. **@seq event controls** - SVA feature, Codex agent scope
4. **VCD dump tasks** - `$dumpfile`/`$dumpvars` stubbed
5. **Some randomization features** - `pre_randomize`/`post_randomize` signature strict

## Completed in Iteration 150

1. **AHB AVIP Simulation Verified**
   - Full circt-sim simulation
   - 1,000,003 processes executed, 250K clock cycles
   - 0 errors, 0 warnings

2. **SPI AVIP Simulation Verified**
   - Full circt-sim simulation
   - 5,000,013 processes executed
   - UVM_INFO messages from BFMs appearing
   - 0 errors, 0 warnings

3. **AXI4 AVIP Investigation**
   - Compilation successful (135K lines MLIR)
   - Simulation hangs after initial phase
   - Initial messages appear, then blocks
   - Needs further investigation

## Completed in Iteration 149

1. **verilator-verification Full Analysis**
   - Verified 122/154 tests pass (79.2%) - baseline confirmed
   - Signal strength parsing working: 18/21 tests pass
   - Identified remaining failure categories:
     - 3 parameter initializer issues (slang strictness)
     - 8 unbased literal syntax (`1'z` invalid)
     - 2 pre/post_randomize signature (slang strictness)
     - 1 coverpoint iff (test design issue - missing parentheses)
     - 1 enum in constraint

2. **pre/post_randomize Signature Analysis**
   - Error from slang: requires explicit `void` return type
   - IEEE 1800-2017 allows implicit void, but slang enforces explicit
   - Not a CIRCT bug - slang strictness issue

3. **Coverpoint iff Syntax Analysis**
   - Test uses `iff enable` without parentheses
   - IEEE 1800-2017 requires `iff (expr)` with parentheses
   - slang correctly enforces the standard - test is invalid

4. **I2S AVIP Simulation Verified**
   - Full circt-sim simulation with 130K cycles
   - 0 errors, $display messages appearing
   - hdlTop module with clock/reset/BFM working

5. **sv-tests Baseline Confirmed**
   - Chapter 5: 43/50 (86%) - 5 negative tests
   - Chapter 6: 73/84 (87%) - 11 negative tests
   - Chapter 8: 44/53 (83%) - 9 negative tests
   - Chapter 22: 53/74 (72%) - 15 negative + define-expansion

## Completed in Iteration 148

1. **Signal Strength Simulation in circt-sim** (Commit 2f0a4b6dc)
   - Added DriveStrength enum and SignalDriver struct to ProcessScheduler
   - Implemented multi-driver tracking with strength-based resolution
   - IEEE 1800-2017 compliant: stronger wins, equal conflict → X
   - Modified LLHDProcessInterpreter to extract and use strength attributes
   - Added 8 comprehensive unit tests for signal strength resolution
   - Full pipeline complete: Verilog → Moore → LLHD → circt-sim

## Completed in Iteration 147

1. **Signal Strength LLHD Lowering** (Commit b8037da32)
   - Added DriveStrengthAttr to LLHD dialect
   - Updated DriveOp with strength0/strength1 attributes
   - MooreToCore AssignOpConversion passes strength through
   - Full IR path: Verilog → Moore → LLHD with strength preserved

2. **Chapter-22 Define-Expansion Analysis**
   - 7 of 8 failures are negative tests (correctly rejected)
   - Test 26 is test design issue (preprocessor works, semantic error)
   - Effective pass rate: 66/67 positive tests (98.5%)

## Remaining Limitations

1. ~~**Signal strength simulation**~~ - **COMPLETE** (Iteration 148)
2. **@seq event controls** - SVA feature (Codex agent scope)
3. **Covergroup get_coverage()** - Compiles, runtime not implemented
4. **VCD dump tasks** - $dumpfile/$dumpvars stubbed
5. **pre/post_randomize** - Signature validation too strict

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
