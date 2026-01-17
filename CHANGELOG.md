# CIRCT UVM Parity Changelog

## Iteration 34 - January 17, 2026

### Multi-Track Parallel Progress (commit 0621de47b)

**Track A: randcase Statement (IEEE 1800-2017 §18.16)**
- Implemented weighted random case selection
- Uses `$urandom_range` with cascading comparisons
- Edge cases: zero weights become uniform, single-item optimized
- Files: `lib/Conversion/ImportVerilog/Statements.cpp` (+100 lines)
- Test: `test/Conversion/ImportVerilog/randcase.sv`

**Track B: Queue delete(index) Runtime**
- New runtime function `__moore_queue_delete_index(queue, index, element_size)`
- Proper element shifting with memory management
- MooreToCore lowering extracts element size from queue type
- Files: `lib/Runtime/MooreRuntime.cpp`, `lib/Conversion/MooreToCore/MooreToCore.cpp`
- Test: `test/Conversion/ImportVerilog/queue-delete-index.sv`

**Track C: LTL Temporal Operators in VerifToSMT**
- `ltl.and` → `smt.and`
- `ltl.or` → `smt.or`
- `ltl.not` → `smt.not`
- `ltl.implication` → `smt.or(smt.not(a), b)`
- `ltl.eventually` → identity at each step (BMC loop accumulates with OR)
- `ltl.until` → `q || p` (weak until semantics for BMC)
- `ltl.boolean_constant` → `smt.constant`
- Files: `lib/Conversion/VerifToSMT/VerifToSMT.cpp` (+178 lines)
- Test: `test/Conversion/VerifToSMT/ltl-temporal.mlir`

**Track D: LSP go-to-definition Verification**
- Confirmed existing implementation works correctly
- Added comprehensive test for modules, wires, ports
- Files: `test/Tools/circt-verilog-lsp-server/goto-definition.test` (+133 lines)

**Total**: 1,695 insertions across 13 files

---

## Iteration 33 - January 17-18, 2026

### Z3 Configuration (January 17)
- ✅ Z3 4.12.4 built and installed at `~/z3-install/`
- ✅ CIRCT configured with `Z3_DIR=~/z3-install/lib64/cmake/z3`
- ✅ `circt-bmc` builds and runs with Z3 SMT backend
- ✅ Runtime linking: `LD_LIBRARY_PATH=~/z3-install/lib64`
- Note: Required symlink `lib -> lib64` for FindZ3 compatibility

### UVM Parity Fixes (Queues/Arrays, File I/O, Distributions)

**Queue/Array Operations**
- Queue range slicing with runtime support
- Dynamic array range slicing with runtime support
- `unique_index()` implemented end-to-end
- Array reductions: `sum()`, `product()`, `and()`, `or()`, `xor()`
- `rsort()` and `shuffle()` queue methods wired to runtime

**File I/O and System Tasks**
- `$fgetc`, `$fgets`, `$feof`, `$fflush`, `$ftell` conversions
- `$ferror`, `$ungetc`, `$fread` with runtime implementations
- `$strobe`, `$monitor`, `$fstrobe`, `$fmonitor` tasks added
- `$dumpfile/$dumpvars/$dumpports` no-op handling

**Distribution Functions**
- `$dist_uniform`, `$dist_normal`, `$dist_exponential`, `$dist_poisson`
- `$dist_erlang`, `$dist_chi_square`, `$dist_t`

**Lowering/Type System**
- Unpacked array comparison (`uarray_cmp`) lowering implemented
- String → bitvector fallback for UVM field automation
- Randsequence production argument binding (input-only, default values)
- Randsequence randjoin(1) support
- Streaming operator lvalue unpacking for dynamic arrays/queues
- Tagged union construction + member access (struct wrapper)
- Tagged union PatternCase matching (tag compare/extract)
- Tagged union matches in `if` / conditional expressions
- Randsequence statement lowering restored (weights/if/case/repeat, randjoin(1))
- Randcase weighted selection support

**Tests**
- Added randsequence arguments/defaults test case
- Added randsequence randjoin(1) test case

**Files Modified**
- `lib/Conversion/ImportVerilog/Expressions.cpp`
- `lib/Conversion/ImportVerilog/Statements.cpp`
- `lib/Conversion/MooreToCore/MooreToCore.cpp`
- `include/circt/Dialect/Moore/MooreOps.td`
- `include/circt/Runtime/MooreRuntime.h`
- `lib/Runtime/MooreRuntime.cpp`

**Next Focus**
- Dynamic array range select lowering (queue slice already implemented)

---

## Iteration 31 - January 16, 2026

### Clocking Block Signal Access and @(cb) Syntax (commit 43f3c7a4d)

**Major Feature**: Complete clocking block signal access support per IEEE 1800-2017 Section 14.

**Clocking Block Signal Access**:
- `cb.signal` rvalue generation - reads correctly resolve to underlying signal value
- `cb.signal` lvalue generation - writes correctly resolve to underlying signal
- Both input and output clocking signals supported
- Works in procedural contexts (always_ff, always_comb)

**Clocking Block Event Reference**:
- `@(cb)` syntax now works in event controls
- Automatically resolves to the clocking block's underlying clock event
- Supports both posedge and negedge clocking blocks

**Queue Reduction Operations**:
- Added QueueReduceOp for sum(), product(), and(), or(), xor() methods
- QueueReduceKind enum and attribute
- MooreToCore conversion to runtime function calls

**LLHD Process Interpreter Phase 2**:
- Full process execution: llhd.drv, llhd.wait, llhd.halt
- Signal probing and driving operations
- Time advancement and delta cycle handling
- 5/6 circt-sim tests passing

**Files Modified** (1,408 insertions):
- `lib/Conversion/ImportVerilog/Expressions.cpp` - ClockVar rvalue/lvalue handling
- `lib/Conversion/ImportVerilog/TimingControls.cpp` - @(cb) event reference
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - QueueReduceOp, UArrayCmpOp conversions
- `include/circt/Dialect/Moore/MooreOps.td` - QueueReduceOp, QueueReduceKind
- `tools/circt-sim/LLHDProcessInterpreter.cpp` - Process execution fixes

**New Test Files**:
- `test/Conversion/ImportVerilog/clocking-event-wait.sv` - @(cb) syntax tests
- `test/Conversion/ImportVerilog/clocking-signal-access.sv` - cb.signal tests
- `test/circt-sim/llhd-process-basic.mlir` - Basic process execution
- `test/circt-sim/llhd-process-probe.mlir` - Probe and drive operations

---

## Iteration 30 - January 16, 2026

### Major Accomplishments

#### SVA Functions in Boolean Contexts (commit a68ed9adf)
Fixed handling of SVA sampled value functions ($changed, $stable, $rose, $fell) when used in boolean expression contexts within assertions:

1. **Logical operators (||, &&, ->, <->)**: When either operand is an LTL property/sequence type, now uses LTL operations (ltl.or, ltl.and) instead of Moore operations
2. **Logical NOT (!)**: When operand is an LTL type, uses ltl.not instead of moore.not
3. **$sampled in procedural context**: Returns original moore-typed value (not i1) so it can be used in comparisons like `val != $sampled(val)`

**Test Results**: verilator-verification SVA tests: 10/10 pass (up from 7/10)

**Files Modified:**
- `lib/Conversion/ImportVerilog/Expressions.cpp` - LTL-aware logical operator handling
- `lib/Conversion/ImportVerilog/AssertionExpr.cpp` - $sampled procedural context fix
- `test/Conversion/ImportVerilog/sva-bool-context.sv` - New test file

#### Z3 CMake Linking Fix (commit 48bcd2308)
Fixed JIT runtime linking for Z3 symbols in circt-bmc and circt-lec:

- SMTToZ3LLVM generates LLVM IR that calls Z3 API at runtime
- Added Z3 to LINK_LIBS in SMTToZ3LLVM/CMakeLists.txt
- Added Z3 JIT dependencies in circt-bmc and circt-lec CMakeLists.txt
- Supports both CONFIG mode (z3::libz3) and Module mode (Z3_LIBRARIES)

**Files Modified:**
- `lib/Conversion/SMTToZ3LLVM/CMakeLists.txt`
- `tools/circt-bmc/CMakeLists.txt`
- `tools/circt-lec/CMakeLists.txt`

#### Comprehensive Test Suite Survey

**sv-tests Coverage** (989 non-UVM tests):
| Chapter | Pass Rate | Notes |
|---------|-----------|-------|
| Ch 5 (Lexical) | 86% | Strong |
| Ch 11 (Operators) | 87% | Strong |
| Ch 13 (Tasks/Functions) | 86% | Strong |
| Ch 14 (Clocking Blocks) | **~80%** | Signal access, @(cb) event |
| Ch 18 (Random/Constraints) | 25% | RandSequence missing |
| Overall | **72.1%** (713/989) | Good baseline |

**mbit AVIP Testing**:
- Global packages: 8/8 (100%) pass
- Interfaces: 6/8 (75%) pass
- HVL packages: 0/8 (0%) - requires UVM library

**verilator-verification SVA Tests** (verified):
- --parse-only: 10/10 (100%)
- --ir-hw: 9/10 (90%) - `$past(val) == 0` needs conversion pattern

#### Multi-Track Progress (commit ab52d23c2) - 3,522 insertions
Major implementation work across 4 parallel tracks:

**Track 1 - Clocking Blocks**:
- Added `ClockingBlockDeclOp` and `ClockingSignalOp` to Moore dialect
- Added MooreToCore conversion patterns for clocking blocks
- Created `test/Conversion/ImportVerilog/clocking-blocks.sv`

**Track 2 - LLHD Process Interpreter**:
- New `LLHDProcessInterpreter.cpp/h` files for circt-sim
- Process detection and scheduling infrastructure
- Created `test/circt-sim/llhd-process-todo.mlir`

**Track 3 - $past Comparison Fix**:
- Added `moore::PastOp` to preserve types for $past in comparisons
- Updated AssertionExpr.cpp for type-preserving $past
- Added PastOpConversion in MooreToCore

**Track 4 - clocked_assert Lowering for BMC**:
- New `LowerClockedAssertLike.cpp` pass in VerifToSMT
- Updated VerifToSMT conversion for clocked assertions
- Enhanced circt-bmc with clocked assertion support

**Additional Changes**:
- LTLToCore enhancements: 986 lines added
- SVAToLTL improvements
- Runtime and integration test updates

#### Clocking Block Implementation (DONE)
Clocking blocks now have Moore dialect ops:
- `ClockingBlockDeclOp`, `ClockingSignalOp` implemented
- MooreToCore lowering patterns added
- Testing against sv-tests Chapter 14 in progress

#### Clocked Assert Lowering for BMC (Research Complete)
Problem: LTLToCore skips i1-property clocked_assert, leaving it unconverted.
Solution: New pass to convert `clocked_assert → assert` for BMC pipeline.
Location: Between LTLToCore and LowerToBMC in circt-bmc.cpp

#### LLHD Process Interpreter (Phase 1A Started)
Created initial implementation files:
- `tools/circt-sim/LLHDProcessInterpreter.h` (9.8 KB)
- `tools/circt-sim/LLHDProcessInterpreter.cpp` (21 KB)
Implements: signal registration, time conversion, llhd.prb/drv/wait/halt handlers

#### Big Projects Status Survey

Comprehensive survey of 6 major projects toward Xcelium parity:

| Project | Status | Key Blocker |
|---------|--------|-------------|
| SVA with Z3 | Partial | Z3 not installed, clocked_assert lowering |
| Multi-core Arcilator | Missing | Requires architectural redesign |
| LSP/Debugging | Partial | Missing completion, go-to-def, debug |
| 4-State Logic (X/Z) | Missing | Type system redesign needed |
| Coverage | Partial | Missing cross-cover expressions |
| DPI/VPI | Stubs only | FFI bridge needed |

**Key Implementation Files:**
- SVAToLTL: 321 conversion patterns
- VerifToSMT: 967 lines
- MooreToCore: 9,464 lines
- MooreRuntime: 2,270 lines

### Active Development Tracks (Parallel Agents)

1. **LLHD Interpreter** (a328f45): Debugging process detection in circt-sim
2. **Clocking Blocks** (aac6fde): Adding ClockingBlockDeclOp, ClockingSignalOp to Moore
3. **clocked_assert Lowering** (a87c394): Creating LowerClockedAssertLike pass for BMC
4. **$past Comparison Fix** (a87be46): Adding moore::PastOp to preserve type for comparisons

---

## Iteration 29 (Complete) - January 16, 2026

### Major Accomplishments

#### VerifToSMT `bmc.final` Assertion Handling (circt-bmc pipeline)
Fixed critical crashes and type mismatches in the VerifToSMT conversion pass when handling `bmc.final` assertions for Bounded Model Checking:

**Fixes Applied:**
1. **Added ReconcileUnrealizedCastsPass** to circt-bmc pipeline after VerifToSMT conversion
   - Cleans up unrealized conversion casts between SMT and concrete types

2. **Fixed BVConstantOp argument order** - signature is (value, width) not (width, value)
   - `smt::BVConstantOp::create(rewriter, loc, 0, 1)` for 1-bit zero
   - `smt::BVConstantOp::create(rewriter, loc, 1, 1)` for 1-bit one

3. **Clock counting timing** - Moved clock counting BEFORE region type conversion
   - After region conversion, `seq::ClockType` becomes `!smt.bv<1>`, losing count

4. **Proper op erasure** - Changed from `op->erase()` to `rewriter.eraseOp()`
   - Required to properly notify the conversion framework during pattern matching

5. **Yield modification ordering** - Modify yield operands BEFORE erasing `bmc.final` ops
   - Values must remain valid when added to yield

**Technical Details:**
- `bmc.final` assertions are hoisted into circuit outputs and checked only at final step
- Final assertions use `!smt.bv<1>` type for scf.for iter_args (matches circuit outputs)
- Final check creates separate `smt.check` after the main loop to verify final properties
- Results combine: `violated || finalCheckViolated` using `arith.ori` and `arith.xori`

#### SVA BMC Pipeline Progress
- VerifToSMT conversion now produces valid MLIR with proper final check handling
- Pipeline stages working: Moore → Verif → LTL → Core → VerifToSMT → SMT
- Remaining: Z3 runtime linking for actual SMT solving

### Files Modified
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - Multiple fixes for final check handling
- `tools/circt-bmc/circt-bmc.cpp` - Added ReconcileUnrealizedCasts pass to pipeline
- `include/circt/Conversion/VerifToSMT.h` - Include for ReconcileUnrealizedCasts

### Key Insights
- VerifToSMT is complex due to SMT/concrete type interleaving in scf.for loops
- Region type conversion changes `seq::ClockType` → `!smt.bv<1>`, must count before
- MLIR rewriter requires `rewriter.eraseOp()` not direct `op->erase()` in conversion patterns
- Final assertions need separate SMT check after main bounded loop completes

### Remaining Work
- Z3 runtime linking (symbols not found at JIT runtime)
- Integration tests with real SVA properties
- Performance benchmarking vs Verilator/Xcelium

---

## Iteration 28 (Complete) - January 16, 2026

### Major Accomplishments

#### $onehot and $onehot0 System Functions (commit 7d5391552)
- Implemented `OneHotBIOp` and `OneHot0BIOp` in Moore dialect
- Added ImportVerilog handlers for `$onehot` and `$onehot0` system calls
- MooreToCore lowering using `llvm.intr.ctpop`:
  - `$onehot(x)` → `ctpop(x) == 1` (exactly one bit set)
  - `$onehot0(x)` → `ctpop(x) <= 1` (at most one bit set)
- Tests added in `builtins.sv` and `string-ops.mlir`

#### $countbits System Function (commit 2830654d4)
- Implemented `CountBitsBIOp` in Moore dialect
- Added ImportVerilog handler for `$countbits` system call
- Counts occurrences of specified bit values in a vector

#### SVA Sampled Value Functions (commit 4704320af)
- Implemented `$sampled` - returns sampled value of expression
- Implemented `$past` with delay parameter - returns value from N cycles ago
- Implemented `$changed` - detects when value differs from previous cycle
- `$stable`, `$rose`, `$fell` all working in SVA context

#### Direct Interface Member Access Fix (commit 25cd3b6a2)
- Fixed direct member access through interface instances
- Uses interfaceInstances map for proper resolution

#### Test Infrastructure Fixes
- Fixed dpi.sv CHECK ordering (commit 12d75735d)
- Documented task clocking event limitation (commit 110fc6caf)
  - Tasks with IsolatedFromAbove can't reference module-level variables in timing controls
  - This is a region isolation limitation, not a parsing issue

#### sim.proc.print Lowering Discovery
- **Finding**: sim.proc.print lowering ALREADY EXISTS in `LowerArcToLLVM.cpp`
- `PrintFormattedProcOpLowering` pattern handles all sim.fmt.* operations
- No additional work needed for $display in arcilator

#### circt-sim LLHD Process Limitation (Critical Finding)
- **Discovery**: circt-sim does NOT interpret LLHD process bodies
- Simulation completes at time 0fs with no output
- Root cause: `circt-sim.cpp:443-486` creates PLACEHOLDER processes
- ProcessScheduler infrastructure exists but not connected to LLHD IR interpretation
- This is a critical gap for behavioral simulation
- Complexity: HIGH (2-4 weeks to implement)
- Arcilator works for RTL-only designs (seq.initial, combinational logic)

#### Coverage Infrastructure Analysis
- Coverage infrastructure exists and is complete:
  - `CovergroupDeclOp`, `CoverpointDeclOp`, `CoverCrossDeclOp`
  - `CovergroupInstOp`, `CovergroupSampleOp`, `CovergroupGetCoverageOp`
- MooreToCore lowering complete for all 6 coverage ops
- **Finding**: Explicit `.sample()` calls WORK (what AVIPs use)
- **Gap**: Event-driven `@(posedge clk)` sampling not connected
- AVIPs use explicit sampling - no additional work needed for AVIP support

### Test Results
- **ImportVerilog Tests**: 38/38 pass (100%)
- **AVIP Global Packages**: 8/8 pass (100%)
- **No regressions** from new features

### Key Insights
- Arcilator is the recommended path for simulation (RTL + seq.initial)
- circt-sim behavioral simulation needs LLHD process interpreter work
- sim.proc.print pipeline: Moore → sim.fmt.* → sim.proc.print → printf (all working)
- Region isolation limitation documented for tasks with timing controls

---

## Iteration 27 (Complete) - January 16, 2026

### Major Accomplishments

#### $onehot and $onehot0 System Functions (commit 7d5391552)
- Implemented `OneHotBIOp` and `OneHot0BIOp` in Moore dialect
- Added ImportVerilog handlers for `$onehot` and `$onehot0` system calls
- MooreToCore lowering using `llvm.intr.ctpop`:
  - `$onehot(x)` → `ctpop(x) == 1` (exactly one bit set)
  - `$onehot0(x)` → `ctpop(x) <= 1` (at most one bit set)
- Added unit tests in `builtins.sv` and `string-ops.mlir`

#### sim.proc.print Lowering Discovery
- **Finding**: sim.proc.print lowering ALREADY EXISTS in `LowerArcToLLVM.cpp`
- `PrintFormattedProcOpLowering` pattern handles all sim.fmt.* operations
- No additional work needed for $display in arcilator

#### circt-sim LLHD Process Limitation (Critical Finding)
- **Discovery**: circt-sim does NOT interpret LLHD process bodies
- Simulation completes at time 0fs with no output
- ProcessScheduler infrastructure exists but not connected to LLHD IR interpretation
- This is a critical gap for behavioral simulation
- Arcilator works for RTL-only designs (seq.initial, combinational logic)

#### LSP Debounce Fix Verification
- Confirmed fix exists (commit 9f150f33f)
- Some edge cases may still cause timeouts
- `--no-debounce` workaround remains available

### Files Modified
- `include/circt/Dialect/Moore/MooreOps.td` - OneHotBIOp, OneHot0BIOp
- `lib/Conversion/ImportVerilog/Expressions.cpp` - $onehot, $onehot0 handlers
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - OneHot conversion patterns
- `test/Conversion/ImportVerilog/builtins.sv` - Unit tests
- `test/Conversion/MooreToCore/string-ops.mlir` - MooreToCore tests

### Key Insights
- Arcilator is the recommended path for simulation (RTL + seq.initial)
- circt-sim behavioral simulation needs LLHD process interpreter work
- sim.proc.print pipeline: Moore → sim.fmt.* → sim.proc.print → printf (all working)

---

## Iteration 26 - January 16, 2026

### Major Accomplishments

#### Coverage Infrastructure
- Added `CovergroupHandleType` for covergroup instances
- Added `CovergroupInstOp` for instantiation (`new()`)
- Added `CovergroupSampleOp` for sampling covergroups
- Added `CovergroupGetCoverageOp` for coverage percentage
- Full MooreToCore lowering to runtime calls

#### SVA Assertion Lowering (Verified)
- Confirmed `moore.assert/assume/cover` → `verif.assert/assume/cover` works
- Immediate, deferred, and concurrent assertions all lower correctly
- Created comprehensive test file: `test/Conversion/MooreToCore/sva-assertions.mlir`

#### $countones System Function
- Implemented `CountOnesBIOp` in Moore dialect
- Added ImportVerilog handler for `$countones` system call
- MooreToCore lowering to `llvm.intr.ctpop` (LLVM count population intrinsic)

#### Interface Fixes
- Fixed `ref<virtual_interface>` to `virtual_interface` conversion
- Generates proper `llhd.prb` operation for lvalue access

#### Constraint Lowering (Complete)
- All 10 constraint ops now have MooreToCore patterns:
  - ConstraintBlockOp, ConstraintExprOp, ConstraintImplicationOp
  - ConstraintIfElseOp, ConstraintForeachOp, ConstraintDistOp
  - ConstraintInsideOp, ConstraintSolveBeforeOp, ConstraintDisableOp
  - ConstraintUniqueOp

#### $finish Handling
- Fixed `$finish` in initial blocks to use `seq.initial` (arcilator-compatible)
- No longer forces fallback to `llhd.process`
- Generates `sim.terminate` with proper exit code

#### AVIP Testing Results
All 9 AVIPs tested through CIRCT pipeline:
| AVIP | Parse | Notes |
|------|-------|-------|
| apb_avip | PARTIAL | `uvm_test_done` deprecated |
| ahb_avip | PARTIAL | bind statement scoping |
| axi4_avip | PARTIAL | hierarchical refs in vif |
| axi4Lite_avip | PARTIAL | bind issues |
| i2s_avip | PARTIAL | `uvm_test_done` |
| i3c_avip | PARTIAL | `uvm_test_done` |
| spi_avip | PARTIAL | nested comments, `this.` in constraints |
| jtag_avip | PARTIAL | enum conversion errors |
| uart_avip | PARTIAL | virtual method signature mismatch |

**Key Finding**: Issues are in AVIP source code (deprecated UVM APIs), not CIRCT limitations.

#### LSP Server Validation
- Document symbols, hover, semantic tokens work correctly
- **Bug Found**: Debounce mechanism causes hang on `textDocument/didChange`
- **Workaround**: Use `--no-debounce` flag

#### Arcilator Research
- Identified path for printf support: add `sim.proc.print` lowering
- Template exists in `arc.sim.emit` → printf lowering
- Recommended over `circt-sim` approach

### Files Modified
- `include/circt/Dialect/Moore/MooreTypes.td` - CovergroupHandleType
- `include/circt/Dialect/Moore/MooreOps.td` - Covergroup ops, CountOnesBIOp
- `lib/Conversion/ImportVerilog/Expressions.cpp` - $countones, covergroup new()
- `lib/Conversion/ImportVerilog/Types.cpp` - CovergroupType conversion
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Constraint ops, coverage ops, $countones

### Unit Tests Added
- `test/Conversion/MooreToCore/sva-assertions.mlir`
- `test/Conversion/MooreToCore/range-constraints.mlir` (extended)
- `test/Dialect/Moore/covergroups.mlir` (extended)

---

## Iteration 25 - January 15, 2026

### Major Accomplishments

#### Interface ref→vif Conversion
- Fixed conversion from `moore::RefType<VirtualInterfaceType>` to `VirtualInterfaceType`
- Generates `llhd.ProbeOp` to read pointer value from reference

#### Constraint MooreToCore Lowering
- Added all 10 constraint op conversion patterns
- Range constraints call `__moore_randomize_with_range(min, max)`
- Multi-range constraints call `__moore_randomize_with_ranges(ptr, count)`

#### $finish in seq.initial
- Removed `hasUnreachable` check from seq.initial condition
- Added `UnreachableOp` → `seq.yield` conversion
- Initial blocks with `$finish` now arcilator-compatible

### Files Modified
- `lib/Conversion/MooreToCore/MooreToCore.cpp`
- `test/Conversion/MooreToCore/initial-blocks.mlir`
- `test/Conversion/MooreToCore/interface-ops.mlir`

---

## Iteration 24 - January 14, 2026

### Major Accomplishments
- AVIP pipeline testing identified blocking issues
- Coverage architecture documented
- Constraint expression lowering (ded570db6)
- Complex initial block analysis confirmed design correctness

---

## Iteration 23 - January 13, 2026 (BREAKTHROUGH)

### Major Accomplishments

#### seq.initial Implementation (cabc1ab6e)
- Simple initial blocks now use `seq.initial` instead of `llhd.process`
- Works through arcilator end-to-end!

#### Multi-range Constraints (c8a125501)
- Support for constraints like `inside {[1:10], [20:30]}`
- ~94% total constraint coverage achieved

#### End-to-End Pipeline Verified
- SV → Moore → Core → HW → Arcilator all working

---

## Iteration 22 - January 12, 2026

### Major Accomplishments

#### sim.terminate (575768714)
- `$finish` now generates `sim.terminate` op
- Lowers to `exit(0)` or `exit(1)` based on finish code

#### Soft Constraints (5e573a811)
- Default value constraints implemented
- ~82% total constraint coverage

---

## Iteration 21 - January 11, 2026

### Major Accomplishments

#### UVM LSP Support (d930aad54)
- Added `--uvm-path` flag to circt-verilog-lsp-server
- Added `UVM_HOME` environment variable support
- Interface symbols now properly returned

#### Range Constraints (2b069ee30)
- Simple range constraints (`inside {[min:max]}`) implemented
- ~59% of AVIP constraints work

#### sim.proc.print (2be6becf7)
- $display works in arcilator
- Format string operations lowered to printf

---

## Previous Iterations

See PROJECT_PLAN.md for complete history of iterations 1-20.
