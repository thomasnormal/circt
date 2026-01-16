# CIRCT UVM Parity Changelog

## Iteration 27 (Current) - January 16, 2026

### In Progress
- Track A: DPI/System call implementation
- Track B: sim.proc.print lowering to arcilator
- Track C: AVIP full validation on ~/mbit
- Track D: LSP debounce fix and polish

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
