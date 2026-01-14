# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-14)

### Overall Progress

**Parsing: ✅ COMPLETE** - UVM core library parses without errors!
**Lowering: ⚠️ BLOCKED** - func.func block terminator issue prevents full conversion

### Session Progress (2026-01-14)

- ✅ All 9 AVIPs parse successfully with UVM
- ⚠️ Test files created for math builtins, covergroup, std::randomize
- 🔄 4 parallel agents working on next sprint tasks

### Critical Blocker: Block Terminator Issue

The UVM package parses successfully but fails during MLIR lowering with ~1,684
"block with no terminator" errors. This is the **#1 priority** to fix.

**Root Cause Analysis:**
- Functions/methods generate MLIR blocks without proper terminators
- First failing class: `uvm_cmdline_set_verbosity` (uvm_cmdline_report.svh:173)
- Cascade effect: uvm_queue → uvm_callbacks_base → uvm_object
- Affected control flow: foreach loops, if/else chains, early returns
- Fix location: `lib/Conversion/ImportVerilog/Statements.cpp`

### Session Progress (35+ commits)

#### Recent Commits (This Session)
- ✅ `58001e3be` - randomize() method handler (ImportVerilog)
- ✅ `dd2b06349` - RandomizeOp lowering + __moore_randomize_basic runtime
- ✅ `2fe8ea6d2` - Error messages for silent conversion failures
- ✅ `ca0c82996` - Interface lowering patterns (4 ops)
- ✅ `a48d88a71` - MLIR error emission improvements
- ✅ `ea985a942` - Block terminator root cause documentation
- ✅ `7f41c2d52` - Warning fixes (switch default cases)

#### Previous Session
- ✅ Fixed `cast<TypedValue<IntType>>` crash in class hierarchy
- ✅ Added `EventTriggerOp` for `->event` syntax
- ✅ Added `QueueConcatOp` for queue concatenation
- ✅ Implemented enum `.name()` method, `$typename`, `$cast`
- ✅ Added `$urandom`, `$urandom_range` with runtime
- ✅ Added constraint block parsing (rand, randc, constraint)
- ✅ Added array locator methods with field-based predicates
- ✅ Added `wait fork`, `disable fork` statement support
- ✅ Fixed covergroup/DPI-C crashes (emit remarks instead)

---

## Remaining Limitations (Xcelium Parity Gaps)

### P0 - Critical (Blocks UVM Execution)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| Block terminators | ❌ BROKEN | ~1,684 errors | Statements.cpp control flow |
| std::randomize() | ⚠️ IR WIP | StdRandomizeOp added, lowering needed | MooreToCore.cpp lowering pattern |

### P1 - High Priority (Blocks Full Simulation)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| Constraint solving | Parse only | No actual solving | Z3/SMT solver integration |
| Covergroups | Skip/remark | No coverage collection | CovergroupDeclOp + runtime |
| DPI-C calls | Skip/remark | No external C linking | LLVM FFI integration |
| Virtual interfaces | Partial | Signal access incomplete | VirtualInterfaceSignalRefOp lowering |

### P2 - Medium Priority (Quality of Life)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| Four-state logic (X/Z) | Missing | Two-state only | Type system extension |
| Assertions (SVA) | Partial | No checking | Assertion dialect work |
| fork/join_any/none | Partial | Basic only | Process management |
| Clocking blocks | Missing | Not parsed | ImportVerilog extension |

### P3 - Lower Priority (Future Work)

| Feature | Status | Notes |
|---------|--------|-------|
| Program blocks | Missing | Rarely used in UVM |
| Bind statements | Missing | Advanced verification |
| Sequence/property | Partial | Complex SVA features |

---

## Track 1: ImportVerilog (Parsing & AST Conversion)

**Status: ✅ FEATURE COMPLETE for UVM parsing**
**Next Focus: Block terminator fix, std::randomize()**

### Completed ✅
- [x] Basic class hierarchy with inheritance
- [x] Parameterized classes (generic classes)
- [x] Static class properties and virtual methods
- [x] Queue, associative array, dynamic array operations
- [x] String operations (len, toupper, tolower, getc, putc, substr)
- [x] Format strings (%s, %d, %h, %b, %p, %m, etc.)
- [x] Event triggers (`->event`)
- [x] Enum `.name()` method, `$typename`, `$cast`
- [x] `$urandom`, `$urandom_range`, `$random`
- [x] Constraint block parsing (rand, randc, constraint)
- [x] `disable fork`, `wait fork` statement support
- [x] Array locator methods with predicates
- [x] **randomize() method handler**
- [x] Covergroup/DPI-C graceful skip

### TODO
- [ ] **Fix block terminator generation** (P0 - CRITICAL)
- [ ] std::randomize() for standalone variables (P0)
- [ ] Full covergroup conversion (P1)
- [ ] Clocking blocks (P2)
- [ ] Program blocks (P3)

### Next Agent Task
**Fix block terminator issue in Statements.cpp** - Ensure all control flow
constructs (if/else, foreach, loops) generate proper MLIR block terminators.

---

## Track 2: MooreToCore (Lowering to LLVM)

**Status: ⚠️ IN PROGRESS - 4 interface ops added this session**
**Next Focus: Complete interface lowering, class vTable**

### Completed ✅
- [x] EventTriggeredOp, WaitConditionOp, EventTriggerOp
- [x] Queue operations (concat, push, pop, unique, min, max)
- [x] UrandomOp, UrandomRangeOp → runtime calls
- [x] String operations lowering
- [x] ArrayLocatorOp → runtime (all operators + field access)
- [x] FormatClassOp, FormatStringOp → sim dialect
- [x] WaitForkOp, DisableForkOp
- [x] DynCastCheckOp → runtime RTTI check
- [x] **RandomizeOp → __moore_randomize_basic runtime call**

### Interface Lowering (ca0c82996) ✅
- [x] InterfaceSignalDeclOp → erase (metadata only)
- [x] ModportDeclOp → erase (metadata only)
- [x] InterfaceInstanceOp → malloc allocation
- [x] VirtualInterfaceGetOp → pass-through pointer
- [x] VirtualInterfaceSignalRefOp → LLVM GEP

### TODO
- [ ] Complete class vTable support (P1)
- [ ] AssocArrayExistsOp (P2)
- [ ] Four-valued logic (X/Z) support (P2)
- [ ] Process/thread management for fork/join (P2)

### MooreToCore Op Audit (2026-01-13) - COMPLETE

**Summary: 70 ops missing lowering patterns out of 233 total (163 have patterns)**

#### P0 - Critical: UVM Ops (31 ops)
All UVM-specific operations need lowering:
- Factory: `UVMObjectCreateOp`, `UVMComponentCreateOp`, `UVMTypeOverrideOp`
- Config: `UVMConfigDbSetOp`, `UVMConfigDbGetOp`
- Sequences: `UVMSequenceStartOp`, `UVMSequenceItemStartOp/FinishOp`
- TLM: `UVMTLMPutOp`, `UVMTLMGetOp`, `UVMTLMTryPutOp/TryGetOp`
- Messaging: `UVMReportOp`

#### P1 - High Priority: Data Structures (8 ops)
- `ArraySizeOp`, `AssocArrayExistsOp`
- Union ops: `UnionCreateOp`, `UnionExtractOp`, `UnionExtractRefOp`
- `StructInjectOp`, `ConcatRefOp`, `NegRealOp`

#### P2 - Medium Priority: Math Builtins (27 ops)
- Trigonometric: `SinBIOp`, `CosBIOp`, `TanBIOp`, etc.
- Exponential: `ExpBIOp`, `LnBIOp`, `Log10BIOp`, `SqrtBIOp`
- Conversion: `Clog2BIOp`, `RealtobitsBIOp`, `BitstorealBIOp`

#### P3 - Already Handled Internally (4 ops)
- `DetectEventOp`, `ForkTerminatorOp`, `NamedBlockTerminatorOp`, `ReturnOp`

Based on audit, categorized by priority:
- **P0**: None remaining (interfaces done!)
- **P1**: ~15 class ops (property access, method calls)
- **P2**: ~30 arithmetic/comparison ops
- **P3**: ~75 advanced ops (coverage, assertions)

### Next Agent Task
**Audit remaining unlowered ops** - Run MooreToCore on UVM IR to identify
which specific ops need lowering patterns next.

---

## Track 3: Moore Runtime Library

**Status: ✅ Comprehensive - randomize() complete!**
**Next Focus: Constraint solver research**

### Completed ✅
- [x] Event operations (`__moore_event_*`)
- [x] Queue operations (push/pop/delete/unique/min/max)
- [x] Associative array operations (`__moore_assoc_*`)
- [x] String operations (`__moore_string_*`)
- [x] Random number generation (`__moore_urandom`, `__moore_urandom_range`)
- [x] Array locator functions (find_eq, find_cmp, find_field_cmp)
- [x] Array min/max/unique functions
- [x] Dynamic cast check (`__moore_dyn_cast_check`)
- [x] **`__moore_randomize_basic`** - Basic field randomization
- [x] Comprehensive unit tests

### TODO
- [ ] `__moore_queue_sort` with comparator (P2)
- [ ] Constraint solver integration (P1 - research phase)
- [ ] Process management functions (P2)
- [ ] Coverage collection runtime (P1)

### Next Agent Task
**Research constraint solver options** - Evaluate Z3, CVC5, or other SMT
solvers for SystemVerilog constraint solving integration.

---

## Track 4: Testing & Integration

**Status: ✅ All AVIPs pass parsing, blocked on lowering**
**Next Focus: Create minimal reproducer for block terminator issue**

### AVIP Testing Results (~/mbit/*)

| AVIP | Parsing | randomize() | With UVM | Status |
|------|---------|-------------|----------|--------|
| AXI4 | ✅ Pass | ✅ 191 uses | ✅ Pass | Complete |
| APB | ✅ Pass | ⚠️ std::randomize | ✅ Pass | Partial |
| AHB | ✅ Pass | ✅ 26 uses | ✅ Pass | Complete |
| SPI | ✅ Pass | ✅ 58 uses | ✅ Pass | Complete |
| I2S | ✅ Pass | ✅ 78 uses | ✅ Pass | Complete |
| I3C | ✅ Pass | ✅ 61 uses | ✅ Pass | Complete |
| JTAG | ✅ Pass | ✅ 3 uses | ✅ Pass | Complete |
| UART | ✅ Pass | ✅ 19 uses | ✅ Pass | Complete |
| AXI4-Lite | ✅ Pass | ✅ 252 uses | ✅ Pass | Complete |

**All 9 AVIP packages pass parsing!**
**APB uses std::randomize() which needs implementation**

### UVM Testbench Testing

| Test Type | Status | Notes |
|-----------|--------|-------|
| UVM package alone | ✅ Parse | Parses completely |
| UVM-style code | ✅ Parse | Generates Moore IR |
| Full UVM testbench | ❌ Lower | Block terminator errors |

### Test Files Added
- `test/Conversion/ImportVerilog/randomize.sv` - randomize() patterns
- `test/Conversion/MooreToCore/random-ops.mlir` - RandomizeOp lowering
- `test/Conversion/MooreToCore/interface-ops.mlir` - Interface lowering

### Next Agent Task
**Create minimal block terminator reproducer** - Extract the simplest SV code
from uvm_cmdline_set_verbosity that triggers the block terminator error.

---

## Xcelium Feature Comparison (Updated)

| Feature | Xcelium | CIRCT Parse | CIRCT Lower | Status |
|---------|---------|-------------|-------------|--------|
| Basic SV | ✅ | ✅ | ✅ | Done |
| Classes | ✅ | ✅ | ⚠️ Partial | P1 |
| Queues | ✅ | ✅ | ✅ | Done |
| Events | ✅ | ✅ | ✅ | Done |
| $urandom | ✅ | ✅ | ✅ | Done |
| $typename | ✅ | ✅ | ✅ | Done |
| UVM Parsing | ✅ | ✅ | N/A | Done |
| wait/disable fork | ✅ | ✅ | ✅ | Done |
| %m format | ✅ | ✅ | ✅ | Done |
| $cast | ✅ | ✅ | ✅ | Done |
| Array locators | ✅ | ✅ | ✅ | Done |
| **randomize()** | ✅ | ✅ | ✅ | **Done!** |
| std::randomize() | ✅ | ❌ | ❌ | P0 |
| Constraint solving | ✅ | ✅ Parse | ❌ | P1 |
| Coverage | ✅ | ⚠️ Skip | ❌ | P1 |
| DPI-C | ✅ | ⚠️ Skip | ❌ | P2 |
| Assertions | ✅ | Partial | ❌ | P2 |
| Interfaces | ✅ | ✅ | ✅ | **Done!** |

---

## Next Sprint Tasks (4 Agents)

### Agent 1: Block Terminator Fix (P0 - CRITICAL)
**Track:** 1 (ImportVerilog)
**Task:** Fix block terminator generation in Statements.cpp
**Files:** `lib/Conversion/ImportVerilog/Statements.cpp`
**Test:** UVM package should convert without "block with no terminator" errors

### Agent 2: Math Builtins Lowering (P2)
**Track:** 2 (MooreToCore)
**Task:** Add lowering patterns for trigonometric/exponential/rounding ops
**Files:** `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Test:** `test/Conversion/MooreToCore/math-builtins.mlir` (created)
**Ops:** sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, ln, log10, sqrt, floor, ceil

### Agent 3: Covergroup Ops Definition (P1)
**Track:** 1+2 (ImportVerilog + MooreToCore)
**Task:** Define CovergroupDeclOp, CoverpointDeclOp, CoverCrossDeclOp in Moore dialect
**Files:** `include/circt/Dialect/Moore/MooreOps.td`, `lib/Conversion/ImportVerilog/Structure.cpp`
**Test:** `test/Conversion/ImportVerilog/covergroup.sv` (created)

### Agent 4: std::randomize() Parsing (P0)
**Track:** 1 (ImportVerilog)
**Task:** Add std::randomize(variable) handler for standalone randomization
**Files:** `lib/Conversion/ImportVerilog/Expressions.cpp`
**Test:** `test/Conversion/ImportVerilog/std-randomize.sv` (created)

---

## Commands

```bash
# Build circt-verilog
ninja -C build bin/circt-verilog

# Test UVM parsing
./build/bin/circt-verilog --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv

# Test with Moore IR output
./build/bin/circt-verilog --ir-moore \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv 2>&1 | head -100

# Test AVIP with UVM
./build/bin/circt-verilog \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  --include-dir=~/mbit/axi4_avip/src/globals \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv \
  ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv

# Run MooreToCore tests
./build/bin/circt-opt --convert-moore-to-core \
  test/Conversion/MooreToCore/random-ops.mlir

# Debug UVM conversion
./build/bin/circt-verilog --debug \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv 2>&1 | \
  grep -E "block with no terminator|empty block" | head -20
```

---

## Milestone Targets

### M1: UVM Parsing (✅ COMPLETE)
- [x] Parse UVM core library without errors
- [x] Parse all AVIP packages
- [x] Generate Moore dialect IR

### M2: Basic UVM Lowering (⚠️ IN PROGRESS)
- [ ] Fix block terminator issue
- [x] Lower randomize() to runtime
- [x] Lower interface operations
- [ ] Lower to executable LLVM IR

### M3: UVM Simulation (FUTURE)
- [ ] Constraint solving with Z3
- [ ] Coverage collection
- [ ] DPI-C integration
- [ ] Full vTable dispatch

### M4: Xcelium Parity (FUTURE)
- [ ] All UVM testbenches run
- [ ] Performance within 2x of Xcelium
- [ ] Full SystemVerilog 2017 support
