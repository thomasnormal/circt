# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-14)

### Overall Progress

**Parsing: ✅ COMPLETE** - UVM core library parses without errors!
**Lowering: ⚠️ BLOCKED** - TypeAlias→Struct type resolution issue

### Sprint 1 Completed (8 commits)

- ✅ `87d1420ca` - atan2, hypot, fneg lowering tests
- ✅ `f5f2882a4` - std::randomize() parsing and lowering
- ✅ `5e8cd2b51` - $clog2 test fix
- ✅ `9ef4b3391` - $clog2 builtin lowering
- ✅ `1f6d550bf` - Covergroup ops (CovergroupDeclOp, CoverpointDeclOp, CoverCrossDeclOp)
- ✅ `e6133b7e8` - Block terminator test
- ✅ `f727d5708` - 18 math builtin lowerings (trig, hyperbolic, exp, rounding)
- ✅ `82e8b2e57` - UVM Parity Plan update

### Current Blocker: TypeAlias→Struct Resolution

The UVM package parses but fails during conversion when a TypeAlias points to
a struct type within a generic class. Specifically:

```
uvm_queue#(uvm_acs_name_struct) names;  // fails
```

**Root Cause:**
- `uvm_acs_name_struct` is a typedef to a struct in `uvm_object_globals.svh`
- When used as a generic class parameter, `getCanonicalType()` returns empty
- This cascades through uvm_component and uvm_queue

**Debug Evidence:**
```
TypeAliasType: T -> targetType: uvm_acs_name_struct (kind: TypeAlias) -> canonical:
```

**Fix Location:** `lib/Conversion/ImportVerilog/Types.cpp:108-128`

### Previous Sprint Commits (35+)

- ✅ randomize() method handler (ImportVerilog)
- ✅ RandomizeOp lowering + __moore_randomize_basic runtime
- ✅ Interface lowering patterns (4 ops)
- ✅ EventTriggerOp, QueueConcatOp, DynCastCheckOp
- ✅ enum .name(), $typename, $cast
- ✅ $urandom, $urandom_range, constraint parsing
- ✅ Array locator methods, wait/disable fork

---

## Remaining Limitations (Xcelium Parity Gaps)

### P0 - Critical (Blocks UVM Execution)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| TypeAlias→Struct | ❌ BROKEN | Generic class param fails | Types.cpp TypeAliasType visitor |
| Block terminators | ✅ Fixed | Tests added | Statements.cpp verified |
| std::randomize() | ✅ Done | Parsing + lowering complete | f5f2882a4 |

### P1 - High Priority (Blocks Full Simulation)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| Constraint solving | Parse only | No actual solving | Z3/SMT solver integration |
| Covergroups | ✅ IR Done | Runtime collection needed | CovergroupDeclOp committed |
| DPI-C calls | ⚠️ Skip | No external C linking | Stub generation or FFI |
| Virtual interfaces | ✅ Done | ca0c82996 | Complete |
| Class property access | Partial | Lowering incomplete | MooreToCore patterns |

### P2 - Medium Priority (Quality of Life)

| Feature | Status | Gap | Fix Required |
|---------|--------|-----|--------------|
| Math builtins | ✅ Done | 18 ops lowered | f727d5708 |
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

**Status: ⚠️ BLOCKED on TypeAlias→Struct resolution**
**Next Focus: Fix generic class struct parameters**

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
- [x] randomize() method handler
- [x] Covergroup ops (CovergroupDeclOp, CoverpointDeclOp, CoverCrossDeclOp)
- [x] **std::randomize() parsing** (f5f2882a4)
- [x] Block terminator test (e6133b7e8)

### TODO
- [ ] **Fix TypeAlias→Struct in generic classes** (P0 - CRITICAL)
- [ ] DPI-C stub generation (P1)
- [ ] Clocking blocks (P2)
- [ ] Program blocks (P3)

### Next Agent Task
**Fix TypeAlias→Struct resolution** - When a typedef points to a struct type
(like `uvm_acs_name_struct`) and is used as a generic class parameter, ensure
`getCanonicalType()` properly resolves the underlying struct type.

---

## Track 2: MooreToCore (Lowering to LLVM)

**Status: ✅ Sprint 1 complete - 20+ ops lowered**
**Next Focus: Class property access, union ops**

### Sprint 1 Completed ✅
- [x] **18 math builtins** (f727d5708): sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, ln, log10, sqrt, floor, ceil
- [x] **$clog2 lowering** (9ef4b3391): Integer ceiling log2
- [x] **atan2, hypot lowering** (87d1420ca): Binary real ops
- [x] **std::randomize lowering** (f5f2882a4): StdRandomizeOp → runtime
- [x] **NegRealOp** (87d1420ca): moore.fneg → arith.negf
- [x] Interface lowering (ca0c82996)
- [x] RandomizeOp → __moore_randomize_basic

### Previous Completed ✅
- [x] EventTriggeredOp, WaitConditionOp, EventTriggerOp
- [x] Queue operations (concat, push, pop, unique, min, max)
- [x] UrandomOp, UrandomRangeOp → runtime calls
- [x] String operations lowering
- [x] ArrayLocatorOp → runtime (all operators + field access)
- [x] FormatClassOp, FormatStringOp → sim dialect
- [x] WaitForkOp, DisableForkOp
- [x] DynCastCheckOp → runtime RTTI check

### TODO
- [ ] Class property access (PropertyRefOp, InstancePropertyRefOp) (P1)
- [ ] Union ops (UnionCreateOp, UnionExtractOp) (P1)
- [ ] AssocArrayExistsOp (P2)
- [ ] BitstorealBIOp, RealtobitsBIOp (P2)
- [ ] Four-valued logic (X/Z) support (P2)
- [ ] Process/thread management for fork/join (P2)

### MooreToCore Op Audit Update

**Summary: ~50 ops still need patterns (down from 70)**

#### Remaining P1 - High Priority (12 ops)
- Class: `PropertyRefOp`, `InstancePropertyRefOp`, `MethodCallOp`
- Union: `UnionCreateOp`, `UnionExtractOp`, `UnionExtractRefOp`
- Struct: `StructInjectOp`, `ConcatRefOp`
- Array: `ArraySizeOp`, `AssocArrayExistsOp`
- Conversion: `BitstorealBIOp`, `RealtobitsBIOp`

#### Completed P2 - Math Builtins (20 ops) ✅
All trigonometric, hyperbolic, exponential, rounding, and binary real ops done!

### Next Agent Task
**Add class property access lowering** - Implement PropertyRefOp and
InstancePropertyRefOp patterns to access class member fields.

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

## Next Sprint Tasks (Sprint 2 - 4 Agents)

### Agent 1: TypeAlias→Struct Fix (P0 - CRITICAL)
**Track:** 1 (ImportVerilog)
**Worktree:** track-a-sim
**Task:** Fix TypeAliasType visitor to handle struct typedefs in generic class parameters
**Files:** `lib/Conversion/ImportVerilog/Types.cpp`
**Test:** Create test with `typedef struct { ... } my_struct; class C#(type T); T data; endclass`
**Bug:** `uvm_queue#(uvm_acs_name_struct)` fails with empty canonical type

### Agent 2: Class Property Access Lowering (P1)
**Track:** 2 (MooreToCore)
**Worktree:** track-b-uvm
**Task:** Implement PropertyRefOp and InstancePropertyRefOp lowering
**Files:** `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Test:** `test/Conversion/MooreToCore/class-property.mlir`
**Approach:** Use LLVM GEP to access class struct fields

### Agent 3: DPI-C Stub Generation (P1)
**Track:** 1 (ImportVerilog)
**Worktree:** track-c-types
**Task:** Generate stub functions for DPI-C imports instead of skipping
**Files:** `lib/Conversion/ImportVerilog/Expressions.cpp`, `Structure.cpp`
**Test:** `test/Conversion/ImportVerilog/dpi-stub.sv`
**Approach:** Create external func declarations that can be linked later

### Agent 4: Union Ops Lowering (P1)
**Track:** 2 (MooreToCore)
**Worktree:** track-d-devex
**Task:** Implement UnionCreateOp, UnionExtractOp, UnionExtractRefOp
**Files:** `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Test:** `test/Conversion/MooreToCore/union-ops.mlir`
**Approach:** Union as largest-member-sized allocation with bitcast

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
