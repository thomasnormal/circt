# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

## Current Status: 🎉 UVM PARSING COMPLETE - MILESTONE M1 ACHIEVED (January 15, 2026)

**Test Command**:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
# Exit code: 0 (SUCCESS!)
```

**Current Blockers / Limitations**:
1. ~~**Mem2Reg dominance errors**~~ ✅ FIXED (b881afe61) - Loop-local variables now excluded from Mem2Reg promotion
2. **Runtime gaps** - Randomization/coverage not implemented; DPI/VPI still stubs; MooreToCore queue globals lowering pending
3. **Multiple input files** - Minor issue with parsing multiple SV files together (empty filename error)

**Recent Fixes (This Session)**:
- **Mem2Reg loop-local variable dominance** ✅ FIXED (b881afe61) - Variables inside loops no longer promoted, fixing 4 dominance errors
- **Static property via instance** ✅ FIXED (a1418d80f) - SystemVerilog allows `obj.static_prop` access. Now correctly generates GetGlobalVariableOp instead of ClassPropertyRefOp.
- **Static property names in parameterized classes** ✅ FIXED (a1418d80f) - Each specialization now gets unique global variable name (e.g., `uvm_pool_1234::m_prop` not `uvm_pool::m_prop`).
- **Abstract class vtable** ✅ FIXED (a1418d80f) - Virtual classes with mixed concrete/pure virtual methods now skip vtable generation instead of emitting error.
- **Time type in Mem2Reg** ✅ FIXED (3c9728047) - `VariableOp::getDefaultValue()` now correctly returns TimeType values instead of l64 constants.
- **Global variable redefinition** ✅ FIXED (a152e9d35) - Fixed duplicate GlobalVariableOp when class type references the variable in methods.
- **Method lookup in parameterized classes** ✅ FIXED (71c80f6bb) - Class bodies now populated via convertClassDeclaration in declareFunction.
- **Property type mismatch** ✅ FIXED - Parameterized class property access uses correct specialized class symbol.

**Previous Blockers FIXED** (Earlier):
1. ~~`$fwrite` unsupported~~ ✅ FIXED (ccfc4f6ca)
2. ~~`$fopen` unsupported~~ ✅ FIXED (ce8d1016a)
3. ~~`next` unsupported~~ ✅ FIXED (2fa392a98) - string assoc array iteration
4. ~~`$fclose` unsupported~~ ✅ FIXED (b4a18d045) - File I/O complete
5. ~~`%20s` width specifier not supported~~ ✅ FIXED (88085cbd7) - String format width
6. ~~String case IntType crash~~ ✅ FIXED (3410de2dc) - String case statement handling

**Note**: Earlier "AVIP passing" tests used wrong UVM path (`~/UVM/distrib/src`).
Correct path is `~/uvm-core/src`. Making good progress on remaining blockers!

---

## Feature Matrix: Current vs Target

| Capability | Current CIRCT | Target (Xcelium Parity) | Status |
|------------|---------------|------------------------|--------|
| **Classes** | Basic OOP + UVM parsing | Full OOP + factory pattern | ✅ Mostly done |
| **Interfaces** | Partial | Virtual interfaces, modports | ✅ Complete |
| **Process Control** | fork/join designed | fork/join, disable, wait | ✅ Designed |
| **File I/O** | $fopen, $fwrite, $fclose | $fopen, $fwrite, $fclose | ✅ Complete |
| **Assoc Arrays** | Int keys work | All key types + iterators | ✅ String keys fixed |
| **Randomization** | Not supported | rand/randc, constraints | ⚠️ Parsing only |
| **Coverage** | Coverage dialect exists | Full functional coverage | ⚠️ Partial |
| **Assertions** | Basic SVA | Full SVA | ✅ SVA dialect |
| **DPI/VPI** | Basic | Full support | ⚠️ Basic only |
| **MooreToCore** | All 9 AVIPs lower | Full UVM lowering | ✅ Complete |

---

## Active Workstreams (keep 4 agents busy)

### Track A: Multi-File Parsing Fix
**Status**: 🔴 IN PROGRESS
**Task**: Fix the empty filename error when parsing multiple SV files together. This blocks AVIP+UVM integration testing.
**Files**: lib/Tools/circt-verilog/, slang integration
**Next**: Investigate why parsing `uvm_pkg.sv` + `apb_global_pkg.sv` together fails with "cannot open input file ''"

### Track B: MooreToCore Lowering Completion
**Status**: 🟡 IN PROGRESS
**Task**: Lower remaining 13 Moore ops for simulation. Previous analysis showed 95.4% coverage (268/281 ops).
**Files**: lib/Conversion/MooreToCore/MooreToCore.cpp
**Next**: Implement lowering patterns for unlowered ops found when running moore-to-core on UVM IR.

### Track C: AVIP Component Testing
**Status**: 🟡 IN PROGRESS
**Task**: Test individual AVIP components (interfaces, BFMs) that don't require UVM imports.
**Files**: ~/mbit/*_avip/src/hdl_top/
**Next**: Parse apb_if.sv, master_agent_bfm.sv, slave_agent_bfm.sv individually.

### Track D: Unit Test Coverage
**Status**: 🟡 IN PROGRESS
**Task**: Add lit tests for recent fixes including Mem2Reg loop-local variable handling.
**Files**: test/Conversion/ImportVerilog/, test/Dialect/Moore/
**Next**: Create test for loop-local variable dominance fix (b881afe61).

---

## Priority Queue

### CRITICAL (Blocking UVM Parsing)
None! UVM parsing complete.

### RECENTLY FIXED ✅ (This Session)
- ~~**Mem2Reg loop-local variable dominance**~~ - ✅ Fixed (b881afe61) - Variables inside loops excluded from promotion
- ~~**Static property via instance**~~ - ✅ Fixed (a1418d80f) - `obj.static_prop` now uses GetGlobalVariableOp
- ~~**Static property names in parameterized classes**~~ - ✅ Fixed (a1418d80f) - Unique names per specialization
- ~~**Abstract class vtable**~~ - ✅ Fixed (a1418d80f) - Mixed concrete/pure virtual methods allowed
- ~~**Time type in Mem2Reg**~~ - ✅ Fixed (3c9728047) - Default values for time variables
- ~~**Method lookup in parameterized classes**~~ - ✅ Fixed (71c80f6bb) - Class body conversion
- ~~**Super.method() dispatch**~~ - ✅ Fixed (09e75ba5a) - Direct dispatch instead of vtable
- ~~**Class upcast with parameterized base**~~ - ✅ Fixed (fbbc2a876) - Generic class lookup
- ~~**Global variable redefinition**~~ - ✅ Fixed (a152e9d35) - Recursive type conversion

### PREVIOUSLY FIXED ✅
- ~~**UVM class declaration issues**~~ - ✅ Fixed (555a78350)
- ~~**String ato* methods**~~ - ✅ Fixed (14dfdbe9f + 34ab7a758)
- ~~**Non-integral assoc array keys**~~ - ✅ Fixed (f6b79c4c7)
- ~~**File I/O ($fopen, $fwrite, $fclose)**~~ - ✅ Fixed

### HIGH (After UVM Parses)
3. **Complete MooreToCore lowering** - All ops must lower for simulation (ato* already done; queue globals pending)
4. **Enum iteration methods** - first(), next(), last(), prev()
5. **MooreSim execution** - Run compiled testbenches
6. **Factory runtime** - Ensure uvm_pool/callback singleton handling matches specialization typing

### MEDIUM (Production Quality)
6. **Coverage groups** - covergroup, coverpoint
7. **Constraint solver (Z3)** - Enable randomization
8. **$fgets** - File read line

### LOW (Future Enhancements)
9. **SVA assertions** - Full property/sequence support
10. **Multi-core simulation** - Performance scaling
11. **Interactive debugger** - circt-debug CLI

---

## Features Completed

### Class Support
- [x] Class declarations and handles
- [x] Class inheritance (extends)
- [x] Virtual methods and vtables
- [x] Static class properties (partial)
- [x] Parameterized classes
- [x] $cast dynamic type checking
- [x] Class handle comparison (==, !=, null)
- [x] new() allocation

### Queue/Array Support
- [x] Queue type and operations
- [x] push_back, push_front, pop_back, pop_front
- [x] delete(), delete(index)
- [x] size(), max(), min(), unique(), sort()
- [x] Dynamic arrays with new[size]
- [x] Associative arrays (int keys)
- [x] exists(), delete(key)
- [x] first(), next(), last(), prev() for string keys (2fa392a98)

### String Support
- [x] String type
- [x] itoa(), len(), getc()
- [x] toupper(), tolower()
- [x] putc() character assignment
- [x] %p format specifier
- [x] String in format strings (emitDefault fix)
- [x] atoi(), atohex(), atooct(), atobin() (14dfdbe9f)

### File I/O ✅ Complete
- [x] $fopen - file open (ce8d1016a)
- [x] $fclose - file close (b4a18d045)
- [x] $fwrite - formatted file write (ccfc4f6ca)
- [x] $fdisplay - file display (ccfc4f6ca - via $fwrite handler)
- [x] $sscanf - string scan (2657ceab7)
- [ ] $fgets - file read line

### Process Control
- [x] fork/join, fork/join_any, fork/join_none
- [x] Named blocks
- [x] disable statement
- [x] wait(condition) statement

### Event Support
- [x] event type (moore::EventType)
- [x] .triggered property
- [x] Event trigger (->)

### Interface Support
- [x] Interface declarations
- [x] Modports
- [x] Virtual interfaces (basic)

### MooreToCore Lowering ✅ Complete
- [x] AssocArrayExistsOp
- [x] Union operations
- [x] Math functions (clog2, atan2, hypot, etc.)
- [x] Real type conversions
- [x] File I/O ops (52511fe46) - FOpenBIOp, FWriteBIOp, FCloseBIOp

---

## AVIP Testing

Test files in ~/mbit/*:
- ahb_avip, apb_avip, axi4_avip, axi4Lite_avip
- i2s_avip, i3c_avip, jtag_avip, spi_avip, uart_avip

**Current blocker**: All AVIPs import UVM, which crashes.
**After crash fix**: Test individual components without UVM macros.

**Test non-UVM components**:
```bash
./build/bin/circt-verilog --ir-moore \
  ~/mbit/apb_avip/src/globals/apb_global_pkg.sv \
  ~/mbit/apb_avip/src/hdl_top/apb_if/apb_if.sv
```

---

## Milestones

| Target | Milestone | Criteria |
|--------|-----------|----------|
| Jan 2026 | M1: UVM Parses | Zero errors parsing uvm_pkg.sv | ✅ ACHIEVED |
| Feb 2026 | M2: File I/O | $fopen, $fwrite, $fclose work |
| Mar 2026 | M3: AVIP Parses | All ~/mbit/* AVIPs parse |
| Q2 2026 | M4: Basic Sim | Simple UVM test runs |
| Q3 2026 | M5: Full UVM | Factory pattern, phasing work |
| Q4 2026 | M6: AVIPs Run | mbits/ahb_avip executes |

---

## Build Commands
```bash
# Build
ninja -C build circt-verilog

# Test UVM
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src

# Test AVIP interface only (no UVM)
./build/bin/circt-verilog --ir-moore \
  ~/mbit/apb_avip/src/globals/apb_global_pkg.sv \
  ~/mbit/apb_avip/src/hdl_top/apb_if/apb_if.sv
```

---

## Recent Commits
- `b881afe61` - [Moore] Don't promote loop-local variables to avoid Mem2Reg dominance errors
- `3c9728047` - [Moore] Fix time type handling in Mem2Reg default value generation
- `a1418d80f` - [ImportVerilog][Moore] Fix static property access and abstract class handling
- `71c80f6bb` - [ImportVerilog] Fix method lookup in parameterized class specializations
- `09e75ba5a` - [ImportVerilog] Use direct dispatch for super.method() calls
- `fbbc2a876` - [ImportVerilog] Fix class upcast with parameterized base classes
- `a152e9d35` - [ImportVerilog] Fix global variable redefinition during recursive type conversion
- `555a78350` - [ImportVerilog] Fix UVM class declaration and statement handling issues
- `34ab7a758` - [MooreToCore] Add lowering for string ato* ops
- `f6b79c4c7` - [ImportVerilog] Fix non-integral assoc array keys and pure virtual methods
- `14dfdbe9f` - [ImportVerilog] Add support for string ato* methods

---

## Architecture Reference

See full plan: `~/.claude/plans/jiggly-tickling-engelbart.md`

Track assignments:
- **Track A (Sim)**: Event kernel, process control, performance
- **Track B (UVM)**: Class parsing, constraints, factory pattern
- **Track C (Types)**: 4-state, coverage, file I/O
- **Track D (DevEx)**: LSP, linting, dashboards
- **Track E (Assert)**: SVA, vacuity detection, debug
