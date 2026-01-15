# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

## Current Status: UVM PARSING - 1 CRASH REMAINING (January 15, 2026)

**Test Command**:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
```

**Current Blockers**:
1. **IntType CRASH** - Still occurs when parsing UVM (different location from case statement fix)
2. **DPI-C imports** - Not supported, UVM uses extensively

**Previous Blockers FIXED**:
1. ~~`$fwrite` unsupported~~ ✅ FIXED (ccfc4f6ca)
2. ~~`$fopen` unsupported~~ ✅ FIXED (ce8d1016a)
3. ~~`next` unsupported~~ ✅ FIXED (2fa392a98) - string assoc array iteration
4. ~~`$fclose` unsupported~~ ✅ FIXED (b4a18d045) - File I/O complete
5. ~~`%20s` width specifier not supported~~ ✅ FIXED (88085cbd7) - String format width
6. ~~String case IntType crash~~ ✅ FIXED (3410de2dc) - String case statement handling

**Note**: Earlier "AVIP passing" tests used wrong UVM path (`~/UVM/distrib/src`).
Correct path is `~/uvm-core/src`. With correct path, UVM still crashes.

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

## Active Workstreams (4 Agents Needed)

### Track A: IntType Crash (track-a-sim)
**Status**: 🔄 NEEDS WORK - Another IntType crash in UVM
**Previous Work**: Enum iteration methods (f8e4b82cf)
**Next Task**: Debug and fix remaining IntType assertion crash in UVM parsing
**Test**: `./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src`
**Files**: Expressions.cpp - likely binary operators or type inference

### Track B: DPI-C Support (track-b-uvm)
**Status**: 🔄 NEEDS WORK - DPI-C imports not supported
**Previous Work**: $fclose (b4a18d045)
**Next Task**: Add stub/placeholder support for DPI-C imports
**Test**: UVM uses DPI for regex, command line args, tool info
**Files**: Statements.cpp, new DPI handling

### Track C: Enum Iteration (track-c-types)
**Status**: 🔄 NEEDS WORK - enum.first/next/last/prev unsupported
**Previous Work**: String case fix (3410de2dc)
**Next Task**: Add enum iteration method support
**Files**: Expressions.cpp - handle EnumeratedTypeMethods

### Track D: MooreToCore Lowering (track-d-devex)
**Status**: 🔄 NEEDS WORK - File I/O ops need lowering
**Previous Work**: String assoc array (2fa392a98)
**Next Task**: Lower FOpenBIOp, FWriteBIOp, FCloseBIOp to LLVM calls
**Files**: MooreToCore.cpp, MooreRuntime.cpp

---

## Priority Queue

### CRITICAL (Blocking UVM Parsing)
1. **Fix remaining IntType Crash** - Another location crashes, not case statements
2. **DPI-C imports** - UVM uses extensively for regex, command line, etc.

### PREVIOUSLY FIXED ✅
- ~~**$fopen**~~ - ✅ Fixed (ce8d1016a)
- ~~**$fwrite**~~ - ✅ Fixed (ccfc4f6ca)
- ~~**$fclose**~~ - ✅ Fixed (b4a18d045)
- ~~**String assoc array next()**~~ - ✅ Fixed (2fa392a98)
- ~~**String case IntType crash**~~ - ✅ Fixed (3410de2dc)
- ~~**%20s format**~~ - ✅ Fixed (88085cbd7)

### HIGH (After UVM Parses)
3. **Complete MooreToCore lowering** - All ops must lower for simulation
4. **Enum iteration methods** - first(), next(), last(), prev()
5. **MooreSim execution** - Run compiled testbenches

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

### File I/O (In Progress)
- [x] $fopen - file open (ce8d1016a)
- [ ] $fclose - file close
- [x] $fwrite - formatted file write (ccfc4f6ca)
- [x] $fdisplay - file display (ccfc4f6ca - via $fwrite handler)
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

### MooreToCore Lowering
- [x] AssocArrayExistsOp
- [x] Union operations
- [x] Math functions (clog2, atan2, hypot, etc.)
- [x] Real type conversions
- [ ] File I/O ops (FOpenBIOp, FWriteBIOp, FCloseBIOp)

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
| Jan 2026 | M1: UVM Parses | Zero errors parsing uvm_pkg.sv |
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
- `88085cbd7` - [ImportVerilog] Add string format specifier with width support
- `b4a18d045` - [ImportVerilog] Add $fclose system task support
- `2fa392a98` - [MooreToCore] Fix string-keyed associative array iteration
- `f8e4b82cf` - [ImportVerilog] Add enum iteration methods (first, next, last, prev)
- `ccfc4f6ca` - [ImportVerilog] Add $fwrite system call support
- `ce8d1016a` - [ImportVerilog] Add $fopen system call support

---

## Architecture Reference

See full plan: `~/.claude/plans/jiggly-tickling-engelbart.md`

Track assignments:
- **Track A (Sim)**: Event kernel, process control, performance
- **Track B (UVM)**: Class parsing, constraints, factory pattern
- **Track C (Types)**: 4-state, coverage, file I/O
- **Track D (DevEx)**: LSP, linting, dashboards
- **Track E (Assert)**: SVA, vacuity detection, debug
