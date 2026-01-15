# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

## Current Status: UVM Parsing - CRASH + 3 Errors (January 15, 2026)

**Test Command**:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
```

**Current Errors (blocking)**:
1. `$fwrite` unsupported (uvm_object.svh:893) - Track C
2. `$fopen` unsupported (uvm_text_tr_database.svh:107) - Track B
3. `next` unsupported - string assoc array iterator (uvm_report_server.svh:514) - Track D
4. **CRASH** - `cast<TypedValue<IntType>>` assertion failure - Track A

---

## Feature Matrix: Current vs Target

| Capability | Current CIRCT | Target (Xcelium Parity) | Status |
|------------|---------------|------------------------|--------|
| **Classes** | Basic OOP + UVM parsing | Full OOP + factory pattern | ✅ Mostly done |
| **Interfaces** | Partial | Virtual interfaces, modports | ✅ Complete |
| **Process Control** | fork/join designed | fork/join, disable, wait | ✅ Designed |
| **File I/O** | Not supported | $fopen, $fwrite, $fclose | ⚠️ In progress |
| **Assoc Arrays** | Int keys work | All key types + iterators | ⚠️ String keys broken |
| **Randomization** | Not supported | rand/randc, constraints | ⚠️ Parsing only |
| **Coverage** | Coverage dialect exists | Full functional coverage | ⚠️ Partial |
| **Assertions** | Basic SVA | Full SVA | ✅ SVA dialect |
| **DPI/VPI** | Basic | Full support | ⚠️ Basic only |

---

## Active Workstreams (4 Agents Running)

### Track A: Simulation & Process Control (track-a-sim)
**Current Task**: Fix IntType assertion crash
**Agent**: a90c570 (running)
**Goal**: Debug and fix the TypedValue<IntType> crash during UVM parsing
**Files**: lib/Conversion/ImportVerilog/Expressions.cpp

### Track B: UVM & File I/O (track-b-uvm)
**Current Task**: Implement $fopen syscall
**Agent**: a6fa14c (running)
**Goal**: Add FOpenBIOp and handler for file open
**Files**: MooreOps.td, Expressions.cpp

### Track C: Types & Coverage (track-c-types)
**Current Task**: Implement $fwrite syscall
**Agent**: abd8dc2 (running)
**Goal**: Add FWriteBIOp and handler for formatted file write
**Files**: MooreOps.td, Statements.cpp

### Track D: Developer Experience (track-d-devex)
**Current Task**: Fix string-keyed associative array next()
**Agent**: a7f1877 (running)
**Goal**: Enable string key iteration in assoc arrays
**Files**: Expressions.cpp, MooreOps.td

---

## Priority Queue

### CRITICAL (Blocking UVM Parsing)
1. **Fix IntType Crash** - Track A - Must fix to proceed
2. **$fopen** - Track B - File I/O needed by UVM
3. **$fwrite** - Track C - File I/O needed by UVM
4. **String assoc array next()** - Track D - Used by UVM report server

### HIGH (Blocking AVIP Runs)
5. **$fclose** - File descriptor cleanup
6. **$fdisplay** - Formatted output to file
7. **Complete MooreToCore lowering** - All ops must lower

### MEDIUM (Production Quality)
8. **DPI-C imports** - For full UVM compatibility
9. **Coverage groups** - covergroup, coverpoint
10. **Constraint solver (Z3)** - Enable randomization

### LOW (Future Enhancements)
11. **SVA assertions** - Full property/sequence support
12. **Multi-core simulation** - Performance scaling
13. **Interactive debugger** - circt-debug CLI

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
- [ ] first(), next(), last(), prev() for string keys

### String Support
- [x] String type
- [x] itoa(), len(), getc()
- [x] toupper(), tolower()
- [x] putc() character assignment
- [x] %p format specifier
- [x] String in format strings (emitDefault fix)

### File I/O (In Progress)
- [ ] $fopen - file open
- [ ] $fclose - file close
- [ ] $fwrite - formatted file write
- [ ] $fdisplay - file display
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
- `e5124660e` - [Docs] Update project plan with current UVM errors
- `10db8cdd0` - [Docs] Update project plan with January 14 session progress
- `9be6c6c4c` - [ImportVerilog] Handle string types in emitDefault
- `4b9b3441d` - [MooreToCore] Add lowering for AssocArrayExistsOp

---

## Architecture Reference

See full plan: `~/.claude/plans/jiggly-tickling-engelbart.md`

Track assignments:
- **Track A (Sim)**: Event kernel, process control, performance
- **Track B (UVM)**: Class parsing, constraints, factory pattern
- **Track C (Types)**: 4-state, coverage, file I/O
- **Track D (DevEx)**: LSP, linting, dashboards
- **Track E (Assert)**: SVA, vacuity detection, debug
