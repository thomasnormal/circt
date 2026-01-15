# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.

## Current Status: UVM Parsing - CRASH + 3 Errors (January 15, 2026)

**Test Command**:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
```

**Current Errors (blocking)**:
1. `$fwrite` unsupported (uvm_object.svh:893)
2. `$fopen` unsupported (uvm_text_tr_database.svh:107)
3. `next` unsupported - string associative array iterator (uvm_report_server.svh:514)
4. **CRASH** - `cast<TypedValue<IntType>>` assertion failure

**Warnings (informational)**:
- `unknown escape sequence '\.'` - Regex in string literal
- `no top-level modules found` - Expected, UVM is a package
- DPI-C imports not yet supported (expected)
- Unreachable code warnings

## Active Workstreams (4 Agents)

### Track A: Fix IntType Crash (track-a-sim)
**Goal**: Fix the TypedValue<IntType> assertion failure
**Status**: Investigating - crash occurs during class method processing
**Next Task**: Add debug tracing to identify exact location and type mismatch

### Track B: Implement $fopen (track-b-uvm)
**Goal**: Add $fopen syscall support
**Status**: Starting fresh
**Next Task**: Add FOpenBIOp to MooreOps.td and handler in Expressions.cpp

### Track C: Implement $fwrite (track-c-types)
**Goal**: Add $fwrite syscall support
**Status**: Had implementation in worktree (a30e1484d) - needs manual merge
**Next Task**: Cherry-pick or reimplement $fwrite handler in Statements.cpp

### Track D: String Assoc Array next() (track-d-devex)
**Goal**: Fix string-keyed associative array iteration
**Status**: Integer key iterators work, string keys fail
**Next Task**: Add string key support to AssocArrayNextOp

## Features Completed

### Class Support
- [x] Class declarations and handles
- [x] Class inheritance (extends)
- [x] Virtual methods and vtables
- [x] Static class properties (partial)
- [x] Parameterized classes
- [x] this_type pattern
- [x] $cast dynamic type checking
- [x] Class handle comparison (==, !=, null)
- [x] new() allocation

### Queue/Array Support
- [x] Queue type and operations
- [x] push_back, push_front, pop_back, pop_front
- [x] delete(), delete(index)
- [x] size(), max(), min(), unique()
- [x] sort()
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

### File I/O (Partial)
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

## Priority Queue

### CRITICAL (Blocking UVM)
1. **Fix IntType Crash** - Track A
2. **$fopen** - Track B
3. **$fwrite** - Track C
4. **String assoc array next()** - Track D

### HIGH (Blocking AVIPs)
5. **$fclose** - File descriptor cleanup
6. **$fdisplay** - Formatted output to file
7. **Complete MooreToCore lowering** - All ops must lower

### MEDIUM
8. **DPI-C imports** - For full UVM compatibility
9. **Coverage groups** - covergroup, coverpoint
10. **Constraints** - Full constraint solver

### LOW
11. **SVA assertions** - Property/sequence support
12. **Bind statements** - Module binding
13. **Program blocks** - program support

## AVIP Testing

Test files in ~/mbit/*:
- ahb_avip, apb_avip, axi4_avip, axi4Lite_avip
- i2s_avip, i3c_avip, jtag_avip, spi_avip, uart_avip

**Current blocker**: All AVIPs import UVM, which crashes.
**After crash fix**: Test individual components without UVM macros.

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

## Recent Commits
- `10db8cdd0` - [Docs] Update project plan
- `9be6c6c4c` - [ImportVerilog] Handle string types in emitDefault
- `4b9b3441d` - [MooreToCore] Add lowering for AssocArrayExistsOp
