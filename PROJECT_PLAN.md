# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

## Current Status: UVM PARSING - 2 ERRORS REMAINING (January 15, 2026)

**Test Command**:
```bash
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
```

**Current Blockers / Limitations**:
1. **Class upcast with parameterized base** - `uvm_reg_mem_hdl_paths_seq extends uvm_reg_sequence #(...)` fails upcast verification because parameterized base class not recognized as base.
2. **Runtime gaps** - Randomization/coverage not implemented; DPI/VPI still stubs; MooreToCore queue globals lowering pending.

**Recent Fixes (This Session)**:
- **Global variable redefinition** ✅ FIXED (a152e9d35) - Fixed duplicate GlobalVariableOp when class type references the variable in methods (uvm_default_line_printer pattern). Uses placeholder type during recursive conversion, then updates to correct type.
- **UVM class declaration** ✅ FIXED (555a78350) - ClassDeclOp SymbolTable block requirement, WaitConditionOp 2-state type, func.return for pure virtual methods
- **String ato* methods** ✅ FIXED (14dfdbe9f + 34ab7a758) - Added atoi, atohex, atooct, atobin support for UVM command-line parsing
- **Non-integral assoc array keys** ✅ FIXED (f6b79c4c7) - String and class handle keys for UVM pools
- **Pure virtual method stubbing** ✅ FIXED (f6b79c4c7) - Default returns for TLM methods

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

### Track A: Import Crash Triage
**Status**: 🔄 IN PROGRESS  
**Task**: Identify and guard the `cast<IntType>` assertion in call lowering (likely `$right/$high`/queue size or format_string path). Add defensive type checks and repro harness.  
**Files**: lib/Conversion/ImportVerilog/Expressions.cpp, Structure.cpp

### Track B: String/Format + Queue Compatibility
**Status**: 🔄 IN PROGRESS  
**Task**: Normalize `format_string` → `string` in queue push_back/concat and string_concat operands; ensure queue element coercions verified.  
**Files**: lib/Conversion/ImportVerilog/Expressions.cpp

### Track C: Abstract Method Return Stubs
**Status**: 🔄 IN PROGRESS  
**Task**: Sweep pure virtual functions (factory/pool/callback classes) to ensure stub returns match signature and verifier passes; adjust return insertions.  
**Files**: lib/Conversion/ImportVerilog/Structure.cpp, Statements.cpp

### Track D: AVIP Regression Testing
**Status**: 🔄 IN PROGRESS  
**Task**: Keep running `~/mbit/*` interface/global packages after each crash fix; capture new failures.  
**Files**: test infra/scripts

---

## Priority Queue

### CRITICAL (Blocking UVM Parsing)
1. **Class upcast with parameterized base** - `uvm_reg_sequence #(...)` not recognized as base of `uvm_reg_mem_hdl_paths_seq`. Need to handle parameterized class inheritance in upcast verification.

### RECENTLY FIXED ✅
- ~~**Global variable redefinition**~~ - ✅ Fixed (a152e9d35) - Recursive type conversion duplicate prevention
- ~~**UVM class declaration issues**~~ - ✅ Fixed (555a78350) - SymbolTable block, 2-state types, pure virtual returns
- ~~**String ato* methods**~~ - ✅ Fixed (14dfdbe9f + 34ab7a758)
- ~~**Non-integral assoc array keys**~~ - ✅ Fixed (f6b79c4c7)
- ~~**Pure virtual method stubbing**~~ - ✅ Fixed (f6b79c4c7)

### PREVIOUSLY FIXED ✅
- ~~**$fopen**~~ - ✅ Fixed (ce8d1016a)
- ~~**$fwrite**~~ - ✅ Fixed (ccfc4f6ca)
- ~~**$fclose**~~ - ✅ Fixed (b4a18d045)
- ~~**String assoc array next()**~~ - ✅ Fixed (2fa392a98)
- ~~**String case IntType crash**~~ - ✅ Fixed (3410de2dc)
- ~~**%20s format**~~ - ✅ Fixed (88085cbd7)

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
- `14dfdbe9f` - [ImportVerilog] Add support for string ato* methods (atoi, atohex, atooct, atobin)
- `2657ceab7` - [ImportVerilog] Add support for $sscanf system function
- `ab38bd7f5` - [Docs] Update blockers - IntType and string replication fixed
- `d16609422` - [ImportVerilog] Add StringReplicateOp for string replication
- `52511fe46` - [MooreToCore] Add lowering for file I/O operations
- `942537c2a` - [ImportVerilog] Return meaningful stub values for DPI-C imports
- `76612d5bd` - [ImportVerilog] Fix IntType assertion crash in ReplicateOp
- `88085cbd7` - [ImportVerilog] Add string format specifier with width support
- `b4a18d045` - [ImportVerilog] Add $fclose system task support
- `2fa392a98` - [MooreToCore] Fix string-keyed associative array iteration

---

## Architecture Reference

See full plan: `~/.claude/plans/jiggly-tickling-engelbart.md`

Track assignments:
- **Track A (Sim)**: Event kernel, process control, performance
- **Track B (UVM)**: Class parsing, constraints, factory pattern
- **Track C (Types)**: 4-state, coverage, file I/O
- **Track D (DevEx)**: LSP, linting, dashboards
- **Track E (Assert)**: SVA, vacuity detection, debug
