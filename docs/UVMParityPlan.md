# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-13)

### 🎉 MILESTONE: UVM Core Library Parses Successfully!

**Overall Progress:** UVM core library parses completely without errors!
Array locator methods with field-based predicates now work. Main focus: MooreToCore lowering and randomize() support.

### Session Progress (26+ commits)
- ✅ Fixed `cast<TypedValue<IntType>>` crash in class hierarchy
- ✅ Added `EventTriggerOp` for `->event` syntax
- ✅ Added `QueueConcatOp` for queue concatenation
- ✅ Fixed default argument `this` reference resolution
- ✅ Fixed dangling reference in recursive class declaration
- ✅ Implemented enum `.name()` method
- ✅ Implemented `$typename` system call
- ✅ Added QueuePushBack/Front, QueuePopBack/Front
- ✅ Added `$urandom`, `$urandom_range` with runtime
- ✅ Added constraint block parsing (rand, constraint)
- ✅ Added runtime unit tests
- ✅ Added `wait fork` statement support
- ✅ Added `%m` format specifier (hierarchical module path)
- ✅ Added `$cast` dynamic casting with RTTI
- ✅ Added `FormatClassOp` for class handle formatting
- ✅ Fixed static member redefinition for parameterized classes
- ✅ Added semaphore/mailbox `new()` construction support
- ✅ Added `disable fork` statement support
- ✅ Fixed `$swrite` with class handles (no format specifier)
- ✅ Added array locator methods (find, find_index, find_first, etc.)
- ✅ Added ArrayLocatorOp lowering (all comparison operators)
- ✅ Added runtime array locator functions
- ✅ Added covergroup skip with remark
- ✅ Fixed DPI-C crash (emits remark instead)
- ✅ Fixed enum .name() lowering with FormatDynStringOp
- ✅ **Added field-based array predicates (`item.field == val`)**
- ✅ **Fixed lit test regressions (type formats, struct sizes)**

### Current Limitations (Xcelium Parity Gaps)

1. **randomize() method** - Not yet implemented
   - Moore dialect has RandomizeOp defined
   - ImportVerilog handler needed
   - MooreToCore lowering needed

2. **Constraint solving** - Constraints parsed but not solved
   - Need: External solver integration (Z3/SMT)

3. **Silent UVM conversion failure** - Full UVM returns exit 1 with no error
   - Lint-only mode works
   - UVM-style code without library works
   - Full library has undiagnosed issue

4. **Coverage/Covergroups** - Parsing skipped with remark
   - Moore dialect has coverage ops
   - No actual coverage collection

5. **~146 Moore ops missing lowering** - Based on audit
   - P0: Interface ops (3 ops)
   - P1: Class ops, some arithmetic
   - P2: Advanced math functions

## Track 1: ImportVerilog (Parsing & AST Conversion)

**Status: ✅ FEATURE COMPLETE for UVM parsing**

### Completed ✅
- [x] Basic class hierarchy with inheritance
- [x] Parameterized classes (generic classes)
- [x] Static class properties
- [x] Virtual methods and method overriding
- [x] Event type support
- [x] Queue operations (push, pop, delete, unique, min, max, concat)
- [x] Associative array operations (first, next, last, prev, delete)
- [x] String operations (len, toupper, tolower, getc, putc, substr)
- [x] Dynamic array support
- [x] Format strings (%s, %d, %h, %b, %p, %m, etc.)
- [x] Event triggers (`->event`)
- [x] Enum `.name()` method
- [x] `$typename` system call
- [x] `$urandom`, `$urandom_range`, `$random`
- [x] Constraint block parsing (rand, randc, constraint)
- [x] `disable fork`, `wait fork` statement support
- [x] Semaphore/mailbox `new()` construction
- [x] Array locator methods with predicates
- [x] `$cast` dynamic casting
- [x] Covergroup skip/remark (graceful degradation)
- [x] DPI-C skip/remark (no crash)

### TODO - High Priority
- [ ] **Implement randomize() handler** - Detect and convert to moore.randomize

### TODO - Medium Priority
- [ ] Full covergroup conversion (not just skip)
- [ ] Clocking blocks
- [ ] Program blocks

**Next Agent Task:** Implement randomize() handler in visitCall()

## Track 2: MooreToCore (Lowering to LLVM)

**Status: ⚠️ IN PROGRESS - Main bottleneck for end-to-end execution**

### Completed ✅
- [x] EventTriggeredOp, WaitConditionOp, EventTriggerOp
- [x] QueueConcatOp, QueuePushBackOp, QueuePushFrontOp
- [x] QueuePopBackOp, QueuePopFrontOp
- [x] QueueUniqueOp, QueueMinOp, QueueMaxOp
- [x] UrandomOp, UrandomRangeOp → runtime calls
- [x] String operations lowering
- [x] Class allocation and virtual dispatch (basic)
- [x] ArrayLocatorOp → runtime (all operators + field access)
- [x] FormatClassOp, FormatStringOp → sim dialect
- [x] WaitForkOp, DisableForkOp
- [x] DynCastCheckOp → runtime RTTI check

### TODO - High Priority (Blocks End-to-End)
- [ ] **RandomizeOp** - Basic lowering to runtime call
- [ ] InterfaceSignalDeclOp, InterfaceInstanceOp, ModportDeclOp
- [ ] Full class virtual dispatch (complete vTable)
- [ ] Debug silent UVM conversion failure

### TODO - Medium Priority
- [ ] Four-valued logic (X/Z) support
- [ ] Process/thread management for fork/join
- [ ] AssocArrayExistsOp

**Next Agent Task:** Add RandomizeOp lowering pattern

## Track 3: Moore Runtime Library

**Status: ✅ Comprehensive - Ready for randomize()**

### Completed ✅
- [x] Event operations (`__moore_event_*`)
- [x] Queue operations (push/pop/delete/unique/min/max)
- [x] Associative array operations (`__moore_assoc_*`)
- [x] String operations (`__moore_string_*`)
- [x] Random number generation (`__moore_urandom`, `__moore_urandom_range`)
- [x] Array locator: `__moore_array_find_eq`, `__moore_array_find_cmp`
- [x] **Array locator: `__moore_array_find_field_cmp` (field-based)**
- [x] Array min/max/unique functions
- [x] Dynamic cast check (`__moore_dyn_cast_check`)
- [x] Comprehensive unit tests

### TODO
- [ ] **`__moore_randomize_basic`** - Basic field randomization
- [ ] `__moore_queue_sort` with comparator
- [ ] Constraint solver integration (future)
- [ ] Process management functions

**Next Agent Task:** Implement __moore_randomize_basic runtime function

## Track 4: Testing & Integration

**Status: ✅ Good - All AVIPs pass, lit tests fixed**

### Test Coverage
- [x] Basic class tests
- [x] Event operation tests
- [x] Queue operation tests
- [x] String operation tests
- [x] Builtin tests ($typename, $urandom, enum .name())
- [x] Runtime unit tests (MooreRuntimeTest.cpp)
- [x] Array locator tests (parsing + lowering + field access)
- [x] **Lit test regressions fixed**

### AVIP Testing Results (~/mbit/*)

#### Summary Table (Updated 2026-01-13)
| AVIP | Globals | Interface | With UVM | Status |
|------|---------|-----------|----------|--------|
| AXI4 | ✅ Pass | ✅ Pass | ✅ Pass | Complete |
| APB | ✅ Pass | ✅ Pass | ✅ Pass | Complete |
| AHB | ✅ Pass | N/A | ✅ Pass | Complete |
| SPI | ✅ Pass | N/A | ✅ Pass | Complete |
| I2S | ✅ Pass | N/A | ✅ Pass | Complete |
| I3C | ✅ Pass | ✅ Pass | ✅ Pass | Complete |
| JTAG | ✅ Pass | N/A | ✅ Pass | Complete |
| UART | ✅ Pass | N/A | ✅ Pass | Complete |
| AXI4-Lite | ✅ Pass | N/A | ✅ Pass | Complete |

**All 9 AVIP packages pass!**

### UVM Testbench Testing
| Test Type | Status | Notes |
|-----------|--------|-------|
| UVM package alone | ✅ Pass | Parses completely |
| UVM-style code | ✅ Pass | Generates Moore + HW IR |
| Full UVM testbench | ⚠️ Silent fail | Exit 1, no error message |

**Next Agent Task:** Debug silent UVM conversion failure

## Xcelium Feature Comparison

| Feature | Xcelium | CIRCT Parse | CIRCT Lower | Gap | Priority |
|---------|---------|-------------|-------------|-----|----------|
| Basic SV | ✅ | ✅ | ✅ | - | - |
| Classes | ✅ | ✅ | ⚠️ Partial | Low | P3 |
| Queues | ✅ | ✅ | ✅ | - | - |
| Events | ✅ | ✅ | ✅ | - | - |
| $urandom | ✅ | ✅ | ✅ | - | - |
| $typename | ✅ | ✅ | ✅ | - | - |
| UVM Parsing | ✅ | ✅ | N/A | - | - |
| wait fork | ✅ | ✅ | ✅ | - | - |
| %m format | ✅ | ✅ | ✅ | - | - |
| Class $swrite | ✅ | ✅ | ✅ | - | - |
| $cast | ✅ | ✅ | ✅ | - | - |
| Array locators | ✅ | ✅ | ✅ | - | - |
| **randomize()** | ✅ | ❌ | ❌ | **High** | **P0** |
| Constraint solving | ✅ | ✅ Parse | ❌ | High | P1 |
| Coverage | ✅ | ⚠️ Skip | ❌ | High | P1 |
| DPI-C | ✅ | ⚠️ Skip | ❌ | Medium | P2 |
| Assertions | ✅ | Partial | ❌ | Medium | P2 |
| fork/join | ✅ | ✅ | Partial | Medium | P2 |
| Interfaces | ✅ | ✅ | ❌ | High | P1 |

## Next Steps by Priority

### P0 - Critical (Blocks UVM Execution)
1. **Implement randomize()** - Most critical missing feature
   - ImportVerilog handler in Expressions.cpp
   - MooreToCore lowering pattern
   - Runtime `__moore_randomize_basic`

2. **Debug silent UVM failure** - Find root cause of exit 1 with no error

### P1 - High (Blocks Full Simulation)
1. **Interface lowering** - InterfaceSignalDeclOp, InterfaceInstanceOp
2. **More class ops lowering** - Complete vTable support

### P2 - Medium (Quality of Life)
1. **Constraint solving** - Z3/SMT integration (future)
2. **DPI-C support** - Link to external C functions
3. **Better diagnostics** - Error propagation for lowering failures

## Commands

```bash
# Build circt-verilog
ninja -C build circt-verilog

# Test UVM parsing
./build/bin/circt-verilog --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv

# Test AVIP with UVM
./build/bin/circt-verilog \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  --include-dir=~/mbit/axi4_avip/src/globals \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv \
  ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv

# Test with Moore IR output
./build/bin/circt-verilog --ir-moore \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv

# Compare with Xcelium
xrun -compile -uvm \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv
```

## Agent Task Assignments

| Track | Agent Task | Priority | Notes |
|-------|-----------|----------|-------|
| Track 1 | Implement randomize() handler | P0 | In Expressions.cpp visitCall() |
| Track 2 | Add RandomizeOp lowering | P0 | Basic runtime call |
| Track 3 | Implement __moore_randomize_basic | P0 | Iterate rand fields |
| Track 4 | Debug silent UVM failure | P0 | Find root cause |
