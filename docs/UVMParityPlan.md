# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-13)

### 🎉 MILESTONE: UVM Core Library Parses Successfully!

**Overall Progress:** UVM core library parses without crashes. Mini-UVM testbenches work.
Full UVM testbenches have a silent conversion failure being investigated.

### Session Progress (12 commits)
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

### Current Blockers
1. **Static member redefinition in parameterized classes** - When `this_type` typedef
   creates specializations, static members get registered multiple times
2. **Silent conversion failure** - Full UVM conversion fails without error message
3. **Task capture of module variables** - Architectural issue with func.func/moore.module

## Track 1: ImportVerilog (Parsing & AST Conversion)

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
- [x] Format strings (%s, %d, %h, %b, %p, etc.)
- [x] Event triggers (`->event`)
- [x] Enum `.name()` method
- [x] `$typename` system call
- [x] `$urandom`, `$urandom_range`
- [x] Constraint block parsing (rand, randc, constraint)

### In Progress
- [ ] Fix static member redefinition bug
- [ ] Add diagnostic output for conversion failures
- [ ] Covergroups and coverage

### TODO - High Priority (UVM Blockers)
- [x] `wait fork` statement support (blocks UVM objection mechanism)
- [x] `%m` format specifier (hierarchical module path)
- [ ] Class object to string conversion in $swrite/$sformat
- [x] `$cast` dynamic casting
- [ ] Full constraint solving (requires external solver)

### TODO - Medium Priority
- [ ] Assertion functions (`$rose`, `$fell`, `$stable`, `$past`)
- [ ] Clocking blocks
- [ ] Program blocks
- [ ] `fork`/`join_any`/`join_none` variations
- [ ] Mailbox/Semaphore operations

**Next Agent Task:** Fix static member redefinition for parameterized class specializations

## Track 2: MooreToCore (Lowering to LLVM)

### Completed ✅
- [x] EventTriggeredOp, WaitConditionOp, EventTriggerOp
- [x] QueueConcatOp, QueuePushBackOp, QueuePushFrontOp
- [x] QueuePopBackOp, QueuePopFrontOp
- [x] QueueUniqueOp, QueueMinOp, QueueMaxOp
- [x] UrandomOp, UrandomRangeOp → runtime calls
- [x] String operations lowering
- [x] Class allocation and virtual dispatch (basic)

### TODO - High Priority
- [ ] QueueSortOp
- [ ] AssocArrayExistsOp
- [ ] Full vTable generation for polymorphism

### TODO - Medium Priority
- [ ] Four-valued logic (X/Z) support
- [ ] Process/thread management

**Next Agent Task:** Add QueueSortOp lowering

## Track 3: Moore Runtime Library

### Completed ✅
- [x] Event operations (`__moore_event_*`)
- [x] Queue operations (`__moore_queue_push/pop/delete/unique/min/max`)
- [x] Associative array operations (`__moore_assoc_*`)
- [x] String operations (`__moore_string_*`)
- [x] Random number generation (`__moore_urandom`, `__moore_urandom_range`)
- [x] Unit tests for runtime functions

### TODO
- [ ] `__moore_queue_sort`
- [ ] Constraint solver integration
- [ ] Process management functions

**Next Agent Task:** Add `__moore_queue_sort` implementation

## Track 4: Testing & Integration

### Test Coverage
- [x] Basic class tests
- [x] Event operation tests
- [x] Queue operation tests (including push/pop)
- [x] String operation tests
- [x] Builtin tests ($typename, $urandom, enum .name())
- [x] Runtime unit tests (MooreRuntimeTest.cpp)
- [x] Random ops lowering tests

### AVIP Testing Results (~/mbit/*)

#### Summary Table
| Category | Status | Notes |
|----------|--------|-------|
| Global packages | 8/8 pass | All AVIP globals compile cleanly |
| Interface files | 7/7 pass | Work with dependencies |
| Assertion files | Pass | Non-UVM assertions work |
| HVL/Testbench | Pass* | Compiles when deps provided; blocked by 3 UVM issues |

*HVL files parse and convert successfully. Remaining errors are in UVM core.

#### Detailed HVL Testing (2026-01-13)

**Test Command:**
```bash
./build/bin/circt-verilog --include-dir=/home/thomas-ahle/uvm-core/src \
  --include-dir=~/mbit/axi4_avip/src/... \
  uvm_pkg.sv globals.sv bfm_files.sv hvl_pkg.sv
```

**AVIPs Tested:**
- AXI4: master_pkg, slave_pkg, env_pkg compile
- APB: master_pkg, slave_pkg, env_pkg compile
- JTAG: controller_pkg, target_pkg compile
- SPI: master_pkg compiles (has user code bug: trailing comma)
- UART: tx_pkg compiles
- I2S: transmitter_pkg compiles
- I3C: globals compile

**Blockers (all from UVM core, not AVIP code):**

1. **`wait fork` statement** (P0 - blocks UVM objection)
   ```
   error: unsupported statement: WaitFork
   wait fork;  // uvm_objection.svh:879
   ```

2. **`%m` format specifier** (P1 - blocks instance scope)
   ```
   error: unsupported format specifier `%m`
   $swrite(uvm_instance_scope, "%m");  // uvm_misc.svh:124
   ```

3. **Class-to-string cast for $swrite** (P1 - blocks debug printing)
   ```
   error: expression of type '!moore.class<...>' cannot be cast to a simple bit vector
   $swrite(v, pool[key]);  // uvm_pool.svh:249
   ```

**Warnings (non-blocking):**
- `uvm_test_done` deprecated UVM 1.1 API (AVIP code issue, not CIRCT)
- Timescale mismatch warnings (user code configuration)
- Reversed range warnings in coverage bins (user code)

### UVM Testbench Testing
| Test Type | Status | Notes |
|-----------|--------|-------|
| UVM package alone | ✅ Pass | Parses without errors |
| Mini-UVM pattern | ✅ Pass | Basic UVM-like classes work |
| Full UVM testbench | ❌ Fail | Silent conversion failure |

**Next Agent Task:** Debug and fix silent UVM conversion failure

## Xcelium Feature Comparison

| Feature | Xcelium | CIRCT | Gap | Priority |
|---------|---------|-------|-----|----------|
| Basic SV | ✅ | ✅ | - | - |
| Classes | ✅ | ✅ | - | - |
| Queues | ✅ | ✅ | - | - |
| Events | ✅ | ✅ | - | - |
| $urandom | ✅ | ✅ | - | - |
| $typename | ✅ | ✅ | - | - |
| UVM Parsing | ✅ | ✅ | - | - |
| rand/constraint parse | ✅ | ✅ | - | - |
| wait fork | ✅ | ✅ | - | - |
| %m format | ✅ | ✅ | - | - |
| Class $swrite | ✅ | ❌ | High | P1 |
| Constraint solving | ✅ | ❌ | High | P1 |
| Coverage | ✅ | ❌ | High | P1 |
| $cast | ✅ | ✅ | - | - |
| Assertions | ✅ | Partial | Medium | P2 |
| fork/join | ✅ | Partial | Medium | P2 |

## Next Steps

### Immediate - UVM Core Blockers (Highest Priority)
1. **`wait fork` statement** - Required for UVM objection drain mechanism
2. **`%m` format specifier** - Required for UVM instance scope tracking
3. **Class $swrite support** - Required for UVM pool debug output

### Short Term
- Fix static member redefinition for this_type pattern
- Add `$cast` dynamic casting
- Complete constraint parsing (expressions)

### Medium Term
- Integrate external constraint solver
- Coverage collection
- Full assertion support

## Commands

```bash
# Build circt-verilog
ninja -C build circt-verilog

# Test UVM parsing (use --include-dir= instead of -I)
./build/bin/circt-verilog --include-dir=/home/thomas-ahle/uvm-core/src \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv

# Test AVIP globals (no UVM needed)
./build/bin/circt-verilog ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv

# Test AVIP HVL with UVM (example for AXI4)
./build/bin/circt-verilog \
  --include-dir=/home/thomas-ahle/uvm-core/src \
  --include-dir=~/mbit/axi4_avip/src/globals \
  --include-dir=~/mbit/axi4_avip/src/hvl_top/master \
  --include-dir=~/mbit/axi4_avip/src/hdl_top/master_agent_bfm \
  /home/thomas-ahle/uvm-core/src/uvm_pkg.sv \
  ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv \
  ~/mbit/axi4_avip/src/hdl_top/master_agent_bfm/axi4_master_driver_bfm.sv \
  ~/mbit/axi4_avip/src/hvl_top/master/axi4_master_pkg.sv

# Run runtime tests
ninja -C build check-circt-unit
```
