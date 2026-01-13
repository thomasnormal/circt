# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-13)

### 🎉 MILESTONE: UVM Core Library Parses Successfully!

**Overall Progress:** UVM core library parses without crashes. Mini-UVM testbenches work.
Full UVM testbenches have a silent conversion failure being investigated.

### Session Progress (10 commits)
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

### TODO - High Priority
- [ ] `$cast` dynamic casting
- [ ] Full constraint solving (requires external solver)
- [ ] Assertion functions (`$rose`, `$fell`, `$stable`, `$past`)

### TODO - Medium Priority
- [ ] Clocking blocks
- [ ] Program blocks
- [ ] `fork`/`join` parallel blocks
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
| Category | Status | Notes |
|----------|--------|-------|
| Global packages | ✅ 8/8 pass | All AVIP globals compile |
| Interface files | ✅ 6/7 pass | Work with dependencies |
| Assertion files | ✅ Pass | Non-UVM assertions work |
| HVL/Testbench | ⚠️ Partial | Need capture fix |

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
| Constraint solving | ✅ | ❌ | High | P0 |
| Coverage | ✅ | ❌ | High | P1 |
| $cast | ✅ | ❌ | Medium | P1 |
| Assertions | ✅ | Partial | Medium | P2 |
| fork/join | ✅ | Partial | Medium | P2 |

## Next Steps

### Immediate (4 Parallel Agents)
1. **Track 1:** Fix static member redefinition for this_type pattern
2. **Track 2:** Add QueueSortOp lowering
3. **Track 3:** Add `__moore_queue_sort` runtime
4. **Track 4:** Debug silent UVM conversion failure

### Short Term
- Fix all blockers for UVM testbench compilation
- Complete constraint parsing (expressions)
- Add `$cast` dynamic casting

### Medium Term
- Integrate external constraint solver
- Coverage collection
- Full assertion support

## Commands

```bash
# Test UVM parsing
./build/bin/circt-verilog -I/home/thomas-ahle/uvm-core/src /home/thomas-ahle/uvm-core/src/uvm_pkg.sv

# Test AVIP
./build/bin/circt-verilog -I~/mbit/axi4_avip/src ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv

# Build
ninja -C build circt-verilog

# Run runtime tests
ninja -C build check-circt-unit
```
