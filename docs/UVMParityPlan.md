# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-13)

**Overall Progress:** UVM core library now compiles without crashes. Hitting feature gaps.

### Recent Fixes (This Session)
- ✅ Fixed `cast<TypedValue<IntType>>` crash in class hierarchy processing
- ✅ Added `EventTriggerOp` for `->event` syntax
- ✅ Added `QueueConcatOp` for queue concatenation `{q1, q2}`
- ✅ Fixed default argument `this` reference resolution
- ✅ Fixed dangling reference in recursive class declaration

### Current Blocker
```
../uvm-core/src/base/uvm_phase.svh:896:0: error: unsupported system call `name`
```

## Track 1: ImportVerilog (Parsing & AST Conversion)

### Completed
- [x] Basic class hierarchy with inheritance
- [x] Parameterized classes (generic classes)
- [x] Static class properties
- [x] Virtual methods and method overriding
- [x] Event type support
- [x] Queue operations (push, pop, delete, unique, min, max)
- [x] Associative array operations (first, next, last, prev, delete)
- [x] String operations (len, toupper, tolower, getc, putc, substr)
- [x] Dynamic array support
- [x] Format strings (%s, %d, %h, %b, %p, etc.)
- [x] Event triggers (`->event`)
- [x] Queue concatenation

### In Progress
- [ ] System calls: `$name`, `$typename`, `$bits`
- [ ] Constraint blocks (randomize, rand, randc)
- [ ] Covergroups and coverage
- [ ] Clocking blocks
- [ ] Program blocks

### TODO - High Priority
- [ ] `$name` system call (returns process/thread name)
- [ ] `$typename` system call (returns type name as string)
- [ ] `$cast` dynamic casting
- [ ] `$urandom`, `$urandom_range` random generation
- [ ] Assertion system functions (`$rose`, `$fell`, `$stable`, `$past`)

### TODO - Medium Priority
- [ ] Sequence/property declarations
- [ ] `fork`/`join` parallel blocks
- [ ] `disable fork` statement
- [ ] `wait fork` statement
- [ ] Mailbox operations
- [ ] Semaphore operations

## Track 2: MooreToCore (Lowering to LLVM)

### Completed
- [x] EventTriggeredOp → runtime call
- [x] WaitConditionOp → runtime call
- [x] EventTriggerOp → runtime call
- [x] QueueConcatOp → runtime call
- [x] QueueUniqueOp, QueueMinOp, QueueMaxOp
- [x] String operations lowering
- [x] Class allocation (malloc-based)
- [x] Virtual method dispatch (basic)

### TODO - High Priority
- [ ] QueuePushBackOp, QueuePushFrontOp
- [ ] QueuePopBackOp, QueuePopFrontOp
- [ ] QueueSortOp
- [ ] AssocArrayExistsOp
- [ ] Full vTable generation for polymorphism

### TODO - Medium Priority
- [ ] Four-valued logic (X/Z) support
- [ ] Out-of-bounds array access handling
- [ ] Process/thread management

## Track 3: Moore Runtime Library

### Completed
- [x] `__moore_event_triggered`
- [x] `__moore_wait_condition`
- [x] `__moore_event_trigger`
- [x] `__moore_queue_*` operations
- [x] `__moore_assoc_*` operations
- [x] `__moore_string_*` operations
- [x] `__moore_dyn_array_new`

### TODO
- [ ] `__moore_queue_push_back`, `__moore_queue_push_front`
- [ ] `__moore_queue_pop_back`, `__moore_queue_pop_front`
- [ ] `__moore_queue_sort`
- [ ] `__moore_process_name` (for `$name`)
- [ ] `__moore_typename` (for `$typename`)
- [ ] Random number generation runtime

## Track 4: Testing & Integration

### Test Coverage
- [x] Basic class tests
- [x] Event operation tests
- [x] Queue operation tests
- [x] String operation tests

### AVIP Testing Results (~/mbit/*)
- Basic package files compile successfully
- Interface files need dependency resolution
- HVL (testbench) files blocked on UVM dependency

### UVM Core Testing
- UVM package parses through most of base classes
- Current blocker: `$name` system call
- Next blockers expected: constraints, coverage

## Xcelium Feature Comparison

| Feature | Xcelium | CIRCT | Gap |
|---------|---------|-------|-----|
| Basic SV | ✅ | ✅ | - |
| Classes | ✅ | ✅ | - |
| Constraints | ✅ | ❌ | High |
| Coverage | ✅ | ❌ | High |
| Assertions | ✅ | Partial | Medium |
| UVM Library | ✅ | Partial | Medium |
| $name, $typename | ✅ | ❌ | High |
| fork/join | ✅ | Partial | Medium |
| Mailbox/Semaphore | ✅ | ❌ | Medium |

## Next Steps

### Immediate (Next 4 Agents)
1. **Agent 1:** Implement `$name` system call
2. **Agent 2:** Implement `$typename` system call
3. **Agent 3:** Add QueuePush/Pop operations
4. **Agent 4:** Test on AVIP files and fix dependency issues

### Short Term
- Complete system call support for UVM
- Implement basic constraint support
- Add fork/join parallel execution

### Medium Term
- Full randomization support
- Coverage collection
- Assertion checking

## Commands

Test UVM:
```bash
./build/bin/circt-verilog -I/home/thomas-ahle/uvm-core/src /home/thomas-ahle/uvm-core/src/uvm_pkg.sv 2>&1 | head -50
```

Test AVIP:
```bash
./build/bin/circt-verilog ~/mbit/[avip]/src/[file].sv 2>&1
```

Build:
```bash
ninja -C build circt-verilog
```
