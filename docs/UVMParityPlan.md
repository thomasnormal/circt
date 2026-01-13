# UVM Parity Plan: CIRCT vs Cadence Xcelium

This document tracks the progress toward bringing CIRCT's SystemVerilog support
to parity with commercial simulators like Cadence Xcelium for running UVM testbenches.

## Current Status (2026-01-13)

### 🎉 MILESTONE: UVM Core Library Parses Successfully!

**Overall Progress:** UVM core library now parses completely without errors or crashes.
The only output is warnings and a remark about class builtins (expected).

```bash
$ ./build/bin/circt-verilog -I/home/thomas-ahle/uvm-core/src /home/thomas-ahle/uvm-core/src/uvm_pkg.sv
<unknown>:0: warning: no top-level modules found in design [-Wmissing-top]
../uvm-core/src/base/uvm_misc.svh:56:15: remark: Class builtin functions...
```

The "no top-level modules" warning is expected - UVM is a package, not a synthesizable design.

### Recent Fixes (This Session - 8 commits)
- ✅ Fixed `cast<TypedValue<IntType>>` crash in class hierarchy processing
- ✅ Added `EventTriggerOp` for `->event` syntax
- ✅ Added `QueueConcatOp` for queue concatenation `{q1, q2}`
- ✅ Fixed default argument `this` reference resolution
- ✅ Fixed dangling reference in recursive class declaration
- ✅ Implemented enum `.name()` method
- ✅ Implemented `$typename` system call
- ✅ Added QueuePushBack/Front, QueuePopBack/Front with lowering & runtime

### Next Phase: Run UVM Testbenches
Now that UVM parses, we need to test actual UVM testbenches that:
1. Import uvm_pkg
2. Define test classes extending uvm_test
3. Use UVM macros and utilities

## Track 1: ImportVerilog (Parsing & AST Conversion)

### Completed ✅
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
- [x] Enum `.name()` method
- [x] `$typename` system call

### In Progress
- [ ] Constraint blocks (randomize, rand, randc) - Critical for UVM
- [ ] Covergroups and coverage
- [ ] `$urandom`, `$urandom_range` random generation

### TODO - High Priority
- [ ] `$cast` dynamic casting
- [ ] Assertion system functions (`$rose`, `$fell`, `$stable`, `$past`)
- [ ] Clocking blocks
- [ ] Program blocks

### TODO - Medium Priority
- [ ] Sequence/property declarations
- [ ] `fork`/`join` parallel blocks (partial)
- [ ] `disable fork` statement
- [ ] `wait fork` statement
- [ ] Mailbox operations
- [ ] Semaphore operations

## Track 2: MooreToCore (Lowering to LLVM)

### Completed ✅
- [x] EventTriggeredOp → runtime call
- [x] WaitConditionOp → runtime call
- [x] EventTriggerOp → runtime call
- [x] QueueConcatOp → runtime call
- [x] QueueUniqueOp, QueueMinOp, QueueMaxOp → runtime calls
- [x] QueuePushBackOp, QueuePushFrontOp → runtime calls
- [x] QueuePopBackOp, QueuePopFrontOp → runtime calls
- [x] String operations lowering
- [x] Class allocation (malloc-based)
- [x] Virtual method dispatch (basic)

### TODO - High Priority
- [ ] QueueSortOp
- [ ] AssocArrayExistsOp
- [ ] Full vTable generation for polymorphism
- [ ] Constraint solving integration

### TODO - Medium Priority
- [ ] Four-valued logic (X/Z) support
- [ ] Out-of-bounds array access handling
- [ ] Process/thread management

## Track 3: Moore Runtime Library

### Completed ✅
- [x] `__moore_event_triggered`
- [x] `__moore_wait_condition`
- [x] `__moore_event_trigger`
- [x] `__moore_queue_push_back`, `__moore_queue_push_front`
- [x] `__moore_queue_pop_back`, `__moore_queue_pop_front`
- [x] `__moore_queue_delete`, `__moore_queue_unique`, `__moore_queue_min`, `__moore_queue_max`
- [x] `__moore_assoc_*` operations
- [x] `__moore_string_*` operations
- [x] `__moore_dyn_array_new`

### TODO
- [ ] `__moore_queue_sort`
- [ ] Random number generation runtime
- [ ] Constraint solver runtime

## Track 4: Testing & Integration

### Test Coverage
- [x] Basic class tests
- [x] Event operation tests
- [x] Queue operation tests (including push/pop)
- [x] String operation tests
- [x] Builtin tests ($typename, enum .name())

### AVIP Testing Results (~/mbit/*)
| Category | Status | Notes |
|----------|--------|-------|
| Global packages | ✅ 8/8 pass | All AVIP globals compile |
| Interface files | ✅ 6/7 pass | Work with deps |
| Assertion files | ✅ Pass | Non-UVM assertions work |
| HVL/Testbench | ❌ Blocked | Need UVM + capture fix |

**Architectural Issue Found:** Tasks containing `@(posedge clk)` cannot capture
module-level variables because `func.func` and `moore.module` are siblings, not
nested. This needs architectural work to fix properly.

### UVM Core Testing
- ✅ UVM package parses completely without errors
- Next: Test with actual UVM testbench

## Xcelium Feature Comparison

| Feature | Xcelium | CIRCT | Gap | Priority |
|---------|---------|-------|-----|----------|
| Basic SV | ✅ | ✅ | - | - |
| Classes | ✅ | ✅ | - | - |
| Queues | ✅ | ✅ | - | - |
| Events | ✅ | ✅ | - | - |
| UVM Parsing | ✅ | ✅ | - | - |
| $typename | ✅ | ✅ | - | - |
| Constraints | ✅ | ❌ | High | P0 |
| Coverage | ✅ | ❌ | High | P1 |
| $urandom | ✅ | ❌ | High | P0 |
| Assertions | ✅ | Partial | Medium | P2 |
| fork/join | ✅ | Partial | Medium | P2 |
| Mailbox/Semaphore | ✅ | ❌ | Medium | P2 |

## Next Steps

### Immediate (Next 4 Agents)
1. **Agent 1:** Implement `$urandom`, `$urandom_range` for randomization
2. **Agent 2:** Start basic constraint block support (parse `rand`, `constraint`)
3. **Agent 3:** Create simple UVM testbench and test end-to-end
4. **Agent 4:** Fix module-level variable capture for tasks

### Short Term
- Complete randomization support (`$urandom`, basic constraints)
- Test actual UVM testbenches end-to-end
- Fix task/module capture architecture

### Medium Term
- Full constraint solving with external solver
- Coverage collection
- Assertion checking

## Commands

Test UVM:
```bash
./build/bin/circt-verilog -I/home/thomas-ahle/uvm-core/src /home/thomas-ahle/uvm-core/src/uvm_pkg.sv 2>&1 | head -50
```

Test AVIP:
```bash
./build/bin/circt-verilog -I~/mbit/axi4_avip/src ~/mbit/axi4_avip/src/globals/axi4_globals_pkg.sv ~/mbit/axi4_avip/src/hdl_top/axi4_if.sv
```

Build:
```bash
ninja -C build circt-verilog
```
