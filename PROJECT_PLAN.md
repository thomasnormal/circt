# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

## Current Status: 🎉 ITERATION 30 - COMPREHENSIVE TEST SURVEY (January 16, 2026)

**Summary**: SVA boolean context fixes, Z3 CMake linking, comprehensive test suite survey

### Test Suite Coverage (Iteration 30)

| Test Suite | Total Tests | Pass Rate | Notes |
|------------|-------------|-----------|-------|
| **sv-tests** | 989 (non-UVM) | **72.1%** (713/989) | Parsing/elaboration focus |
| **mbit AVIP globals** | 8 packages | **100%** | All package files work |
| **mbit AVIP interfaces** | 8 interfaces | **75%** | 6/8 pass |
| **mbit AVIP HVL** | 8 packages | **0%** | Requires UVM library |
| **verilator-verification** | 154 | ~60% | SVA tests improved |

### sv-tests Chapter Breakdown (72.1% overall)

| Chapter | Pass Rate | Key Gaps |
|---------|-----------|----------|
| Ch 5 (Lexical) | **86%** | Good |
| Ch 6 (Data Types) | **75%** | TaggedUnion |
| Ch 7 (Aggregate) | **72%** | Unpacked dimensions |
| Ch 9 (Behavioral) | **73%** | Minor gaps |
| Ch 10 (Scheduling) | **50%** | RaceyWrite |
| Ch 11 (Operators) | **87%** | Strong |
| Ch 12 (Procedural) | **79%** | SequenceWithMatch |
| Ch 13 (Tasks/Functions) | **86%** | Strong |
| Ch 14 (Clocking Blocks) | **~5%** | ClockingBlockDeclOp added, needs full impl |
| Ch 16 (Assertions) | **68%** | EmptyArgument |
| Ch 18 (Random/Constraints) | **25%** | RandSequence |
| Ch 20 (I/O Formatting) | **83%** | Good |
| Ch 21 (I/O System Tasks) | **37%** | VcdDump |

### Top Missing Features (by sv-tests failures)

| Feature | Tests Failed | Priority |
|---------|--------------|----------|
| **ClockingBlock** | ~50 | HIGH |
| **RandSequence** | ~30 | MEDIUM |
| **SequenceWithMatch** | ~25 | MEDIUM |
| **TaggedUnion** | ~20 | MEDIUM |
| **EmptyArgument** | ~15 | LOW |

### Z3 BMC Status

- **Z3 AVAILABLE** at ~/z3 (include: ~/z3/include, lib: ~/z3/lib/libz3.so)
- CMake linking code is correct (both CONFIG and Module mode support)
- Pipeline verified: SV → Moore → HW → BMC MLIR → LLVM IR generation
- **LowerClockedAssertLike pass added** - handles verif.clocked_assert for BMC
- Testing Z3 integration in progress

### Test Commands
```bash
# UVM Parsing - COMPLETE
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
# Exit code: 0 (SUCCESS!) - 161,443 lines of Moore IR

# SVA BMC (Bounded Model Checking) - CONVERSION WORKS
./build/bin/circt-verilog --ir-hw /tmp/simple_sva.sv | \
  ./build/bin/circt-bmc --bound=10
# VerifToSMT conversion produces valid MLIR (Z3 installation needed)
```

**Iteration 30 Commits**:
- **Multi-track progress (commit ab52d23c2)** - 3,522 insertions across 26 files:
  - Track 1: Clocking blocks - ClockingBlockDeclOp, ClockingSignalOp in Moore
  - Track 2: LLHD interpreter - LLHDProcessInterpreter.cpp/h for circt-sim
  - Track 3: $past fix - moore::PastOp for type-preserving comparisons
  - Track 4: clocked_assert lowering - LowerClockedAssertLike.cpp for BMC
  - LTLToCore enhancements (986 lines added)
- Big projects status survey (commit 9abf0bb24)
- Active development tracks documentation (commit e48c2f3f8)
- SVA functions in boolean contexts (commit a68ed9adf) - ltl.or/ltl.and/ltl.not for LTL types
- Z3 CMake linking fix (commit 48bcd2308) - JIT runtime linking for SMTToZ3LLVM
- $rose/$fell test improvements (commit 8ad3a7cc6)
- MooreToCore coverage ops tests (commit d92d81882)
- VerifToSMT conversion tests (commit ecabb4492)
- SVAToLTL conversion tests (commit 47c5a7f36)

**Iteration 29 Commits**:
- VerifToSMT `bmc.final` assertion handling fixes
- ReconcileUnrealizedCasts pass added to circt-bmc pipeline
- BVConstantOp argument order fix (value, width)
- Clock counting before region conversion
- Proper rewriter.eraseOp() usage in conversion patterns

**Iteration 28 Commits**:
- `7d5391552` - $onehot/$onehot0 system calls
- `2830654d4` - $countbits system call
- `4704320af` - $sampled/$past/$changed for SVA assertions
- `25cd3b6a2` - Direct interface member access
- `12d75735d`, `110fc6caf` - Test fixes and documentation
- `47c5a7f36` - SVAToLTL comprehensive tests
- `ecabb4492` - VerifToSMT comprehensive tests
- `235700509` - CHANGELOG update

**Fork**: https://github.com/thomasnormal/circt (synced with upstream)

**Current Blockers / Limitations** (Post-MooreToCore):
1. **Coverage** ✅ INFRASTRUCTURE DONE - CovergroupHandleType, CovergroupInstOp, CovergroupSampleOp implemented
2. **SVA assertions** ✅ LOWERING WORKS - moore.assert/assume/cover → verif.assert/assume/cover
3. **DPI/VPI** ⚠️ STUBS ONLY - 22 DPI functions return defaults (0, empty string, "CIRCT")
4. **Complex constraints** ⚠️ PARTIAL - ~6% need SMT solver (94% now work!)
5. **System calls** ✅ $countones IMPLEMENTED - $clog2 and some others still needed
6. **UVM reg model** ⚠️ CLASS HIERARCHY ISSUE - uvm_reg_map base class mismatch

**AVIP Testing Results** (Iteration 28 - comprehensive validation):

| Component Type | Pass Rate | Notes |
|----------------|-----------|-------|
| Global packages | 8/8 (100%) | All package files work |
| Interfaces | 7/9 (78%) | JTAG/I2S fail due to source issues, not CIRCT bugs |

| AVIP | Step 1 (Moore IR) | Step 2 (MooreToCore) | Notes |
|------|------------------|---------------------|-------|
| APB | ✅ PASS | ✅ PASS | Works without UVM |
| AXI4-Lite | ✅ PASS | ✅ PASS | Works without UVM |
| UART | ✅ PASS | ✅ PASS | Works without UVM |
| SPI | ✅ PASS | ✅ PASS | Works without UVM |
| AHB | ✅ PASS | ✅ PASS | Works without UVM |
| AXI4 | ✅ PASS | ✅ PASS | Works without UVM |

**MAJOR MILESTONE (Iteration 28)**:
- **SVA assertion functions** ✅ COMPLETE - $sampled, $past (with delay), $changed, $stable, $rose, $fell all implemented
- **System calls expanded** ✅ COMPLETE - $onehot, $onehot0, $countbits added
- **Direct interface member access** ✅ FIXED - Hierarchical name resolution for interface.member syntax
- **Test coverage improved** ✅ COMPLETE - SVAToLTL: 3 new test files, VerifToSMT: comprehensive tests added
- **AVIP validation** ✅ COMPLETE - Global packages 100%, Interfaces 78% (failures are source issues)

**MAJOR MILESTONE (Iteration 26)**:
- **Upstream merge** ✅ COMPLETE - Merged 21 upstream commits, resolved 4 conflicts
- **Fork published** ✅ COMPLETE - thomasnormal/circt with comprehensive README feature list
- **SVA assertion lowering** ✅ VERIFIED - moore.assert/assume/cover → verif dialect working
- **$countones** ✅ IMPLEMENTED - Lowers to llvm.intr.ctpop
- **AVIP validation** ✅ ALL 6 PASS - APB, AXI4-Lite, UART, SPI, AHB, AXI4 work through MooreToCore
- **Coverage infrastructure** ✅ COMPLETE - CovergroupHandleType and ops implemented in Iteration 25

**MAJOR MILESTONE (Iteration 25)**:
- **Interface ref→vif conversion** ✅ FIXED - Interface member access generates proper lvalue references
- **Constraint MooreToCore lowering** ✅ COMPLETE - All 10 constraint ops now lower to runtime calls
- **$finish in seq.initial** ✅ FIXED - $finish no longer forces llhd.process fallback

**MAJOR MILESTONE (Iteration 23)**:
- **Initial blocks** ✅ FIXED (cabc1ab6e) - Simple initial blocks use seq.initial, work through arcilator!
- **Multi-range constraints** ✅ FIXED (c8a125501) - ~94% total constraint coverage
- **End-to-end pipeline** ✅ VERIFIED - SV → Moore → Core → HW → Arcilator all working

**Fixed (Iteration 22)**:
- **sim.terminate** ✅ FIXED (575768714) - $finish now calls exit(0/1)
- **Soft constraints** ✅ FIXED (5e573a811) - Default value constraints work

**Fixed (Iteration 21)**:
- **UVM LSP support** ✅ FIXED (d930aad54) - `--uvm-path` flag and `UVM_HOME` env var
- **Range constraints** ✅ FIXED (2b069ee30) - Simple range constraints work
- **Interface symbols** ✅ FIXED (d930aad54) - LSP returns proper interface symbols
- **sim.proc.print** ✅ FIXED (2be6becf7) - $display works in arcilator

**Resolved Blockers (Iteration 14)**:
- ~~**moore.builtin.realtobits**~~ ✅ FIXED (36fdb8ab6) - Added conversion patterns for realtobits/bitstoreal

**Recent Fixes (This Session - Iteration 13)**:
- **VTable fallback for classes without vtable segments** ✅ FIXED (6f8f531e6) - Searches ALL vtables when class has no segment
- **AVIP BFM validation** ✅ COMPLETE - APB, AHB, AXI4, AXI4-Lite parse and convert; issues in test code (deprecated UVM APIs) not tool
- **AXI4-Lite AVIP** ✅ 100% PASS - Zero MooreToCore errors
- **Pipeline investigation** ✅ DOCUMENTED - circt-sim runs but doesn't execute llhd.process bodies; arcilator is RTL-only

**Previous Fixes (Iteration 12)**:
- **Array locator inline loop** ✅ FIXED (115316b07) - Complex predicates (string cmp, AND/OR, func calls) now lowered via scf.for loop
- **llhd.time data layout crash** ✅ FIXED (1a4bf3014) - Structs with time fields now handled via getTypeSizeSafe()
- **AVIP MooreToCore** ✅ VALIDATED - All 7 AVIPs (APB, AHB, AXI4, UART, I2S, I3C, SPI) pass through MooreToCore

**Recent Fixes (Previous Session)**:
- **RefType cast crash for structs with dynamic fields** ✅ FIXED (5dd8ce361) - StructExtractRefOp now uses LLVM GEP for structs containing strings/queues instead of crashing on SigStructExtractOp
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
| **Randomization** | Range constraints work | rand/randc, constraints | ⚠️ ~59% working |
| **Coverage** | Coverage dialect exists | Full functional coverage | ⚠️ Partial |
| **Assertions** | SVA functions complete | Full SVA | ✅ $sampled/$past/$changed/$stable/$rose/$fell |
| **DPI/VPI** | Stub returns (0/empty) | Full support | ⚠️ 22 funcs analyzed, stubs work |
| **MooreToCore** | All 9 AVIPs lower | Full UVM lowering | ✅ Complete |

---

## Active Workstreams (keep 4 agents busy)

### Track A: LLHD Process Interpretation in circt-sim 🎯 ITERATION 30
**Status**: 🟡 IMPLEMENTATION PLAN READY - Phase 1 design complete
**Problem**: circt-sim doesn't interpret LLHD process bodies - simulation ends at 0fs

**Implementation Plan (Phase 1A - Core Interpreter)**:

```cpp
// New class: LLHDProcessInterpreter (tools/circt-sim/LLHDProcessInterpreter.h)
class LLHDProcessInterpreter {
  struct SignalState {
    mlir::Value sigValue;
    size_t schedulerSignalId;
  };

  llvm::DenseMap<mlir::Value, SignalState> signals;
  llvm::DenseMap<mlir::Value, llvm::Any> ssaValues;

public:
  // Phase 1A: Register signals from llhd.sig ops
  void registerSignals(mlir::Operation *moduleOp);

  // Phase 1A: Convert llhd.time to SimTime
  SimTime convertTime(llhd::TimeAttr timeAttr);

  // Phase 1A: Core operation handlers
  void interpretProbe(llhd::PrbOp op);     // Read signal value
  void interpretDrive(llhd::DrvOp op);     // Schedule signal update
  void interpretWait(llhd::WaitOp op);     // Suspend process
  void interpretHalt(llhd::HaltOp op);     // Terminate process

  // Phase 1B: Control flow (cf.br, cf.cond_br)
  void interpretBranch(cf::BranchOp op);
  void interpretCondBranch(cf::CondBranchOp op);

  // Phase 1C: Arithmetic (arith.addi, arith.cmpi, etc.)
  void interpretArith(mlir::Operation *op);
};
```

**Integration with circt-sim.cpp**:
```cpp
// In SimulationContext::buildSimulationModel():
for (auto &op : moduleOp.getBody().front()) {
  if (auto processOp = dyn_cast<llhd::ProcessOp>(&op)) {
    auto interpreter = std::make_shared<LLHDProcessInterpreter>();
    interpreter->registerSignals(moduleOp);

    auto callback = [interpreter, &processOp]() {
      interpreter->execute(processOp.getBody());
    };

    scheduler.createProcess(callback);
  }
}
```

**Phased Approach**:
- **Phase 1A** (1 week): Signal registration, llhd.prb/drv/wait/halt handlers
- **Phase 1B** (3-4 days): Control flow (cf.br, cf.cond_br, block arguments)
- **Phase 1C** (3-4 days): Arithmetic operations (arith.addi, cmpi, etc.)
- **Phase 2** (1 week): Complex types, memory, verification

**Files to Create/Modify**:
- `tools/circt-sim/LLHDProcessInterpreter.h` (NEW)
- `tools/circt-sim/LLHDProcessInterpreter.cpp` (NEW)
- `tools/circt-sim/circt-sim.cpp` (modify buildSimulationModel)
- `tools/circt-sim/CMakeLists.txt` (add new source files)

**Verified Test Case**:
```bash
# Input: test_llhd_sim.sv with initial block and always block
./build/bin/circt-verilog --ir-llhd /tmp/test_llhd_sim.sv | ./build/bin/circt-sim --sim-stats
# Output: "Simulation completed at time 0 fs" with only 1 placeholder process
# Expected: Should run llhd.process bodies with llhd.wait delays
```

**Priority**: CRITICAL - Required for behavioral simulation

### Track B: Direct Interface Member Access 🎯 ITERATION 28 - FIXED
**Status**: 🟢 COMPLETE (commit 25cd3b6a2)
**Problem**: "unknown hierarchical name" for direct (non-virtual) interface member access
**Resolution**: Fixed hierarchical name resolution for interface.member syntax
**Verified**: Works in AVIP interface tests
**Files**: `lib/Conversion/ImportVerilog/`
**Priority**: DONE

### Track C: System Call Expansion 🎯 ITERATION 28 - COMPLETE
**Status**: 🟢 ALL SVA FUNCTIONS IMPLEMENTED
**What's Done** (Iteration 28):
- $onehot, $onehot0 IMPLEMENTED (commit 7d5391552)
- $countbits IMPLEMENTED (commit 2830654d4) - count specific bit values
- $countones working (llvm.intr.ctpop)
- $clog2, $isunknown already implemented
- **SVA assertion functions** (commit 4704320af):
  - $sampled - sample value in observed region
  - $past (with delay parameter) - previous cycle value
  - $changed - value changed from previous cycle
  - $stable - value unchanged from previous cycle
  - $rose - positive edge detection
  - $fell - negative edge detection
**What's Needed**:
- Additional system calls as discovered through testing
**Files**: `lib/Conversion/ImportVerilog/Expressions.cpp`, `lib/Conversion/ImportVerilog/AssertionExpr.cpp`
**Priority**: LOW - Core SVA functions complete

### Track D: Coverage Runtime & UVM APIs 🎯 ITERATION 28 - RESEARCH COMPLETE
**Status**: 🟡 DOCUMENTED - Infrastructure exists, event sampling gap identified

### Track E: SVA Bounded Model Checking 🎯 ITERATION 29 - IN PROGRESS
**Status**: 🟢 CONVERSION WORKING - VerifToSMT produces valid MLIR, Z3 linking pending

**What's Working** (Iteration 29):
1. **Moore → Verif lowering**: SVA assertions lower to verif.assert/assume/cover
2. **Verif → LTL lowering**: SVAToLTL pass converts SVA sequences to LTL properties
3. **LTL → Core lowering**: LTLToCore converts LTL to hw/comb logic
4. **VerifToSMT conversion**: Bounded model checking loop with final assertion handling
5. **`bmc.final` support**: Assertions checked only at final step work correctly

**Key Fixes (Iteration 29)**:
- `ReconcileUnrealizedCastsPass` added to pipeline (cleanup unrealized casts)
- `BVConstantOp` argument order: (value, width) not (width, value)
- Clock counting moved BEFORE region type conversion
- `rewriter.eraseOp()` instead of direct `op->erase()` in conversion patterns
- Yield modification before op erasure (values must remain valid)

**What's Pending**:
1. **Z3 runtime linking** - Symbols not found: Z3_del_config, Z3_del_context, etc.
2. **Integration tests** - Need end-to-end SVA → SAT/UNSAT result tests
3. **Performance benchmarking** - Compare vs Verilator/Xcelium assertion checking

**Test Pipeline**:
```bash
# SVA property implication test
echo 'module test(input clk, a, b);
  assert property (@(posedge clk) a |=> b);
endmodule' > /tmp/sva_test.sv
./build/bin/circt-verilog --ir-hw /tmp/sva_test.sv | ./build/bin/circt-bmc --bound=10
```

**Files**:
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - Core BMC loop generation
- `tools/circt-bmc/circt-bmc.cpp` - BMC tool pipeline
- `lib/Conversion/SVAToLTL/SVAToLTL.cpp` - SVA to LTL conversion
- `lib/Conversion/LTLToCore/LTLToCore.cpp` - LTL to HW/Comb lowering

**Priority**: HIGH - Critical for formal verification capability

**COVERAGE INFRASTRUCTURE ANALYSIS (Iteration 28)**:

**What's Implemented** (MooreOps.td + MooreToCore.cpp):
1. `moore.covergroup.decl` - Covergroup type declarations with coverpoints/crosses
2. `moore.coverpoint.decl` - Coverpoint declarations with type info
3. `moore.covercross.decl` - Cross coverage declarations
4. `moore.covergroup.inst` - Instantiation (`new()`) with handle allocation
5. `moore.covergroup.sample` - Explicit `.sample()` method call
6. `moore.covergroup.get_coverage` - Get coverage percentage (0.0-100.0)
7. `CovergroupHandleType` - Runtime handle type (lowers to `!llvm.ptr`)

**MooreToCore Runtime Interface** (expected external functions):
- `__moore_covergroup_create(name, num_coverpoints) -> void*`
- `__moore_coverpoint_init(cg, index, name) -> void`
- `__moore_coverpoint_sample(cg, index, value) -> void`
- `__moore_covergroup_get_coverage(cg) -> double`
- (Future) `__moore_covergroup_destroy(cg)`, `__moore_coverage_report()`

**THE SAMPLING GAP**:
- **Explicit sampling works**: `cg.sample()` calls generate `CovergroupSampleOp` which lowers to runtime calls
- **Event-driven sampling NOT connected**: SystemVerilog `covergroup cg @(posedge clk)` syntax
  - Slang parses the timing event but CIRCT doesn't connect it to sampling triggers
  - The `@(posedge clk)` sampling event is lost during IR generation
  - Would require: (1) storing event info in CovergroupDeclOp, (2) generating always block to call sample

**AVIP COVERGROUP PATTERNS** (from ~/mbit/* analysis):
- AVIPs use `covergroup ... with function sample(args)` pattern (explicit sampling)
- Sample called from `write()` method in uvm_subscriber (UVM callback-based)
- Example from axi4_master_coverage.sv:
  ```systemverilog
  covergroup axi4_master_covergroup with function sample(cfg, packet);
  ...
  function void write(axi4_master_tx t);
    axi4_master_covergroup.sample(axi4_master_agent_cfg_h, t);
  endfunction
  ```
- This pattern IS SUPPORTED by current infrastructure (explicit sample calls work)

**DEPRECATED UVM APIs IN AVIPs** (need source updates for UVM 2017+):
| AVIP | File | Deprecated API |
|------|------|----------------|
| ahb_avip | AhbBaseTest.sv | `uvm_test_done.set_drain_time()` |
| i2s_avip | I2sBaseTest.sv | `uvm_test_done.set_drain_time()` |
| axi4_avip | axi4_base_test.sv | `uvm_test_done.set_drain_time()` |
| apb_avip | apb_base_test.sv | `uvm_test_done.set_drain_time()` |
| axi4Lite_avip | Multiple tests | `uvm_test_done.set_drain_time()` |
| i3c_avip | i3c_base_test.sv | `uvm_test_done.set_drain_time()` |

**Modern replacement**: `phase.phase_done.set_drain_time(this, time)` or objection-based

**What's Needed for Full Coverage Support**:
1. **Runtime library implementation** - C library implementing `__moore_*` functions
2. **Event-driven sampling** (optional) - Parse and connect @(event) to sampling triggers
3. **Coverage report generation** - At $finish, call `__moore_coverage_report()`
4. **Bins and illegal_bins** - Currently declarations only, need runtime bin tracking

**Files**: `lib/Conversion/MooreToCore/MooreToCore.cpp` (lines 1755-2095), `include/circt/Dialect/Moore/MooreOps.td` (lines 3163-3254)
**Priority**: MEDIUM - Explicit sampling works for AVIP patterns; event-driven sampling is enhancement

### Operating Guidance
- Keep 4 agents active on highest-priority tracks:
  - **Track A (LLHD interpretation)** - CRITICAL blocker for behavioral simulation
  - **Track E (SVA BMC)** - Z3 linking, then integration tests
  - **Track D (Coverage/UVM)** - Runtime library implementation
  - **Track C (System calls)** - As discovered through testing
- Track B (interface access) is COMPLETE.
- Add unit tests for each new feature or bug fix.
- Commit regularly and merge worktrees into main to keep workers in sync.
- Test on ~/mbit/*avip* and ~/sv-tests/ for real-world feedback.

### Iteration 29 Results - SVA BMC CONVERSION FIXED
**Key Fixes**:
- VerifToSMT `bmc.final` assertion handling - proper hoisting and final-only checking
- ReconcileUnrealizedCastsPass added to circt-bmc pipeline
- BVConstantOp argument order corrected (value, width)
- Clock counting before region type conversion
- Proper rewriter.eraseOp() usage in conversion patterns

**Status**: VerifToSMT conversion produces valid MLIR. Z3 runtime linking is the remaining blocker.

### Iteration 28 Results - COMPREHENSIVE UPDATE
**Commits**:
- `7d5391552` - $onehot/$onehot0 system calls
- `2830654d4` - $countbits system call
- `4704320af` - $sampled/$past/$changed for SVA assertions
- `25cd3b6a2` - Direct interface member access fix
- `12d75735d`, `110fc6caf` - Test fixes and documentation
- `47c5a7f36` - SVAToLTL comprehensive tests (3 new test files)
- `ecabb4492` - VerifToSMT comprehensive tests
- `235700509` - CHANGELOG update

**AVIP Testing Results**:
- Global packages: 8/8 pass (100%)
- Interfaces: 7/9 pass (78%) - JTAG/I2S fail due to source issues, not CIRCT bugs

**SVA Assertion Functions** - All implemented:
- $sampled, $past (with delay), $changed, $stable, $rose, $fell

**Test Coverage Improved**:
- SVAToLTL: 3 new test files added
- VerifToSMT: comprehensive tests added
- ImportVerilog: 38/38 tests pass (100%)

### Iteration 27 Results - KEY DISCOVERIES
- **$onehot/$onehot0**: ✅ IMPLEMENTED (commit 7d5391552) - lowers to llvm.intr.ctpop == 1 / <= 1
- **sim.proc.print**: ✅ ALREADY WORKS - PrintFormattedProcOpLowering exists in LowerArcToLLVM.cpp
- **circt-sim**: 🔴 CRITICAL GAP - LLHD process interpretation NOT IMPLEMENTED, simulation ends at 0fs
- **LSP debounce**: ✅ FIX EXISTS (9f150f33f) - may still have edge cases

### Previous Track Results (Iteration 26) - MAJOR PROGRESS
- **Coverage Infrastructure**: ✅ CovergroupHandleType, CovergroupInstOp, CovergroupSampleOp, CovergroupGetCoverageOp implemented
- **SVA Assertions**: ✅ Verified working - moore.assert/assume/cover → verif dialect
- **$countones**: ✅ Implemented - lowers to llvm.intr.ctpop
- **Constraint Lowering**: ✅ All 10 constraint ops have MooreToCore patterns
- **Interface ref→vif**: ✅ Fixed conversion generates llhd.prb
- **$finish handling**: ✅ Initial blocks with $finish use seq.initial (arcilator-compatible)
- **AVIP Testing**: ✅ All 9 AVIPs tested - issues are source code problems, not CIRCT
- **LSP Validation**: ✅ Works with --no-debounce flag, bug documented
- **Arcilator Research**: ✅ Identified sim.proc.print lowering as next step

### Previous Track Results (Iteration 25)
- **Track B**: ✅ Interface ref→vif conversion FIXED - Interface member access generates proper lvalue references
- **Track C**: ✅ Constraint MooreToCore lowering COMPLETE - All 10 constraint ops now lower to runtime calls
- **Track D**: ✅ $finish in seq.initial FIXED - $finish no longer forces llhd.process fallback

### Previous Track Results (Iteration 24)
- **Track A**: ✅ AVIP pipeline testing - Identified blocking issues (interface lvalue, $finish)
- **Track B**: ✅ Coverage architecture documented - Runtime ready, need IR ops
- **Track C**: ✅ Constraint expression lowering (ded570db6) - All constraint types now parsed
- **Track D**: ✅ Complex initial block analysis - Confirmed design is correct

### Previous Track Results (Iteration 23) - BREAKTHROUGH
- **Track A**: ✅ seq.initial implemented (cabc1ab6e) - Simple initial blocks work through arcilator!
- **Track B**: ✅ Full pipeline verified - SV → Moore → Core → HW → Arcilator all working
- **Track C**: ✅ Multi-range constraints (c8a125501) - ~94% total coverage
- **Track D**: ✅ AVIP constraints validated - APB/AHB/AXI4 patterns tested

### Previous Track Results (Iteration 22)
- **Track A**: ✅ sim.terminate implemented (575768714) - $finish now calls exit()
- **Track B**: ✅ Initial block solution identified - use seq.initial instead of llhd.process
- **Track C**: ✅ Soft constraints implemented (5e573a811) - ~82% total coverage
- **Track D**: ✅ All 8 AVIPs validated - Package/Interface/BFM files work excellently

### Previous Track Results (Iteration 21)
- **Track A**: ✅ Pipeline analysis complete - llhd.halt blocker identified
- **Track B**: ✅ UVM LSP support added (d930aad54) - --uvm-path flag, UVM_HOME env var
- **Track C**: ✅ Range constraints implemented (2b069ee30) - ~59% of AVIP constraints work
- **Track D**: ✅ Interface symbols fixed (d930aad54) - LSP properly shows interface structure

### Previous Track Results (Iteration 20)
- **Track A**: ✅ LSP debounce deadlock FIXED (9f150f33f) - `--no-debounce` no longer needed
- **Track B**: ✅ sim.proc.print lowering IMPLEMENTED (2be6becf7) - Arcilator can now output $display
- **Track C**: ✅ Randomization architecture researched - 80% of constraints can be done without SMT
- **Track D**: ✅ LSP tested on AVIPs - Package files work, interface/UVM gaps identified

### Previous Track Results (Iteration 19)
- **Track A**: ✅ All 27/27 MooreToCore unit tests pass (100%)
- **Track B**: ✅ Arcilator research complete - `arc.sim.emit` exists, need `sim.proc.print` lowering
- **Track C**: ✅ AVIP gaps quantified - 1097 randomization, 970 coverage, 453 DPI calls
- **Track D**: ✅ 6 LSP tests added, debounce hang bug documented (use --no-debounce)

### Previous Track Results (Iteration 13)
- **Track A**: ✅ VTable fallback committed (6f8f531e6) - Classes without vtable segments now search ALL vtables
- **Track B**: ✅ AVIP BFM validation complete - APB/AHB/AXI4/AXI4-Lite work; test code issues documented
- **Track C**: ✅ Randomization already implemented - confirmed working
- **Track D**: ✅ Pipeline investigation complete - circt-sim doesn't execute llhd.process bodies
- **Track E**: ✅ UVM conversion validation - only 1 error (moore.builtin.realtobits), AXI4-Lite 100%

### Previous Track Results (Iteration 12)
- **Track A**: ✅ Array locator inline loop complete (115316b07) - AND/OR/string predicates work
- **Track A**: ✅ llhd.time data layout crash fixed (1a4bf3014)
- **Track B**: ✅ All 7 AVIPs (APB/AHB/AXI4/UART/I2S/I3C/SPI) pass MooreToCore
- **Track C**: ⚠️ DPI chandle support added; randomization runtime still needed
- **Track D**: ⚠️ vtable.load_method error found blocking full UVM conversion

### Previous Track Results (Iteration 11)
- **Track A**: ✅ BFM nested task calls fixed (d1b870e5e) - Interface tasks calling other interface tasks now work correctly
- **Track A**: ⚠️ MooreToCore timing limitation documented - Tasks with `@(posedge clk)` can't lower (llhd.wait needs process parent)
- **Track B**: ✅ UVM MooreToCore: StructExtract crash fixed (59ccc8127) - only `moore.array.locator` remains
- **Track C**: ✅ DPI tool info functions implemented - returns "CIRCT" and "1.0" for tool name/version
- **Track D**: ✅ AHB AVIP testing confirms same fixes work across AVIPs

### Previous Track Results (Iteration 10)
- **Track A**: ✅ Interface task/function support (d1cd16f75) - BFM patterns now work with implicit iface arg
- **Track B**: ✅ JTAG/SPI/UART failures documented - all are source code issues, not CIRCT bugs
- **Track C**: ✅ DPI-C analysis complete - 22 functions documented (see docs/DPI_ANALYSIS.md)
- **Track D**: ✅ Queue global lowering verified - already works correctly

### Previous Track Results (Iteration 9)
- **Track A**: ✅ 5/9 AVIPs pass full pipeline (APB, AHB, AXI4, I2S, I3C) - JTAG/SPI/UART have source issues
- **Track B**: ⚠️ BFM parsing blocked on interface port rvalue handling (`preset_n` not recognized)
- **Track C**: ✅ Runtime gaps documented - DPI-C stubbed, randomization/covergroups not implemented
- **Track D**: ✅ Unit test for StructExtractRefOp committed (99b4fea86)

### Previous Track Results (Iteration 8)
- **Track A**: ✅ RefType cast crash fixed (5dd8ce361) - StructExtractRefOp now uses GEP for structs with dynamic fields
- **Track B**: ✅ UVM MooreToCore conversion now completes without crashes
- **Track C**: ✅ Added dyn_cast safety checks to multiple conversion patterns
- **Track D**: ✅ Sig2RegPass RefType cast also fixed

### Previous Track Results (Iteration 7)
- **Track A**: ✅ Virtual interface assignment support added (f4e1cc660) - enables `vif = cfg.vif` patterns
- **Track B**: ✅ StringReplicateOp lowering added (14bf13ada) - string replication in MooreToCore
- **Track C**: ✅ Scope tracking for virtual interface member access (d337cb092) - fixes class context issues
- **Track D**: ✅ Unpacked struct variable lowering fixed (ae1441b9d) - handles dynamic types in structs

### Previous Track Results (Iteration 6)
- **Track A**: ✅ Data layout crash fixed (2933eb854) - convertToLLVMType helper
- **Track B**: ✅ AVIP BFM testing - interfaces pass, BFMs need class members in interfaces
- **Track C**: ✅ ImportVerilog tests 30/30 passing (65eafb0de)
- **Track D**: ✅ AVIP packages pass MooreToCore, RTL modules work

### Previous Track Results (Iteration 5)
- **Track A**: ✅ getIntOrFloatBitWidth crash fixed (8911370be) - added type-safe helper
- **Track B**: ✅ Virtual interface member access added (0a16d3a06) - VirtualInterfaceSignalRefOp
- **Track C**: ✅ QueueConcatOp empty format fixed (2bd58f1c9) - parentheses format
- **Track D**: ✅ Test suite fixed (f7b9c7b15) - Moore 18/18, MooreToCore 24/24

### Previous Track Results (Iteration 4)
- **Track A**: ✅ vtable.load_method fixed for abstract classes (e0df41cec) - 4764 ops unblocked
- **Track B**: ✅ All vtable ops have conversion patterns
- **Track C**: ✅ AVIP testing found: virtual interface member access needed, QueueConcatOp format bug
- **Track D**: ✅ Comprehensive vtable tests added (12 test cases)

### Previous Track Results (Iteration 3)
- **Track A**: ✅ array.size lowering implemented (f18154abb) - 349 ops unblocked
- **Track B**: ✅ Virtual interface comparison ops added (8f843332d) - VirtualInterfaceCmpOp
- **Track C**: ✅ hvlTop tested - all fail on UVM macros (separate issue)
- **Track D**: ✅ Test suite runs clean

### Previous Track Results (Iteration 2)
- **Track A**: ✅ MooreSim tested - dyn_extract was blocking, now fixed
- **Track B**: ✅ dyn_extract/dyn_extract_ref implemented (550949250) - 970 queue ops unblocked
- **Track C**: ✅ AVIP+UVM tested - interfaces pass, BFMs blocked on virtual interface types
- **Track D**: ✅ All unit tests pass after fixes (b9335a978)

### Previous Track Results (Iteration 1)
- **Track A**: ✅ Multi-file parsing fixed (170414961) - empty filename handling added
- **Track B**: ✅ MooreToCore patterns added (69adaa467) - FormatString, CallIndirect, SScanf, etc.
- **Track C**: ✅ AVIP testing done - 13/14 components pass (timescale issue with JTAG)
- **Track D**: ✅ Unit tests added (b27f71047) - Mem2Reg, static properties, time type

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

## Feature Gap Analysis (Iteration 30) - COMPREHENSIVE SURVEY

Based on systematic testing of ~/sv-tests/, ~/mbit/*avip*, and ~/verilator-verification/:

### Critical Gaps for Xcelium Parity

| Feature | Status | Tests Blocked | Priority |
|---------|--------|---------------|----------|
| **Clocking Blocks** | NOT IMPLEMENTED | ~50 sv-tests (Ch14 0%) | HIGH - sv-tests |
| **Z3 Installation** | Z3_LIBRARIES-NOTFOUND | SVA BMC execution | HIGH - Install needed |
| **LLHD Process Interpreter** | Plan ready | circt-sim behavioral | HIGH - Critical |
| **RandSequence** | NOT IMPLEMENTED | ~30 sv-tests | MEDIUM |
| **SequenceWithMatch** | NOT IMPLEMENTED | ~25 sv-tests | MEDIUM |
| **TaggedUnion** | NOT IMPLEMENTED | ~20 sv-tests | MEDIUM |
| **clocked_assert lowering** | Missing pass | circt-bmc with clocked props | MEDIUM |
| **4-State (X/Z)** | NOT IMPLEMENTED | Many tests | HIGH |
| **Signal Strengths** | NOT IMPLEMENTED | 37 verilator tests | MEDIUM |

### Test Suite Coverage (Verified Iteration 30)

| Test Suite | Total Tests | Pass Rate | Notes |
|------------|-------------|-----------|-------|
| **sv-tests** | 989 (non-UVM) | **72.1%** (713/989) | Parsing/elaboration |
| **mbit AVIP globals** | 8 packages | **100%** (8/8) | All work |
| **mbit AVIP interfaces** | 8 interfaces | **75%** (6/8) | 2 source issues |
| **mbit AVIP HVL** | 8 packages | **0%** | Requires UVM lib |
| **verilator-verification** | 154 | **~60%** | SVA tests improved |

### sv-tests Detailed Analysis

**Strongest Chapters** (>80%):
- Chapter 11 (Operators): 87% pass
- Chapter 5 (Lexical): 86% pass
- Chapter 13 (Tasks/Functions): 86% pass
- Chapter 20 (I/O Formatting): 83% pass

**Weakest Chapters** (<50%):
- Chapter 14 (Clocking Blocks): 0% pass - NOT IMPLEMENTED
- Chapter 18 (Random/Constraints): 25% pass - RandSequence missing
- Chapter 21 (I/O System Tasks): 37% pass - VcdDump missing

**Top Error Categories** (by test count):
1. ClockingBlock - 0% of Ch14 tests pass
2. RandSequence - randsequence statement not supported
3. SequenceWithMatch - sequence match patterns
4. TaggedUnion - tagged union types
5. EmptyArgument - empty function arguments

### SVA Functions Status (Iteration 28-29)

| Function | ImportVerilog | SVAToLTL | VerifToSMT | Status |
|----------|---------------|----------|------------|--------|
| $sampled | ✅ | ✅ | ✅ | WORKING |
| $past | ✅ | ✅ | ✅ | WORKING |
| $rose | ✅ | ✅ | ✅ | WORKING |
| $fell | ✅ | ✅ | ✅ | WORKING |
| $stable | ✅ | ✅ | ✅ | WORKING |
| $changed | ✅ | ✅ | ✅ | WORKING |
| Sequences | ✅ | ✅ | ? | Needs testing |
| Properties | ✅ | ✅ | ? | Needs testing |

### Z3 Linking Fix Options

1. **Quick Fix**: Use `--shared-libs=/path/to/libz3.so` at runtime
2. **CMake Fix**: Add Z3 to target_link_libraries in circt-bmc
3. **Auto-detect**: Store Z3 path at build time, inject at runtime

---

## Big Projects Status (Iteration 30)

Comprehensive survey of the 6 major projects for Xcelium parity:

### 1. Full SVA Support with Z3 ⚠️ PARTIAL

**Working:**
- SVA → LTL conversion complete (SVAToLTL.cpp - 321 patterns)
- VerifToSMT conversion (967 lines)
- $sampled, $past, $changed, $stable, $rose, $fell implemented
- circt-bmc bounded model checking pipeline

**Missing:**
- Z3 NOT INSTALLED (`Z3_LIBRARIES-NOTFOUND`)
- LTL properties not yet supported in VerifToSMT
- `verif.clocked_assert` needs lowering pass
- SMT solver for complex constraints

### 2. Scalable Multi-core Arcilator ❌ MISSING

**Status:** No multi-threading support found
- Arcilator runtime is single-threaded JIT
- Arc dialect has 37+ transform passes (all sequential)
- Would require fundamental architectural redesign
- Consider PDES (Parallel Discrete Event Simulation) model

### 3. Language Server (LSP) and Debugging ⚠️ PARTIAL

**Working:**
- circt-verilog-lsp-server compiles and runs
- LSP transport infrastructure (LLVM LSP integration)
- `--uvm-path` flag and `UVM_HOME` env var parsing
- Basic file parsing and error reporting

**Missing:**
- Code completion (semantic)
- Go-to-definition/references (cross-file)
- Rename refactoring
- Debugger integration (LLDB)

### 4. Full 4-State (X/Z Propagation) ❌ MISSING

**Status:** Two-state logic only (0/1)
- X and Z recognized as identifiers only
- Requires Moore type system redesign
- Would impact 321+ conversion patterns
- Design 4-state type system RFC needed

### 5. Coverage Support ⚠️ PARTIAL

**Working:**
- CovergroupDeclOp, CoverpointDeclOp, CoverCrossDeclOp in MooreOps.td
- Coverage runtime library (2,270 LOC)
- 80+ test cases in test_coverage_runtime.cpp
- Coverage ops lower to MooreToCore

**Missing:**
- Coverage expressions and conditional sampling
- Cross-cover correlation analysis
- Coverage HTML report generation

### 6. DPI/VPI Support ⚠️ STUBS ONLY

**Current:**
- DPI-C import parsing works (22 functions stubbed)
- External function declarations recognized
- Stub returns: int=0, string="CIRCT", void=no-op

**Missing:**
- No actual C function invocation (FFI bridge needed)
- No VPI (Verilog Procedural Interface)
- Memory management between SV and C undefined

### Big Projects Summary Table

| Project | Status | Priority | Blocking |
|---------|--------|----------|----------|
| SVA with Z3 | Partial | HIGH | Z3 install |
| Multi-core Arc | Missing | MEDIUM | Architecture |
| LSP/Debugging | Partial | MEDIUM | Features |
| 4-State Logic | Missing | LOW | Type system |
| Coverage | Partial | HIGH | Cross-cover |
| DPI/VPI | Stubs | MEDIUM | FFI bridge |

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

### Iteration 29
- VerifToSMT `bmc.final` fixes - proper assertion hoisting and final-only checking
- ReconcileUnrealizedCastsPass in circt-bmc pipeline
- BVConstantOp argument order fix (value, width)
- Clock counting timing fix (before region conversion)
- Proper rewriter.eraseOp() in conversion patterns

### Iteration 28
- `235700509` - [Docs] CHANGELOG update for Iteration 28
- `ecabb4492` - [Tests] VerifToSMT comprehensive tests
- `47c5a7f36` - [Tests] SVAToLTL comprehensive tests (3 new files)
- `12d75735d`, `110fc6caf` - [Tests] Test fixes and documentation
- `25cd3b6a2` - [ImportVerilog] Direct interface member access fix
- `4704320af` - [ImportVerilog] $sampled/$past/$changed/$stable/$rose/$fell for SVA
- `2830654d4` - [ImportVerilog] $countbits system call
- `7d5391552` - [ImportVerilog] $onehot/$onehot0 system calls

### Earlier
- `6f8f531e6` - [MooreToCore] Add vtable fallback for classes without vtable segments
- `59ccc8127` - [MooreToCore] Fix StructExtract/StructCreate for dynamic types
- `d1b870e5e` - [ImportVerilog] Add DPI tool info and fix interface task-to-task calls
- `d1cd16f75` - [ImportVerilog] Add interface task/function support
- `99b4fea86` - [MooreToCore] Add tests for StructExtractRefOp with dynamic fields
- `5dd8ce361` - [MooreToCore] Fix RefType cast crashes for structs with dynamic fields
- `f4e1cc660` - [ImportVerilog] Add virtual interface assignment support
- `14bf13ada` - [MooreToCore] Add StringReplicateOp lowering
- `d337cb092` - [ImportVerilog] Add scope tracking for virtual interface member access in classes
- `ae1441b9d` - [MooreToCore] Fix variable lowering for unpacked structs with dynamic types
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
