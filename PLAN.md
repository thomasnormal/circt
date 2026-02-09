# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 9, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 206/206 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 696/696 eligible pass+xfail | 100% |
| sv-tests xfail (runtime gaps) | ~35 | See breakdown below |
| sv-tests compile-only | 73 | Class-only + SVA UVM |
| AVIPs passing | 9/9 | APB, AHB, UART, I2S, I3C, SPI, AXI4, AXI4Lite, JTAG |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |
| Coverage collection | Working | Parametric `with function sample()` fixed |

### Recent Fixes (This Session)

1. **Constraint extraction improvements**: zext/sext lookup, compound range decomposition, constraint inheritance, VariableOp support for static blocks
2. **Per-object RNG state**: `__moore_class_srandom(objPtr, seed)` with std::mt19937 per object address
3. **Virtual interface clock propagation**: ContinuousAssignOp at module level → llhd.process for signal watching
4. **Distribution constraint extraction**: `traceToPropertyName()` replaces BlockArgument-based lookup
5. **Parametric covergroup sampling**: Fix 0% coverage - evaluate per-coverpoint expressions with bound formal parameters

### xfail Breakdown (~35 tests remaining)

| Category | Count | Tests | Difficulty |
|----------|-------|-------|------------|
| Foreach/array-reduction constraints | 2 | 18.5.8.* | Hard |
| Global constraints | 1 | 18.5.9 | Hard |
| Functions in constraints | 1 | 18.5.12 | Hard |
| Soft constraint priorities | 2 | 18.5.14.1 | Medium |
| Infeasible constraint detection | 2 | 18.6.3 | Hard |
| Inline constraints (randomize with) | 4 | 18.7.* | Medium-Hard |
| Inline variable/constraint checker | 4 | 18.11.* | Medium-Hard |
| Random stability/seeding | 7 | 18.13-18.15 | Medium |
| UVM virtual interface clock | 6 | uvm_agent_*, uvm_monitor_*, uvm_scoreboard_* | Hard |
| UVM resource_db | 1 | uvm_resource_db_read_by_name | Medium |
| SVA runtime assertions | 17 | 16.2-16.17 (compile-only) | Hard |

## Workstreams

### Track 1: Constraint Solver Improvements (**ACTIVE**)
**Goal**: Fix the constraint extraction pipeline and add missing constraint features.

**Recently Completed**:
- ✅ Fix `getPropertyName()` zext/sext lookup
- ✅ Compound range decomposition (`and(uge, ule)` → min/max)
- ✅ Constraint inheritance (parent class hierarchy walking)
- ✅ Distribution constraint extraction via `traceToPropertyName()`
- ✅ VariableOp support for static constraint blocks
- ✅ Constraint guard null checks
- ✅ Soft range constraints
- ✅ Implication/if-else/set-membership constraints
- ✅ rand_mode receiver resolution

**Next Tasks (priority order)**:
1. **Inline constraint ConstraintExprOp extraction** - Fixes 18.7.* tests (4 tests)
2. **Infeasible constraint detection** - Return 0 from randomize() when unsatisfiable (2 tests)
3. **Foreach iterative constraints** - Loop over array elements (2 tests)
4. **Functions in constraints** - Evaluate function calls during solving (1 test)
5. **Cross-variable constraint solving** - `b2 == b` patterns (hard)

### Track 2: Random Stability & Seeding (**ACTIVE**)
**Goal**: Implement deterministic random number generation per IEEE 1800-2017 §18.13-18.14.

**Recently Completed**:
- ✅ Per-object RNG state (std::mt19937 per object address)
- ✅ `srandom(seed)` method → `__moore_class_srandom` in ImportVerilog + interpreter
- ✅ Pending seed bridging for legacy `@srandom_NNNN` stubs

**Next Tasks**:
1. **`get_randstate()` / `set_randstate()`** - Save/restore RNG state as string (2 tests)
2. **Thread stability** - Fork contexts get isolated RNG state (3 tests)
3. **Manual seed in `randomize(seed)`** - Override RNG for one call (2 tests)

### Track 3: Coverage Collection (**ACTIVE**)
**Goal**: Get coverage working end-to-end for all AVIPs.

**Recently Completed**:
- ✅ Parametric covergroup `with function sample()` fix (was causing 0% coverage)
- ✅ Coverage reporting framework (`__moore_coverage_report()`)
- ✅ Coverpoint expression evaluation with formal parameter binding

**Next Tasks**:
1. **Recompile AVIPs** with coverage fix to verify non-zero hits
2. **Coverpoint `iff` guard evaluation** - Currently stored as metadata string only
3. **Automatic sampling triggers** - `@(posedge clk)` event-driven sampling
4. **Wildcard bin matching** - Pattern matching for `wildcard bins`
5. **Cross coverage verification** - Validate cross bins hit counting

### Track 4: UVM Testbench Fixes (**ACTIVE**)
**Goal**: Fix remaining UVM test failures.

**Recently Completed**:
- ✅ Virtual interface clock propagation (ContinuousAssignOp → llhd.process)
- ✅ Expanded traceToMemoryPtr (struct_create/extract, extractvalue/insertvalue)
- ✅ Edge detection (posedge/negedge/anyedge)

**Next Tasks**:
1. **Virtual interface DUT clock sensitivity** - `always @(posedge vif.clk)` in DUT modules
2. **resource_db read_by_name** - Extend interceptor for UVM resource_db pattern
3. **SVA concurrent assertions** - Runtime eval for `assert property` (17 compile-only tests)
4. **Process await/kill** - `process::await()` and `process::kill()` (sv-tests timeouts)

## Priority Matrix

| Priority | Track | Next Task | Impact |
|----------|-------|-----------|--------|
| P0 | Track 3 | Recompile AVIPs with coverage fix | 9 AVIPs get real coverage |
| P0 | Track 1 | Inline constraint extraction | Unblocks 4 tests |
| P1 | Track 2 | get/set_randstate | Unblocks 2 tests |
| P1 | Track 4 | Virtual interface DUT clock | Unblocks 6 UVM tests |
| P2 | Track 1 | Infeasible constraint detection | Unblocks 2 tests |
| P2 | Track 3 | Coverpoint iff evaluation | AVIP coverage accuracy |
| P3 | Track 4 | SVA concurrent assertions | 17 compile-only tests |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 206+ pass |
| ImportVerilog | `python3 build/bin/llvm-lit test/Conversion/ImportVerilog/ -v` | 268 pass |
| sv-tests sim | `bash utils/run_sv_tests_circt_sim.sh ~/sv-tests` | 0 fail, 0 timeout |
| AVIPs | `circt-sim X.mlir --top Y --max-time=500000000` | All 9 exit 0 |
| sv-tests BMC | `BMC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_bmc.sh ~/sv-tests` | 26/26 |
| sv-tests LEC | `LEC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_lec.sh ~/sv-tests` | 23/23 |

## Test Suites

- `~/sv-tests/` - IEEE 1800 compliance (750+ tests)
- `~/mbit/*avip*/` - AVIP testbenches (9 protocols)
- `~/verilator-verification/` - Verilator reference tests
- `~/yosys/tests/` - Yosys SVA tests
- `~/opentitan/` - OpenTitan formal (Codex handles)
