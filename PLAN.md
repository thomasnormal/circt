# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 9, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 212/212 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 696/696 eligible pass+xfail | 100% |
| sv-tests xfail (runtime gaps) | ~54 | See breakdown below |
| sv-tests compile-only | 73 | Class-only + SVA UVM |
| AVIPs (hvl_top only) | 9/9 pass | All reach full phase lifecycle (build→run→report); no transactions without hdl_top |
| AVIPs with transactions | 0/9 | **Blocked**: BFM gap — no hdl_top simulated → no bus transactions, 0% coverage |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |
| Coverage collection | Working | Parametric `with function sample()` fixed |

### Recent Fixes (This Session)

1. **Associative array deep copy** (UVM phase livelock fix): `aa1 = aa2` now calls `__moore_assoc_copy_into(dst, src)` instead of shallow pointer copy. Root cause of phase livelock: `uvm_phase::add()` copies predecessor maps between phase nodes then calls `.delete()` on the source, corrupting all shared references.
2. **Call stack restoration fix**: Three bugs fixed — innermost-first frame processing, `waitConditionSavedBlock` derivation from outermost frame's callOp, `outermostCallOp` fallback for non-wait_condition suspensions.
3. **Per-process RNG for random stability**: Replaced global std::rand/urandom with per-process RNG (IEEE 1800-2017 §18.13)
4. **Coverpoint iff guard lowering**: Added iff_conditions to CovergroupSampleOp with AttrSizedOperandSegments
5. **Class get/set_randstate lowering**: ImportVerilog emits __moore_class_get_randstate/__moore_class_set_randstate calls
6. **Debug output cleanup**: Removed all [PH-TRACE] and [DBG-] debug logging from interpreter
7. **Fork RNG hierarchical seeding**: Child threads seeded from parent's RNG per IEEE 1800-2017 §18.13-18.14
8. **Deferred sim.terminate fix**: Normal $finish no longer kills forked children (phase hopper); only UVM_FATAL sets terminationRequested
9. **Suspension propagation**: call_indirect failure with waiting state treated as valid suspension, not error

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

**Recently Completed** (cont.):
- ✅ `get_randstate()` / `set_randstate()` - Save/restore RNG state as string
- ✅ ImportVerilog: `__moore_class_get_randstate(objPtr)` / `__moore_class_set_randstate(objPtr, stateStruct)`
- ✅ Interpreter: Serialize/deserialize `std::mt19937` state via `ostringstream`/`istringstream`
- ✅ Test: `random-get-set-randstate.sv` validates round-trip state preservation

**Next Tasks**:
1. **Thread stability** - Fork contexts get isolated RNG state (3 tests)
2. **Manual seed in `randomize(seed)`** - Override RNG for one call (2 tests)

### Track 3: Coverage Collection (**BLOCKED on Track 5**)
**Goal**: Get coverage working end-to-end for all AVIPs.
**Status**: Coverage infrastructure works (130+ runtime functions), but no AVIP transactions
flow to trigger sampling. Blocked on dual-top simulation (Track 5).

**Recently Completed**:
- ✅ Parametric covergroup `with function sample()` fix (was causing 0% coverage)
- ✅ Coverage reporting framework (`__moore_coverage_report()`)
- ✅ Coverpoint expression evaluation with formal parameter binding
- ✅ Coverpoint `iff` guard evaluation (conditional branch in MooreToCore)
- ✅ Per-object RNG (`get_randstate`/`set_randstate`)

**Next Tasks** (after Track 5 unblocks):
1. **Automatic sampling triggers** - `@(posedge clk)` event-driven sampling
2. **Coverage start()/stop()** - Lower to `set_sample_enabled()` runtime calls
3. **Wildcard bin matching** - X/Z pattern matching in `checkIgnoreBinsInternal`
4. **Default bins** - `bins others = default` catch-all
5. **Transition bin execution** - State machine tracking in runtime
6. **Verify coverage vs Xcelium reference** - Compare APB 21-30% baseline

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

### Track 5: Dual-Top Simulation (**P0 BLOCKER**)
**Goal**: Simulate both `hvl_top` and `hdl_top` together so BFMs are available.

**Root cause**: All 9 AVIPs use proxy-BFM split architecture. Driver/monitor proxies
in `hvl_top` call `uvm_config_db::get(virtual apb_master_driver_bfm)` to get BFM handles
from `hdl_top`. Without `hdl_top`, these `get()` calls fail with UVM_FATAL at time 0.
No transactions flow, no scoreboard comparisons happen, coverage is always 0%.

**What works now** (after vtable override fix):
- UVM phases run correctly: build → connect → **run_phase dispatches to derived class**
- `apb_base_test::run_phase`, `apb_master_driver_proxy::run_phase` etc. all execute
- Correct UVM_FATAL: "cannot get() apb_master_drv_bfm_h" — expected without hdl_top
- Phase sequencing: per-process map, master_phase_process detection, join_any polling
- All 6 available AVIPs (.mlir) reach run_phase and exit cleanly (exit 0)

**Multi-`--top` infrastructure already exists** in circt-sim:
- `cl::list<std::string> topModules` accepts multiple `--top` flags
- `SimulationContext::initialize()` loops through all tops, builds simulation model for each
- All tops share the same `LLHDProcessInterpreter` (same scheduler, same config_db)
- Config_db uses global key-value store → handles cross-module automatically

**Tasks**:
1. **Compile APB AVIP with both tops**: `circt-verilog -f apb_compile.f hdl_top.sv hvl_top.sv`
2. **Test dual-top simulation**: `circt-sim combined.mlir --top hvl_top --top hdl_top`
3. **Validate BFM exchange**: hdl_top `initial` sets virtual interface → hvl_top `build_phase` gets it
4. **Verify transactions**: Check sim time > 0, UVM_FATAL count = 0, coverage > 0%
5. **Repeat for all 9 AVIPs**

**Xcelium reference** (APB `apb_8b_write_test`): 21-30% coverage, real bus transactions,
0 UVM errors, 130ns sim time.

## Priority Matrix

| Priority | Track | Next Task | Impact |
|----------|-------|-----------|--------|
| P0 | Track 5 | Dual-top AVIP compilation + simulation | ALL AVIP coverage testing blocked without this |
| P0 | Track 1 | Inline constraint extraction | Unblocks 4 tests |
| P1 | Track 3 | Automatic sampling (`@(posedge clk)`) | AVIP implicit coverage |
| P1 | Track 3 | Coverage start()/stop() | Dynamic coverage control |
| P1 | Track 4 | Virtual interface DUT clock | Unblocks 6 UVM tests |
| P2 | Track 1 | Infeasible constraint detection | Unblocks 2 tests |
| P2 | Track 3 | Wildcard/default bins | Full bin support |
| P3 | Track 4 | SVA concurrent assertions | 17 compile-only tests |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 212 pass |
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
