# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 10, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 221/221 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 696/696 eligible pass+xfail | 100% |
| sv-tests xfail | 8 | All UVM VIF signal propagation |
| sv-tests compile-only | 46 | Class-only + SVA UVM |
| sv-tests pass | 115 Ch18/UVM tests | Constraints, random stability, phases, resource_db all working |
| AVIPs (hvl_top only) | 9/9 pass | All reach full phase lifecycle (build→run→report); no transactions without hdl_top |
| AVIPs with transactions | 0/9 | **Blocked**: BFM gap — no hdl_top simulated → no bus transactions, 0% coverage |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |
| Coverage collection | Working | Parametric `with function sample()` fixed |

### Recent Fixes (This Session)

1. **`randomize(null)` check-only mode** (IEEE §18.11.1): Checks if current values satisfy constraints without randomizing. Returns 0/1.
2. **`randomize(var_list)` argument filtering** (IEEE §18.11): Only randomize listed variables; all others treated as state variables.
3. **Class-level dynamic constraint extraction**: `extractClassDynamicConstraints()` for property-to-property constraints.
4. **`arith::MinSIOp`/`MaxSIOp` interpreter handlers**: Needed for check-only constraint evaluation.
5. **`__moore_stream_unpack_bits` interpreter handler**: Unblocked 6 UVM testbench tests.
6. **5 additional sv-tests now pass**: foreach constraints, array reduction, inline constraint range, static rand_mode, resource_db read_by_name.

### xfail Breakdown (8 tests remaining)

| Category | Count | Tests | Root Cause |
|----------|-------|-------|------------|
| UVM VIF signal mismatch | 3 | uvm_agent_env, uvm_agent_passive, uvm_monitor_env | VIF writes to memory, not signals |
| UVM VIF delta overflow | 3 | uvm_scoreboard_env, uvm_scoreboard_monitor_*, uvm_scoreboard_monitor_env | Clock-driven delta loop |
| UVM resource_db + VIF | 1 | uvm_agent_active | resource_db + VIF clock |
| UVM sequencer interface | 1 | uvm_driver_sequencer_env | Sequencer interface pattern |

**Root cause (all 8)**: Virtual interface writes via VIF pointers modify heap memory but don't drive LLHD signals. The VIF pointer (obtained via `resource_db::set/get`) is a probed snapshot, not a live signal reference. Architectural fix needed.

### Tests Now Passing (previously xfail)

These categories were all fixed and moved from xfail to pass:
- Constraint inheritance (18.5.2), distribution (18.5.4), global (18.5.9), functions (18.5.12)
- Soft constraints (18.5.14), soft priorities (18.5.14.1), discarding soft (18.5.14.2)
- Infeasible detection (18.6.3), pre/post_randomize (18.6.2)
- Inline constraints with ranges/variables (18.7 tests 0,2,4,6), local scope (18.7.1)
- Inline constraint checker (18.11.1 tests 0,1), variable control (18.11 tests 0,1)
- rand_mode (18.8 tests 0-3), constraint_mode (18.9), dynamic modification (18.10)
- Foreach iterative constraints (18.5.8.1), array reduction (18.5.8.2)
- Scope variables (18.12, 18.12.1)
- Random stability: srandom (18.13.3), get/set_randstate (18.13.4-5)
- Thread/object stability (18.14, 18.14.2, 18.14.3), manual seed (18.15)
- UVM: urandom (18.13.1-2), resource_db basic + read_by_name, run_test

## Workstreams

### Track 1: Constraint Solver Improvements (**COMPLETE**)
**Goal**: Fix remaining constraint features. All constraint tests now pass.

**Completed** (all Ch18 constraint tests pass):
- ✅ Constraint inheritance, guards, soft, implication/if-else/set-membership
- ✅ Distribution, compound ranges, zext/sext, VariableOp support
- ✅ rand_mode receiver, constraint_mode, dynamic modification
- ✅ Inline constraints with class-property bounds and constant bounds
- ✅ Inline soft constraint priority override
- ✅ Infeasible constraint detection (randomize returns 0)
- ✅ `randomize(null)` check-only mode (§18.11.1)
- ✅ `randomize(v, w)` argument filtering (§18.11)
- ✅ Inline constraint var-to-var (§18.7)
- ✅ Foreach iterative constraints (§18.5.8.1)
- ✅ Array reduction constraints (§18.5.8.2)
- ✅ Static rand_mode sharing (§18.8)

### Track 2: Random Stability & Seeding (**COMPLETE**)
**Goal**: Implement deterministic random number generation per IEEE 1800-2017 §18.13-18.14.

**All tasks completed** - All random stability tests now pass:
- ✅ Per-object RNG state (std::mt19937 per object address)
- ✅ `srandom(seed)`, `get_randstate()`, `set_randstate()`
- ✅ Thread stability (fork contexts get isolated RNG)
- ✅ Object stability (same seed → same values)
- ✅ Manual seed in `randomize()` call
- ✅ Fork RNG hierarchical seeding

### Track 3: Coverage Collection (**BLOCKED on Track 5**)
**Goal**: Get coverage working end-to-end for all AVIPs.
**Status**: Coverage infrastructure works (130+ runtime functions), but no AVIP transactions
flow to trigger sampling. Blocked on dual-top simulation (Track 5).

**Completed**:
- ✅ Parametric covergroup `with function sample()` fix
- ✅ Coverage reporting framework
- ✅ Coverpoint expression evaluation with formal parameter binding
- ✅ Coverpoint `iff` guard evaluation

**Next Tasks** (after Track 5 unblocks):
1. **Automatic sampling triggers** - `@(posedge clk)` event-driven sampling
2. **Coverage start()/stop()** - Lower to `set_sample_enabled()` runtime calls
3. **Wildcard bin matching** - X/Z pattern matching
4. **Default bins** - `bins others = default` catch-all
5. **Verify coverage vs Xcelium reference** - Compare APB 21-30% baseline

### Track 4: UVM Testbench Fixes (**BLOCKED on VIF signal propagation**)
**Goal**: Fix remaining 8 UVM testbench xfail tests.

**Completed**:
- ✅ Virtual interface clock propagation (module-level continuous assigns)
- ✅ Edge detection (posedge/negedge/anyedge)
- ✅ UVM phase sequencing, objection handling
- ✅ config_db set/get/exists interceptor
- ✅ UVM run_test with grace period
- ✅ stream_unpack interpreter handler
- ✅ resource_db read_by_name

**Remaining**: All 8 xfail tests fail because VIF writes (from UVM agents) modify heap memory
but don't drive LLHD signals. Architectural fix needed to bridge VIF memory writes → signal drives.

**Possible approaches**:
1. Register interface struct fields as signals in interpreter
2. Create continuous drives back to original signals at compile time
3. Use llhd.sig for each interface field instead of malloc'd struct

**Also pending**:
- **SVA concurrent assertions** - Runtime eval for `assert property` (26 compile-only tests, not xfail)

### Track 5: Dual-Top Simulation (**P0 BLOCKER**)
**Goal**: Simulate both `hvl_top` and `hdl_top` together so BFMs are available.

**Root cause**: All 9 AVIPs use proxy-BFM split architecture. Without `hdl_top`, BFM
`get()` calls fail with UVM_FATAL. No transactions flow, coverage is 0%.

**Multi-`--top` infrastructure already exists** in circt-sim.

**Tasks**:
1. **Recompile AVIP .mlir files** - Current .mlir files missing; need fresh compilation from .sv
2. **Compile APB AVIP with both tops** into single .mlir
3. **Test dual-top simulation**: `circt-sim combined.mlir --top hvl_top --top hdl_top`
4. **Validate BFM exchange**: hdl_top sets virtual interface → hvl_top gets it
5. **Repeat for all 9 AVIPs**

## Priority Matrix

| Priority | Track | Next Task | Impact |
|----------|-------|-----------|--------|
| P0 | Track 5 | Recompile AVIPs + dual-top simulation | ALL AVIP coverage testing blocked |
| P0 | Track 4 | VIF signal propagation fix | Last 8 xfail tests |
| P1 | Track 3 | Coverage verification after Track 5 | End-to-end coverage numbers |
| P2 | Track 4 | SVA concurrent assertions | 26 compile-only tests |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 221 pass |
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
