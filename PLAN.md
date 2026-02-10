# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 10, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 221/221 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 912 total, 850 pass, 0 fail | 99.9% |
| sv-tests xfail | 7 UVM phase sequencing + 54 compile-only/negative | |
| sv-tests xpass | 1 (uvm_agent_active) | NEW: resolveDrivers fix |
| AVIPs (hvl_top only) | 9/9 pass | Full phase lifecycle; no transactions without hdl_top |
| AVIPs with transactions | 0/9 | **Blocked**: BFM gap — no hdl_top simulated |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |
| Coverage collection | Working | Parametric `with function sample()` fixed |

### Recent Fixes (This Session)

1. **`resolveDrivers()` multi-bit fix** (ProcessScheduler.h): Fixed signal resolution for FourStateStruct signals. Old code used `getLSB()` to classify drivers — broken for `hw.struct<value, unknown>` where the value bit is at MSB. Both `0b00` (val=0) and `0b10` (val=1) had LSB=0, so signals never changed value. Fix: group drivers by full APInt value, resolve by effective strength.

2. **VIF shadow signals** (LLHDProcessInterpreter.cpp/h): Created per-field runtime signals for interface struct fields. Three components:
   - `createInterfaceFieldShadowSignals()`: scans `valueToSignal` for `llhd.sig` holding `!llvm.ptr`, finds GEP users of interface structs, creates shadow signals per field
   - Store interception in `interpretLLVMStore`: drives shadow signal when interface field written
   - Sensitivity expansion in `interpretWait` (case 1 + case 3): expands interface ptr signals to field shadow signals

3. **`uvm_agent_active` now passes**: Was XFAIL due to VIF signal mismatch. The resolveDrivers fix + VIF shadow signals resolved it.

### xfail Breakdown (7 UVM tests remaining)

| Category | Count | Tests | Root Cause |
|----------|-------|-------|------------|
| UVM phase sequencing | 4 | uvm_agent_env, uvm_agent_passive, uvm_monitor_env, uvm_scoreboard_env | Processes stuck at time 0 — deeper UVM infra issue |
| UVM phase sequencing | 2 | uvm_scoreboard_monitor_agent_env, uvm_scoreboard_monitor_env | Same |
| UVM sequencer interface | 1 | uvm_driver_sequencer_env | Needs resource_db + sequencer interface |

**Root cause (all 7)**: Signal resolution is fixed, but these tests have deeper UVM phase sequencing issues. Processes spawn but never advance past time 0 — likely missing `exec_task` dispatch for specific UVM component types or incomplete phase-to-process mapping.

### Tests Now Passing (previously xfail)

All Ch18 constraint, random stability, and basic UVM tests pass:
- Constraint inheritance, distribution, global, functions, guards, soft, implication/if-else
- Foreach iterative constraints, array reduction constraints
- Infeasible detection, pre/post_randomize, inline constraints with ranges/variables
- rand_mode (all), constraint_mode, dynamic modification, scope variables
- Random stability: srandom, get/set_randstate, thread/object stability, manual seed
- UVM: urandom, resource_db, run_test, **uvm_agent_active** (NEW)

## Workstreams

### Track 1: Constraint Solver Improvements (**COMPLETE**)
**Goal**: Fix remaining constraint features. All constraint tests now pass.

**Completed** (all Ch18 constraint tests pass):
- ✅ All constraint types: inheritance, guards, soft, implication, if-else, set-membership
- ✅ Distribution, compound ranges, zext/sext, VariableOp support
- ✅ rand_mode, constraint_mode, dynamic modification
- ✅ Inline constraints with class-property bounds and constant bounds
- ✅ Infeasible detection, randomize(null) check-only, randomize(var_list) filtering
- ✅ Foreach iterative constraints, array reduction constraints
- ✅ Static rand_mode sharing across instances

### Track 2: Random Stability & Seeding (**COMPLETE**)
**Goal**: Deterministic random number generation per IEEE 1800-2017 §18.13-18.14.

**All tasks completed** - All random stability tests now pass.

### Track 3: Coverage Collection (**BLOCKED on Track 5**)
**Goal**: Get coverage working end-to-end for all AVIPs.
**Status**: Coverage infrastructure works, but no AVIP transactions flow to trigger sampling.

**Next Tasks** (after Track 5 unblocks):
1. **Automatic sampling triggers** - `@(posedge clk)` event-driven sampling
2. **Coverage start()/stop()** - Lower to `set_sample_enabled()` runtime calls
3. **Wildcard bin matching** - X/Z pattern matching
4. **Default bins** - `bins others = default` catch-all
5. **Verify coverage vs Xcelium reference** - Compare APB 21-30% baseline

### Track 4: UVM Testbench Fixes (**7 xfail remaining**)
**Goal**: Fix remaining 7 UVM testbench xfail tests.

**Completed**:
- ✅ Virtual interface clock propagation (module-level continuous assigns)
- ✅ Edge detection (posedge/negedge/anyedge)
- ✅ UVM phase sequencing, objection handling
- ✅ config_db set/get/exists interceptor
- ✅ UVM run_test with grace period
- ✅ stream_unpack interpreter handler
- ✅ resource_db read_by_name
- ✅ resolveDrivers() multi-bit fix for FourStateStruct signals
- ✅ VIF shadow signals (interface field → signal bridging)
- ✅ uvm_agent_active test now passes

**Remaining 7 xfail**: Processes stuck at time 0 — UVM phase sequencing/dispatch issue.

**Investigation needed**:
- Compare process creation/suspension patterns between uvm_agent_active (passes) and uvm_agent_env (fails)
- Check if `exec_task` dispatch correctly handles all UVM component types
- Verify phase-to-process mapping for multi-agent environments

**Also pending**:
- **SVA concurrent assertions** - Runtime eval for `assert property` (26 compile-only tests)

### Track 5: Dual-Top Simulation (**P0 BLOCKER**)
**Goal**: Simulate both `hvl_top` and `hdl_top` together so BFMs are available.

**Tasks**:
1. **Recompile AVIP .mlir files** - Current .mlir files missing; need fresh compilation
2. **Compile APB AVIP with both tops** into single .mlir
3. **Test dual-top simulation**: `circt-sim combined.mlir --top hvl_top --top hdl_top`
4. **Validate BFM exchange**: hdl_top sets virtual interface → hvl_top gets it
5. **Repeat for all 9 AVIPs**

### Track 6: Performance & Tech Debt (**ONGOING**)
**Goal**: Keep simulation fast and code clean.

**Current**: ~171 ns/s APB (10us sim in 59s wall-clock)

**Targets**:
- Remove unused debug infrastructure
- Profile hot paths in interpreter loop
- Optimize signal resolution for multi-driver signals
- Reduce memory allocations in process execution

## Priority Matrix

| Priority | Track | Next Task | Impact |
|----------|-------|-----------|--------|
| P0 | Track 5 | Recompile AVIPs + dual-top simulation | ALL AVIP coverage testing blocked |
| P0 | Track 4 | Investigate 7 UVM xfail root causes | Last 7 xfail tests |
| P1 | Track 3 | Coverage verification after Track 5 | End-to-end coverage numbers |
| P2 | Track 4 | SVA concurrent assertions | 26 compile-only tests |
| P3 | Track 6 | Performance optimization | Faster simulation |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 221 pass |
| ImportVerilog | `python3 build/bin/llvm-lit test/Conversion/ImportVerilog/ -v` | 268 pass |
| sv-tests sim | `bash utils/run_sv_tests_circt_sim.sh` | 0 fail, 1 xpass, 7 UVM xfail |
| AVIPs | `circt-sim X.mlir --top Y --max-time=500000000` | All 9 exit 0 |
| sv-tests BMC | `BMC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_bmc.sh` | 26/26 |
| sv-tests LEC | `LEC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_lec.sh` | 23/23 |

## Test Suites

- `~/sv-tests/` - IEEE 1800 compliance (912 tests)
- `~/mbit/*avip*/` - AVIP testbenches (9 protocols)
- `~/verilator-verification/` - Verilator reference tests
- `~/yosys/tests/` - Yosys SVA tests
- `~/opentitan/` - OpenTitan formal (Codex handles)
