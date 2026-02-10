# CIRCT UVM Parity Project Plan

Goal: Bring `circt-sim` to parity with Cadence Xcelium for running UVM testbenches.

## Current Status (Feb 10, 2026)

| Metric | Count | Rate |
|--------|-------|------|
| circt-sim unit tests | 223/223 | 100% |
| ImportVerilog tests | 268/268 | 100% |
| sv-tests simulation | 912 total, 856 pass, 0 fail | 99.9% |
| sv-tests xfail | 1 UVM + 54 compile-only/negative | |
| sv-tests xpass | 7 (4 agent/monitor + 3 scoreboard) | resolveSignalId + analysis port fixes |
| AVIPs (hvl_top only) | 9/9 pass | Full phase lifecycle; no transactions without hdl_top |
| AVIPs with transactions | 0/9 | **Blocked**: BFM gap — no hdl_top simulated |
| sv-tests BMC | 26/26 | 100% (Codex) |
| sv-tests LEC | 23/23 | 100% (Codex) |
| Coverage collection | Working | Parametric `with function sample()` fixed |

### Recent Fixes (This Session)

1. **`resolveSignalId` cast+probe tracing** (LLHDProcessInterpreter.cpp): Fixed signal resolution for interface pointer signals passed through module ports. The pattern `unrealized_conversion_cast(llhd.prb(sig))` was not traced — `resolveSignalId` now follows through the cast to the probe to find the underlying signal. This unblocked DUT processes from delta overflow when using `always @(posedge in_if.clk)`.

2. **`resolveDrivers()` multi-bit fix** (ProcessScheduler.h): Fixed signal resolution for FourStateStruct signals. Old code used `getLSB()` to classify drivers — broken for `hw.struct<value, unknown>` where the value bit is at MSB. Fix: group drivers by full APInt value, resolve by effective strength.

3. **VIF shadow signals** (LLHDProcessInterpreter.cpp/h): Created per-field runtime signals for interface struct fields. Three components:
   - `createInterfaceFieldShadowSignals()`: scans `valueToSignal` for `llhd.sig` holding `!llvm.ptr`, finds GEP users of interface structs, creates shadow signals per field
   - Store interception in `interpretLLVMStore`: drives shadow signal when interface field written
   - Sensitivity expansion in `interpretWait` (case 1 + case 3): expands interface ptr signals to field shadow signals

4. **4 UVM tests now pass**: `uvm_agent_active` (resolveDrivers + VIF shadows), `uvm_agent_env`, `uvm_agent_passive`, `uvm_monitor_env` (resolveSignalId fix).

### xfail Breakdown (1 UVM test remaining)

| Category | Count | Tests | Root Cause |
|----------|-------|-------|------------|
| Sequencer interface | 1 | uvm_driver_sequencer_env | Needs sequencer interface implementation |

**Root cause**: Missing `get_next_item()`/`item_done()` sequencer interface implementation.

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
- ✅ uvm_agent_active, uvm_agent_env, uvm_agent_passive, uvm_monitor_env now pass
- ✅ resolveSignalId cast+probe tracing for interface module ports
- ✅ uvm_scoreboard_env, uvm_scoreboard_monitor_env, uvm_scoreboard_monitor_agent_env now pass
- ✅ Analysis port connect/write interceptor with chain-following dispatch

**Remaining 1 xfail**: uvm_driver_sequencer_env (sequencer interface).

**Investigation needed**:
- Implement sequencer interface (`get_next_item()`, `item_done()`)

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
| P0 | Track 4 | Fix last UVM xfail (sequencer interface) | Last 1 xfail test |
| P1 | Track 3 | Coverage verification after Track 5 | End-to-end coverage numbers |
| P2 | Track 4 | SVA concurrent assertions | 26 compile-only tests |
| P3 | Track 6 | Performance optimization | Faster simulation |

## Testing Targets

| Suite | Command | Expected |
|-------|---------|----------|
| circt-sim unit | `python3 build/bin/llvm-lit test/Tools/circt-sim/ -v` | 223 pass |
| ImportVerilog | `python3 build/bin/llvm-lit test/Conversion/ImportVerilog/ -v` | 268 pass |
| sv-tests sim | `bash utils/run_sv_tests_circt_sim.sh` | 0 fail, 7 xpass, 1 UVM xfail |
| AVIPs | `circt-sim X.mlir --top Y --max-time=500000000` | All 9 exit 0 |
| sv-tests BMC | `BMC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_bmc.sh` | 26/26 |
| sv-tests LEC | `LEC_SMOKE_ONLY=1 bash utils/run_sv_tests_circt_lec.sh` | 23/23 |

## Test Suites

- `~/sv-tests/` - IEEE 1800 compliance (912 tests)
- `~/mbit/*avip*/` - AVIP testbenches (9 protocols)
- `~/verilator-verification/` - Verilator reference tests
- `~/yosys/tests/` - Yosys SVA tests
- `~/opentitan/` - OpenTitan formal (Codex handles)
