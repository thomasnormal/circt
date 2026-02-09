# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/mbit/*_avip` testbenches using only CIRCT tools and the library ~/uvm-core.
Secondary goal: Get to 100% in the ~/sv-tests/ and ~/verilator-verification/ test suites.

---

## Current Status - February 9, 2026

### Test Results

| Mode | Eligible | Pass | Fail | Rate |
|------|----------|------|------|------|
| Parsing | 853 | 853 | 0 | **100%** |
| Elaboration | 1028 | 1021+ | 7 | **99.3%+** |
| Simulation (full) | 696 | 696 | 0 | **100%** (0 unexpected failures) |
| BMC (full Z3) | 26 | 26 | 0 | **100%** |
| LEC (full Z3) | 23 | 23 | 0 | **100%** |
| circt-sim lit | 206 | 206 | 0 | **100%** |
| ImportVerilog lit | 268 | 268 | 0 | **100%** |

### AVIP Status

All 9 AVIPs compile and simulate end-to-end. Performance: ~171 ns/s (APB 10us in 59s).
Coverage collection now works for parametric covergroups (requires AVIP recompilation).

| AVIP | Status | Notes |
|------|--------|-------|
| APB | WORKS | apb_base_test, 500ns sim time |
| AHB | WORKS | AhbBaseTest, 500ns sim time |
| UART | WORKS | UartBaseTest, 500ns sim time |
| I2S | WORKS | I2sBaseTest, 500ns sim time |
| I3C | WORKS | i3c_base_test, 500ns sim time |
| SPI | WORKS | SpiBaseTest, 500ns sim time |
| AXI4 | WORKS | hvl_top, 57MB MLIR, passes sim |
| AXI4Lite | WORKS | Axi4LiteBaseTest, exit code 0 |
| JTAG | WORKS | HvlTop, 500ns sim time, regex DPI fixed |

### Workstream Status

| Track | Owner | Status | Next Steps |
|-------|-------|--------|------------|
| **Track 1: Constraint Solver** | Agent | Active | Inline constraints, infeasible detection, foreach |
| **Track 2: Random Stability** | Agent | Active | get/set_randstate, thread stability |
| **Track 3: Coverage Collection** | Agent | Active | Recompile AVIPs, iff guards, auto-sampling |
| **Track 4: UVM Test Fixes** | Agent | Active | VIF clock sensitivity, resource_db, SVA runtime |
| **BMC/LEC** | Codex | Active | Structured Slang event-expression metadata (DO NOT TOUCH) |

### Feature Gap Table — Road to Xcelium Parity

**Goal: Eliminate ALL xfail tests. Every feature Xcelium supports, we support.**

| Feature | Status | Blocking Tests | Priority |
|---------|--------|----------------|----------|
| **Constraint solver** | PARTIAL | ~15 sv-tests | **P0** |
| - Constraint inheritance | **DONE** | 0 | Parent class hierarchy walking |
| - Distribution constraints | **DONE** | 0 | `traceToPropertyName()` fix |
| - Static constraint blocks | **DONE** | 0 | VariableOp support |
| - Soft constraints | **DONE** | 0 | `isSoft` flag extraction |
| - Constraint guards (null) | **DONE** | 0 | `ClassHandleCmpOp`+`ClassNullOp` |
| - Implication/if-else/inside | **DONE** | 0 | Conditional range application |
| - Inline constraints (`with`) | MISSING | `18.7--*_0/2/4/6` | 4 tests |
| - Foreach iterative constraints | MISSING | `18.5.8.1`, `18.5.8.2` | 2 tests |
| - Functions in constraints | MISSING | `18.5.12` | 1 test |
| - Infeasible detection | MISSING | `18.6.3--*_2/3` | 2 tests |
| - Global constraints | MISSING | `18.5.9` | 1 test |
| **rand_mode / constraint_mode** | **DONE** | 0 | Receiver resolution fixed |
| **Random stability** | PARTIAL | 7 sv-tests | **P1** |
| - srandom seed control | **DONE** | 0 | Per-object RNG via `__moore_class_srandom` |
| - Per-object RNG | **DONE** | 0 | `std::mt19937` per object address |
| - get/set_randstate | MISSING | `18.13.4`, `18.13.5` | 2 tests |
| - Thread/object stability | MISSING | `18.14--*` | 3 tests |
| - Manual seeding | MISSING | `18.15--*` | 2 tests |
| **Coverage collection** | PARTIAL | 0 (AVIPs) | **P0** |
| - Basic covergroups | **DONE** | 0 | Implicit + parametric sample() |
| - Parametric sample() | **DONE** | 0 | Expression binding with visitSymbolReferences |
| - Coverpoint iff guard | MISSING | — | Metadata string only, not evaluated |
| - Auto sampling (@posedge) | MISSING | — | Event-driven trigger not connected |
| - Wildcard bins | MISSING | — | Pattern matching logic needed |
| - start()/stop() | MISSING | — | Runtime stubs only |
| **SVA concurrent assertions** | MISSING | 17 sv-tests | **P1** |
| - assert/assume/cover property | MISSING | `16.2--*-uvm` | Runtime eval |
| - Sequences with ranges | MISSING | `16.7--*-uvm` | `##[1:3]` delay |
| - expect statement | MISSING | `16.17--*-uvm` | Blocking check |
| **UVM virtual interface** | PARTIAL | 6 sv-tests | **P1** |
| - Signal propagation | **DONE** | 0 | ContinuousAssignOp → llhd.process |
| - DUT clock sensitivity | MISSING | `uvm_agent_*`, etc. | `always @(posedge vif.clk)` |
| **UVM resource_db** | PARTIAL | 1 sv-test | **P2** |
| **Inline constraint checker** | MISSING | 4 sv-tests | **P2** |
| **pre/post_randomize** | **DONE** | 0 | Fixed |
| **Class property initializers** | **DONE** | 0 | Fixed |

See CHANGELOG.md on recent progress.

### Project-Plan Logging Policy
- `PROJECT_PLAN.md` now keeps intent/roadmap-level summaries only.
- `CHANGELOG.md` is the source of truth for execution history, validations, and
  command-level evidence.
- Future iterations should add:
  - concise outcome and planning impact in `PROJECT_PLAN.md`
  - detailed implementation + validation data in `CHANGELOG.md`

### Active Formal Gaps (Near-Term)
- Lane-state:
  - Add recursive refresh trust-evidence capture (peer cert chain + issuer
    linkage + pin material) beyond sidecar field matching.
  - Move metadata trust from schema + static policy matching to active
    transport-chain capture/verification in refresh tooling (issuer/path
    validation evidence).
  - Extend checkpoint granularity below lane-level where ROI is high.
- BMC capability closure:
  - Close known local-variable and `disable iff` semantic mismatches.
  - Reduce multi-clock edge-case divergence.
  - Expand full (not filtered) regular closure cadence on core suites.
- LEC capability closure:
  - Reduce/eliminate `LEC_ACCEPT_XPROP_ONLY=1` dependence for OpenTitan
    `aes_sbox_canright`.
  - Improve 4-state/X-prop semantic alignment and diagnostics.
- DevEx/CI:
  - Promote lane-state inspector to required pre-resume CI gate.
  - Add per-lane historical trend dashboards and automatic anomaly detection.

### Iteration 733 (OpenTitan Parity Gate)
- Added `utils/run_opentitan_formal_e2e.sh` to run non-smoke OpenTitan
  end-to-end checks across:
  - full-IP simulation targets
  - full parse targets
  - AES S-Box LEC
- Hardened `utils/run_opentitan_circt_sim.sh` so wall-clock timeout markers are
  treated as failures (prevents false PASS).
- Planning impact:
  - OpenTitan parity claims are now backed by explicit non-smoke E2E evidence,
    not smoke-only checks.
  - Gaps remain visible via machine-readable `results.tsv` outputs and can be
    tracked to closure.
- Detailed logs and validation commands are tracked in `CHANGELOG.md`.
