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
| **Mutation coverage (Certitude-style)** | IN_PROGRESS | New toolchain | **P0** |
| - Native mutation harness (`run_mutation_cover.sh`) | **DONE** | 0 | Added with formal pre-qualification + reporting |
| - 4-way classes (`not_activated`, `not_propagated`, `propagated_not_detected`, `detected`) | **DONE** | 0 | Mutant + pair-level artifacts |
| - Formal activation/propagation pruning | **DONE** | 0 | Per test-mutant pair pre-qualification |
| - Global formal propagation filter | **DONE** | 0 | Added per-mutant `--formal-global-propagate-cmd` relevance pruning before pair runs |
| - LEC-native global relevance helper | **DONE** | 0 | Added built-in `--formal-global-propagate-circt-lec` mode with `LEC_RESULT=EQ/NEQ` classification |
| - BMC-native differential global relevance helper | **DONE** | 0 | Added built-in `--formal-global-propagate-circt-bmc` mode comparing orig vs mutant `BMC_RESULT` |
| - Improvement + metric report modes | **DONE** | 0 | `improvement.tsv` + `metrics.tsv` outputs |
| - Single-host parallel scheduler/resume | **DONE** | 0 | Added `--jobs` + `--resume` with deterministic report rebuild |
| - Qualification cache reuse across iterations | **DONE** | 0 | Added `--reuse-pair-file` + `reused_pairs` metrics/JSON tracking |
| - Detection-order hint reuse | **DONE** | 0 | Added `--reuse-summary-file` + `hinted_mutants`/`hint_hits` metrics |
| - Reuse compatibility manifests and policy | **DONE** | 0 | Added sidecar compat hash manifests + `--reuse-compat-mode` guards |
| - Content-addressed reuse cache | **DONE** | 0 | Added `--reuse-cache-dir` + cache read/read-write modes keyed by compat hash |
| - Yosys-backed mutation list generation | **DONE** | 0 | Added `generate_mutations_yosys.sh` + `--generate-mutations` flow |
| - Multi-mode mutation mix generation | **DONE** | 0 | Added `--mutations-modes` / `--modes` to combine arithmetic/control mutation modes deterministically |
| - Native mutation CLI frontend (`circt-mut`) | IN_PROGRESS | — | Added initial `circt-mut` tool with `cover|matrix|generate` subcommands that dispatch to existing mutation scripts; migration of shell logic into C++ subcommands is pending |
| - Native mutation operator expansion (arithmetic/control-depth) | IN_PROGRESS | — | Added mutate profile presets (`--mutations-profiles`), weighted mode allocations (`--mutations-mode-counts`), deterministic mode-family expansion (`arith/control/balanced/all` -> `inv/const0/const1/cnot0/cnot1`), plus `-cfg`/select controls (`--mutations-cfg`, `--mutations-select`) across generator/cover/matrix; deeper operator families still pending |
| - CI lane integration across AVIP/sv-tests/verilator/yosys/opentitan | IN_PROGRESS | — | Added `run_mutation_matrix.sh` with generated lanes, parallel lane-jobs, reuse-pair/summary pass-through, reuse cache pass-through, reuse-compat policy pass-through, generated-lane mode/profile/mode-count/cfg/select controls, default/lane global formal propagation filters, full default/lane circt-lec global filter controls (`args`, `c1/c2`, `z3`, `assume-known-inputs`, `accept-xprop-only`), default/lane circt-bmc global filter controls (including `ignore_asserts_until`), and default/lane chained LEC/BMC global filtering (`--formal-global-propagate-circt-chain lec-then-bmc|bmc-then-lec|consensus|auto`) with chain telemetry metrics (`chain_lec_unknown_fallbacks`, `chain_bmc_resolved_not_propagated_mutants`, `chain_bmc_unknown_fallbacks`, `chain_lec_resolved_not_propagated_mutants`, `chain_consensus_not_propagated_mutants`, `chain_consensus_disagreement_mutants`, `chain_consensus_error_mutants`, `chain_auto_parallel_mutants`); added built-in differential BMC original-design cache reuse (`.global_bmc_orig_cache`) with `bmc_orig_cache_hit_mutants`/`bmc_orig_cache_miss_mutants` and runtime telemetry (`bmc_orig_cache_saved_runtime_ns`/`bmc_orig_cache_miss_runtime_ns`), bounded cache controls (`--bmc-orig-cache-max-entries`, `--bmc-orig-cache-max-bytes`, `--bmc-orig-cache-max-age-seconds`), configurable count/byte eviction policy (`--bmc-orig-cache-eviction-policy lru|fifo|cost-lru`), age-aware pruning telemetry (`bmc_orig_cache_pruned_age_entries`/`bmc_orig_cache_pruned_age_bytes`, including persisted-cache variants), and cross-run cache publication status (`bmc_orig_cache_write_status`) via `--reuse-cache-dir/global_bmc_orig_cache`; added matrix default/lane cache-limit pass-through controls (`--default-bmc-orig-cache-max-entries`, `--default-bmc-orig-cache-max-bytes`, `--default-bmc-orig-cache-max-age-seconds`, `--default-bmc-orig-cache-eviction-policy`, lane TSV overrides), plus lane selection filters (`--include-lane-regex`, `--exclude-lane-regex`) for targeted CI slicing; BMC orig-cache key now includes original-design SHA-256 to prevent stale reuse when design content changes at the same path; added compatibility-guarded global filter reuse from prior `pair_qualification.tsv` (`test_id=-`) with `reused_global_filters` metric; built-in global filters now conservatively treat formal `UNKNOWN` as propagated (not pruned); full external-suite wiring still pending |
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

### Non-Smoke OpenTitan End-to-End Parity Plan

#### Scope (Required Lanes)
1. `SIM` lane via `utils/run_opentitan_circt_sim.sh` on full-IP targets.
2. `VERILOG` lane via `utils/run_opentitan_circt_verilog.sh --ir-hw` on
   full-IP parse targets.
3. `LEC` lane via `utils/run_opentitan_circt_lec.py` with
   `LEC_SMOKE_ONLY=0` and strict handling of `XPROP_ONLY` by default.

#### Fixed Target Matrix (Parity Gate Set)
1. `SIM`: `gpio`, `uart`, `usbdev`, `i2c`, `spi_host`, `spi_device`.
2. `VERILOG`: `gpio`, `uart`, `usbdev`, `i2c`, `spi_device`, `dma`,
   `keymgr_dpe`.
3. `LEC`: all AES S-Box implementations selected by default in
   `run_opentitan_circt_lec.py` (unmasked by default, masked enabled in
   separate lane).

#### Gate Rules
1. OpenTitan parity claims are allowed only when the full matrix above runs
   through `utils/run_opentitan_formal_e2e.sh` with zero unexpected failures.
2. Smoke-only OpenTitan runs cannot be used as parity evidence.
3. `XPROP_ONLY` results count as failures in strict parity mode.
4. `--allow-xprop-only` is permitted only for transitional tracking and must be
   marked as non-parity status in summaries.

#### Current Integration Status
1. `utils/run_opentitan_formal_e2e.sh` is the canonical non-smoke OpenTitan
   parity runner.
2. `utils/run_formal_all.sh` exposes this lane as `opentitan/E2E` via
   `--with-opentitan-e2e`, with case-level failure export to
   `opentitan-e2e-results.txt` for expected-failure case tracking.
3. All command-level validation and run evidence remain in `CHANGELOG.md`.
4. `VERILOG i2c` and `VERILOG spi_device` singleton-array parse failures are
   closed (LLHD singleton index normalization in MooreToCore).
5. `VERILOG usbdev` parse closure landed (`prim_sec_anchor_*` dependencies).
6. `VERILOG dma` and `VERILOG keymgr_dpe` target support is now implemented in
   `run_opentitan_circt_verilog.sh`.
7. OpenTitan LEC no longer depends on `LEC_ACCEPT_XPROP_ONLY=1` for
   `aes_sbox_canright` in default OpenTitan flow (`LEC_X_OPTIMISTIC` default
   enabled in OpenTitan LEC harness).
8. `SIM i2c` timeout is closed in non-smoke E2E by short-circuiting TL-UL BFM
   response wait when `a_ready` never handshakes.
9. OpenTitan non-smoke E2E gate is currently clean (`pass=12 fail=0`).
10. `run_formal_all.sh` now supports an explicit strict OpenTitan LEC audit
    lane (`opentitan/LEC_STRICT`, `--with-opentitan-lec-strict`) to track the
    remaining strict X-prop parity gap independently from default parity lanes.
11. OpenTitan LEC case artifacts now retain mismatch diagnostics in the
    artifact field (e.g. `#XPROP_ONLY`) for case-level expected-failure
    tracking and strict-lane triage.
12. `run_formal_all.sh` expected-failure case matching now supports regex-based
    selectors (`id_kind=base_regex|path_regex`) for stable strict-lane
    diagnostic tracking across non-deterministic artifact paths.
13. Mutation differential-BMC original-cache policy now includes age-based
    pruning (`--bmc-orig-cache-max-age-seconds`) with cover/matrix telemetry
    for age-specific eviction accounting.
14. Mutation differential-BMC original-cache now supports configurable
    count/byte eviction policy
    (`--bmc-orig-cache-eviction-policy lru|fifo|cost-lru`)
    with matrix default/lane pass-through and runtime telemetry
    (`bmc_orig_cache_saved_runtime_ns`, `bmc_orig_cache_miss_runtime_ns`);
    built-in global-filter UNKNOWN telemetry is exported as
    `global_filter_lec_unknown_mutants` and
    `global_filter_bmc_unknown_mutants`.
15. Mutation matrix now supports lane ID include/exclude regex selectors
    (`--include-lane-regex`, `--exclude-lane-regex`) for targeted CI slices;
    no-global-filter mutation lanes no longer trip missing
    `global_propagate.log` parsing in mutation cover.
16. OpenTitan E2E now exposes explicit LEC X-semantic controls
    (`--lec-x-optimistic`, `--lec-strict-x`, `--lec-assume-known-inputs`);
    `run_formal_all.sh` pins OpenTitan E2E to x-optimistic mode and forwards
    `--lec-assume-known-inputs` into the E2E lane.
17. OpenTitan LEC artifact paths are now deterministic in both formal-all and
    OpenTitan E2E harnesses (`opentitan-lec-work`, `opentitan-lec-strict-work`,
    `opentitan-formal-e2e/lec-workdir`) for stable case-level gating/triage.
18. `run_formal_all.sh` expected-failure case matching now supports
    `id_kind=base_diag` (`<base>#<DIAG>`), enabling stable strict OpenTitan
    diagnostic tracking (`XPROP_ONLY`) without path-regex coupling.
18. Mutation built-in circt-lec/circt-bmc filters now support automatic tool
    discovery (PATH, then `<circt-root>/build/bin`) when paths are omitted or
    set to `auto`, including chained and matrix default flows.
19. Added initial `circt-mut` binary frontend with
    `cover|matrix|generate` subcommands to provide a stable mutation CLI while
    preserving script backend compatibility during migration.

#### Current Open Non-Smoke Gaps (latest parity tracking)
1. Strict non-optimistic (`LEC_X_OPTIMISTIC=0`) 4-state parity for
   `aes_sbox_canright` still reports `XPROP_ONLY`; default OpenTitan LEC path
   now uses x-optimistic equivalence.

#### Closure Workflow
1. Keep one issue per failing lane target with owner, reproducer, and expected
   check-in test.
2. Require each fix to add/extend a lit test when feasible plus re-run the full
   OpenTitan E2E gate.
3. Move target from `failing` to `passing` only after two consecutive clean E2E
   runs with archived artifacts.

#### Tracking and Artifacts
1. Canonical result file: `<out-dir>/results.tsv` from
   `utils/run_opentitan_formal_e2e.sh`.
2. Store per-target logs under `<out-dir>/logs/` and keep failure signatures in
   `CHANGELOG.md`.
3. Track matrix status in a table with columns:
   `lane`, `target`, `status`, `owner`, `blocking_issue`, `last_clean_run`.

### Tabby-CAD-Level Formal Parity Plan (P0-P2)

#### P0 (Baseline Commercial Capability)
1. SVA semantic correctness:
   local vars, `disable iff`, multi-clock edge cases, sampled-value semantics.
2. Proof strength:
   robust unbounded flow (IC3/PDR + k-induction) with deterministic outcomes.
3. 4-state/X-prop soundness:
   consistent BMC/LEC treatment and no parity waivers for core benchmarks.
4. Practical LEC:
   retiming/clock-gating/reset-delta friendly equivalence with clear mismatch
   diagnostics.
5. Constraint soundness:
   over-constraint and contradiction detection with actionable diagnostics.

#### P1 (Adoption and Closure Efficiency)
1. Coverage/vacuity stack in CI outputs.
2. Compositional proving and partitioned closure.
3. Capacity features (abstraction/refinement and scaling controls).
4. Better debug UX (trace minimization, mismatch localization, replay tooling).

#### P2 (Beyond Baseline)
1. Formal app layer (connectivity/security/reset-focused app checks).
2. Advanced liveness/fairness closure flow.
3. Distributed formal execution with deterministic resume/replay.
4. Cross-run analytics and trend-based release gates.

#### Test and Progress Framework (Mandatory)
1. Tiered test model:
   unit/lit semantic tests, differential corpus tests, full-suite external
   regressions (`~/sv-tests`, `~/verilator-verification`,
   `~/yosys/tests/sva`, `~/mbit/*avip*`, `~/opentitan`).
2. Each epic (`P0-*`, `P1-*`, `P2-*`) must define:
   `entry criteria`, `exit criteria`, `required suites`, `required metrics`.
3. Progress metrics:
   semantic mismatch count, non-smoke OpenTitan fail count, strict LEC
   `XPROP_ONLY` count, full-suite pass rates, flaky rate.
4. Status discipline:
   roadmap intent in `PROJECT_PLAN.md`; command-level evidence and results in
   `CHANGELOG.md`.
