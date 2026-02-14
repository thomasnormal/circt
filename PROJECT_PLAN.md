# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/mbit/*_avip` testbenches using only CIRCT tools and the library ~/uvm-core.
Secondary goal: Get to 100% in the ~/sv-tests/ and ~/verilator-verification/ test suites.

---

## Simulation Workstream (circt-sim) — February 12, 2026

### Current Status
- **sv-tests simulation**: 856 pass, 0 xfail, 1 compile-only, 9 skip, 7 xpass (Feb 10)
- **circt-sim unit tests**: 230/230 (165 pass, 65 timeout-UVM, 0 failures)
- **ImportVerilog tests**: 268/268 (100%)
- **AVIP dual-top**: APB runs with full virtual sequence + sub-sequence dispatch, 0 UVM_FATAL, 0 UVM_ERROR, BFM drives IDLE→SETUP. Coverage 0% (BFM stuck at SETUP→ACCESS transition).
- **Performance**: ~171 ns/s simulated time

### Recently Completed (Iteration 1171, Feb 12, 2026)
1. **randomize() vtable corruption fix**: `__moore_randomize_basic` was filling entire class object memory with random bytes, corrupting class_id (bytes 0-3) and vtable pointer (bytes 4-11). Now a no-op — individual rand fields set by `_with_range`/`_with_dist`. This fixed sub-sequence body dispatch (factory-created sequences now call derived body()).
2. **Blocking finish_item / item_done handshake** (iter 1165): Direct-wake mechanism, process-level retry fix.
3. **Phase hopper objection fix**: Broadened interceptors for `uvm_phase_hopper::` variants.
4. **`wait_for` wasEverRaised tracking**: Prevents premature phase completion.
5. **Dual-top sim.terminate handling**: Graceful shutdown when HVL finishes before HDL.
6. **Stack size increase**: 64MB stack for deep UVM sequence nesting.
7. **Diagnostic cleanup**: Removed all 21 DIAG-* debug logging blocks.
8. **config_db native write fix for dynamic arrays**: `uvm_config_db::get` writeback now reliably updates heap-backed dynamic-array slots (`new[]`) across all interpreter interception paths; added `test/Tools/circt-sim/config-db-dynamic-array-native.sv`.

### Feature Gap Table (Simulation)

| Feature | Status | Notes |
|---------|--------|-------|
| UVM phase sequencing | DONE | All 9 function phase IMPs complete in order |
| Phase hopper objections | DONE | get/raise/drop/wait_for for uvm_phase_hopper |
| VIF shadow signals | DONE | Interface field propagation + store interception |
| Sequencer interface | DONE | start_item/finish_item(blocking)/get/item_done; direct-wake handshake |
| Analysis port write | DONE | Chain-following BFS dispatch via vtable |
| Associative array deep copy | DONE | Prevents UVM phase livelock |
| Runtime vtable override | DONE | All 3 call_indirect paths check runtime vtable |
| Per-process RNG | DONE | IEEE 1800-2017 §18.13 random stability |
| Coverage collection | DONE | Covergroups + coverpoints + iff guards |
| Constraint solver | DONE | Soft/hard, inheritance, inline, dynamic bounds |
| Stack size (64MB) | DONE | Handles deep UVM sequence nesting |
| Sub-sequence body dispatch | DONE | Fixed by randomize_basic no-op (vtable preservation) |
| BFM/driver transactions | DONE | finish_item blocks until item_done; direct-wake handshake |
| SVA concurrent assertions | NOT STARTED | 26 compile-only tests (deferred) |
| BFM APB state machine | IN PROGRESS | IDLE→SETUP works, SETUP→ACCESS blocked (investigating) |
| Multi-AVIP coverage | IN PROGRESS | Handshake done; needs AVIP recompile + end-to-end test |

### Next Steps (Simulation)
1. **Fix BFM SETUP→ACCESS transition**: The APB BFM drives IDLE→SETUP but never transitions to ACCESS (penable=1). Investigate whether a clock edge wait or signal drive is being missed in the interpreter.
2. **Recompile AVIPs**: All AVIPs (APB, AHB, SPI, I2S, I3C, JTAG, AXI4, AXI4Lite, UART) need recompilation with latest circt-verilog to include all recent fixes. Then run end-to-end to verify coverage.
3. **Performance optimization**: Target >500 ns/s for practical AVIP runs.
4. **SVA concurrent assertions**: Needed for 26 compile-only SVA tests.

### Known Limitations (Simulation)
- BFM APB state machine: SETUP→ACCESS transition not completing (coverage remains 0%)
- AVIPs need recompilation with latest circt-verilog for end-to-end coverage testing
- SVA concurrent assertions not simulated (26 tests compile-only)
- Xcelium APB reference: 21-30% coverage, 130ns sim time — our target baseline

---

## Formal Workstream (circt-mut) — February 12, 2026

Historical note: prior `### Formal Closure Snapshot Update (...)` entries were
migrated to `CHANGELOG.md` under `Historical Migration - February 14, 2026`.

### Milestone Progress Update (February 14, 2026)

1. Launch-telemetry strict governance completed for formal lanes:
   - absolute launch event limits (`--fail-on-any-{bmc,lec}-launch-events`)
   - bounded launch event limits (`--max-{bmc,lec}-launch-event-rows`)
   - launch reason-key drift checks
     (`--fail-on-new-{bmc,lec}-launch-reason-keys`)
2. Launch reason-key allowlist support is now first-class and strict-gate aware:
   - `--{bmc,lec}-launch-reason-key-allowlist-file`
   - allowlisted reason keys no longer cause strict-gate prefix drift on
     dynamic `*_launch_reason_*_events` counters.
3. OpenTitan FPV target-selection governance completed:
   - added baseline/update/fail/allowlist controls for
     `opentitan-fpv-target-manifest.tsv` drift.
   - strict-gate now auto-enforces target-manifest drift when a target-manifest
     baseline is configured.
4. Remaining limitation vs Jasper/VCF-style mature flows:
   - per-assertion semantic status parity remains partial (especially vacuity/
     unreachable classification and richer task-level diagnostics).
5. Next formal milestone (OpenTitan-aligned, backend-generic):
   - strengthen compile-contract parity and FPV task-policy coverage:
     - tighten stopat/blackbox task-profile propagation checks (`FpvSecCm`).
     - add strict drift governance for task-policy metadata alongside compile
       contracts and target manifests.

### OpenTitan DVSIM-Equivalent Formal Plan (CIRCT Backend) — February 14, 2026

#### Goal

Enable a CIRCT-native command flow that is operationally equivalent to:

- `util/dvsim/dvsim.py hw/top_earlgrey/formal/top_earlgrey_fpv_*.hjson --select-cfgs ...`
- `util/dvsim/dvsim.py hw/top_earlgrey/formal/chip_conn_cfg*.hjson`

while keeping the backend generic (reusable across OpenTitan, sv-tests,
verilator-verification, and yosys corpora).

#### Architecture Commitments (long-term)

1. Build a generic formal orchestration core with pluggable project adapters.
2. Treat OpenTitan as the first full adapter, not a hard-coded one-off path.
3. Keep artifacts schema-versioned and strict-gate-governed from day one.
4. Preserve deterministic run identity (inputs, options, contracts, toolchain).

#### 1) Native OpenTitan HJSON ingestion with `--select-cfgs` semantics

1. Deliverables:
   - new adapter module for OpenTitan formal cfg ingestion
     (target cfg HJSON + imported cfgs).
   - canonical target-selection semantics matching dvsim behavior:
     - `--select-cfgs` exact target names
     - category cfg expansion (`ip`, `prim`, `sec_cm`)
     - deterministic selection order and dedup.
   - emitted normalized manifest:
     - `formal-target-manifest.tsv/json` (target, dut, flow kind, task, rel_path,
       fusesoc core, cfg source).
2. Implementation steps:
   - parse HJSON graph and `import_cfgs` with cycle checks.
   - normalize cfg entries into a common lane model consumed by formal runners.
   - add CLI entry-point compatible subset:
     - `run_formal_all.sh --opentitan-fpv-cfg <hjson> --select-cfgs <list>`.
3. Tests:
   - lit parser fixtures for malformed/ambiguous cfgs.
   - parity fixtures against known OpenTitan cfg subsets.
   - strict-gate checks for selection drift (`target list` and `target metadata`).
4. Exit criteria:
   - selecting any target from `top_earlgrey_fpv_ip_cfgs.hjson`,
     `top_earlgrey_fpv_prim_cfgs.hjson`, and `top_earlgrey_fpv_sec_cm_cfgs.hjson`
     yields deterministic target manifests with stable IDs.

#### 2) Full FuseSoC/dvsim-style filelist + defines/include resolution

1. Deliverables:
   - formal input resolution layer that produces canonical compile contracts:
     - ordered fileset
     - include dirs
     - defines
     - top module, package roots, generated files.
   - per-target resolved compile artifact:
     - `resolved-compile-contract.tsv/json`.
2. Implementation steps:
   - implement adapter-backed resolution path for OpenTitan FPV targets
     (FuseSoC core + cfg imports + overrides).
   - preserve exact ordering semantics and generated-file handling.
   - include contract fingerprint in strict-gate baseline rows.
3. Tests:
   - fixture-based parity tests comparing resolved contracts against reference
     manifests for representative `ip`, `prim`, and `sec_cm` targets.
   - regression tests for include/define precedence and duplicate suppression.
4. Exit criteria:
   - resolved compile contracts are stable and reproducible per target, and
     strict-gate detects any unintended contract drift.

#### 3) FPV task support parity (`task`, `stopat`, `blackbox`, sec_cm flow)

1. Deliverables:
   - task profile model in formal lane contracts:
     - default FPV
     - `FpvSecCm`
     - connectivity-specific profile hooks.
   - stopat/blackbox transformation support for sec_cm workflows.
2. Implementation steps:
   - implement task-profile interpreter from cfg metadata.
   - add stopat injection pipeline:
     - stopat net selection and symbolic fault-driving wrapper generation.
   - add blackbox policy support:
     - designated modules replaced with abstract stubs during formal elaboration.
   - version and export effective task policy per case in resolved contracts.
3. Tests:
   - synthetic sec_cm fixtures asserting stopat/blackbox policy activation.
   - OpenTitan sec_cm smoke targets with policy provenance assertions.
   - strict-gate checks for task-policy drift.
4. Exit criteria:
   - sec_cm targets run through CIRCT with explicit policy provenance
     (`task`, `stopats`, `blackboxes`) and deterministic outcomes.

#### 4) FPV-compatible per-assertion reporting

1. Deliverables:
   - per-assertion result artifact:
     - `assertion-results.tsv/json` with stable assertion IDs and source locs.
   - summary counters aligned with FPV expectations:
     - proven, failing, vacuous, covered, unreachable, timeout, unknown.
2. Implementation steps:
   - attribute assertion/check IDs through lowering and BMC pipelines.
   - execute per-property classification runs as needed:
     - bounded check
     - induction check (where configured)
     - cover reachability checks for vacuity/coverage classification.
   - add report adapters producing FPV-like summary views.
3. Tests:
   - targeted assertion micro-benches for each status class.
   - OpenTitan module subset with known assertion outcomes.
   - strict-gate rules on assertion-status deltas by assertion ID.
4. Exit criteria:
   - target-level reports are assertion-granular and machine-comparable across
     runs; strict-gate flags assertion-level regressions, not only lane totals.

#### 5) Connectivity formal integration (`chip_conn_cfg*.hjson` + CSV)

1. Deliverables:
   - connectivity adapter that ingests chip conn cfg HJSON and CSV rules.
   - generated connectivity check suite and per-connection result artifacts.
2. Implementation steps:
   - parse connectivity CSV schema into normalized connection contracts.
   - synthesize formal checks (assert/cover assumptions) from connection rules.
   - route through shared orchestration pipeline with lane/task metadata.
3. Tests:
   - synthetic CSV fixture tests for parser + check generation.
   - OpenTitan chip connectivity subset validation with deterministic summaries.
   - strict-gate drift checks for connection-level IDs/results.
4. Exit criteria:
   - `chip_conn_cfg*.hjson` flows execute with connection-level pass/fail reports
     and strict-gate compatibility.

#### 6) Scalability model for large formal target sets

1. Deliverables:
   - shardable execution planner across targets and assertion groups.
   - run-local immutable toolchain pinning and input contract pinning.
   - content-addressed cache layers for frontend/elaboration artifacts.
2. Implementation steps:
   - add scheduler model:
     - shard by target, then by assertion batches for heavy jobs.
   - add robust process isolation:
     - run-local tool copies, per-shard workdirs, bounded retries.
   - add artifact durability:
     - normalized pathing, schema-versioned reports, resumable checkpoints.
3. Tests:
   - stress fixtures for parallel lane execution and shard replay.
   - deterministic artifact hash checks under concurrent execution.
   - throughput regressions and timeout-budget trend checks in strict-gate.
4. Exit criteria:
   - multi-target OpenTitan FPV batches run reproducibly with bounded variance,
     stable artifacts, and actionable strict-gate output.

#### Execution Phases and Milestones

1. Phase A (ingestion + selection):
   - complete item 1, deliver `--select-cfgs` parity and target manifests.
2. Phase B (compile contract parity):
   - complete item 2, deliver resolved compile contracts with fingerprints.
3. Phase C (task parity, sec_cm):
   - complete item 3, deliver stopat/blackbox policy execution + provenance.
4. Phase D (assertion-granular FPV reports):
   - complete item 4, deliver per-assertion status and drift gating.
5. Phase E (connectivity):
   - complete item 5, deliver chip connection cfg+CSV execution path.
6. Phase F (scale hardening):
   - complete item 6, deliver sharding, artifact stability, and runtime SLAs.

#### Execution Status (February 14, 2026)

1. Phase A is complete:
   - `run_formal_all.sh` supports `--opentitan-fpv-cfg` + `--select-cfgs`
     selection semantics with deterministic target manifests.
   - `--opentitan-fpv-cfg` is now repeatable, enabling deterministic
     multi-bundle target selection across FPV category cfgs (`ip`, `prim`,
     `sec_cm`) in a single planning run.
   - duplicate target names across cfg bundles are now fail-closed when payload
     metadata diverges, preventing ambiguous target resolution.
   - unfiltered FPV execution is now available as an explicit operator choice
     (`--opentitan-fpv-allow-unfiltered`) with target-budget governance
     (`--opentitan-fpv-max-targets`).
   - `utils/select_opentitan_formal_cfgs.py` resolves cfg import graphs and
     emits stable selected-target artifacts.
2. Phase B is complete:
   - FuseSoC-backed compile-contract resolution is integrated.
   - OpenTitan FPV execution lane bootstrap is landed:
     - `--with-opentitan-fpv-bmc` (`opentitan/FPV_BMC`)
     - contracts are executable via `utils/run_opentitan_fpv_circt_bmc.py`.
3. Phase C core execution semantics are landed:
   - new generic HW pass `--hw-externalize-modules` converts selected
     `hw.module` symbols to `hw.module.extern`.
   - OpenTitan FPV BMC runner now applies task blackbox policy via
     `BMC_PREPARE_CORE_PASSES` + `hw-externalize-modules`.
   - task stopats (including hierarchical selectors) are now lowered through
     `--hw-stopat-symbolic` and exercised end-to-end in OpenTitan FPV runner
     and `run_formal_all` lanes.
4. Phase D is in progress:
   - OpenTitan FPV runner supports FPV-style assertion summary artifacts.
   - generic pairwise BMC runner now has optional assertion-granular execution
     (`BMC_ASSERTION_GRANULAR=1`) with per-assertion result artifacts.
   - assertion-granular classification now consumes explicit
     `BMC_ASSERTION_STATUS` tags, including first-class `VACUOUS` rows.
   - cover-granular execution (`BMC_COVER_GRANULAR=1`) now provides concrete
     `covered` / `unreachable` evidence for FPV summaries.
   - `run_formal_all.sh` now forwards OpenTitan FPV summary-drift governance
     controls (`baseline/drift/allowlist/fail/update`) into `opentitan/FPV_BMC`,
     with strict-gate auto-enabling drift failure when a baseline is configured.
   - OpenTitan FPV BMC launch-retry/fallback telemetry is now surfaced through
     pairwise artifacts (`BMC_LAUNCH_EVENTS_OUT`) and summarized in
     `run_formal_all.sh` lane metrics (`bmc_launch_*` counters).
   - non-FPV launch telemetry surfacing is now wired for standard formal lanes:
     - `run_sv_tests_circt_bmc.sh` emits `BMC_LAUNCH_EVENTS_OUT`.
     - `run_verilator_verification_circt_bmc.sh` and
       `run_yosys_sva_circt_bmc.sh` now emit `BMC_LAUNCH_EVENTS_OUT` with
       retry/fallback events for frontend launch retries and copy-fallback.
     - `run_sv_tests_circt_lec.sh`,
       `run_verilator_verification_circt_lec.sh`, and
       `run_yosys_sva_circt_lec.sh` emit `LEC_LAUNCH_EVENTS_OUT`.
     - `run_formal_all.sh` now forwards and summarizes:
       - `bmc_launch_*` in `sv-tests/BMC` and `sv-tests-uvm/BMC_SEMANTICS`
       - `bmc_launch_*` in `verilator-verification/BMC` and
         `yosys/tests/sva/BMC`
       - `lec_launch_*` in `sv-tests/LEC`,
         `verilator-verification/LEC`, and `yosys/tests/sva/LEC`.
   - strict-gate launch-counter policy coupling is now enabled by default:
     - `--strict-gate` auto-enables `--fail-on-new-bmc-counter-prefix bmc_launch_`
       and `--fail-on-new-lec-counter-prefix lec_launch_`.
5. Phase E execution lane is landed:
   - new connectivity adapter utility:
     - `utils/select_opentitan_connectivity_cfg.py`
   - supports OpenTitan connectivity cfg import-graph composition and
     placeholder expansion (`{proj_root}`, `{conn_csvs_dir}`, etc.).
   - parses connectivity CSV rows into normalized rule contracts:
     - `CONNECTION`
     - `CONDITION`
   - emits deterministic artifacts:
     - `opentitan-connectivity-target-manifest.tsv`
     - `opentitan-connectivity-rules-manifest.tsv`
   - `run_formal_all.sh` now supports connectivity planning lane wiring:
     - `--with-opentitan-connectivity-parse` (`opentitan/CONNECTIVITY_PARSE`)
     - `--opentitan-connectivity-cfg`
     - `--opentitan-connectivity-target-manifest`
     - `--opentitan-connectivity-rules-manifest`
   - connectivity BMC execution lane is now wired:
     - `--with-opentitan-connectivity-bmc` (`opentitan/CONNECTIVITY_BMC`)
     - `--opentitan-connectivity-rule-filter`
     - `--opentitan-connectivity-bmc-rule-shard-count`
     - `--opentitan-connectivity-bmc-rule-shard-index`
   - connectivity BMC now emits connectivity-status evidence beyond pass/fail:
     - per-rule guard-activation cover objectives are synthesized with each
       generated connectivity checker.
     - `utils/run_opentitan_connectivity_circt_bmc.py` now supports
       cover-granular delegation into `run_pairwise_circt_bmc.py`
       (`BMC_COVER_GRANULAR` + deterministic cover shard controls).
     - `run_formal_all.sh` now records connectivity BMC cover counters in lane
       summaries (`bmc_cover_total`, `bmc_cover_covered`,
       `bmc_cover_unreachable`, ...).
   - connectivity BMC status drift governance is now wired:
     - per-rule status summary artifact:
       - `opentitan-connectivity-bmc-status-summary.tsv`
     - baseline/drift/allowlist/fail/update controls through
       `run_formal_all.sh`:
       - `--opentitan-connectivity-bmc-status-baseline-file`
       - `--opentitan-connectivity-bmc-status-drift-file`
       - `--opentitan-connectivity-bmc-status-drift-allowlist-file`
       - `--update-opentitan-connectivity-bmc-status-baseline`
       - `--fail-on-opentitan-connectivity-bmc-status-drift`
     - strict-gate auto-enables connectivity BMC status drift failure when a
       baseline is configured.
   - connectivity LEC execution lane is now wired:
     - `--with-opentitan-connectivity-lec` (`opentitan/CONNECTIVITY_LEC`)
     - `--opentitan-connectivity-rule-filter`
     - `--opentitan-connectivity-lec-rule-shard-count`
     - `--opentitan-connectivity-lec-rule-shard-index`
   - connectivity LEC status drift governance is now wired:
     - per-rule status summary artifact:
       - `opentitan-connectivity-lec-status-summary.tsv`
     - baseline/drift/allowlist/fail/update controls through
       `run_formal_all.sh`:
       - `--opentitan-connectivity-lec-status-baseline-file`
       - `--opentitan-connectivity-lec-status-drift-file`
       - `--opentitan-connectivity-lec-status-drift-allowlist-file`
       - `--update-opentitan-connectivity-lec-status-baseline`
     - `--fail-on-opentitan-connectivity-lec-status-drift`
     - strict-gate auto-enables connectivity LEC status drift failure when a
       baseline is configured.
   - connectivity cross-lane status parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_status_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-status-parity-file`
       - `--opentitan-connectivity-status-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-status-parity`
     - strict-gate auto-enables parity failure when both connectivity lanes
       are active.
   - connectivity cross-lane contract-fingerprint parity governance is now
     wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_contract_fingerprint_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_CONTRACT_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-contract-parity-file`
       - `--opentitan-connectivity-contract-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-contract-parity`
     - strict-gate auto-enables contract-parity failure when both
       connectivity lanes are active.
   - connectivity cross-lane cover-counter parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_cover_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_COVER_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-cover-parity-file`
       - `--opentitan-connectivity-cover-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-cover-parity`
     - checker compares only shared cover counters across lanes (no synthetic
       placeholders); strict-gate auto-enables cover-parity failure when both
       connectivity lanes are active.
   - connectivity cross-lane objective-level parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_objective_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_OBJECTIVE_PARITY`
   - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-objective-parity-file`
       - `--opentitan-connectivity-objective-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-objective-parity`
       - `--opentitan-connectivity-objective-parity-include-missing`
       - `--opentitan-connectivity-objective-parity-missing-policy`
    - checker now emits structured objective metadata rows
      (`objective_id/objective_class/objective_key/rule_id/kind`) and compares
      normalized objective-level statuses across shared objective IDs by
      default, with explicit missing-objective policy controls
      (`ignore|case|all`).
   - new connectivity runner:
     - `utils/run_opentitan_connectivity_circt_bmc.py`
     - `utils/run_opentitan_connectivity_circt_lec.py`
   - `CONNECTION` rules are synthesized into bind-check cases and executed via
     generic `run_pairwise_circt_bmc.py` with deterministic rule sharding.
   - `CONDITION` rows are now consumed as guard semantics on the owning
     connectivity assertion:
     - conditions are associated to the preceding `CONNECTION` in CSV order.
     - `--opentitan-connectivity-rule-filter` now matches both connection rule
       IDs/names and associated condition rule IDs/names.
     - guarded implication lowering uses:
       - conjunction of all expected-true condition checks
       - OR fallback over expected-false condition checks (when provided).
6. Phase F scalability bootstrap is landed for deterministic sharded execution:
   - `run_opentitan_fpv_circt_bmc.py` now supports deterministic target sharding
     (`--target-shard-count`, `--target-shard-index`).
   - `run_opentitan_fpv_circt_bmc.py` now forwards deterministic per-target case
     sharding into pairwise execution:
     - `--case-shard-count`
     - `--case-shard-index`
   - `run_pairwise_circt_bmc.py` now supports deterministic sharding at three
     levels:
     - case-level (`--case-shard-count`, `--case-shard-index`)
     - assertion-level (`--assertion-shard-count`, `--assertion-shard-index`)
     - cover-level (`--cover-shard-count`, `--cover-shard-index`)
   - `run_formal_all.sh` now forwards OpenTitan FPV shard controls via:
     - `--opentitan-fpv-bmc-target-shard-count`
     - `--opentitan-fpv-bmc-target-shard-index`
     - `--opentitan-fpv-bmc-case-shard-count`
     - `--opentitan-fpv-bmc-case-shard-index`
   - empty shard partitions now return deterministic zero-work success, which
     enables stable static sharding across heterogeneous target counts.

#### Success Definition (program-level)

1. CIRCT can run selected OpenTitan FPV targets from official formal cfg HJSON
   with dvsim-equivalent target selection semantics.
2. Reports are assertion-granular and strict-gate-enforced.
3. Connectivity cfg+CSV flows run in the same governance plane.
4. The implementation remains adapter-based and reusable outside OpenTitan.

### Remaining Formal Limitations (BMC/LEC/mutation focus)

1. **FPV status derivation depth gap**: explicit assertion statuses (`PROVEN/FAILING/VACUOUS/UNKNOWN`) and cover-granular `covered/unreachable` evidence are now wired, but fully automatic vacuity/coverage derivation for targets that do not emit explicit status tags is still pending.
2. **BMC/LEC operational robustness**: launcher retry/copy-fallback controls are now telemetry-visible in OpenTitan FPV and non-FPV formal lanes (`bmc_launch_*`, `lec_launch_*`), and strict-gate now enforces launch-counter regression via prefix gates; remaining work is absolute-budget/nonzero policies with per-reason allowlisting for sustained multi-agent contention.
3. **Frontend triage ergonomics**: sv-tests BMC now preserves frontend error logs via `KEEP_LOGS_DIR`, and launch retry is in place for transient launcher failures; host-side tool relink contention can still surface as launcher-level `Permission denied`/ETXTBSY noise until binaries stabilize.
4. **Frontend scalability blocker on semantic closure buckets**: `sv-tests` UVM `16.11` (sequence-subroutine) and `16.13` (multiclock) currently hit frontend OOM in `circt-verilog` during import; this blocks clean semantic closure measurement for those buckets.
5. **Assertion/cover-granular scalability gap**: deterministic objective sharding is now available, but adaptive batch sizing, runtime-budget aware shard planning, and strict-gate policy guardrails for large targets are still pending.
6. **LEC provenance parity**: BMC resolved-contract fingerprinting is stronger than LEC/mutation lanes; strict-gate cross-lane provenance equivalence remains incomplete.
7. **Mutation cross-lane governance**: mutation strict gates are lane-scoped, but deeper policy coupling to BMC/LEC semantic buckets and resolved contracts is still pending.
8. **Connectivity parity depth gap**: connectivity cross-lane parity now covers
   per-rule case counters, resolved-contract fingerprints, shared
   cover-counters, and objective-level status drift with structured metadata
   and explicit missing-objective policies; OpenTitan FPV BMC now also has
   evidence-vs-summary objective parity governance in the same strict-gate
   plane; remaining gap is cross-lane FPV objective parity (BMC vs LEC) over
   vacuous/covered/unreachable classes once FPV LEC evidence artifacts land.

### Next Long-Term Features (best long-term path)

1. Add absolute launch-event budget policies (nonzero/threshold) for BMC/LEC launch counters with per-reason allowlisting, then extend retry classification beyond ETXTBSY (selected transient I/O launch races).
2. Extend resolved-contract artifact/fingerprint semantics to LEC and mutation runners, then enforce strict-gate drift checks on shared `(case_id, fingerprint)` tuples.
3. Add dedicated OpenTitan+sv-tests semantic-closure dashboards in strict-gate summaries (multiclock/sequence-subroutine/disable-iff/local-var buckets) to drive maturity from semantic evidence, not pass-rate alone.
4. Implement FPV cross-lane objective parity (BMC vs LEC) on normalized
   assertion/cover evidence classes (`vacuous`, `covered`, `unreachable`) and
   enforce strict-gate drift policies with allowlist support.
