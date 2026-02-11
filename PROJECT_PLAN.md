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
| - Native mutation CLI frontend (`circt-mut`) | IN_PROGRESS | — | Added native `circt-mut` `init|run|report|cover|matrix|generate` flows; target architecture is MCY/Certitude-style campaign UX (`init`/project config + `run`/`report`-grade flows) with staged migration of script logic into native C++ subcommands. `circt-mut init` bootstraps campaign templates (`circt-mut.toml`, `tests.tsv`, `lanes.tsv`) with overwrite guards (`--force`), `circt-mut run` consumes project config to dispatch native-preflight-backed `cover`/`matrix` flows (`--mode cover|matrix|all`) and now supports generated-mutation config (`generate_mutations` + mode/mode-count/mode-weight/profile/cfg/select/Yosys keys), native cover prequalify/probe pass-through keys (`native_global_filter_prequalify`, `native_global_filter_prequalify_only`, `native_global_filter_prequalify_pair_file`, `native_global_filter_probe_mutant`, `native_global_filter_probe_log`), plus expanded formal/gate config pass-through (including strict boolean parsing for both gate toggles and cover formal bool flags), and `circt-mut report` now aggregates cover/matrix artifacts into normalized key/value campaign summaries (stdout and optional `--out` TSV) including formal-global-filter telemetry rollups (timeouts/unknowns/chain/runtime/cache metrics) across matrix lanes and baseline comparison (`--compare`) with numeric diff rows (`diff.<metric>.delta`/`pct_change`) plus regression gate thresholds (`--fail-if-delta-gt`, `--fail-if-delta-lt`), history snapshot workflows (`--compare-history-latest`, `--append-history`), native history-trend summaries/gates (`--trend-history`, `--trend-window`, `--fail-if-trend-delta-gt`, `--fail-if-trend-delta-lt`), and policy bundles (`--policy-profile formal-regression-basic|formal-regression-trend|formal-regression-matrix-basic|formal-regression-matrix-trend|formal-regression-matrix-guard|formal-regression-matrix-trend-guard|formal-regression-matrix-guard-smoke|formal-regression-matrix-guard-nightly|formal-regression-matrix-guard-strict|formal-regression-matrix-stop-on-fail-strict|formal-regression-matrix-full-lanes-strict`). Native `circt-mut generate` executes Yosys mutation-list generation directly (mode/mode-count/mode-weight/profile/cfg/select/top-up dedup) and includes native `--cache-dir` behavior (content-addressed cache hit/miss, metadata-based saved-runtime reporting, lock wait/contended telemetry), with script fallback for unsupported future flags. `circt-mut cover` now performs native global-filter preflight: built-in tool resolution/rewrite for `--formal-global-propagate-circt-lec` / `--formal-global-propagate-circt-bmc` (including bare `auto` forms), built-in Z3 resolution for `--formal-global-propagate-z3` / `--formal-global-propagate-bmc-z3`, chain-mode validation/default engine injection (`--formal-global-propagate-circt-chain`), cover mutation-source consistency checks (`--mutations-file` vs `--generate-mutations`), mutation-generator Yosys resolution (`--mutations-yosys`), generated-mutation mode/profile/allocation/seed validation (`--mutations-modes`, `--mutations-profiles`, `--generate-mutations`, `--mutations-seed`, `--mutations-mode-counts`, `--mutations-mode-weights`, including mode-name checks), early mode-conflict diagnostics, native numeric/cache validation for cover formal controls (`--formal-global-propagate-*_timeout-seconds`, `--formal-global-propagate-bmc-bound`, `--formal-global-propagate-bmc-ignore-asserts-until`, `--bmc-orig-cache-max-*`, `--bmc-orig-cache-eviction-policy`), and PATH-accurate `timeout` preflight for non-zero active global-filter timeout settings, plus native single-mutant runtime global-filter probing (`--native-global-filter-probe-mutant`, optional `--native-global-filter-probe-log`) for command-mode and built-in global filters, and native campaign prequalification handoff (`--native-global-filter-prequalify`, optional `--native-global-filter-prequalify-pair-file`) for command-mode and built-in circt-lec/circt-bmc/chain classification feeding script `--reuse-pair-file` dispatch across both static and generated mutation sources, with a no-test-dispatch mode (`--native-global-filter-prequalify-only`) for formal-only batch triage and `--jobs`-parallelized prequalification that preserves deterministic pair-row ordering. `circt-mut matrix` now performs the analogous default-global-filter preflight/rewrite (`--default-formal-global-propagate-cmd`, `--default-formal-global-propagate-circt-{lec,bmc,chain}` plus default Z3 options), native default Yosys resolution (`--default-mutations-yosys`), native default generated-mutation mode/profile/allocation/seed validation (`--default-mutations-modes`, `--default-mutations-profiles`, `--default-mutations-mode-counts`, `--default-mutations-mode-weights`, `--default-mutations-seed`, including mode-name checks), lane mutation-source consistency checks (`mutations_file` vs `generate_count`), lane generated-mutation preflight from `--lanes-tsv` (modes/profiles, yosys, `generate_count`, `mutations_seed`, mode-count/weight syntax/conflict/total/mode-name checks), lane formal tool preflight (`global_propagate_cmd`, `global_propagate_circt_lec`, `global_propagate_circt_bmc`, `global_propagate_z3`, `global_propagate_bmc_z3`) plus lane timeout/cache/gate override validation (timeouts, BMC bound/ignore-assert, BMC orig-cache limits/policy, lane skip/fail booleans), lane formal boolean validation (`global_propagate_assume_known_inputs`, `global_propagate_accept_xprop_only`, `global_propagate_bmc_run_smtlib`, `global_propagate_bmc_assume_known_inputs` with `1|0|true|false|yes|no|-`), native validation for matrix default numeric/cache controls (`--default-formal-global-propagate-*_timeout-seconds`, `--default-formal-global-propagate-bmc-bound`, `--default-formal-global-propagate-bmc-ignore-asserts-until`, `--default-bmc-orig-cache-max-*`, `--default-bmc-orig-cache-eviction-policy`), PATH-accurate `timeout` preflight for non-zero effective default/lane timeout settings with active effective global-filter modes, native matrix lane prequalification dispatch (`--native-global-filter-prequalify`) that runs per-lane native prequalify-only, materializes lane reuse pair files, dispatches script matrix with a rewritten lanes manifest, exports aggregated matrix prequalification telemetry counters (`native_matrix_prequalify_summary_lanes`, `native_matrix_prequalify_summary_missing_lanes`, `native_matrix_prequalify_total_mutants`, `native_matrix_prequalify_not_propagated_mutants`, `native_matrix_prequalify_propagated_mutants`, plus error/cmd-source counters), forwards matrix gate defaults (`--skip-baseline`, `--fail-on-undetected`, `--fail-on-errors`) into native lane cover dispatch with per-lane TSV override precedence (`skip_baseline`, `fail_on_undetected`, `fail_on_errors`), supports native lane selection filters (`--include-lane-regex`, `--exclude-lane-regex`) with regex validation and deterministic lane skipping before dispatch, supports native lane-level parallel dispatch (`--jobs`) with deterministic `results.tsv` row ordering and `native_matrix_dispatch_lane_jobs` telemetry, and now reports explicit SKIP accounting in campaign summaries (`matrix.lanes_skip`, `matrix.gate_skip`) for native stop-on-fail cut lanes plus always-on skip-budget counters (`matrix.skip_budget_rows_total`, `matrix.skip_budget_rows_stop_on_fail`, `matrix.skip_budget_rows_non_stop_on_fail`). Default mutation materialization no longer depends on `~/mcy/scripts/create_mutated.sh` (now uses in-repo `utils/create_mutated_yosys.sh`), and mutation scripts are installed to `<prefix>/share/circt/utils` for compatibility during migration. Next steps: migrate full matrix lane scheduling loops natively, expand skip-lane budget policy bundles across compare/trend governance, publish policy-pack docs with recommended profile mapping for smoke/nightly/strict regressions, wire lane-trend policy bundles into CI history workflows by default, and adopt `--history --history-bootstrap` in first-run bootstrap jobs. |
| - Native mutation operator expansion (arithmetic/control-depth) | IN_PROGRESS | — | Added mutate profile presets (`--mutations-profiles`, including `fault-basic`/`fault-stuck`/`fault-connect`), weighted mode allocations (`--mutations-mode-counts`, `--mutations-mode-weights`), deterministic mode-family expansion (`arith/control/balanced/all/stuck/invert/connect` -> concrete mutate modes), deterministic seed-rotated remainder allocation across both top-level mode groups and concrete family expansion (`--mutations-seed`) in native/script generators, strict mode-name validation in both native and legacy generator paths (`--mode`/`--modes`, mode-count/weight keys), plus `-cfg`/select controls (`--mutations-cfg`, `--mutations-select`) across generator/cover/matrix; deeper operator families still pending |
| - CI lane integration across AVIP/sv-tests/verilator/yosys/opentitan | IN_PROGRESS | — | Added `run_mutation_matrix.sh` with generated lanes, parallel lane-jobs, reuse-pair/summary pass-through, reuse cache pass-through, reuse-compat policy pass-through, generated-lane mode/profile/mode-count/mode-weight/cfg/select controls, default/lane global formal propagation filters, full default/lane circt-lec global filter controls (`args`, `c1/c2`, `z3|auto`, `assume-known-inputs`, `accept-xprop-only`), default/lane circt-bmc global filter controls (including `ignore_asserts_until`, `z3|auto`), and default/lane chained LEC/BMC global filtering (`--formal-global-propagate-circt-chain lec-then-bmc|bmc-then-lec|consensus|auto`) with chain telemetry metrics (`chain_lec_unknown_fallbacks`, `chain_bmc_resolved_not_propagated_mutants`, `chain_bmc_resolved_propagated_mutants`, `chain_bmc_unknown_fallbacks`, `chain_lec_resolved_not_propagated_mutants`, `chain_lec_resolved_propagated_mutants`, `chain_lec_error_fallbacks`, `chain_bmc_error_fallbacks`, `chain_consensus_not_propagated_mutants`, `chain_consensus_disagreement_mutants`, `chain_consensus_error_mutants`, `chain_auto_parallel_mutants`, `chain_auto_short_circuit_mutants`) and conservative single-engine-error fallback (never prune on sole non-propagation evidence when the peer engine errors); added per-mutant global formal timeout controls (`--formal-global-propagate-timeout-seconds`) plus per-engine overrides (`--formal-global-propagate-lec-timeout-seconds`, `--formal-global-propagate-bmc-timeout-seconds`) with matrix default/lane overrides and timeout telemetry (`global_filter_timeout_mutants`, `global_filter_lec_timeout_mutants`, `global_filter_bmc_timeout_mutants`) plus runtime telemetry (`global_filter_lec_runtime_ns`, `global_filter_bmc_runtime_ns`, `global_filter_cmd_runtime_ns`, `global_filter_lec_runs`, `global_filter_bmc_runs`, `global_filter_cmd_runs`); added built-in differential BMC original-design cache reuse (`.global_bmc_orig_cache`) with `bmc_orig_cache_hit_mutants`/`bmc_orig_cache_miss_mutants` and runtime telemetry (`bmc_orig_cache_saved_runtime_ns`/`bmc_orig_cache_miss_runtime_ns`), bounded cache controls (`--bmc-orig-cache-max-entries`, `--bmc-orig-cache-max-bytes`, `--bmc-orig-cache-max-age-seconds`), configurable count/byte eviction policy (`--bmc-orig-cache-eviction-policy lru|fifo|cost-lru`), age-aware pruning telemetry (`bmc_orig_cache_pruned_age_entries`/`bmc_orig_cache_pruned_age_bytes`, including persisted-cache variants), and cross-run cache publication status (`bmc_orig_cache_write_status`) via `--reuse-cache-dir/global_bmc_orig_cache`; generated mutation-list cache telemetry now exported in cover/matrix metrics (`generated_mutations_cache_status`, `generated_mutations_cache_hit`, `generated_mutations_cache_miss`); added matrix default/lane cache-limit pass-through controls (`--default-bmc-orig-cache-max-entries`, `--default-bmc-orig-cache-max-bytes`, `--default-bmc-orig-cache-max-age-seconds`, `--default-bmc-orig-cache-eviction-policy`, lane TSV overrides), strict gate pass-through controls (`--skip-baseline`, `--fail-on-undetected`, `--fail-on-errors`) plus per-lane overrides (`skip_baseline`, `fail_on_undetected`, `fail_on_errors`) with explicit boolean validation (`1|0|true|false|yes|no|-`), gate-summary export (`--gate-summary-file`, default `<out-dir>/gate_summary.tsv`), plus lane selection filters (`--include-lane-regex`, `--exclude-lane-regex`) for targeted CI slicing; BMC orig-cache key now includes original-design SHA-256 to prevent stale reuse when design content changes at the same path; added compatibility-guarded global filter reuse from prior `pair_qualification.tsv` (`test_id=-`) with `reused_global_filters` metric; built-in global filters now conservatively treat formal `UNKNOWN` as propagated (not pruned); run_mutation_matrix script path now validates default generated mode/profile/allocation config upfront, flags malformed generated-lane mutation config as `CONFIG_ERROR` before launching lane cover runs, and exports `results.tsv` `config_error_code` + `config_error_reason` for deterministic lane-failure diagnostics; full external-suite wiring still pending |
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
- Mutation/report governance closure (next long-term mutation tranche):
  - Composite matrix policy bundles are now available
    (`formal-regression-matrix-nightly|strict`) with dedicated lane-drift
    bundles (`formal-regression-matrix-lane-drift-nightly|strict`), but
    native matrix lane scheduling is only partially migrated:
    `--native-matrix-dispatch` now exists as an opt-in scaffold, but lane-job
    parallelism and lane-level gate override parity are still script-backed.
  - Wire the composite matrix policy bundles into CI defaults for
    nightly/strict matrix report jobs.
  - Add bounded-history defaults to matrix report jobs in CI bootstrap wiring
    (`--history --history-bootstrap --history-max-runs`) for stable trend data.
- Lane-state:
  - Add recursive refresh trust-evidence capture (peer cert chain + issuer
    linkage + pin material) beyond sidecar field matching.
  - Move metadata trust from schema + static policy matching to active
    transport-chain capture/verification in refresh tooling (issuer/path
    validation evidence).
  - Extend checkpoint granularity below lane-level where ROI is high.
- BMC capability closure:
  - Caller-owned filter policy hardening:
    - `verilator-verification` BMC/LEC and `yosys/tests/sva` BMC/LEC direct
      runner scripts now require explicit `TEST_FILTER` (no implicit
      full-suite fallback).
    - `run_formal_all.sh` now also requires explicit OpenTitan lane filters
      when those lanes are enabled:
      `--opentitan-lec-impl-filter` for `opentitan/LEC|LEC_STRICT`,
      `--opentitan-e2e-impl-filter` for
      `opentitan/E2E|E2E_STRICT|E2E_MODE_DIFF`.
  - Close remaining local-variable and `disable iff` semantic mismatches.
  - Reduce multi-clock edge-case divergence.
  - Expand full (not filtered) regular closure cadence on core suites.
  - Keep strict-gate semantic-tag coverage checks active without blocking
    legitimate closure wins (tagged-case regression is now fail-like-budget
    aware when fail-like rows decrease).
  - Strict gate now has an opt-in absolute no-drop mode
    (`--strict-gate-no-drop-remarks`) to require zero dropped-syntax remarks
    across BMC/LEC lanes during closure-hardening runs.
  - Keep strict-gate unclassified semantic-bucket growth checks active so
    new fail-like rows cannot silently bypass bucket tracking.
  - Keep strict-gate BMC abstraction provenance checks active for both:
    token-set growth and provenance-record volume growth.
  - Remaining harness limitation:
    semantic-bucket coverage is now complete on active fail-like rows across
    `sv-tests/BMC`, `verilator-verification/BMC`, and `yosys/tests/sva/BMC`
    (`unclassified_cases=0` in the latest full lane sweep).
  - Remaining closure gap is now semantic correctness (reducing fail-like rows
    themselves), not bucket attribution coverage.
  - Syntax-tree completeness gaps to close next:
    - BMC/ImportVerilog: implicit built-in class methods are now preserved as
      declarations (no generic drop remark); continue auditing remaining
      dropped-syntax emit sites to keep "no intentional drops" true end-to-end.
    - BMC: continue the "no implicit drops" audit by ensuring clock/semantic
      matching helpers do not materialize transient IR when comparing syntax
      trees (`LowerToBMC.cpp` now side-effect-free for explicit-clock lookup).
    - BMC `ExternalizeRegisters`: single-clock mode now rejects only true
      *used-register* clock conflicts (not mere presence of extra clock ports);
      follow-up remains better diagnostics for derived-clock/source conflicts.
    - BMC SMT-LIB export: live-cone gating now ignores dead LLVM ops in
      `verif.bmc` regions; remaining closure is to lower/replace *live*
      unsupported LLVM ops instead of rejecting them.
    - BMC `ExternalizeRegisters`: register initial values from `seq.initial`
      now accept foldable constant expressions (not just direct
      `hw.constant`); remaining gap is non-foldable dynamic initial logic.
    - BMC `LowerToBMC`: single-clock mode now rejects multiple explicit clock
      ports only when multiple explicit domains are actually used; remaining
      gap is full semantics for intentionally used independent multi-clock
      domains without `allow-multi-clock`.
    - BMC `LowerToBMC`: multi-clock reject diagnostics now report the used
      explicit clock names (and unresolved clock-expression presence) to speed
      semantic triage on remaining closure failures.
- LEC capability closure:
  - Keep no-waiver OpenTitan LEC policy (`XPROP_ONLY` remains fail-like).
  - Keep strict-gate X-prop counter drift checks active in CI.
  - Improve 4-state/X-prop semantic alignment and diagnostics.
  - Keep generic LEC counter drift gates available across all `LEC*` lanes via
    `--fail-on-new-lec-counter` / `--fail-on-new-lec-counter-prefix`
    (`lec_status_*`, `lec_diag_*` from case rows).
  - LEC harness rows now carry explicit structured columns:
    `status, base, path, suite, mode, diag` for sv-tests/verilator/yosys lanes,
    and `run_formal_all.sh` consumes explicit `diag` before `#DIAG` path tags.
  - OpenTitan LEC case rows now also emit explicit `diag` as a dedicated
    column (while retaining path-tag compatibility for downstream consumers).
  - OpenTitan LEC producer now emits deterministic fallback diagnostics when
    solver tags are absent (`EQ`/`NEQ`/`UNKNOWN`/`PASS`/`SMOKE_ONLY`) and
    stage-specific failure diagnostics (`CIRCT_VERILOG_ERROR`,
    `CIRCT_OPT_ERROR`, `CIRCT_LEC_ERROR`).
  - Strict-gate now supports dedicated LEC diag-taxonomy drift checks via
    `--fail-on-new-lec-diag-keys`; global `--strict-gate` enables it with a
    baseline-aware safeguard for legacy baseline rows.
  - Strict-gate now also tracks diag provenance fallback drift via
    `--fail-on-new-lec-diag-path-fallback-cases` (enabled by `--strict-gate`)
    and optional absolute zero gate
    `--fail-on-any-lec-diag-path-fallback-cases`.
  - Strict-gate now also tracks missing explicit LEC diag rows via
    `--fail-on-new-lec-diag-missing-cases` (enabled by `--strict-gate`) and
    optional absolute zero gate `--fail-on-any-lec-diag-missing-cases`.
  - Non-OpenTitan LEC producers (`sv-tests`, `verilator-verification`,
    `yosys/tests/sva`) now emit explicit diag tokens for all emitted rows
    (`EQ`/`NEQ`/`TIMEOUT`/`ERROR`, parse-only `LEC_NOT_RUN`, and compile-step
    error tokens), eliminating avoidable missing-diag rows in these lanes.
  - Remaining diagnostics gap: keep phasing out `#DIAG` path-tag fallback in
    favor of fully explicit per-case diag fields for all
    producers/fixtures (remaining concentration: compatibility fixtures and
    OpenTitan path-tag consumers).
  - Keep optional absolute no-drop gates available for closure runs:
    `--fail-on-any-bmc-drop-remarks`, `--fail-on-any-lec-drop-remarks`.
  - Syntax-tree completeness gaps to close next:
    - LLHD signal/ref lowering still has unsupported probe/drive/cast patterns
      that currently fail with explicit diagnostics (`StripLLHDInterfaceSignals.cpp`).
    - LLVM struct conversion now supports `llvm.mlir.zero` defaults for
      4-state struct casts/extracts; remaining unsupported edges are more
      complex aggregate reconstruction paths (`LowerLECLLVM.cpp`).
- DevEx/CI:
  - Promote lane-state inspector to required pre-resume CI gate.
  - Add per-lane historical trend dashboards and automatic anomaly detection.
- Keep explicit caller-owned lane filters for non-OpenTitan BMC/LEC runs in
  `run_formal_all.sh`; new callsites must pass filters explicitly (`.*` for
  intentionally full-lane sweeps).

### BMC Semantic Closure Plan (Next Execution Track)
1. Target semantics to close:
   `disable iff` timing/enable semantics.
2. Target semantics to close:
   local variable lifetime/sampling in assertions/sequences.
3. Target semantics to close:
   multi-clock sequence/event semantics.
4. Target semantics to close:
   4-state unknown handling consistency (`X`/`Z`) in proofs.
5. Execution sequence:
   run full (non-filtered) `sv-tests`, `verilator-verification`,
   `yosys/tests/sva`, and OpenTitan lanes; classify remaining failures by the
   four buckets above.
6. Execution sequence:
   land fixes bucket-by-bucket with focused lit/unit tests per semantic
   mismatch before expanding to full-suite reruns.
7. Closure criteria:
   known mismatches are fixed or intentionally scoped.
8. Closure criteria:
   regression tests exist for each fix.
9. Closure criteria:
   full non-smoke suites stay green (`sv-tests`,
   `verilator-verification`, `yosys/tests/sva`, OpenTitan lanes).
10. Current baseline status (February 9, 2026):
    no reproducing fail-like mismatches in the four semantic buckets across
    full non-filtered BMC suites plus OpenTitan parity lanes.
11. Immediate follow-up:
    expand explicit multi-clock and 4-state `X`/`Z` semantic stress coverage
    where current full-suite signal is sparse.
12. Candidate next-batch semantic coverage additions from `sv-tests`:
    `16.13--sequence-multiclock-uvm`,
    `16.15--property-iff-uvm`,
    `16.15--property-iff-uvm-fail`,
    `16.10--property-local-var-uvm`,
    `16.10--sequence-local-var-uvm`,
    `16.11--sequence-subroutine-uvm`.
13. Harness hardening landed for this expansion:
    - `circt-bmc` now registers SCF dialect so UVM-derived IR containing
      `scf.if` is accepted.
    - sv-tests BMC/LEC harnesses now auto-resolve UVM path to
      `lib/Runtime/uvm-core/src` (fallback), not only legacy `.../uvm`.
14. Current expanded-candidate status (February 10, 2026 revalidation):
    - With `FORCE_BMC=1 ALLOW_MULTI_CLOCK=1`, the 6-test UVM semantic
      candidate set above is currently `0/6 pass` (`error=6`).
    - All six fail with the same backend issue:
      LLVM translation failure on `builtin.unrealized_conversion_cast`
      rooted at 4-state `hw.struct_create` bridging in BMC lowering.
15. SMTLIB hardening status (February 10, 2026):
    - `convert-verif-to-smt(for-smtlib-export=true)` emits an explicit
      capability diagnostic when `verif.bmc` regions still contain LLVM ops.
    - The same 6-test candidate set in SMTLIB mode currently fails fast with
      that explicit guard (`for-smtlib-export ... found 'llvm.mlir.undef'`),
      confirming this unsupported path is not JIT-only.
16. Harness/orchestrator hardening (February 10, 2026):
    - `utils/run_formal_all.sh` now has first-class
      `--bmc-allow-multi-clock` control and forwards it to all BMC lanes
      (`sv-tests`, `verilator-verification`, `yosys/tests/sva`) so
      multiclock closure cadence is script-native.
17. Next closure feature for this bucket:
    - legalize/eliminate mixed concrete (`i1`) <-> symbolic (`!smt.bv<1>`)
      bridge casts from 4-state `hw.struct` paths in BMC lowering.
18. Bridge-cast lowering progress (February 10, 2026):
    - `LowerSMTToZ3LLVM` now lowers `builtin.unrealized_conversion_cast`
      from concrete integers (`iN`) to `!smt.bv<N>` directly to
      `Z3_mk_unsigned_int64` (for `N<=64`), with conversion tests.
19. Remaining prioritized blocker in this bucket:
    - reverse bridge materialization (`!llvm.ptr` -> `!smt.bv<1>` -> `i1`)
      still appears on 4-state UVM paths and currently blocks the 6-case
      multiclock/local-var/`disable iff` candidate set (`pass=0 error=6`).
20. Feasibility check (February 10, 2026):
    direct reuse of `circt-lec` LLHD interface-stripping passes in the BMC
    pipeline is not drop-in; the attempt fails with
    `LLHD operations are not supported by circt-lec` in current BMC flow.
21. Next closure implementation target:
    build a BMC-native LLHD/interface-storage elimination step for 4-state
    bridge paths (`smt.bv` <-> `i1` round-trips) before SMT-to-Z3 LLVM
    lowering, with focused regression on the 6-case UVM candidate set.
22. Updated status (February 10, 2026, current branch):
    - `circt-bmc` LLHD flow now reuses targeted LEC preprocessing
      (`lower-llhd-ref-ports` + `strip-llhd-interface-signals` with
      `require-no-llhd=false`) without running full `lower-lec-llvm`.
    - Revalidation on the 6-case UVM semantic candidate set with
      `FORCE_BMC=1 ALLOW_MULTI_CLOCK=1`:
      5/6 no longer hit LLVM bridge-cast translation errors and now produce
      real BMC outcomes (`SAT` / pass-fail classification), including
      `16.15--property-iff-uvm-fail` passing.
    - Remaining blocker:
      `16.13--sequence-multiclock-uvm` fails with multi-clock metadata
      legalization (`bmc_reg_clocks` / `bmc_reg_clock_sources`) and is now the
      primary multiclock closure item.
23. Multi-clock metadata closure progress (February 10, 2026):
    - `lower-to-bmc` now remaps `bmc_reg_clock_sources.arg_index` when derived
      clock inputs are prepended.
    - New regression test:
      `test/Tools/circt-bmc/lower-to-bmc-reg-clock-sources-shift.mlir`.
    - `16.13--sequence-multiclock-uvm` no longer fails with metadata
      legalization diagnostics and now reaches solver semantics
      (`BMC_RESULT=SAT`).
24. Updated 6-case semantic-candidate status after metadata fix:
    - `FORCE_BMC=1 ALLOW_MULTI_CLOCK=1` set is now `pass=1 fail=5 error=0`.
    - Remaining positive-test semantic mismatches (all currently SAT):
      `16.10--property-local-var-uvm`,
      `16.10--sequence-local-var-uvm`,
      `16.11--sequence-subroutine-uvm`,
      `16.13--sequence-multiclock-uvm`,
      `16.15--property-iff-uvm`.
25. `disable iff` closure progress (February 10, 2026, current branch):
    - Fixed constant-guard `disable iff` handling in LTL-to-core lowering to
      avoid a spurious first-sample violation when the disable guard is
      statically true.
    - Added dedicated regression:
      `test/Tools/circt-bmc/circt-bmc-disable-iff-constant.mlir`.
    - Revalidated sv-tests pair:
      `16.15--property-disable-iff` now PASS and
      `16.15--property-disable-iff-fail` now FAIL under BMC.
26. Implication-delay closure progress (February 10, 2026, current branch):
    - Fixed implication tautology folding for delayed consequents in
      `LTLToCore` by using folded OR construction for implication safety/final
      checks (prevents spurious first-sample failures when consequent is
      logically true but not yet canonicalized to a constant op).
    - Added regression:
      `test/Tools/circt-bmc/circt-bmc-implication-delayed-true.mlir`.
27. LLHD process-abstraction limitation identified (February 10, 2026):
    - Remaining 6-case semantic-candidate revalidation stays
      `pass=1 fail=5 error=0`.
    - Root cause evidence from minimal reproducer (`/tmp/min-local-var-direct`)
      and emitted BMC IR:
      dynamic LLHD process results are abstracted as unconstrained
      `llhd_process_result*` solver inputs, allowing spurious SAT witnesses for
      otherwise deterministic assertion checks.
28. Pipeline hardening landed (February 10, 2026):
    - `circt-bmc` LLHD pipeline now runs `strip-llhd-processes` after LLHD
      lowering/simplification passes (instead of before), so reducible process
      semantics are preserved as far as possible before fallback abstraction.
29. LLHD abstraction observability hardening (February 10, 2026):
    - `strip-llhd-processes` now tags modules with
      `circt.bmc_abstracted_llhd_process_results = <count>` when process
      results are abstracted to unconstrained inputs.
    - `lower-to-bmc` now propagates this to
      `verif.bmc` as `bmc_abstracted_llhd_process_results` and emits a warning
      that SAT witnesses may be spurious when this abstraction is active.
    - Purpose: make semantic-risk boundaries explicit while continuing closure
      on local-var/`disable iff`/multiclock buckets.
30. LEC X-prop diagnostic hardening (February 10, 2026):
    - `circt-lec --diagnose-xprop` / `--accept-xprop-only` now emit explicit
      machine-readable recheck status:
      `LEC_DIAG_ASSUME_KNOWN_RESULT=<UNSAT|SAT|UNKNOWN>`.
    - This improves strict/no-waiver triage by distinguishing true XPROP_ONLY
      mismatches (`UNSAT` under assume-known-inputs) from persistent
      mismatches (`SAT`/`UNKNOWN`).
31. OpenTitan LEC dominance blocker closure (February 10, 2026):
    - Root cause: `llhd-unroll-loops` alloca hoisting could place
      `llvm.alloca` before its hoisted count operand constant in entry blocks,
      triggering `operand #0 does not dominate this use` on
      `aes_sbox_canright` (`aes_pkg::aes_mvm` path).
    - Fix landed in `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp`:
      hoisted allocas are now placed after their entry-block operand defs.
    - Result: `opentitan/LEC` and `opentitan/LEC_STRICT` are both green again
      in focused and full LEC-lane reruns.
32. Remaining formal closure priorities after this fix:
    - BMC semantic closure: close positive-test SAT mismatches for local-var,
      sequence-subroutine, multiclock, and `disable iff` UVM semantics.
    - LEC hardening: continue strict no-waiver `XPROP_ONLY` gating and deepen
      4-state mismatch diagnostics (unknown-source provenance and reduction
      paths).
33. BMC LLHD abstraction hardening (February 10, 2026):
    - `strip-llhd-processes` now drops process-result drive uses when the
      driven signal has no observable consumers (dead probes / drive-only use),
      instead of introducing unconstrained `llhd_process_result*` module inputs.
    - This reduces avoidable over-approximation noise while preserving
      conservative behavior for actually observed signal paths.
34. Status after hardening rerun:
    - full BMC lane aggregates remain unchanged
      (`sv-tests` 23/26, `verilator-verification` 12/17,
      `yosys/tests/sva` 7/14).
    - 6-case UVM semantic candidate bucket remains semantically blocked by
      observed LLHD process/interface abstraction paths (`pass=1 fail=5`); the
      next closure target is lowering the residual `llhd.wait yield` process
      result pattern (clock/cycle helper processes) without unconstrained
      primary inputs.
35. Interface-abstraction diagnostics hardening (February 10, 2026):
    - `strip-llhd-interface-signals` now records
      `circt.bmc_abstracted_llhd_interface_inputs = <count>` per `hw.module`
      whenever LLHD interface stripping introduces unconstrained inputs.
    - `lower-to-bmc` now propagates this to `verif.bmc` as
      `bmc_abstracted_llhd_interface_inputs` and emits an explicit warning that
      SAT witnesses may be spurious.
- Stateful-probe semantic closure hardening (February 10, 2026):
    - Removed `llhd-sig2reg` from the BMC LLHD lowering pipeline in
      `tools/circt-bmc/circt-bmc.cpp` because this step could collapse
      stateful probe-driven recurrences to init constants in straight-line LLHD
      forms (named-property + sampled-value patterns), effectively dropping
      meaningful semantics.
    - The BMC flow now relies on `strip-llhd-interface-signals` for LLHD signal
      elimination in this path, preserving read-before-write recurrence
      behavior used by `$changed` and named property checks.
    - Added end-to-end regression:
      `test/Tools/circt-bmc/sva-stateful-probe-order-unsat-e2e.sv`.
- Updated baseline after pipeline hardening (February 10, 2026):
    - `sv-tests/BMC`: `26/26` pass.
    - `verilator-verification/BMC`: `17/17` pass.
    - `yosys/tests/sva/BMC`: `12/14` pass with `2` skip, `0` fail.
    - `opentitan/LEC` + `opentitan/LEC_STRICT`: both `1/1` pass.
- Remaining near-term hardening limitation:
    - `circt-bmc --print-counterexample` dominance verifier failure is now
      closed in `LowerSMTToZ3LLVM` by filtering model-print declarations to
      values that dominate the `smt.check` site.
    - Remaining debug limitation is completeness (not correctness): declarations
      that do not dominate the check site are currently omitted from printed
      model-value lists until we land explicit rematerialization for them.
- Full-syntax-tree closure policy target:
    - keep reducing `llhd_process_result*` and
      `signal_requires_abstraction` fallback usage so semantic closure is
      achieved by explicit lowering, not abstraction, on the core BMC lanes.
36. Remaining near-term formal limitations and next build targets:
    - BMC: positive-test SAT mismatches still cluster in local-var /
      multiclock / `disable iff` UVM semantics where residual interface
      abstraction (`_field*` inputs) can over-approximate environment behavior.
    - LEC: strict/no-waiver lanes are green, but 4-state diagnostics still need
      deeper provenance (which abstracted input and which LLHD store/read path
      introduced it) to speed root-cause closure.
    - Next feature for semantic closure cadence:
      add per-input abstraction provenance metadata and strict drift gates on
      abstraction-count/provenance deltas in BMC/LEC formal lanes.
37. Interface-provenance closure progress (February 10, 2026):
    - `strip-llhd-interface-signals` now emits structured per-input details in
      `circt.bmc_abstracted_llhd_interface_input_details` (name/base/type) in
      addition to the existing abstraction count.
    - `lower-to-bmc` propagates these details into `verif.bmc` as
      `bmc_abstracted_llhd_interface_input_details`, enabling machine-readable
      SAT risk triage per abstracted interface input.
38. Current capability limits after provenance landing:
    - Provenance currently captures insertion-level metadata
      (input name/base/type), but not yet source-operation paths
      (which store/read chain forced abstraction).
    - Formal lane gating is still aggregate-count based; per-input provenance
      drift gating in `run_formal_all.sh` remains to be implemented.
39. Next long-term feature sequence for BMC/LEC closure:
    - add source-path provenance (`signal`, `field`, `reason`, source loc) for
      each abstracted interface input;
    - add optional fail-on-drift checks for provenance deltas in BMC/LEC lanes;
    - mirror the same structured provenance model for process-result
      abstractions to unify SAT risk diagnostics.
40. Source-path provenance extension landed (February 10, 2026):
    - `strip-llhd-interface-signals` now records per-input abstraction metadata:
      `reason`, `signal`, `field`, and `loc` in
      `circt.bmc_abstracted_llhd_interface_input_details` alongside
      `name`/`base`/`type`.
    - This upgrades interface abstraction diagnostics from count-only to
      machine-readable path context for BMC/LEC triage.
41. Updated limitations after this landing:
    - provenance drift is observable but not yet policy-gated in
      `run_formal_all.sh` (no fail-on-new-provenance mode yet).
    - process-result abstraction still lacks matching structured provenance
      fields (`reason`, source path, location).
42. Next highest-ROI build target:
    - add formal-lane drift gates for abstraction provenance
      (`count + details`) with allowlist controls, then mirror this format for
      process-result abstractions.
43. BMC provenance-drift gate landed (February 10, 2026):
    - `lower-to-bmc` now emits machine-readable warning tokens for each
      abstracted LLHD interface input:
      `BMC_PROVENANCE_LLHD_INTERFACE reason=... signal=... field=... name=...`.
    - BMC harnesses now collect these tokens per case into lane TSVs:
      `sv-tests-bmc-abstraction-provenance.tsv`,
      `verilator-bmc-abstraction-provenance.tsv`,
      `yosys-bmc-abstraction-provenance.tsv`.
    - `run_formal_all.sh` now persists per-suite/mode provenance token sets in
      baselines and supports strict gating with
      `--fail-on-new-bmc-abstraction-provenance`.
44. Remaining near-term formal limitations after provenance gate:
    - BMC semantic closure is still blocked by real solver mismatches in
      local-var and `disable iff` related positive tests (not infrastructure
      visibility gaps).
    - LEC strict no-waiver posture is in place, but process-result abstraction
      provenance is still count-only and should be upgraded to the same
      source-path token model.
45. Process-result provenance closure landed (February 10, 2026):
    - `strip-llhd-processes` now emits structured
      `circt.bmc_abstracted_llhd_process_result_details`
      (`name`, `base`, `type`, `reason`, `result`, optional `signal`, `loc`)
      for each abstracted process result.
    - `lower-to-bmc` now propagates this detail array as
      `bmc_abstracted_llhd_process_result_details` and emits machine-readable
      warning tokens:
      `BMC_PROVENANCE_LLHD_PROCESS reason=... result=... signal=... name=...`.
    - Existing BMC abstraction drift gate now covers both interface and
      process provenance via the unified token stream.
46. Updated formal limitations and long-term build targets:
    - BMC semantic gaps remain in the three `sv-tests` fail cases
      (`16.10--property-local-var-fail`,
      `16.10--sequence-local-var-fail`,
      `16.15--property-disable-iff-fail`) and corresponding UVM-positive
      semantics where LLHD process abstraction is still active.
    - LEC strict lanes are green, but 4-state diagnostic depth is still
      limited by process/interface abstraction provenance not yet being
      correlated into a single source-path chain in user-facing reports.
    - Next high-ROI feature: add provenance allowlist/prefix controls in
      `run_formal_all.sh` so known abstraction classes can be scoped while
      still failing on newly introduced semantic-risk tokens.
47. Provenance allowlist controls landed (February 10, 2026):
    - `run_formal_all.sh` now supports
      `--bmc-abstraction-provenance-allowlist-file <FILE>` for strict-gate
      filtering of known-safe abstraction tokens.
    - Allowlist format supports:
      `exact:<token>` (or bare token), `prefix:<prefix>`, `regex:<pattern>`.
    - `--fail-on-new-bmc-abstraction-provenance` now fails only on
      non-allowlisted token deltas, preserving regression sensitivity while
      avoiding noisy re-fails on known legacy abstractions.
48. Updated near-term closure priorities:
    - Continue semantic closure on the three failing `sv-tests` BMC cases
      with provenance evidence now separating known abstraction classes from
      genuinely new risk.
    - Add source-chain correlation in diagnostics (map each process/interface
      provenance token to user-visible assertion/sequence paths) to accelerate
      local-var and `disable iff` mismatch root-cause loops.
49. BMC provenance case-correlation report landed (February 10, 2026):
    - `run_formal_all.sh` now emits
      `bmc-abstraction-provenance-case-map.tsv` per run, joining:
      suite/mode/case/status/path with aggregated provenance tokens.
    - Each row includes `is_fail_like` and token cardinality to prioritize
      semantic closure on fail-like cases first.
50. Current BMC/LEC limitation snapshot after this landing:
    - The correlated map confirms all current `sv-tests` BMC fail cases
      (`16.10--property-local-var-fail`,
      `16.10--sequence-local-var-fail`,
      `16.15--property-disable-iff-fail`) share the same process abstraction
      token class (`observable_signal_use` on `clk`).
    - Next long-term feature target: push correlation one step deeper from
      case-level to assertion/sequence-level attribution so solver witnesses
      can be tied directly to specific semantic lowering paths.
51. Token-level provenance prioritization landed (February 10, 2026):
    - `run_formal_all.sh` now emits
      `bmc-abstraction-provenance-token-summary.tsv` per run, aggregating
      each provenance token into fail-like vs non-fail-like case counts and
      case ID lists.
    - This directly surfaces token classes that are failure-dominant and
      therefore highest priority for semantic-closure work.
52. Updated closure guidance from current token summary:
    - Current `sv-tests/BMC` token summary shows both active process
      provenance tokens (`observable_signal_use` on `clk`) are
      `fail_like_cases=3`, `non_fail_like_cases=0`.
    - Next feature build should therefore target eliminating or refining this
      specific abstraction class in local-var/`disable iff` lowering paths
      before broadening to secondary buckets.
53. Assertion/sequence attribution report landed (February 10, 2026):
    - `run_formal_all.sh` now emits
      `bmc-abstraction-provenance-assertion-attribution.tsv`, joining each
      provenance-correlated case with extracted source assertion/sequence sites
      (`L<line>:<snippet>`).
    - This closes the gap between token-level provenance and concrete
      user-facing property/sequence definitions for triage.
54. Updated limitations after assertion attribution:
    - Attribution is currently source-pattern based (SV text extraction), not
      yet guaranteed one-to-one with final lowered `verif.*` checks.
    - Next long-term feature should add lowered-op stable IDs (or source loc
      propagation) so witness/provenance can be tied to exact backend checks.
55. IR-check attribution landed (February 10, 2026):
    - BMC lane scripts now emit per-case IR check signatures from generated
      MLIR (`verif.assert` / `verif.clocked_assert` / assume / cover ops).
    - `run_formal_all.sh` now emits
      `bmc-abstraction-provenance-ir-check-attribution.tsv` and extends case/
      assertion attribution reports with:
      `ir_check_count`, `ir_check_kinds`, `ir_check_sites`.
56. Updated long-term closure focus after IR-check attribution:
    - The three remaining `sv-tests` BMC fail cases now correlate to a single
      `verif.clocked_assert` IR check each plus the same process abstraction
      token class (`observable_signal_use` on `clk`).
    - Next long-term feature target is stable lowered-check IDs propagated into
      BMC diagnostics so SAT witnesses/provenance can reference exact backend
      check IDs across pipeline stages.
57. BMC semantic-closure cadence expansion landed (February 10, 2026):
    - `utils/run_formal_all.sh` now supports a dedicated
      `sv-tests-uvm/BMC_SEMANTICS` lane (targeted 6-case set for local-var,
      `disable iff`, and multiclock semantics), enabled via
      `--with-sv-tests-uvm-bmc-semantics`.
    - Lane policy is fixed to semantic-closure intent:
      `INCLUDE_UVM_TAGS=1`, `ALLOW_MULTI_CLOCK=1`, and a curated
      `TEST_FILTER` containing:
      `16.10--property-local-var-uvm`,
      `16.10--sequence-local-var-uvm`,
      `16.11--sequence-subroutine-uvm`,
      `16.13--sequence-multiclock-uvm`,
      `16.15--property-iff-uvm`,
      `16.15--property-iff-uvm-fail`.
58. Current near-term limitations after this expansion:
    - This lane is intentionally curated (6 tests), not yet broad UVM-assertion
      corpus closure.
    - LEC long-term closure remains focused on strict no-waiver X-prop policy
      and deeper 4-state diagnostic precision.
59. IR-check attribution hardening landed (February 10, 2026):
    - `run_formal_all.sh` provenance reports now include
      `ir_check_fingerprints` (`chk_<sha1-12>`) derived from normalized
      lowered check kind+snippet.
    - This provides stable-ish check identity across per-run check reindexing
      and improves long-term BMC triage/debug joins.
60. Remaining limitation after fingerprint landing:
    - Fingerprints are content-based approximations, not first-class backend
      check IDs propagated through lowering and solver witness diagnostics.
    - Next long-term feature target remains explicit stable check IDs in
      lowering/diagnostics so witness mapping is exact, not heuristic.
61. IR-check extraction fidelity hardening landed (February 10, 2026):
    - BMC lane scripts now preserve full normalized `verif.*` check lines in
      `BMC_CHECK_ATTRIBUTION_OUT` (no early 200-char truncation).
    - `run_formal_all.sh` now truncates only display rendering for
      `ir_check_sites` while computing `ir_check_fingerprints` from full check
      text.
62. Updated long-term implication:
    - Fingerprint collision risk from truncated check text is reduced, but true
      end-to-end check-ID propagation remains the target capability.
63. Structured IR-check key stratification landed (February 10, 2026):
    - BMC provenance reports now include:
      `ir_check_keys` and `ir_check_key_modes`.
    - Key selection order is now explicit:
      `label` (non-empty) -> `loc` -> `fingerprint` fallback.
64. Remaining limitation after structured-key landing:
    - Current sv-tests UVM semantic lane still resolves to `fingerprint` mode
      because lowered checks mostly lack non-empty labels/source locs.
    - Next long-term feature remains backend-owned stable check IDs propagated
      through lowering and solver witness reporting.
65. Strict-gate fallback-key drift hardening landed (February 10, 2026):
    - `run_formal_all.sh` now supports
      `--fail-on-new-bmc-ir-check-fingerprint-cases`.
    - `--strict-gate` enables this by default.
    - BMC lane summaries now emit explicit check-key mode counters:
      `bmc_ir_check_key_mode_{fingerprint,label,loc}_{checks,cases}`.
66. Remaining limitation after fallback-key drift gate:
    - The gate currently tracks fallback-key drift at case granularity, not
      per-check witness semantics.
    - Next long-term closure target remains native stable backend check IDs
      that eliminate heuristic fallback identity in strict-gate policy.
67. BMC semantic-bucket strict-gate hardening landed (February 10, 2026):
    - `run_formal_all.sh` now emits BMC fail-like semantic bucket counters in
      lane summaries:
      `bmc_semantic_bucket_{fail_like,disable_iff,local_var,multiclock,four_state,unclassified}_cases`.
    - Added strict-gate option
      `--fail-on-new-bmc-semantic-bucket-cases` and enabled it under
      `--strict-gate` defaults.
68. BMC semantic-lane strict-gate collector parity fix:
    - strict-gate failure-case and abstraction-provenance collectors now
      include `sv-tests-uvm/BMC_SEMANTICS` files, matching baseline-update
      telemetry coverage.
69. Remaining near-term BMC closure limitation after this hardening:
    - bucket counts are name/path-based classification heuristics, not direct
      solver-IR semantic tags; deep closure still requires backend-emitted
      semantic category metadata for exact attribution.
70. Cross-suite semantic-bucket coverage hardening landed (February 10, 2026):
    - `run_yosys_sva_circt_bmc.sh` now writes deterministic case rows to
      `OUT` (`STATUS base path suite mode`) and no longer leaves
      `yosys-bmc-results.txt` empty during normal runs.
    - Result: `yosys/tests/sva/BMC` now emits
      `bmc_semantic_bucket_*_cases` counters in `run_formal_all.sh` summaries,
      enabling uniform strict-gate bucket telemetry across all BMC lanes.
71. Updated cross-suite limitation snapshot after yosys row-emission fix:
    - `sv-tests/BMC` still carries concrete semantic signal
      (`disable_iff=1`, `local_var=2` in current rerun).
    - `sv-tests-uvm/BMC_SEMANTICS` remains green (`6/6`) with zero fail-like
      bucket counts.
    - `verilator-verification/BMC` and `yosys/tests/sva/BMC` fail-like rows are
      still mostly `unclassified` by current name/path heuristics.
72. Next long-term closure feature from this point:
    - add backend- or harness-emitted semantic bucket tags (instead of
    name/path regex only) so strict-gate counters reflect true semantic
    classes for `verilator`/`yosys` fail-like rows.
73. Tag-aware semantic bucket classifier landed (February 10, 2026):
    - `run_formal_all.sh` semantic bucket summarization now accepts explicit
      per-case bucket tags from result rows (for example
      `semantic_buckets=disable_iff,multiclock`, `bucket=four_state`) and
      falls back to name/path regex only when tags are absent.
    - New counters now split attribution source:
      `bmc_semantic_bucket_classified_cases`,
      `bmc_semantic_bucket_tagged_cases`,
      `bmc_semantic_bucket_regex_cases`.
74. Updated limitation snapshot after tag-aware classifier:
    - current real suite rows still report `tagged_cases=0` across
      `sv-tests`, `sv-tests-uvm`, `verilator`, and `yosys`; coverage remains
      regex-driven/unclassified until runners or backend diagnostics emit
      explicit semantic tags.
75. Next concrete implementation target:
    - add first-class semantic-tag emission in BMC runners (or backend
    diagnostics) for known closure categories (`disable iff`, local-var,
    multiclock, 4-state) so strict-gate can track semantic drift without
    relying on filename heuristics.
76. sv-tests runner semantic-tag emission landed (February 10, 2026):
    - `run_sv_tests_circt_bmc.sh` now supports
      `BMC_SEMANTIC_TAG_MAP_FILE` and emits tagged case rows
      (`suite=sv-tests`, `mode=BMC`, `semantic_buckets=...`) for mapped cases.
    - `run_formal_all.sh` now forwards
      `SV_TESTS_BMC_SEMANTIC_TAG_MAP_FILE` to both `sv-tests/BMC` and
      `sv-tests-uvm/BMC_SEMANTICS` lanes.
77. Initial map rollout status:
    - new map file `utils/sv-tests-bmc-semantic-tags.tsv` tags known
      local-var/disable-iff/multiclock closure cases.
    - current real run signal:
      `sv-tests/BMC` moved to `tagged_cases=3 regex_cases=0` for fail-like
      rows, while `verilator` and `yosys` remain untagged.
78. Next long-term step after sv-tests map rollout:
    - add analogous semantic-tag sources for `verilator-verification` and
      `yosys/tests/sva` (runner map or backend diagnostics), then optionally
      gate on `bmc_semantic_bucket_tagged_cases` floor in strict mode.
79. BMC semantic-bucket strict-gate coverage now tracks all emitted bucket
    counters under `--fail-on-new-bmc-semantic-bucket-cases`, including:
    `sampled_value`, `property_named`, `implication_timing`,
    and `hierarchical_net` (in addition to legacy
    `disable_iff`/`local_var`/`multiclock`/`four_state`).
    - This closes drift blind spots where non-legacy bucket regressions could
      appear without tripping strict-gate policy.
80. BMC semantic bucket triage artifacts now export per-lane, case-level
    bucket attribution files:
    - `sv-tests-bmc-semantic-buckets.tsv`
    - `verilator-bmc-semantic-buckets.tsv`
    - `yosys-bmc-semantic-buckets.tsv`
    Each row is `(status, case_id, path, suite, mode, semantic_bucket, source)`
    where `source` is `tagged`, `regex`, or `unclassified`.
    - This gives direct machine-readable bucket-to-case joins for closure
      planning and strict-gate investigation without re-parsing logs.
81. `run_formal_all.sh` now emits a merged cross-lane BMC semantic case map:
    `bmc-semantic-bucket-case-map.tsv` with a stable tabular schema:
    `(status, case_id, path, suite, mode, semantic_bucket, source)`.
    - This unifies sv-tests / verilator / yosys semantic fail-like attribution
      into one artifact for bucket-priority closure planning and CI trend
      ingestion.
82. OpenTitan LEC X-prop diagnostics now include explicit
    `LEC_DIAG_ASSUME_KNOWN_RESULT` attribution in both per-case artifacts and
    summary counters.
    - `run_formal_all.sh` now emits
      `opentitan-lec-xprop-case-map.tsv` (merged `LEC` + `LEC_STRICT` rows)
      with `(status, implementation, mode, diag, lec_result, counters, log_dir,
      assume_known_result, source_file)`.
    - This closes a strict/no-waiver triage gap by making 4-state
      assume-known behavior machine-readable and strict-gate-addressable.
83. Strict-gate default OpenTitan LEC X-prop key-prefix policy now includes
    `xprop_assume_known_result_` (in addition to diag/status/result/counter).
    - This closes a governance gap where assume-known semantic drift could
      bypass strict mode unless users manually provided extra prefix flags.
84. Latest cross-suite BMC/LEC closure snapshot (February 10, 2026):
    - `sv-tests/BMC`: `pass=23 fail=3` (all fail-like rows remain tagged and
      classified; `disable_iff=1`, `local_var=2`, `unclassified=0`).
    - `verilator-verification/BMC`: `pass=12 fail=5`
      (`sampled_value=3`, `property_named=2`, `unclassified=0`).
    - `yosys/tests/sva/BMC`: `pass=7 fail=5 skip=2`
      (`disable_iff=2`, `four_state=1`, `sampled_value=1`,
      `implication_timing=2`, `hierarchical_net=1`, `unclassified=0`).
    - `opentitan/LEC_STRICT`: `pass=1 fail=0`.
    - Near-term semantic closure remains focused on reducing those fail-like
      rows (not coverage attribution, which remains complete on active
      fail-like cases).
85. `run_formal_all.sh` now forces `BMC_RUN_SMTLIB=1` for `sv-tests` BMC
    lanes (`sv-tests/BMC` and `sv-tests-uvm/BMC_SEMANTICS`) to avoid known
    JIT/Z3-LLVM backend divergence on local-variable/`disable iff` semantics.
    - Post-landing closure snapshot:
      `sv-tests/BMC` moved from `23/26` to `26/26` pass in the same lane set,
      while `verilator-verification/BMC` and `yosys/tests/sva/BMC` remained
      unchanged.
    - Remaining long-term limitation:
      JIT-vs-SMTLIB backend parity is still open and should be tracked as a
      backend correctness hardening item (not a harness attribution gap).
86. Added optional sv-tests BMC backend parity drift instrumentation in
    `run_formal_all.sh` (`--sv-tests-bmc-backend-parity`) plus strict-gate
    drift counter support (`--fail-on-new-bmc-backend-parity-mismatch-cases`).
    - New artifact:
      `sv-tests-bmc-backend-parity.tsv` with per-case statuses
      (`status_smtlib`, `status_jit`) and classification.
    - Current measured baseline (February 10, 2026):
      `bmc_backend_parity_mismatch_cases=3`, all
      `bmc_backend_parity_jit_only_fail_cases=3`, in:
      `16.10--property-local-var-fail`,
      `16.10--sequence-local-var-fail`,
      `16.15--property-disable-iff-fail`.
    - Near-term closure target:
      drive this parity mismatch counter to `0` while preserving
      `sv-tests/BMC` pass on SMT-LIB backend.

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
3. `XPROP_ONLY` results count as failures in parity runs.
4. `--allow-xprop-only` is removed from OpenTitan E2E parity flow and cannot be
   used for parity status.

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
9. Latest canonical OpenTitan dual-lane run via `run_formal_all.sh`
   (`^opentitan/(E2E|E2E_STRICT|E2E_MODE_DIFF)$`) reports:
   - `E2E`: `pass=12 fail=0`
   - `E2E_STRICT`: `pass=12 fail=0`
   - `E2E_MODE_DIFF`: `strict_only_fail=0 strict_only_pass=0`
10. Strict non-optimistic OpenTitan LEC closure is now in place:
    `opentitan/LEC_STRICT` runs `LEC_X_OPTIMISTIC=0` without
    `aes_sbox_canright#XPROP_ONLY` by default (known-input assumptions in the
    OpenTitan LEC harness strict mode path).
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
17. OpenTitan E2E no longer supports `--allow-xprop-only`; `XPROP_ONLY` rows
    are hard failures in parity runs.
17. OpenTitan LEC artifact paths are now deterministic in both formal-all and
    OpenTitan E2E harnesses (`opentitan-lec-work`, `opentitan-lec-strict-work`,
    `opentitan-formal-e2e/lec-workdir`) for stable case-level gating/triage.
18. `run_formal_all.sh` expected-failure case matching now supports
    `id_kind=base_diag` (`<base>#<DIAG>`), enabling stable strict OpenTitan
    diagnostic tracking (`XPROP_ONLY`) without path-regex coupling.
19. `run_formal_all.sh` expected-failure case matching now supports
    `id_kind=base_diag_regex` for one-to-many strict diagnostic matching across
    implementation sets while remaining path-independent.
22. `run_formal_all.sh` strict-gate now tracks fail-like case IDs via baseline
    `failure_cases` telemetry and flags diagnostic drift even when aggregate
    fail counts stay flat.
24. OpenTitan LEC lanes in `run_formal_all.sh` now require direct case TSV
    output from `run_opentitan_circt_lec.py`; missing case output is recorded as
    a hard lane error (`missing_results=1`) instead of log-based fallback
    inference.
25. `run_formal_all.sh` expected-failure case ingestion/refresh now reads
    detailed `yosys/tests/sva` BMC case rows (`yosys-bmc-results.txt`) in
    addition to summary counters, enabling per-case BMC expectations without
    collapsing to aggregate-only IDs.
26. Strict OpenTitan LEC lane now supports first-class unknown-source dump
    control via `run_formal_all.sh --opentitan-lec-strict-dump-unknown-sources`
    (wired to `LEC_DUMP_UNKNOWN_SOURCES=1` in the strict lane harness).
27. OpenTitan E2E LEC mode is now selectable from `run_formal_all.sh`:
    `--opentitan-e2e-lec-x-optimistic` or
    `--opentitan-e2e-lec-strict-x` (mutually exclusive), enabling strict
    parity audits through the canonical E2E control plane.
28. `run_formal_all.sh` now supports a dedicated strict OpenTitan E2E audit
    lane (`opentitan/E2E_STRICT`, `--with-opentitan-e2e-strict`) that can run
    alongside `opentitan/E2E` and exports case-level rows to
    `opentitan-e2e-strict-results.txt` for expected-failure and strict-gate
    case tracking.
29. When both OpenTitan E2E lanes are enabled, `run_formal_all.sh` now emits
    a normalized mode-diff artifact (`opentitan-e2e-mode-diff.tsv`) and
    fail-like case export (`opentitan-e2e-mode-diff-results.txt`,
    `mode=E2E_MODE_DIFF`) so strict-only behavioral drift is directly trackable
    through existing expected-failure and strict-gate flows.
30. `E2E_MODE_DIFF` now exports classification telemetry as both structured
    metrics (`opentitan-e2e-mode-diff-metrics.tsv`) and summary counters
    (`strict_only_fail`, `same_status`, `status_diff`, missing-case classes),
    enabling trend-friendly CI analytics without parsing ad hoc logs.
31. `run_formal_all.sh` now supports a dedicated strict gate for OpenTitan
    mode-diff regressions:
    `--fail-on-new-e2e-mode-diff-strict-only-fail` (also enabled by
    `--strict-gate`) to fail when `opentitan/E2E_MODE_DIFF` strict-only-fail
    count increases vs baseline window.
32. OpenTitan E2E case export in `run_formal_all.sh` now preserves fail-like
    statuses (`FAIL`, `ERROR`, `XFAIL`, `XPASS`, `EFAIL`) instead of collapsing
    all non-pass rows to `FAIL`, enabling real `status_diff` tracking between
    default and strict E2E lanes.
33. `run_formal_all.sh` now supports
    `--fail-on-new-e2e-mode-diff-status-diff` (also enabled by `--strict-gate`)
    to fail when `opentitan/E2E_MODE_DIFF` `status_diff` increases vs baseline
    window.
34. `run_formal_all.sh` strict-gate now supports and validates all currently
    exported OpenTitan E2E mode-diff drift classes:
    `strict_only_fail`, `status_diff`, `strict_only_pass`,
    `missing_in_e2e`, and `missing_in_e2e_strict`; parser telemetry extraction
    now correctly supports metric keys containing digits (for example
    `missing_in_e2e*`).
35. `run_opentitan_circt_sim.sh` now auto-recovers local tool execute-bit
    drift for `circt-verilog`/`circt-sim` (attempts `chmod +x` before run),
    preventing transient OpenTitan E2E SIM failures from local file-mode
    skew while keeping explicit failure on non-recoverable tool state.
36. OpenTitan LEC lanes now emit machine-readable X-prop diagnostics:
    - `utils/run_opentitan_circt_lec.py` supports `OUT_XPROP_SUMMARY` /
      `--xprop-summary-file` and writes per-implementation XPROP rows
      (`status`, `diag`, `LEC_RESULT`, parsed counter summary, log path).
    - `run_formal_all.sh` now provisions dedicated artifacts for both lanes:
      `opentitan-lec-xprop-summary.tsv` and
      `opentitan-lec-strict-xprop-summary.tsv`.
37. `run_formal_all.sh` now aggregates OpenTitan LEC XPROP counters directly
    into lane summaries (`summary.tsv` / baseline result field), including:
    `xprop_cases`, `xprop_diag_*`, `xprop_result_*`, `xprop_status_*`,
    `xprop_counter_*`, plus per-implementation keys
    `xprop_impl_<impl>_{cases,status_*,diag_*,result_*,counter_*}`.
38. Strict-gate now supports regression gating on strict OpenTitan LEC XPROP
    counters via repeatable
    `--fail-on-new-opentitan-lec-strict-xprop-counter <key>` against baseline
    windows, enabling long-term 4-state parity trend enforcement beyond coarse
    fail-count tracking.
39. Strict OpenTitan LEC counter gating now supports per-implementation drift
    tracking (for example
    `xprop_impl_aes_sbox_canright_counter_input_unknown_extracts`) so parity
    regressions can be attributed and gated at implementation granularity.
40. Strict OpenTitan LEC counter gating now also supports prefix-based drift
    enforcement via
    `--fail-on-new-opentitan-lec-strict-xprop-counter-prefix <prefix>`,
    so newly introduced strict-lane X-prop counters cannot bypass drift gates
    by appearing under previously unseen keys.
41. Strict OpenTitan LEC gating now also supports generic X-prop summary-key
    prefix drift checks via
    `--fail-on-new-opentitan-lec-strict-xprop-key-prefix <prefix>`, enabling
    policy gates on class-level diagnostics (`xprop_diag_*`,
    `xprop_result_*`, `xprop_status_*`, or implementation-specific prefixes)
    without enumerating each concrete key.
42. `--strict-gate` now auto-enables strict OpenTitan LEC key-prefix drift
    checks when `--with-opentitan-lec-strict` is active:
    `xprop_diag_*`, `xprop_status_*`, `xprop_result_*`,
    and `xprop_counter_*`.
    - This turns strict-mode LEC X-prop drift detection on by default instead
      of requiring extra per-run flags, while retaining explicit override
      options for narrower or broader policies.
43. BMC lane summaries in `run_formal_all.sh` now include case-derived drift
    counters when detailed case files are available:
    - `bmc_timeout_cases`
    - `bmc_unknown_cases`
44. Strict-gate now supports explicit BMC drift gates:
    - `--fail-on-new-bmc-timeout-cases`
    - `--fail-on-new-bmc-unknown-cases`
    and enables both by default under `--strict-gate`.
18. Mutation built-in circt-lec/circt-bmc filters now support automatic tool
    discovery (PATH, then `<circt-root>/build/bin`) when paths are omitted or
    set to `auto`, including chained and matrix default flows.
19. Added initial `circt-mut` binary frontend with
    `cover|matrix|generate` subcommands to provide a stable mutation CLI while
    preserving script backend compatibility during migration.
20. Mutation generation now supports content-addressed list caching under
    `--reuse-cache-dir/generated_mutations`, reducing repeated Yosys
    `mutate -list` work across iterative cover/matrix runs.
21. Mutation cover/matrix metrics now export generated-list cache telemetry
    (`generated_mutations_cache_status`, `generated_mutations_cache_hit`,
    `generated_mutations_cache_miss`) to make cache effectiveness directly
    visible in CI trend data.
23. Mutation generation now tracks runtime telemetry in cover/matrix artifacts
    (`generated_mutations_runtime_ns`,
    `generated_mutations_cache_saved_runtime_ns`), and matrix `results.tsv`
    now surfaces generated-cache status/hit/miss/saved-runtime columns for
    lane-level CI triage without opening per-lane metrics files.
24. Generated mutation-list caching now uses per-key process locking in
    `generate_mutations_yosys.sh`, preventing duplicate `yosys mutate -list`
    synthesis across concurrent matrix lanes targeting the same cache key.
25. Generated mutation cache lock-contention telemetry is now exported through
    generator/cover/matrix artifacts
    (`generated_mutations_cache_lock_wait_ns`,
    `generated_mutations_cache_lock_contended`) and surfaced in matrix
    `results.tsv` for lane-level cache hotspot diagnosis.

#### Current Open Non-Smoke Gaps (latest parity tracking)
1. No currently reproducing OpenTitan non-smoke parity gaps in canonical
   dual-lane runs (`E2E`, `E2E_STRICT`, and mode-diff all clean).
2. Keep strict-gate drift checks enabled so any future reintroduction of
   timeout/X-prop deltas trips immediately.

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
   X-prop drift counters, full-suite pass rates, flaky rate.
4. Status discipline:
   roadmap intent in `PROJECT_PLAN.md`; command-level evidence and results in
   `CHANGELOG.md`.

### Latest BMC Backend-Parity + No-Drop Status (February 10, 2026)
1. Removed intentional semantic dropping in `VerifToSMT` BMC lowering:
   `verif.assume` is now always lowered to SMT assertions (or conversion fails),
   never silently discarded.
2. Fixed SMTLIB crash introduced by inline-assume lowering by avoiding
   clone-and-erase of temporary `verif.assume` ops in inline `verif.bmc`
   regions; lowering now uses mapped operands directly.
3. Current `sv-tests/BMC` parity remains clean after no-drop hardening:
   `bmc_backend_parity_mismatch_cases=0`,
   `bmc_backend_parity_status_diff_cases=0`
   (`/tmp/formal-bmc-no-drop-svtests-fix-20260210-133224`).
4. Local-var/`disable iff` semantic closure update (February 10, 2026):
   `sv-tests/BMC` now reports `total=26 pass=26 fail=0 error=0`
   (`/tmp/sv-bmc-full-after-disableiff-fix.tsv`).
5. The previously failing semantic-closure trio now passes in both backends:
   - `16.10--property-local-var-fail`
   - `16.10--sequence-local-var-fail`
   - `16.15--property-disable-iff-fail`
   via JIT and SMT-LIB (`/tmp/sv-bmc-3cases-after-fix2.tsv`,
   `/tmp/sv-bmc-3cases-after-fix2-smt.tsv`).
6. Broader BMC/LEC snapshot after this closure/hardening:
   - `sv-tests/BMC`: `pass=26 fail=0`
   - `verilator-verification/BMC`: `pass=12 fail=5`
   - `yosys/tests/sva/BMC`: `pass=7 fail=5 skip=2`
   - `opentitan/LEC`: `pass=1 fail=0`
   - `opentitan/LEC_STRICT`: `pass=1 fail=0`
7. Next execution target:
   continue capability closure/hardening on non-sv-tests BMC fail-like rows
   while preserving strict no-drop semantics.

### Latest BMC/LEC No-Drop Interface Status (February 10, 2026)
1. Hardened `strip-llhd-interface-signals` interface fallback to honor
   `require-no-llhd` for unresolved interface-field cases:
   in `require-no-llhd=false` mode (used by BMC), unresolved reads are no longer
   force-abstracted to unconstrained module inputs.
2. Added regression:
   `test/Tools/circt-lec/lec-strip-llhd-interface-require-no-llhd.mlir`
   to lock default abstraction vs residual-LLHD behavior split.
3. Revalidated targeted semantic-closure set with
   `FORCE_BMC=1 ALLOW_MULTI_CLOCK=1` on:
   `16.10--property-local-var-uvm`,
   `16.10--sequence-local-var-uvm`,
   `16.11--sequence-subroutine-uvm`,
   `16.13--sequence-multiclock-uvm`,
   `16.15--property-iff-uvm`,
   `16.15--property-iff-uvm-fail`
   -> `total=6 pass=5 fail=1 error=0`.
4. Current near-term blocker remains SMT-LIB closure on this bucket:
   `for-smtlib-export` still rejects residual LLVM ops in `verif.bmc` regions
   (for example `llvm.mlir.constant`), so this remains the next syntax-tree
   completeness target for BMC formal parity.
5. New no-drop guardrail available in sv-tests BMC harness:
   `utils/run_sv_tests_circt_bmc.sh` now reports
   `drop_remark_cases` for frontend diagnostics matching
   `"will be dropped during lowering"` and supports opt-in enforcement via
   `FAIL_ON_DROP_REMARKS=1`.
6. Formal orchestration now tracks this guardrail in strict-gate telemetry:
   `utils/run_formal_all.sh` captures `bmc_drop_remark_cases` for all active
   BMC lanes:
   `sv-tests/BMC`, `sv-tests-uvm/BMC_SEMANTICS`,
   `verilator-verification/BMC`, and `yosys/tests/sva/BMC`, and can gate
   regression via `--fail-on-new-bmc-drop-remark-cases`
   (enabled by `--strict-gate`).
7. SMT-LIB syntax-tree closure progress (February 10, 2026):
   `convert-verif-to-smt(for-smtlib-export=true)` now legalizes
   `llvm.mlir.constant` (scalar integer/float) inside `verif.bmc` regions to
   `arith.constant` before unsupported-op checks.
8. After this legalization, the 6-case UVM semantic SMT-LIB blocker moved from
   `llvm.mlir.constant` to `llvm.call` (`malloc` path), making the next closure
   target explicit: eliminate or legalize call/pointer constructs in BMC
   regions for SMT-LIB export.
9. Cross-suite no-drop telemetry validation (February 10, 2026):
   focused `run_formal_all.sh` lane reruns now report
   `bmc_drop_remark_cases=0` on:
   - `sv-tests/BMC` (`26/26` pass)
   - `verilator-verification/BMC` (`12/17` pass)
   - `yosys/tests/sva/BMC` (`7/14` pass, `2` skip)
   confirming no new dropped-syntax remark signal while semantic fail-like
   closure continues on non-sv-tests suites.
10. Cross-suite runner no-drop parity hardening (February 10, 2026):
    - `utils/run_verilator_verification_circt_bmc.sh` now emits
      `verilator-verification dropped-syntax summary: drop_remark_cases=...`
      and supports `FAIL_ON_DROP_REMARKS=1`.
    - `utils/run_yosys_sva_circt_bmc.sh` now emits
      `yosys dropped-syntax summary: drop_remark_cases=...`
      and supports `FAIL_ON_DROP_REMARKS=1`, with per-test dedup across
      pass/fail modes.
11. Remaining no-drop limitation after this landing:
    strict-gate drift is still count-based (`bmc_drop_remark_cases`) and does
    not yet gate on newly affected case identities.
12. Next closure feature for full-syntax-tree governance:
    add optional per-case drop-remark artifact export from all BMC runners and
    strict-gate support for "new dropped-syntax cases" deltas.
13. Per-case dropped-syntax artifact closure landed (February 10, 2026):
    - BMC runners now export optional case-level drop-remark artifacts via
      `BMC_DROP_REMARK_CASES_OUT`:
      `sv-tests`, `verilator-verification`, `yosys/tests/sva`.
    - `yosys/tests/sva` deduplicates case IDs across pass/fail mode executions
      before counting and artifact emission.
    - `run_formal_all.sh` now passes lane-local drop-case artifact paths for
      all BMC lanes and persists case IDs in baseline rows
      (`bmc_drop_remark_case_ids`).
    - new strict-gate option:
      `--fail-on-new-bmc-drop-remark-case-ids`
      (enabled by `--strict-gate`) fails on growth in dropped-syntax-affected
      case IDs, not just count drift.
14. Updated no-drop limitation after case-ID gate:
    drift detection is now case-aware, but still tied to warning-pattern
    detection (`"will be dropped during lowering"`) rather than first-class
    lowering provenance tags emitted directly by the frontend/lowering passes.
15. Case-reason dropped-syntax provenance landed (February 10, 2026):
    - BMC runners now optionally emit normalized drop reasons per case via
      `BMC_DROP_REMARK_REASONS_OUT` in addition to case IDs:
      `sv-tests`, `verilator-verification`, `yosys/tests/sva`.
    - Reasons are normalized in-runner to reduce path/line-number churn
      (location prefix stripping, whitespace collapse, number normalization).
    - `run_formal_all.sh` now captures these artifacts into lane-local
      `*-drop-remark-reasons.tsv` files and persists baseline tuples in
      `bmc_drop_remark_case_reason_ids`.
    - new strict-gate option:
      `--fail-on-new-bmc-drop-remark-case-reasons`
      (enabled by `--strict-gate`) fails on growth in dropped-syntax
      case+reason tuples.
16. Remaining no-drop limitation after case-reason gate:
    reason extraction is still log-derived; final target remains first-class
    frontend/lowering provenance tags (structured reason/category/op/path)
    emitted directly from lowering instead of warning-text parsing.
17. LEC no-drop parity hardening landed (February 10, 2026):
    - LEC runners now export optional dropped-syntax artifacts in all three
      lanes via:
      `LEC_DROP_REMARK_CASES_OUT` and `LEC_DROP_REMARK_REASONS_OUT`
      (`sv-tests`, `verilator-verification`, `yosys/tests/sva`).
    - LEC lane logs now emit:
      `* LEC dropped-syntax summary: drop_remark_cases=...`.
18. Formal strict-gate LEC governance landed:
    - `run_formal_all.sh` now records `lec_drop_remark_cases` in summary
      telemetry for non-OpenTitan LEC lanes and persists:
      `lec_drop_remark_case_ids`,
      `lec_drop_remark_case_reason_ids` in baselines.
    - New strict-gate knobs:
      `--fail-on-new-lec-drop-remark-cases`,
      `--fail-on-new-lec-drop-remark-case-ids`,
      `--fail-on-new-lec-drop-remark-case-reasons`
      (enabled by `--strict-gate`).
19. Current no-drop limitation after LEC parity:
    - OpenTitan LEC lanes are still governed via strict X-prop counters/keys,
    not drop-remark case/reason artifacts.
    - Both BMC and LEC reason telemetry remain warning-pattern/log-derived
    rather than first-class lowering provenance tags.
20. OpenTitan LEC no-drop parity landed (February 10, 2026):
    - `run_opentitan_circt_lec.py` now exports optional dropped-syntax
      artifacts:
      `LEC_DROP_REMARK_CASES_OUT`, `LEC_DROP_REMARK_REASONS_OUT`.
    - `run_formal_all.sh` now wires these for:
      `opentitan/LEC` and `opentitan/LEC_STRICT`.
    - LEC drop-remark strict-gate case/reason checks now apply to all
      `LEC*` modes (including `LEC_STRICT`), not just plain `LEC`.
21. Updated no-drop limitation after OpenTitan LEC parity:
    - OpenTitan E2E lanes remain governed by E2E status/mode-diff gates,
      not dropped-syntax case/reason artifacts.
    - Drop reasons are still log-derived; long-term target remains
      first-class lowering provenance tags.
22. Absolute no-drop closure gate landed (February 10, 2026):
    - `run_formal_all.sh` now supports:
      `--fail-on-any-bmc-drop-remarks`,
      `--fail-on-any-lec-drop-remarks`.
    - These fail the run if current `*_drop_remark_cases` is non-zero,
      regardless of baseline drift.
23. Remaining policy decision:
    decide when to make absolute no-drop gating default in strict CI
    versus opt-in for targeted closure runs.
24. BMC struct-clock closure progress (February 10, 2026):
    - `lower-to-bmc` now accepts single `seq.clock` fields nested in
      `hw.struct` inputs by routing them through derived-clock synthesis
      (`seq.from_clock` materialization + prepended BMC clock input + assume
      equality wiring), instead of hard-failing.
    - Added regression:
      `test/Tools/circt-bmc/lower-to-bmc-struct-seq-clock-input.mlir`.
25. BMC mixed-clock closure progress (February 10, 2026):
    - `lower-to-bmc` now accepts mixed explicit top-level clocks plus
      struct-carried clocks when `allow-multi-clock=true`.
    - Struct-carried clocks are rewritten through synthesized BMC clock inputs;
      explicit top-level clock uses remain on their native BMC clock inputs.
    - Added regression:
      `test/Tools/circt-bmc/lower-to-bmc-mixed-clock-inputs.mlir`.
26. BMC single-clock conservatism reduction (February 10, 2026):
    - mixed explicit+struct clock designs in single-clock mode no longer fail
      solely because struct clock fields exist in the input type.
    - single-clock mode now rejects mixed designs only when a struct-carried
      clock domain is actually active in lowering (`clockInputs` non-empty).
    - Added regression:
      `test/Tools/circt-bmc/lower-to-bmc-mixed-clock-unused-struct.mlir`.
    - Remaining limitation:
      if both explicit and struct-derived domains are active and semantically
      equivalent only via non-trivial assumptions, single-clock mode still
      treats them as multiple clocks.
27. BMC SMT-LIB export fallback hardening for semantic-closure lanes
    (February 10, 2026):
    - `utils/run_sv_tests_circt_bmc.sh` now retries a failed
      `BMC_RUN_SMTLIB=1` invocation with JIT (`--run`/`--shared-libs`) when
      the failure is the known unsupported-export diagnostic:
      `for-smtlib-export does not support LLVM dialect operations inside
      verif.bmc regions`.
    - This removes non-semantic `ERROR` noise for UVM-heavy tests that still
      carry `llvm.call` in `verif.bmc` regions, while preserving SMT-LIB as the
      default primary backend.
    - Semantic lane impact:
      `sv-tests-uvm/BMC_SEMANTICS` now reports `fail=1 error=0` (previously
      `fail=0 error=1`), so remaining signal is true semantic outcome instead
      of backend-export capability mismatch.
28. Filter-discipline hardening for concurrent formal runs (February 10, 2026):
    - `run_sv_tests_circt_bmc.sh` and `run_sv_tests_circt_lec.sh` now require
      explicit caller filtering (`TAG_REGEX` or `TEST_FILTER`) and no longer
      rely on implicit default tag regexes.
    - `run_formal_all.sh` now passes explicit `TAG_REGEX` values to
      `sv-tests/BMC`, `sv-tests/LEC`, and `sv-tests-uvm/BMC_SEMANTICS` lanes.
    - Purpose:
      keep lane selection deterministic across parallel agents and avoid silent
      scope drift from implicit defaults.
29. Top-level explicit sv-tests filter controls in `run_formal_all.sh`
    (February 10, 2026):
    - Added explicit CLI knobs so callers own sv-tests lane scope:
      `--sv-tests-bmc-tag-regex`,
      `--sv-tests-lec-tag-regex`,
      `--sv-tests-uvm-bmc-semantics-tag-regex`.
    - Removed internal default tag-regex injection for sv-tests BMC/LEC/UVM
      BMC lanes; lane scope is now caller-configured at orchestration time.
    - Added regex validation for the new options to fail fast on malformed
      filter expressions.
30. sv-tests lane accounting hardening in `run_formal_all.sh`
    (February 10, 2026):
    - `sv-tests/BMC`, `sv-tests/LEC`, and `sv-tests-uvm/BMC_SEMANTICS` now
      emit explicit error rows when the underlying runner exits without
      producing a parseable summary line.
    - This closes a silent-drop path where lane failures could previously
      disappear from `summary.tsv` if the runner errored before summary output.
31. Optional strict preflight for explicit sv-tests filters in orchestration
    (February 10, 2026):
    - Added `--require-explicit-sv-tests-filters` in `run_formal_all.sh`.
    - When enabled, selected sv-tests lanes fail fast unless caller supplies
      explicit filters:
      - `sv-tests/BMC` requires `--sv-tests-bmc-tag-regex` or `TEST_FILTER`
      - `sv-tests/LEC` requires `--sv-tests-lec-tag-regex` or `TEST_FILTER`
      - `sv-tests-uvm/BMC_SEMANTICS` requires
        `--sv-tests-uvm-bmc-semantics-tag-regex` or `TEST_FILTER`
    - Purpose:
      enforce explicit lane scope policy across concurrent agents without
      immediately breaking existing default invocations.
32. Always-on explicit sv-tests filter contract in orchestration
    (February 10, 2026):
    - `run_formal_all.sh` now enforces explicit caller-owned filtering for
      selected sv-tests lanes by default (no opt-in gate required).
    - Added lane-specific base-name filter CLI knobs:
      - `--sv-tests-bmc-test-filter`
      - `--sv-tests-lec-test-filter`
      - `--sv-tests-uvm-bmc-semantics-test-filter`
    - sv-tests lane preflight now requires one explicit filter per selected lane:
      tag regex (`--sv-tests-*-tag-regex`) or test filter
      (`--sv-tests-*-test-filter`).
    - Removed implicit UVM semantic-lane case-name fallback in
      `run_formal_all.sh`; lane scope is now fully caller-defined.
33. Lane-scoped non-sv filter forwarding in orchestration
    (February 10, 2026):
    - Added lane-specific filter knobs in `run_formal_all.sh` for non-sv suites:
      - `--verilator-bmc-test-filter`
      - `--verilator-lec-test-filter`
      - `--yosys-bmc-test-filter`
      - `--yosys-lec-test-filter`
    - Orchestrator now passes lane-local `TEST_FILTER` values to
      `verilator-verification` and `yosys/tests/sva` BMC/LEC runners, removing
      implicit dependence on a shared process-level `TEST_FILTER`.
    - Purpose:
      prevent cross-agent filter collisions and keep lane selection
      deterministic in multi-runner formal workflows.
