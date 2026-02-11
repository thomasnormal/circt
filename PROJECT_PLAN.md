# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/mbit/*_avip` testbenches using only CIRCT tools and the library ~/uvm-core.
Secondary goal: Get to 100% in the ~/sv-tests/ and ~/verilator-verification/ test suites.

---

## Current Status - February 11, 2026

### Formal Closure Snapshot Update (February 11, 2026, 23:55)

1. Added deterministic multiclock BMC e2e parity anchors:
   - `test/Tools/circt-bmc/sva-multiclock-assume-domains-unsat-e2e.sv`
   - `test/Tools/circt-bmc/sva-multiclock-assume-domains-sat-e2e.sv`
   - expected outcomes validated in both JIT and SMT-LIB:
     - unsat case: `BMC_RESULT=UNSAT`
     - sat case: `BMC_RESULT=SAT`
2. Remaining formal limitations:
   - this multiclock pair currently uses assumption-anchored domain contracts;
     we still need a non-trivial cross-domain implication anchor that is
     deterministic without circular assumption structure.
   - LEC `semantic_diag_error` decomposition is still not split into stable
     parser/lowering/solver families.
   - mutation policy is currently point-in-time scoped; trend-aware scoped
     semantic gating remains open.
3. Next long-term features:
   - add a second multiclock pair targeting sequence-event-list semantics with
     strict SAT/UNSAT determinism and bucket tracking.
   - extend `run_formal_all.sh` semantic bucketing with finer LEC
     `semantic_diag_error` subfamilies and strict drift gates.
   - add `circt-mut` scoped semantic trend profiles over
     `verilator_verification/LEC` + `yosys_tests_sva/LEC` counters.

### Formal Closure Snapshot Update (February 11, 2026, 23:40)

1. Added a second scoped external-formal semantic guard profile in
   `circt-mut`:
   - `formal-regression-matrix-external-formal-semantic-guard-yosys`
   - rule enforces
     `external_formal.summary_counter_by_suite_mode.yosys_tests_sva.LEC.lec_error_bucket_semantic_diag_error_cases <= 0`.
2. Current formal limitations still open:
   - BMC deterministic multiclock semantic parity anchor is still missing in
     the e2e regression set and policy bundles.
   - LEC semantic decomposition is still coarse inside
     `semantic_diag_error` (needs parser/lowering/solver family split).
   - mutation policy still lacks trend-aware scoped semantic guard bundles for
     sustained closure quality (not just point-in-time strictness).
3. Next long-term features to build:
   - add deterministic multiclock SAT/UNSAT BMC e2e pair and wire into
     BMC semantic-bucket drift gates.
   - split `lec_error_bucket_semantic_diag_error_cases` into stable reason
     families and expose counters per suite/mode.
   - add scoped semantic trend profiles in `circt-mut` that combine
     `verilator` + `yosys` guard keys with history-window deltas.

### Formal Closure Snapshot Update (February 11, 2026, 23:20)

1. Added first native scoped formal-semantic policy profile in `circt-mut`:
   - `formal-regression-matrix-external-formal-semantic-guard`
   - currently targets `verilator_verification/LEC`
     `lec_error_bucket_semantic_diag_error_cases`.
2. Long-term impact:
   - moves mutation governance from global external-formal budgets toward
     lane-aware semantic debt enforcement.
3. Remaining limitations:
   - profile currently gates one scoped key; broader suite coverage and trend
     semantics are still needed.
   - BMC multiclock deterministic parity anchor still pending.
4. Next long-term features:
   - add companion scoped guards for `yosys/tests/sva/LEC` and selected BMC
     semantic counters.
   - add trend-aware variants for scoped formal semantic keys.
   - couple policy outcomes to mutator-family budget reallocation.

### Formal Closure Snapshot Update (February 11, 2026, 23:05)

1. `circt-mut report` now exports per-suite/per-mode external formal counters:
   - `external_formal.summary_counter_by_suite_mode.<suite>.<mode>.<key>`
2. Long-term value:
   - enables mutation policy and profile ranking to target active failing
     formal families at lane granularity (for example `verilator/LEC`
     semantic buckets) instead of only global aggregate deltas.
3. Remaining limitations:
   - we still do not have built-in policy profiles that consume the new scoped
     keys directly.
   - LEC semantic diagnostics still need deeper decomposition beneath
     `semantic_diag_error`.
4. Next long-term features:
   - add native policy profiles for strict-formal modes keyed to scoped
     `external_formal.summary_counter_by_suite_mode.*` thresholds.
   - add deterministic multiclock SAT/UNSAT BMC pair and track those counters
     in policy budgets.
   - connect scoped formal deltas to mutator-family auto-prioritization.

### Formal Closure Snapshot Update (February 11, 2026, 22:50)

1. Mutation/reporting now consumes formal summary bucket counters natively:
   - `circt-mut report` emits
     `external_formal.summary_counter.<key>` from external `summary.tsv`.
2. Long-term impact:
   - enables mutation policy profiles to gate/prioritize directly on formal
     semantic families (`lec_error_bucket_*`, `bmc_semantic_bucket_*`) without
     ad-hoc parsers.
3. Remaining limitations:
   - counter export is global sum only; no per-suite/per-mode split yet in
     `circt-mut` telemetry rows.
   - semantic granularity still depends on upstream `run_formal_all.sh`
     classifier quality.
4. Next long-term features:
   - add per-lane/per-suite prefixed counter exports in `circt-mut` for finer
     mutator routing.
   - define native mutation policy profiles keyed to
     `external_formal.summary_counter.lec_error_bucket_semantic_diag_error_cases`
     and BMC multiclock/subroutine counters.
   - keep tightening LEC semantic decomposition so mutation deltas map to
     actionable compiler subsystems.

### Formal Closure Snapshot Update (February 11, 2026, 22:35)

1. LEC semantic error telemetry now decomposes coarse `semantic_other`:
   - added semantic sub-buckets (`semantic_diag_error`,
     `semantic_result_tag`, `semantic_diag_missing`) plus `infra_config`.
2. Real signal impact:
   - filtered `verilator-verification/LEC` (`assert_changed`) now lands in
     `lec_error_bucket_semantic_diag_error_cases=1` instead of broad
     `semantic_other`.
3. Remaining limitations:
   - `semantic_diag_error` is still a transitional bucket; we still need finer
     parser/lowering/solver family attribution for stronger triage.
   - BMC multiclock deterministic SAT/UNSAT pair is still missing.
4. Next long-term features:
   - split `semantic_diag_error` into stable reason families (for example
     parser_import, lowering_pipeline, solver_contract) and add case-ID drift
     gates per family.
   - add deterministic multiclock SAT/UNSAT e2e parity pair and fold into
     strict-cadence profiles.
   - consume refined BMC/LEC bucket deltas directly in mutation profile ranking.

### Formal Closure Snapshot Update (February 11, 2026, 22:20)

1. OpenTitan LEC filtered no-match handling is now explicit and non-failing:
   - when no AES S-Box implementations match an impl filter, lane emits
     `SKIP` with `no_matching_impl_filter` instead of synthetic
     `missing_results` failure.
2. Real sampled cadence signal across target suites:
   - `avip/apb_avip` compile: PASS
   - `sv-tests/LEC` (selected non-UVM filter): empty/skip slice
   - `verilator-verification/LEC`: still `ERROR` with
     `lec_error_bucket_semantic_other_cases=1`
   - `opentitan/LEC` (`rv_timer|otp_ctrl` filter): now clean skip classification.
3. Remaining limitations:
   - `verilator` LEC semantic failures are still in the coarse
     `semantic_other` bucket and need finer reason-family decomposition.
   - BMC deterministic multiclock parity pair still missing.
4. Next long-term features:
   - split `lec_error_bucket_semantic_other_cases` into stable semantic
     subfamilies and add per-family case-ID drift gates.
   - add deterministic multiclock SAT/UNSAT e2e pair and wire it into strict
     formal cadence.
   - feed per-family BMC/LEC drift deltas into mutation profile ranking.

### Formal Closure Snapshot Update (February 11, 2026, 22:05)

1. Added deterministic BMC e2e micro-tests for sequence-subroutine semantics:
   - `sva-sequence-subroutine-call-unsat-e2e.sv`
   - `sva-sequence-subroutine-call-sat-e2e.sv`
2. Concrete signal now available:
   - JIT and SMT-LIB both agree on expected UNSAT/SAT outcomes for this
     subroutine family.
3. Remaining limitations:
   - these tests currently rely on solver-enabled environments and are marked
     unsupported in this local lit configuration.
   - multiclock semantic closure still lacks the same compact SAT/UNSAT pairing
     style used here for subroutine semantics.
4. Next long-term features:
   - add deterministic multiclock UNSAT/SAT pair tests with explicit
     backend-parity checks.
   - evolve from remark-based “ignored subroutine” behavior toward structured
     accounting (for example counters/reason families) so we can measure
     semantic impact vs benign ignored side-effect calls.
   - connect these deterministic BMC families to mutation profile scoring so
     mutator priorities track active formal semantic debt.

### Formal Closure Snapshot Update (February 11, 2026, 21:35)

1. Added focused BMC semantic drift governance:
   - `--fail-on-new-bmc-semantic-bucket-case-ids-for <bucket>` in
     `run_formal_all.sh` (repeatable).
2. Practical usage:
   - allows strict closure on selected families such as `multiclock` and
     `sequence_subroutine` while keeping other semantic buckets in observation
     mode.
3. Remaining limitations:
   - BMC closure signal for `sequence_subroutine` and `multiclock` still relies
     on harness-scale runs; we still need smaller deterministic e2e corpus for
     these families with explicit expected SAT/UNSAT outcomes.
   - semantic bucket taxonomy still has `unclassified` pressure in some paths,
     reducing prioritization precision.
4. Next long-term features:
   - build dedicated `circt-bmc` e2e micro-suite for multiclock/subroutine
     corner semantics and wire into nightly strict profile.
   - expand semantic bucketing from regex+metadata heuristics to compiler
     emitted stable category IDs where possible.
   - feed bucket-specific drift deltas into mutation profile ranking so mutator
     investment tracks active formal semantic debt.

### Formal Closure Snapshot Update (February 11, 2026, 21:20)

1. Added strict LEC error-bucket drift governance in `run_formal_all.sh`:
   - new summary counters: `lec_error_bucket_*_cases`
   - new baseline field: `lec_error_bucket_case_ids`
   - new gate: `--fail-on-new-lec-error-bucket-case-ids`
   - included in `--strict-gate` defaults.
2. Long-term impact:
   - separates infra/tool-pipeline churn from semantic LEC errors in baseline
     drift checks, reducing false-positive gate noise.
   - creates a stable bridge for mutation prioritization keyed by formal
     reason-family deltas.
3. Remaining limitations:
   - `semantic_other` is still broad; we need finer stable semantic buckets
     (for example, lowering, unsupported feature, and solver-model mismatch).
   - external filtered samples still include many empty/skip selections, so
     cadence runs need suite-specific non-empty target sets for stronger signal.
4. Next features to build (best long-term ROI):
   - split `semantic_other` into deterministic semantic families and gate new
     case IDs per family.
   - add BMC semantic bucket governance for multiclock and sequence-subroutine
     closure (case-ID drift and parity checks).
   - wire mutation scoring to formal bucket deltas so mutator profiles focus on
     active failing semantic families.

### Formal Closure Snapshot Update (February 11, 2026, 21:00)

1. Hardened LEC runner wrapper-launch retries for `Permission denied` in
   addition to `Text file busy` across `sv-tests`, `verilator`, and `yosys`
   LEC runners.
2. Concrete observed impact:
   - filtered `sv-tests/LEC` (`16.15--property-iff-uvm`) moved from
     runner-command infra error to clean pass with explicit `build-test/bin`
     toolchain.
3. Remaining limitations and next long-term targets:
   - `verilator` filtered LEC sample still fails via non-wrapper `ERROR`
     classification; semantic/lowering diagnostics remain the next closure
     area.
   - runner-command infra governance is now broad, but we still need
     lane-specific budget policies (for example, allow/deny by suite and trend).
   - continue BMC multiclock + sequence-subroutine semantic bucket closure and
     mutation-priority integration using formal reason-family deltas.

### Formal Closure Snapshot Update (February 11, 2026, 20:50)

1. Added explicit LEC infra budget governance for runner-command failures:
   - new aggregate summary counter: `lec_runner_command_cases`
   - new gates:
     - `--fail-on-new-lec-runner-command-cases`
     - `--fail-on-any-lec-runner-command-cases`
2. Strict-gate now covers runner-command drift at four granularities:
   - reason-key (`lec_runner_command_reason_*_cases`)
   - case IDs (`lec_runner_command_case_ids`)
   - case+reason tuples (`lec_runner_command_case_reasons`)
   - aggregate count (`lec_runner_command_cases`)
3. Remaining limitations and next long-term targets:
   - current filtered `sv-tests` and `verilator` LEC smokes still report
     `runner_command_permission_denied` infra errors
     (`lec_runner_command_cases=1`), so zero-any-case budgets are not yet
     generally deployable.
   - add lane-level infra budget policy (per-suite allowed burst/rate) instead
     of only absolute/new-count gates.
   - continue LEC semantic classification for remaining non-infra `ERROR`
     populations in `verilator` and `yosys` filtered lanes.
   - continue BMC semantic closure for multiclock + sequence-subroutine buckets
     with strict case-ID drift gates and mutation-priority integration.

### Formal Closure Snapshot Update (February 11, 2026, 20:30)

1. Added dedicated formal-governance counters for infra runner failures:
   - `lec_runner_command_reason_*_cases` emitted from LEC summaries in
     `run_formal_all.sh`.
2. Added new strict-gate control:
   - `--fail-on-new-lec-runner-command-reason-keys`
   - now enabled by default under `--strict-gate`.
3. Why this is long-term useful:
   - separates infrastructure execution regressions
     (`runner_command_not_found/permission_denied/text_file_busy/...`) from
     semantic/lowering reason-family drift in dashboards and gates.
4. Current limitations and next build targets:
   - continue reducing duplicated runner-side reason parsing (move toward shared
     helper/module and compiler-emitted category IDs).
   - add formal lane-level surfacing for `runner_command_*` case IDs/reasons
     (not just summary counter keys) to improve mutation prioritization input.
   - continue BMC semantic closure on multiclock + sequence-subroutine buckets
     with strict case-ID drift gates.

### Formal Closure Snapshot Update (February 11, 2026, 23:59)

1. Completed opt-stage parity with verilog-stage runner hardening:
   - stable infra reason taxonomy for `CIRCT_OPT_ERROR`
   - ETXTBSY retry with fallback copied binary in all three LEC runners.
2. Impact:
   - removes remaining high-cardinality wrapper/path noise from opt-stage error
     telemetry and improves strict-gate drift stability.
3. External filtered cadence signal after this change:
   - no immediate wrapper-class `CIRCT_OPT_ERROR`/`CIRCT_VERILOG_ERROR` churn in
     sampled `sv-tests`/`verilator`/`yosys` filtered runs.
   - current failures in filtered `verilator`/`yosys` samples are now LEC-stage
     `ERROR` outcomes, which is the next semantic/debugging target.
4. Remaining long-term features and limitations:
   - formal summaries still do not explicitly bucket infra `runner_command_*`
     families; add dedicated counters in `run_formal_all.sh`.
   - reason extraction remains log-heuristic; move toward compiler-emitted
     stable diagnostic category IDs.
   - continue BMC semantic closure on multiclock + sequence-subroutine with
     strict bucket case-ID drift gating.
   - mutation pipeline should consume formal reason-family deltas directly for
     profile gating and mutator prioritization.

### Formal Closure Snapshot Update (February 11, 2026, 23:58)

1. Added LEC verilog-front-end ETXTBSY hardening across all formal runners:
   - retry on wrapper `Text file busy`
   - fallback to a copied temp binary before classifying as error
2. Current signal after fix:
   - filtered external lanes now progress beyond `CIRCT_VERILOG_ERROR`
     ETXTBSY and fail later at `CIRCT_OPT_ERROR` timeout-wrapper stage.
3. Why this is high leverage:
   - removes a high-noise infrastructure failure mode that previously obscured
     real semantic/lowering bottlenecks in external cadence runs.
4. Remaining limitations and next long-term features:
   - `CIRCT_OPT_ERROR` reason taxonomy still contains wrapper/path-derived
     timeout strings; apply the same canonicalization and retry hardening used
     for verilog stage.
   - add explicit `runner_command_*` counter families in `run_formal_all.sh`
     summaries to separate infra churn from semantic regressions.
   - continue BMC semantic closure on multiclock + sequence-subroutine buckets
     with strict bucket case-ID drift gating.
   - mutation generation/reporting should consume formal reason-family deltas to
     prioritize mutator profiles against active breakage classes.

### Formal Closure Snapshot Update (February 11, 2026, 23:59)

1. Stabilized LEC `CIRCT_VERILOG_ERROR` reason normalization in all primary
   formal runners (`sv-tests`, `verilator`, `yosys`) for wrapper/infra faults:
   - canonical tokens now include:
     - `runner_command_not_found`
     - `runner_command_permission_denied`
     - `runner_failed_to_run_command`
     - `command_timeout`
     - `command_oom`
2. Why this matters long term:
   - removes host/path-dependent reason token churn that pollutes strict-gate
     drift checks and makes baseline history noisy.
3. Current concrete limitation signal:
   - filtered external LEC smoke cases still fail with
     `runner_command_permission_denied` (infra-level execution issue), now
     clearly surfaced as one stable bucket instead of high-cardinality tokens.
4. Next high-leverage features:
   - move reason taxonomy from log heuristics to compiler-emitted stable
     diagnostic category IDs (ImportVerilog/lowering/LEC pipeline).
   - add structured infra-failure counters in `run_formal_all.sh` for
     `runner_command_*` classes so infra regressions are visible separately from
     semantic failures.
   - continue BMC semantic closure on multiclock + sequence-subroutine with
     focused filtered lanes and strict bucket case-ID drift gates.
   - mutation pipeline: consume formal reason-family counters to prioritize
     mutators against active failure families.

### Formal Closure Snapshot Update (February 11, 2026, 23:55)

1. Completed strict-gate parity for LEC `CIRCT_VERILOG_ERROR` identity drift:
   - new gates:
     - `--fail-on-new-lec-circt-verilog-error-case-ids`
     - `--fail-on-new-lec-circt-verilog-error-case-reasons`
   - both are now included in `--strict-gate` defaults.
2. Baseline state now persists full verilog-error identity telemetry:
   - `lec_circt_verilog_error_case_ids`
   - `lec_circt_verilog_error_case_reasons`
3. This closes the previously noted governance gap:
   - verilog-error is now tracked at reason-key, case-id, and case+reason
     levels, matching opt-error governance depth.
4. Remaining long-term limitations and next features:
   - error reason extraction is still log-derived string normalization; we need
     compiler-emitted stable diagnostic category IDs for low-churn governance.
   - formal runner telemetry helpers are duplicated; extract shared library
     helpers for reason parsing/cache keying to reduce drift and maintenance
     cost.
   - mutation generation/reporting should ingest formal reason-family deltas
     (opt/verilog case+reason) to prioritize mutator families that target active
     breakage buckets.
   - next BMC semantic closure focus remains multiclock and sequence-subroutine
     buckets with strict drift gates on bucket case IDs.

### Formal Closure Snapshot Update (February 11, 2026, 23:30)

1. Added reason-level governance for LEC `CIRCT_VERILOG_ERROR`:
   - runner reason extraction in `sv-tests` / `verilator` / `yosys` LEC scripts
   - summary counters:
     - `lec_circt_verilog_error_reason_<token>_cases`
   - new strict gate:
     - `--fail-on-new-lec-circt-verilog-error-reason-keys`
2. Strict-gate now governs both front-end and middle-end error families:
   - `CIRCT_VERILOG_ERROR` reason-key drift
   - `CIRCT_OPT_ERROR` case-ID / reason-key / case+reason drift
3. Why this is high leverage:
   - prevents silent migration of failure pressure between compile stages by
     making reason-family churn explicit in lane governance.
4. Remaining limitations and next long-term features:
   - reason extraction still normalizes free-form logs; long-term stable
     category IDs from compiler diagnostics are needed.
   - next governance step should add `CIRCT_VERILOG_ERROR` case-id and
     case+reason tuple drift (not just reason keys) for parity with opt-error.
   - mutation reporting should ingest these reason counters to prioritize
     mutators against active formal breakage families.

### Formal Closure Snapshot Update (February 11, 2026, 23:10)

1. Added strict governance for LEC `CIRCT_OPT_ERROR` case+reason identity
   drift in `run_formal_all.sh`:
   - new gate: `--fail-on-new-lec-circt-opt-error-case-reasons`
   - new baseline field: `lec_circt_opt_error_case_reasons`
2. Strict-gate now enforces three tiers for this class:
   - case IDs (`lec_circt_opt_error_case_ids`)
   - reason keys (`lec_circt_opt_error_reason_*_cases`)
   - case+reason tuples (`lec_circt_opt_error_case_reasons`)
3. Long-term impact:
   - this closes a governance blind spot where stable case sets could still
     drift into new failure reasons undetected.
4. Remaining limitations and next high-leverage features:
   - reason extraction is still log-heuristic; migrate to stable, structured
     reason categories emitted by lowering/pipeline stages.
   - formal runner helper duplication (cache/reason normalization) should be
     consolidated into shared utilities.
   - mutation pipeline should start ingesting these LEC reason-family counters
     for profile gating and targeted mutator prioritization.

### Formal Closure Snapshot Update (February 11, 2026, 22:55)

1. Added reason-level telemetry for LEC `CIRCT_OPT_ERROR` in runner outputs:
   - `run_sv_tests_circt_lec.sh`
   - `run_verilator_verification_circt_lec.sh`
   - `run_yosys_sva_circt_lec.sh`
2. Added reason-key summarization and governance in `run_formal_all.sh`:
   - summary keys:
     - `lec_circt_opt_error_reason_<token>_cases`
   - new gate:
     - `--fail-on-new-lec-circt-opt-error-reason-keys`
   - enabled by default under `--strict-gate`.
3. Added regression locks for reason-key drift and runner extraction behavior.
4. Why this is high leverage:
   - shifts LEC `CIRCT_OPT_ERROR` tracking from opaque aggregate errors to
     structured reason-family drift, improving prioritization and long-term
     closure planning.
5. Remaining limitations and next features:
   - reason tokens are still first-line heuristics from logs; we should migrate
     toward explicit machine-stable categories emitted by `circt-opt`/pipeline
     stages (or sidecar metadata) to avoid normalization churn.
   - duplicate cache/reason helper code across runner scripts remains technical
     debt; shared helper extraction should be next for maintainability.
   - formal summary should add top-K reason/case joins per lane for faster
     triage at scale.

### Formal Closure Snapshot Update (February 11, 2026, 22:35)

1. Added strict governance for LEC `CIRCT_OPT_ERROR` identity drift in
   `utils/run_formal_all.sh`:
   - new gate: `--fail-on-new-lec-circt-opt-error-case-ids`
   - new baseline field: `lec_circt_opt_error_case_ids`
2. Strict-gate hardening:
   - `--strict-gate` now includes this LEC case-ID drift check by default.
3. Added regression locks:
   - `test/Tools/run-formal-all-strict-gate-lec-circt-opt-error-case-ids.test`
   - `test/Tools/run-formal-all-strict-gate-lec-circt-opt-error-case-ids-defaults.test`
4. Why this matters long term:
   - cache improvements reduce runtime cost, but this gate makes persistent
     `CIRCT_OPT_ERROR` churn auditable as concrete case-identity regressions,
     not just aggregate error counts.
5. Remaining limitations and next features:
   - still need root-cause taxonomy for `CIRCT_OPT_ERROR` classes in lowering
     (shape/clocking/diag-source buckets) to prioritize real semantic fixes.
   - duplicated cache helper logic across runners is still technical debt;
     shared helper extraction remains next maintainability step.
   - formal lane governance should add per-lane cache effectiveness rollups and
     bounded cache policy controls for CI longevity.

### Formal Closure Snapshot Update (February 11, 2026, 22:10)

1. Extended LEC front-end MLIR caching to external-suite runners:
   - `utils/run_verilator_verification_circt_lec.sh`
   - `utils/run_yosys_sva_circt_lec.sh`
   - shared env toggle: `LEC_MLIR_CACHE_DIR`
2. Added dedicated regression tests:
   - `test/Tools/run-verilator-verification-circt-lec-mlir-cache.test`
   - `test/Tools/run-yosys-sva-circt-lec-mlir-cache.test`
3. Focused external sanity (single-case run pairs, explicit build-test tools):
   - `verilator-verification/assert_changed`: run1 miss/store, run2 hit
   - `yosys/tests/sva/basic00`: run1 miss/store, run2 hit
4. Limitation signal remains unchanged:
   - both sampled external cases still end in `CIRCT_OPT_ERROR`, indicating
     semantic/lowering problems orthogonal to caching.
5. Long-term features and tech debt priorities:
   - close `CIRCT_OPT_ERROR` root causes with stage-specific diagnostics and
     bucketed provenance in `run_formal_all.sh`.
   - deduplicate runner-local cache helper logic (`hash_key/hash_file`) into a
     shared formal helper script to prevent drift.
   - add cache budget governance (`max bytes` / eviction policy) and per-lane
     cache effectiveness rollups in formal summaries.

### Formal Closure Snapshot Update (February 11, 2026, 21:45)

1. Added sv-tests LEC front-end compile cache in
   `utils/run_sv_tests_circt_lec.sh`:
   - opt-in via `LEC_MLIR_CACHE_DIR`
   - deterministic keying over tool identity, flags, include/define context, and
     source content hashes.
2. Added LEC cache telemetry:
   - `sv-tests frontend cache summary: hits=... misses=... stores=...`
3. Added regression lock:
   - `test/Tools/run-sv-tests-lec-mlir-cache.test`
4. Focused external sanity signal (`/home/thomas-ahle/sv-tests`, explicit
   build-test tools):
   - same filtered case run twice demonstrates compile cache reuse
     (miss/store on run1, hit on run2).
   - case still ends in `CIRCT_OPT_ERROR` in this workspace snapshot, indicating
     a semantic/toolchain issue orthogonal to cache behavior.
5. Remaining limitations and next long-term features:
   - unify BMC/LEC cache key schema into shared helper to reduce runner drift
     and tech debt.
   - extend cache to verilator/yosys LEC runners for portfolio-scale closure
     loops.
   - add cache budget controls (max bytes/entries + eviction policy) and lane
     telemetry roll-up in `run_formal_all.sh`.

### Formal Closure Snapshot Update (February 11, 2026, 21:25)

1. Added sv-tests BMC front-end compile cache in
   `utils/run_sv_tests_circt_bmc.sh`:
   - opt-in via `BMC_MLIR_CACHE_DIR`
   - deterministic keying over tool flags + file content hashes
   - cache summary telemetry emitted per run (hits/misses/stores).
2. New regression lock:
   - `test/Tools/run-sv-tests-bmc-mlir-cache.test`
   - validates compile dedup across repeated identical runs.
3. Long-term impact:
   - reduces repeated closure-loop front-end cost for stable case sets, which is
     a prerequisite for practical multiclock/subroutine semantic closure at
     scale.
4. Current limitation signal:
   - external filtered `verilator-verification` and `yosys/tests/sva` formal
     slices are currently red in this workspace snapshot (broad errors/fails),
     suggesting global environment/tool drift outside this cache-only patch.
5. Next high-leverage features:
   - extend the same cache pattern to LEC runners (`sv-tests`/`verilator`/`yosys`)
     with shared key schema and invalidation policy.
   - add cache eviction policy + size guardrails (`max entries` / `max bytes`)
     for long-running CI workers.
   - tie cache hit/miss telemetry into `run_formal_all.sh` lane summaries for
     governance on performance regressions (not only semantic regressions).

### Formal Closure Snapshot Update (February 11, 2026, 21:05)

1. Added explicit BMC semantic coverage for sequence-subroutine failures:
   - new semantic bucket: `sequence_subroutine`
   - new summary counter:
     - `bmc_semantic_bucket_sequence_subroutine_cases`
2. Strict-gate integration:
   - `--fail-on-new-bmc-semantic-bucket-cases` now also checks
     `sequence_subroutine` bucket drift.
3. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-bmc-semantic-bucket-cases-sequence-subroutine.test`
4. Focused validation:
   - lit semantic-bucket stack remains green (7/7).
   - external filtered smokes (`verilator-verification`, `yosys/tests/sva`)
     remain passing with explicit `build-test/bin` tools.
5. Current limitation signal:
   - sv-tests UVM `sequence-subroutine`/`multiclock` focused reruns are still
     dominated by front-end compile latency in this environment, limiting
     per-iteration semantic signal throughput.
6. Long-term next features (highest leverage):
   - introduce direct BMC lane support for precompiled/partitioned sv-tests UVM
     front-end artifacts (IR cache keying by case + tool hash) to cut closure
     loop time.
   - add semantic-bucket case-map provenance join with timeout metadata to
     distinguish semantic misses vs compile-time pressure automatically.
   - extend mutation report pipeline with profile-aware incremental cache reuse
     and deterministic shard scheduling for AVIP-scale runs.

### Formal Closure Snapshot Update (February 11, 2026, 20:45)

1. Added semantic-bucket identity drift governance in `run_formal_all.sh`:
   - new gate: `--fail-on-new-bmc-semantic-bucket-case-ids`
   - new baseline field: `bmc_semantic_bucket_case_ids`
   - identity format: `bucket::case_id`
2. Strict-gate hardening:
   - `--strict-gate` now enables BMC semantic-bucket case-ID drift checks by
     default.
3. Legacy baseline compatibility:
   - semantic-bucket case-ID drift enforcement activates only when baseline rows
     already carry `bmc_semantic_bucket_case_ids`.
4. New regression locks:
   - `test/Tools/run-formal-all-strict-gate-bmc-semantic-bucket-case-ids.test`
   - `test/Tools/run-formal-all-strict-gate-bmc-semantic-bucket-case-ids-defaults.test`
5. Focused validation:
   - lit strict-gate semantic bucket stack: PASS (8/8).
   - external filtered smokes:
     - `/home/thomas-ahle/verilator-verification` BMC/LEC: PASS.
     - `/home/thomas-ahle/yosys/tests/sva` BMC/LEC: PASS.
6. Remaining long-term limitations and next features:
   - BMC still needs semantic closure implementation for multiclock and
     sequence-subroutine lowering/runtime behavior (beyond drift governance).
   - LEC timeout taxonomy still needs model-size/proof-shape provenance to
     reduce `unknown` class.
   - Mutation generation/reporting still needs deterministic
     shard+cache lineage (IR hash + policy profile + external-formal digest),
     plus profile-aware incremental invalidation to keep AVIP-scale runs fast.

### Formal Closure Snapshot Update (February 11, 2026, 20:20)

1. Added BMC timeout identity telemetry and drift gating in
   `run_formal_all.sh`:
   - new counter/key: `bmc_timeout_case_ids`
   - new gate: `--fail-on-new-bmc-timeout-case-ids`
2. Strict-gate hardening:
   - `--strict-gate` now also enables BMC timeout case-ID drift checks by
     default, aligning BMC governance with LEC timeout identity controls.
3. Baseline compatibility:
   - baseline TSV now stores `bmc_timeout_case_ids`.
   - legacy rows remain compatible (drift gate only enforces when baseline rows
     carry timeout-ID telemetry).
4. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-bmc-timeout-case-ids.test`
5. External sanity:
   - `/home/thomas-ahle/yosys/tests/sva` BMC slice (`basic00|basic01`) is
     green with explicit `build-test/bin` tools and emits
     `bmc_timeout_case_ids=` in summary telemetry.
6. Remaining long-term limitations and next feature priorities:
   - BMC semantic closure still needs concrete lowering/runtime work for
     multiclock and sequence-subroutine tails (not just governance).
   - LEC timeout classes still need model-size evidence wiring from solver logs
     to reduce `unknown` attribution and improve remediation.
   - Mutation generation/reporting needs deterministic shard/cache policy for
     large AVIP portfolios (IR hash + policy fingerprint + external-formal
     digest lineage).

### Formal Closure Snapshot Update (February 11, 2026, 20:00)

1. Upgraded LEC timeout provenance from heuristic-only to runner-authored
   classification:
   - `run_sv_tests_circt_lec.sh`
   - `run_verilator_verification_circt_lec.sh`
   - `run_yosys_sva_circt_lec.sh`
2. Timeout status normalization:
   - preprocessing stage timeouts now emit `TIMEOUT` + class `preprocess`.
   - solver-stage LEC timeouts now emit `TIMEOUT` + class `solver_budget`.
3. Baseline summarization hardening in `run_formal_all.sh`:
   - optional row-level timeout class override (7th column) now overrides
     diag-token inference in `summarize_lec_case_file()`.
4. Regression lock:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout-class-override.test`
5. External sanity (explicit `build-test/bin` tools) remains green:
   - `/home/thomas-ahle/sv-tests` LEC slice
     (`16.15--property-iff-uvm|16.15--property-iff-uvm-fail`)
   - `/home/thomas-ahle/verilator-verification` LEC slice
     (`assert_past|assert_stable`)
   - `/home/thomas-ahle/yosys/tests/sva` LEC slice (`basic00|basic01`)
6. Remaining long-term limitations and next build targets:
   - BMC semantic closure still has uncovered multiclock and
     sequence-subroutine edge buckets; add strict-gated semantic bucket maps
     for those ids.
   - LEC timeout root-cause taxonomy still lacks model-size evidence from
     solver transcripts; add structured timeout telemetry extraction to reduce
     `unknown` bucket share.
   - Mutation pipeline should add deterministic sharding + cache-key lineage
     (IR hash + policy fingerprint + external-formal digest) to scale report
     generation on large AVIP portfolios.

### Formal Closure Snapshot Update (February 11, 2026, 19:14)

1. Added normalized LEC timeout-class telemetry in `run_formal_all.sh`:
   - `lec_timeout_class_solver_budget_cases`
   - `lec_timeout_class_preprocess_cases`
   - `lec_timeout_class_model_size_cases`
   - `lec_timeout_class_unknown_cases`
2. Added strict-gate timeout-class drift control:
   - `--fail-on-new-lec-timeout-class-cases`
   - detects class-count increases vs baseline for `LEC*` lanes.
3. Strict-gate behavior:
   - `--strict-gate` now enables timeout-class drift checks.
   - legacy baseline compatibility preserved (if historical rows predate class
     counters, strict mode skips class drift enforcement until telemetry exists).
4. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout-class-cases.test`
5. External sanity:
   - `/home/thomas-ahle/verilator-verification` LEC filtered slice
     (`assert_past|assert_stable`) remains PASS and now emits zeroed timeout
     class counters explicitly.
6. Remaining strategic limitations:
   - class taxonomy currently inferred from diag tags only; it still needs
     richer provenance wiring from runners/passes for higher-fidelity root cause
     attribution (pipeline stage + solver transcript class + model-size hints).
   - BMC multiclock/subroutine semantic closure still needs additional targeted
     attribution and strict-gate coverage.

### Formal Closure Snapshot Update (February 11, 2026, 19:12)

1. Added LEC timeout provenance buckets in `run_formal_all.sh` summaries:
   - `lec_timeout_diag_<diag>_cases`
   - `lec_timeout_diag_missing_cases`
2. Added strict-gate drift control for timeout provenance keys:
   - `--fail-on-new-lec-timeout-diag-keys`
   - detects new timeout diagnostic buckets across `LEC*` lanes vs baseline.
3. Strict-gate behavior:
   - `--strict-gate` now enables timeout diagnostic key drift checks.
   - legacy baseline compatibility preserved (skip drift enforcement when
     baseline predates timeout-diagnostic key emission).
4. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout-diag-keys.test`
5. External sanity:
   - `/home/thomas-ahle/verilator-verification` LEC filtered slice
     (`assert_past|assert_stable`) remains green and now emits
     `lec_timeout_diag_missing_cases=0` in summary telemetry.
6. Remaining strategic gaps:
   - convert timeout provenance from diag-key buckets into higher-level
     semantic classes (solver-budget / preprocessing / model-size / unknown).
   - drive BMC multiclock + sequence-subroutine closure with dedicated
     semantic-bucket strict gates and attribution.
   - mutation report scaling: add incremental ingest cache invalidation keyed
     by external-formal artifact hash + profile policy fingerprint.

### Formal Closure Snapshot Update (February 11, 2026, 19:10)

1. Added hard zero-timeout gate for LEC lanes in `run_formal_all.sh`:
   - `--fail-on-any-lec-timeouts`
2. New strict policy capability:
   - this enforces `lec_timeout_cases == 0` across any selected `LEC*` lanes
     in the current run (independent of baseline-window drift).
3. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout-any.test`
4. External sanity:
   - `/home/thomas-ahle/verilator-verification` LEC filtered slice
     (`assert_past|assert_stable`) with `--fail-on-any-lec-timeouts` is
     passing and reports `lec_timeout_cases=0`.
5. Next long-term build priorities after this governance step:
   - LEC timeout root-cause taxonomy/provenance (bucketed by solver budget,
     pipeline stage, and semantic class) to make timeout regressions actionable.
   - BMC semantic closure for multiclock + sequence-subroutine tails.
   - mutation matrix scalability hardening (incremental caching/sharding for
     external formal result ingestion).

### Formal Closure Snapshot Update (February 11, 2026, 19:02)

1. Tightened LEC timeout governance in `run_formal_all.sh`:
   - added `--fail-on-new-lec-timeout-case-ids`
   - gate now detects growth in timeout *identity set*, not just counts.
2. Baseline telemetry extension:
   - baseline TSV now records `lec_timeout_case_ids` per LEC lane.
   - legacy baseline compatibility is preserved (empty/missing field => no
     case-ID drift enforcement until data exists).
3. Strict-gate default hardening:
   - `--strict-gate` now automatically enables
     `--fail-on-new-lec-timeout-case-ids`.
4. New regression lock:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout-case-ids.test`.
5. Remaining long-term limitations after this slice:
   - LEC still needs semantic/perf closure on UVM sequence-subroutine and
     multiclock-heavy cases that currently converge to TIMEOUT.
   - BMC still needs deeper bucket closure for multiclock/subroutine edges.
   - mutation matrix still needs stronger external-formal ingestion scaling
     controls (cache/shard/provenance compaction) for large portfolios.

### Formal Closure Snapshot Update (February 11, 2026, 18:52)

1. Added LEC timeout strict-gate control in `run_formal_all.sh`:
   - `--fail-on-new-lec-timeout-cases`
2. Strict-gate parity improvement:
   - this mirrors existing BMC timeout drift checks and now fails when
     `lec_timeout_cases` grows vs baseline window.
   - compatibility preserved for legacy summary counters via fallback to
     `lec_status_timeout_cases` when needed.
3. LEC case-summary normalization:
   - `summarize_lec_case_file()` now emits explicit `lec_timeout_cases`
     aggregate in addition to status-scoped counters.
4. New regression locks:
   - `test/Tools/run-formal-all-strict-gate-lec-timeout.test`
   - help surface updated in `test/Tools/run-formal-all-help.test`
5. External sanity (live suite):
   - `/home/thomas-ahle/yosys/tests/sva` filtered BMC+LEC
     (`basic00|basic01`) passed with explicit timeout controls
     (`--bmc-timeout-secs 120 --lec-timeout-secs 120`).

### Formal Closure Snapshot Update (February 11, 2026, 17:05)

1. Verilator sampled-value closure:
   - `verilator-verification/tests/asserts/assert_stable.sv` now passes in BMC
     (`BMC_RESULT=UNSAT`) with explicit `build-test/bin` tools.
   - filtered suite run (`assert_stable$`) now reports:
     `total=1 pass=1 fail=0 xfail=0`.
2. Root-cause closure:
   - fixed LLHD register-state probe stripping in
     `StripLLHDInterfaceSignals.cpp` (single unguarded zero-time
     register-backed signal drives now resolve probes to register state).
3. Debt retirement:
   - removed `assert_stable` from `utils/verilator-bmc-xfails.txt`.
4. New regression lock:
   - `test/Tools/circt-bmc/sva-stable-toggling-unsat-e2e.sv`.

### Formal Closure Snapshot Update (February 11, 2026, 17:35)

1. Yosys BMC lane drift-hardening landed:
   - `run_yosys_sva_circt_bmc.sh` now emits explicit mode/profile diagnostics
     per row (`PASS_KNOWN`, `FAIL_KNOWN`, `PASS_XPROP`, `FAIL_XPROP`, ...).
2. Long-term impact:
   - keeps `base` IDs stable while enabling unambiguous `base_diag` matching in
     expected-failure-case governance (`run_formal_all.sh`) for dual-mode rows.
   - prevents pass-mode/fail-mode row conflation for identical test base IDs.
3. Focused external sanity:
   - `yosys/tests/sva` filtered slice (`basic00|basic01`, known profile) remains
     fully passing in both pass/fail modes with mode-qualified diagnostics.
4. Regression locks updated:
   - `test/Tools/run-yosys-sva-bmc-out-file.test`
   - `test/Tools/run-yosys-sva-bmc-semantic-tag-map.test`

### Formal Closure Snapshot Update (February 11, 2026, 17:50)

1. `run_formal_all.sh` yosys BMC lane now preserves runner-reported
   `xfail/xpass` counters in lane summaries and `summary.tsv`.
2. New regression lock:
   - `test/Tools/run-formal-all-yosys-bmc-xfail-summary.test`
3. This closes a formal-driver accounting gap where expected-failure signals in
   yosys BMC runs could be silently flattened to zero in aggregate dashboards.

### Formal Closure Snapshot Update (February 11, 2026, 17:58)

1. Added explicit yosys BMC profile selection in `run_formal_all.sh`:
   - `--yosys-bmc-profile auto|known|xprop` (default `auto`)
2. Profile-to-runner forwarding is now deterministic:
   - `known` -> `BMC_ASSUME_KNOWN_INPUTS=1`
   - `xprop` -> `BMC_ASSUME_KNOWN_INPUTS=0`
   - `auto` -> preserve previous global-flag-driven behavior
3. New regression lock:
   - `test/Tools/run-formal-all-yosys-bmc-profile-forwarding.test`
4. Focused external sanity (`/home/thomas-ahle/yosys/tests/sva`,
   `basic00|basic01`) remains coherent:
   - known profile: `2 tests, failures=0, xfail=0`
   - xprop profile: `2 tests, failures=0, xfail=2`

### Formal Closure Snapshot Update (February 11, 2026, 18:08)

1. LEC runner diagnostics hardened for silent `circt-opt` failures:
   - `run_sv_tests_circt_lec.sh`
   - `run_verilator_verification_circt_lec.sh`
   - `run_yosys_sva_circt_lec.sh`
   now emit:
   - `error: circt-opt failed without diagnostics for case '<base>'`
2. New regression lock:
   - `test/Tools/run-sv-tests-lec-silent-opt-diagnostic.test`
3. Focused external sv-tests repro with explicit build-test tools:
   - `16.11--sequence-subroutine-uvm`: `CIRCT_OPT_ERROR` (now with fallback diag)
   - `16.13--sequence-multiclock-uvm`: `CIRCT_OPT_ERROR` (now with fallback diag)
   - this confirms the next LEC semantic blocker is still
     `strip-llhd-interface-signals` pass behavior on UVM-heavy sequence cases.

### Formal Closure Snapshot Update (February 11, 2026, 18:14)

1. `strip-llhd-interface-signals` pass now emits non-silent context on failure:
   - `StripLLHDInterfaceSignals.cpp` `runOnOperation()` emits explicit
     op-local errors before `signalPassFailure()`.
   - strict comb-loop regression now also checks:
     `failed to lower llhd.combinational for LEC`
     (`test/Tools/circt-lec/lec-strict-llhd-comb-loop.mlir`).
2. Focused external sv-tests repro (`16.11--sequence-subroutine-uvm`):
   - `circt-opt` per-case log now reports:
     `error: expected llhd.signal in hw.module for LEC`
   - runner classification remains `CIRCT_OPT_ERROR` (as expected), but
     root-cause diagnostics are now pass-local and actionable.

### Formal Closure Snapshot Update (February 11, 2026, 18:40)

1. LEC preprocessing closure landed for UVM sequence stress cases:
   - `StripLLHDInterfaceSignals.cpp` now scopes LLHD stripping/no-LLHD checks
     to `hw.module` regions (ignores helper-region LLHD).
   - `circt-lec` now registers `circt::moore::MooreDialect`, removing parser
     failures on residual `moore.*` helper ops.
2. New regression locks:
   - `test/Tools/circt-lec/lec-strip-llhd-ignore-non-hw-module.mlir`
   - `test/Tools/circt-lec/lec-moore-dialect-parse.mlir`
3. Focused external sv-tests outcome shift:
   - `16.11--sequence-subroutine-uvm` and `16.13--sequence-multiclock-uvm`
     no longer hit `CIRCT_OPT_ERROR`.
   - forced LEC mode now reaches solver and reports `TIMEOUT`, which is the
     new remaining blocker for these cases.

### Formal Closure Snapshot Update (February 11, 2026, 18:46)

1. Added formal-driver timeout governance in `run_formal_all.sh`:
   - `--bmc-timeout-secs N`
   - `--lec-timeout-secs N`
2. The driver now forwards lane-specific `CIRCT_TIMEOUT_SECS` overrides to:
   - BMC lanes: sv-tests/verilator/yosys
   - LEC lanes: sv-tests/verilator/yosys
3. Timeout policies are now part of lane-state config hashing:
   - `bmc_timeout_secs`
   - `lec_timeout_secs`
   so resume/provenance captures timeout-tuning changes explicitly.
4. New regression lock:
   - `test/Tools/run-formal-all-timeout-forwarding.test`

### Formal Closure Snapshot (February 11, 2026, 16:22)

1. sv-tests focused semantic closure (SMT-LIB lane mode, explicit build-test tools):
   - PASS `16.10--property-local-var-uvm`
   - PASS `16.10--sequence-local-var-uvm`
   - PASS `16.11--sequence-subroutine-uvm`
   - PASS `16.13--sequence-multiclock-uvm`
   - PASS `16.15--property-iff-uvm`
   - XFAIL `16.15--property-iff-uvm-fail` (known metadata/content mismatch)
2. verilator-verification targeted formal slice:
   - BMC: `assert_past` PASS, `assert_stable` PASS (`BMC_RESULT=UNSAT`)
   - LEC: `assert_past` PASS, `assert_stable` PASS (`LEC_RESULT=EQ`)
3. yosys/tests/sva targeted formal slice:
   - BMC mode check: `basic00` PASS, `basic01` mismatch in pass-profile
     (`utils/yosys-sva-bmc-expected.txt` currently marks `basic01 pass xfail`)
   - LEC: `basic00` PASS, `basic01` PASS (`LEC_RESULT=EQ`)
4. AVIP smoke cadence:
   - `apb_avip` compile pass with `CIRCT_VERILOG=build-test/bin/circt-verilog`
     (exit code 0, emitted `avip-circt-verilog.log`)

### Formal Limitations And Long-Term Build Targets

1. BMC semantic limits still open:
   - Yosys SVA expectation governance still needs strict lane-owned
     expected-case checks that fail fast on profile drift in CI.
   - `run_formal_all` still lacks yosys-specific tag-regex controls
     (currently only test-filter is lane-specific), which can make targeted
     non-empty slices less ergonomic than sv-tests lanes.
   - mixed 4-state/2-state expectations still leak into targeted external lanes
2. LEC hardening limits still open:
   - UVM-heavy sequence cases (`16.11--sequence-subroutine-uvm`,
     `16.13--sequence-multiclock-uvm`) now clear preprocessing but timeout in
     forced LEC mode; next closure is solver/runtime tractability.
   - next long-term step: add timeout-budget governance in strict gate mode
     (LEC timeout regression checks analogous to existing BMC timeout gating).
   - maintain strict no-waiver X-prop governance while improving diagnostics depth
   - keep cross-suite drop-remark accounting deterministic in strict gates
3. Mutation generation limits still open:
   - complete native matrix scheduling migration (script parity -> native parity)
   - strengthen lane-level trend drift guardrails for nightly/strict policy bundles
4. Priority implementation order (formal-only tracks):
   - P0: add strict lane-owned expected-case policy checks for yosys BMC/LEC
     (profile-aware fail-fast gates in formal driver paths)
   - P1: normalize yosys `basic01` pass/fail profile expectation accounting
   - P1: add strict lane-level formal closure report artifact generation
   - P2: extend mutation report trend governance with prequalify/bucket deltas

### Recent Formal Infra Upgrade (February 11, 2026)

1. Added explicit Verilator BMC expected-failure forwarding in
   `utils/run_formal_all.sh`:
   - new option: `--verilator-bmc-xfails FILE`
   - forwards as runner env: `XFAILS=FILE`
   - validates missing files early and includes value in lane-state config hash
2. Added regression lock:
   - `test/Tools/run-formal-all-verilator-bmc-xfails-forwarding.test`

### Formal Driver Stabilization Update (February 11, 2026)

1. Added default Verilator-BMC expected-failure file:
   - `utils/verilator-bmc-xfails.txt`
   - originally seeded with sampled-value mismatch cases; now intentionally
     kept minimal (current file is empty after `assert_stable` closure).
2. `utils/run_formal_all.sh` now auto-loads that default list when present
   alongside the script, while keeping explicit override support:
   - `--verilator-bmc-xfails FILE`
3. Added coverage:
   - `test/Tools/run-formal-all-verilator-bmc-default-xfails-forwarding.test`
4. External validation:
   - forwarding path remains validated via
     `test/Tools/run-formal-all-verilator-bmc-default-xfails-forwarding.test`.
5. Semantic closure status:
   - this infra path is now in steady state; `$stable` parity is closed.

### Formal Backend Hardening (February 11, 2026)

1. Closed a VerifToSMT known-input modeling gap for alias-wrapped 4-state types:
   - `lib/Conversion/VerifToSMT/VerifToSMT.cpp` now unwraps
     `!hw.typealias<...>` when detecting `{value, unknown}` 4-state structs.
   - `--assume-known-inputs` now consistently constrains aliased inputs
     (`unknown == 0`) for both BMC and LEC lowering.
2. Added conversion regressions:
   - `test/Conversion/VerifToSMT/bmc-assume-known-inputs.mlir`
   - `test/Conversion/VerifToSMT/lec-assume-known-inputs.mlir`
   - `test/Conversion/VerifToSMT/four-state-input-warning.mlir`
3. Remaining top semantic limiter remains unchanged:
   - superseded: `assert_stable` is now closed in BMC and LEC.

### MooreToCore `$past` Initialization Hardening (February 11, 2026)

1. `moore.past` lowering now emits deterministic initialized delay registers in
   clocked assertion contexts, rather than unconstrained startup state.
2. Aggregate/4-state payloads are now handled via integer bitcast register
   storage with deterministic integer initialization and bitcast-back.
3. Regression lock:
   - `test/Conversion/MooreToCore/past-assert-compare.sv` checks
     `seq.compreg ... initial` appears in lowered IR.
4. Post-change sanity:
   - `sv-tests` BMC `16.10--property-local-var-uvm` remains PASS.
5. Remaining limitation:
   - superseded: `$stable` mismatch is now closed; next BMC P0 is yosys
     expectation drift (`basic01` pass-profile accounting).

### BMC Provenance Hardening (February 11, 2026)

1. `StripLLHDProcesses` now emits interface abstraction attrs for unresolved
   dynamic process drives:
   - `circt.bmc_abstracted_llhd_interface_inputs`
   - `circt.bmc_abstracted_llhd_interface_input_details`
   with per-row reason `dynamic_drive_resolution_unknown`.
2. `StripLLHDInterfaceSignals` now preserves and extends pre-existing LLHD
   interface abstraction metadata instead of dropping it when no new rows are
   added in the same pass.
3. New/updated regressions:
   - `test/Tools/circt-bmc/strip-llhd-processes.mlir`
   - `test/Tools/circt-lec/lec-strip-llhd-interface-abstraction-attr.mlir`
4. Remaining high-priority limitation:
   - LLHD abstraction metadata is in place; next work is provenance surfacing
     and drift diagnostics for yosys/verilator formal lanes.

### Test Results

| Mode | Eligible | Pass | Fail | Rate |
|------|----------|------|------|------|
| Parsing | 853 | 853 | 0 | **100%** |
| Elaboration | 1028 | 1021+ | 7 | **99.3%+** |
| Simulation (full) | 912 | 856 | 0 | **99.9%** (0 xfail, 7 xpass, 1 compile-only, 9 skip) |
| BMC (full Z3) | 26 | 26 | 0 | **100%** |
| LEC (full Z3) | 23 | 23 | 0 | **100%** |
| circt-sim lit | 225 | 225 | 0 | **100%** (call-depth-protection fixed) |
| ImportVerilog lit | 268 | 268 | 0 | **100%** |

### sv-tests Raw Results (unfiltered, from stale results file)

| Status | Count | Notes |
|--------|-------|-------|
| PASS | 856+ | 2 previously listed as FAIL now pass (stale results file) |
| FAIL | 0 | `18.5.6--implication_0` and `18.7--randomize_3` both pass now |
| XFAIL | 52 | 34 negative tests + 9 constraint solver + 9 other |
| XPASS | 1 | `16.15--property-iff-uvm-fail` (promote to pass) |
| NO_TOP | 1 | `18.5.2--pure-constraint_3` |

**Note**: The `sv-tests-sim-results.txt` file is stale. Need to re-run full suite.

### AVIP Status

**APB dual-top: SEQBDYZMB fixed, 0 UVM errors, phases run correctly.** Feb 11.
Sequences now execute body() properly; blocks on BFM/driver `finish_item`.
Performance: ~171 ns/s (APB 10us in 59s).

| AVIP | Compilation | Blocker | Notes |
|------|------------|---------|-------|
| APB | **RUNNING** (494K lines LLHD) | BFM/driver transaction completion (finish_item blocks) | 0 SEQBDYZMB, 0 UVM errors, phases run correctly |
| AHB | Fails | Missing `\`timescale` directives | All packages/interfaces/modules |
| UART | Fails | Missing `\`timescale` directives | Also has escape sequence warnings |
| SPI | Fails | Missing `\`timescale` directives | Same as AHB |
| I2S | Not tested | Needs recompile | Stale MLIR had parse errors |
| I3C | Not tested | Needs recompile | Stale MLIR had parse errors |
| AXI4 | Not tested | Needs recompile | Previously: 57MB MLIR, hvl_top-only worked |
| AXI4Lite | Not tested | Needs recompile | Previously: exit code 0 |
| JTAG | Not tested | Needs recompile | Previously: 500ns sim, regex DPI fixed |

### Workstream Status

| Track | Owner | Status | Next Steps |
|-------|-------|--------|------------|
| **Track 1: Constraint Solver** | Agent | **DONE** | All FAIL tests fixed; implication_0 and randomize_3 pass |
| **Track 2: AVIP Recompilation** | Agent | **DONE** | APB recompiled; AHB/SPI/UART need timescale fixes |
| **Track 3: Interpreter Robustness** | Agent | **DONE** | abort→failure audit (6 sites), get_root short-circuit, join type fix |
| **Track 4: AVIP Transaction Flow** | Agent | Active | SEQBDYZMB fixed; next: BFM driver `item_done` path for transaction completion |
| **Track 5: Coverage + SVA** | Agent | Pending | iff guard runtime eval, auto-sampling, SVA concurrent assertions |
| **BMC/LEC** | Codex | Active | Structured Slang event-expression metadata (DO NOT TOUCH) |

### BMC Semantic-Closure Update (February 11, 2026)

1. Local-var / `disable iff` edge hardening:
   - added explicit mid-flight abort/no-abort e2e checks:
     - `test/Tools/circt-bmc/sva-local-var-disable-iff-midflight-abort-unsat-e2e.sv`
     - `test/Tools/circt-bmc/sva-local-var-disable-iff-midflight-no-abort-sat-e2e.sv`
   - validated both JIT and SMT-LIB parity:
     - abort case: `BMC_RESULT=UNSAT`
     - no-abort control: `BMC_RESULT=SAT`
2. Focused UVM semantic slice (explicit caller-owned filters) is green:
   - `16.10--property-local-var-uvm`: PASS
   - `16.10--sequence-local-var-uvm`: PASS
   - `16.15--property-iff-uvm`: PASS
   - `16.15--property-iff-uvm-fail`: XFAIL (known sv-tests metadata/content mismatch)

### Feature Gap Table — Road to Xcelium Parity

**Goal: Eliminate ALL xfail tests. Every feature Xcelium supports, we support.**

| Feature | Status | Blocking Tests | Priority |
|---------|--------|----------------|----------|
| **Constraint solver** | PARTIAL | 0 FAIL + ~5 XFAIL | **P1** |
| - Constraint inheritance | **DONE** | 0 | Parent class hierarchy walking |
| - Distribution constraints | **DONE** | 0 | `traceToPropertyName()` fix |
| - Static constraint blocks | **DONE** | 0 | VariableOp support |
| - Soft constraints | **DONE** | 0 | `isSoft` flag extraction |
| - Constraint guards (null) | **DONE** | 0 | `ClassHandleCmpOp`+`ClassNullOp` |
| - Implication/if-else/inside | **DONE** | 0 | Conditional range application |
| - Inline constraints (`with`) | **DONE** | 0 | Range extraction from inline blocks |
| - Foreach iterative constraints | **DONE** | 0 | Per-element constraints |
| - Functions in constraints | MISSING | `18.5.12` | 1 XFAIL |
| - Infeasible detection | **DONE** | 0 | intersect hard bounds per property |
| - Global constraints | MISSING | `18.5.9` | 1 XFAIL |
| - Implication constraint_0 | **DONE** | 0 | Was stale FAIL — now passes |
| - Inline randomize_3 | **DONE** | 0 | Was stale FAIL — now passes |
| **rand_mode / constraint_mode** | **DONE** | 0 | Receiver resolution fixed |
| **Random stability** | PARTIAL | ~5 XFAIL | **P1** |
| - srandom seed control | **DONE** | 0 | Per-object RNG via `__moore_class_srandom` |
| - Per-object RNG | **DONE** | 0 | `std::mt19937` per object address |
| - get/set_randstate | **DONE** | 0 | Serialize/deserialize mt19937 state |
| - Thread/object stability | XFAIL | `18.14--*` | 3 tests |
| - Manual seeding | XFAIL | `18.15--*` | 2 tests |
| **Coverage collection** | PARTIAL | 0 (AVIPs) | **P0** |
| - Basic covergroups | **DONE** | 0 | Implicit + parametric sample() |
| - Parametric sample() | **DONE** | 0 | Expression binding with visitSymbolReferences |
| - Coverpoint iff guard | **DONE** | 0 | AttrSizedOperandSegments lowering |
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
| - Native mutation CLI frontend (`circt-mut`) | IN_PROGRESS | — | Added native `circt-mut` `init|run|report|cover|matrix|generate` flows; target architecture is MCY/Certitude-style campaign UX (`init`/project config + `run`/`report`-grade flows) with staged migration of script logic into native C++ subcommands. `circt-mut init` bootstraps campaign templates (`circt-mut.toml`, `tests.tsv`, `lanes.tsv`) with overwrite guards (`--force`), `circt-mut run` consumes project config to dispatch native-preflight-backed `cover`/`matrix` flows (`--mode cover|matrix|all`), supports optional post-run report emission (`--with-report`, `--with-report-on-fail`, `--report-mode`) and now also supports direct CLI governance overrides for post-run reporting (`--report-compare`, `--report-compare-history-latest`, `--report-history`, `--report-append-history`, `--report-trend-history`, `--report-trend-window`, `--report-history-max-runs`, `--report-out`, `--report-history-bootstrap`, `--report-no-history-bootstrap`, `--report-policy-profile` repeatable, `--report-policy-mode`, `--report-policy-stop-on-fail`, `--report-fail-on-prequalify-drift`, `--report-no-fail-on-prequalify-drift`, `--report-fail-if-value-gt`, `--report-fail-if-value-lt`, `--report-fail-if-delta-gt`, `--report-fail-if-delta-lt`, `--report-fail-if-trend-delta-gt`, `--report-fail-if-trend-delta-lt`) with strict validation and precedence over config fallbacks plus guardrails that reject report override flags unless report emission is enabled (`--with-report` or `[run] with_report = true`); this prevents silent no-op CI misconfiguration. It forwards `[run] report_*` policy/history/compare/output controls into the report subcommand (`report_policy_profile(s)`, `report_policy_mode`, `report_policy_stop_on_fail`, compare/history/trend files, lane/skip budget outputs, `report_history_max_runs`, `report_trend_window`, `report_history_bootstrap`, `report_fail_on_prequalify_drift`, `report_fail_if_value_gt`, `report_fail_if_value_lt`, `report_fail_if_delta_gt`, `report_fail_if_delta_lt`, `report_fail_if_trend_delta_gt`, `report_fail_if_trend_delta_lt`) with strict run-level validation for policy-mode/stop-on-fail combinations and explicit bool override forwarding (`--fail-on-prequalify-drift` / `--no-fail-on-prequalify-drift`), while `circt-mut report` now supports config-level gate override (`[report] fail_on_prequalify_drift`) even when policy profiles enable drift gating; this closes a long-term governance gap where policy bundles were not tunable without replacing profiles wholesale. It also supports generated-mutation config (`generate_mutations` + mode/mode-count/mode-weight/profile/cfg/select/Yosys keys), matrix native-dispatch/prequalify toggles (`native_matrix_dispatch`, `native_global_filter_prequalify`), native cover prequalify/probe pass-through keys (`native_global_filter_prequalify`, `native_global_filter_prequalify_only`, `native_global_filter_prequalify_pair_file`, `native_global_filter_probe_mutant`, `native_global_filter_probe_log`), plus expanded formal/gate config pass-through (including strict boolean parsing for both gate toggles and cover formal bool flags), and `circt-mut report` now aggregates cover/matrix artifacts into normalized key/value campaign summaries (stdout and optional `--out` TSV) including formal-global-filter telemetry rollups (timeouts/unknowns/chain/runtime/cache metrics) across matrix lanes and baseline comparison (`--compare`) with numeric diff rows (`diff.<metric>.delta`/`pct_change`) plus regression gate thresholds (`--fail-if-delta-gt`, `--fail-if-delta-lt`), history snapshot workflows (`--compare-history-latest`, `--append-history`), native history-trend summaries/gates (`--trend-history`, `--trend-window`, `--fail-if-trend-delta-gt`, `--fail-if-trend-delta-lt`), and policy bundles (`--policy-profile formal-regression-basic|formal-regression-trend|formal-regression-matrix-basic|formal-regression-matrix-trend|formal-regression-matrix-guard|formal-regression-matrix-trend-guard|formal-regression-matrix-guard-smoke|formal-regression-matrix-guard-nightly|formal-regression-matrix-stop-on-fail-guard-smoke|formal-regression-matrix-stop-on-fail-guard-nightly|formal-regression-matrix-guard-strict|formal-regression-matrix-stop-on-fail-basic|formal-regression-matrix-stop-on-fail-trend|formal-regression-matrix-stop-on-fail-strict|formal-regression-matrix-full-lanes-strict|formal-regression-matrix-runtime-nightly|formal-regression-matrix-runtime-trend`). Native `circt-mut generate` executes Yosys mutation-list generation directly (mode/mode-count/mode-weight/profile/cfg/select/top-up dedup) and includes native `--cache-dir` behavior (content-addressed cache hit/miss, metadata-based saved-runtime reporting, lock wait/contended telemetry), with script fallback for unsupported future flags. `circt-mut cover` now performs native global-filter preflight: built-in tool resolution/rewrite for `--formal-global-propagate-circt-lec` / `--formal-global-propagate-circt-bmc` (including bare `auto` forms), built-in Z3 resolution for `--formal-global-propagate-z3` / `--formal-global-propagate-bmc-z3`, chain-mode validation/default engine injection (`--formal-global-propagate-circt-chain`), cover mutation-source consistency checks (`--mutations-file` vs `--generate-mutations`), mutation-generator Yosys resolution (`--mutations-yosys`), generated-mutation mode/profile/allocation/seed validation (`--mutations-modes`, `--mutations-profiles`, `--generate-mutations`, `--mutations-seed`, `--mutations-mode-counts`, `--mutations-mode-weights`, including mode-name checks), early mode-conflict diagnostics, native numeric/cache validation for cover formal controls (`--formal-global-propagate-*_timeout-seconds`, `--formal-global-propagate-bmc-bound`, `--formal-global-propagate-bmc-ignore-asserts-until`, `--bmc-orig-cache-max-*`, `--bmc-orig-cache-eviction-policy`), and PATH-accurate `timeout` preflight for non-zero active global-filter timeout settings, plus native single-mutant runtime global-filter probing (`--native-global-filter-probe-mutant`, optional `--native-global-filter-probe-log`) for command-mode and built-in global filters, and native campaign prequalification handoff (`--native-global-filter-prequalify`, optional `--native-global-filter-prequalify-pair-file`) for command-mode and built-in circt-lec/circt-bmc/chain classification feeding script `--reuse-pair-file` dispatch across both static and generated mutation sources, with a no-test-dispatch mode (`--native-global-filter-prequalify-only`) for formal-only batch triage and `--jobs`-parallelized prequalification that preserves deterministic pair-row ordering. `circt-mut matrix` now performs the analogous default-global-filter preflight/rewrite (`--default-formal-global-propagate-cmd`, `--default-formal-global-propagate-circt-{lec,bmc,chain}` plus default Z3 options), native default Yosys resolution (`--default-mutations-yosys`), native default generated-mutation mode/profile/allocation/seed validation (`--default-mutations-modes`, `--default-mutations-profiles`, `--default-mutations-mode-counts`, `--default-mutations-mode-weights`, `--default-mutations-seed`, including mode-name checks), lane mutation-source consistency checks (`mutations_file` vs `generate_count`), lane generated-mutation preflight from `--lanes-tsv` (modes/profiles, yosys, `generate_count`, `mutations_seed`, mode-count/weight syntax/conflict/total/mode-name checks), lane formal tool preflight (`global_propagate_cmd`, `global_propagate_circt_lec`, `global_propagate_circt_bmc`, `global_propagate_z3`, `global_propagate_bmc_z3`) plus lane timeout/cache/gate override validation (timeouts, BMC bound/ignore-assert, BMC orig-cache limits/policy, lane skip/fail booleans), lane formal boolean validation (`global_propagate_assume_known_inputs`, `global_propagate_accept_xprop_only`, `global_propagate_bmc_run_smtlib`, `global_propagate_bmc_assume_known_inputs` with `1|0|true|false|yes|no|-`), native validation for matrix default numeric/cache controls (`--default-formal-global-propagate-*_timeout-seconds`, `--default-formal-global-propagate-bmc-bound`, `--default-formal-global-propagate-bmc-ignore-asserts-until`, `--default-bmc-orig-cache-max-*`, `--default-bmc-orig-cache-eviction-policy`), PATH-accurate `timeout` preflight for non-zero effective default/lane timeout settings with active effective global-filter modes, native matrix lane prequalification dispatch (`--native-global-filter-prequalify`) that runs per-lane native prequalify-only, materializes lane reuse pair files, dispatches script matrix with a rewritten lanes manifest, exports aggregated matrix prequalification telemetry counters (`native_matrix_prequalify_summary_lanes`, `native_matrix_prequalify_summary_missing_lanes`, `native_matrix_prequalify_total_mutants`, `native_matrix_prequalify_not_propagated_mutants`, `native_matrix_prequalify_propagated_mutants`, plus error/cmd-source counters), forwards matrix gate defaults (`--skip-baseline`, `--fail-on-undetected`, `--fail-on-errors`) into native lane cover dispatch with per-lane TSV override precedence (`skip_baseline`, `fail_on_undetected`, `fail_on_errors`), supports native lane selection filters (`--include-lane-regex`, `--exclude-lane-regex`) with regex validation and deterministic lane skipping before dispatch, supports native lane-level parallel dispatch (`--jobs`) with deterministic `results.tsv` row ordering and `native_matrix_dispatch_lane_jobs` telemetry plus native runtime counters (`native_matrix_dispatch_executed_lanes`, `native_matrix_dispatch_runtime_ns`, `native_matrix_dispatch_avg_lane_runtime_ns`), direct `results.tsv` runtime_ns emission for each lane, and report aggregation that prefers row-local runtime_ns with sidecar fallback (`matrix.runtime_ns_sum`, `matrix.runtime_ns_avg`, `matrix.runtime_ns_max`), and now reports explicit SKIP accounting in campaign summaries (`matrix.lanes_skip`, `matrix.gate_skip`) for native stop-on-fail cut lanes plus always-on skip-budget counters (`matrix.skip_budget_rows_total`, `matrix.skip_budget_rows_stop_on_fail`, `matrix.skip_budget_rows_non_stop_on_fail`) and per-lane skip diagnostics (`matrix.lane_budget.lane.<id>.is_skip`, `.is_stop_on_fail_skip`, `.skip_reason_code`, `.skip_reason`) for robust CI trend/debug workflows. Default mutation materialization no longer depends on `~/mcy/scripts/create_mutated.sh` (now uses in-repo `utils/create_mutated_yosys.sh`), and mutation scripts are installed to `<prefix>/share/circt/utils` for compatibility during migration. Next steps: migrate full matrix lane scheduling loops natively, adopt policy profiles in CI wrappers via these new run-level CLI governance overrides for smoke/nightly/strict lanes, publish policy-pack docs with recommended profile mapping for smoke/nightly/strict regressions, and adopt `--history --history-bootstrap` in first-run bootstrap jobs. |
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
- Latest mutation-governance milestone: `circt-mut report` now supports direct
  CLI matrix policy-mode selection (`--policy-mode` + optional
  `--policy-stop-on-fail`) with precedence over config policy profiles.
- Latest mutation-governance milestone (this slice): composite matrix policy
  bundles are now available for single-flag CI policy selection
  (`formal-regression-matrix-composite-*`), reducing manual profile
  composition drift between smoke/nightly/strict governance lanes.
- Latest mutation-governance milestone (current): policy-mode mapping now emits
  a single composite matrix profile (instead of paired guard/runtime profiles),
  simplifying CI policy configuration and reducing multi-profile precedence
  edge-cases in run/report flows.
- Latest mutation-governance milestone (current): trend governance now has
  single-profile composite bundles
  (`formal-regression-matrix-composite-trend-{nightly,strict}`) that compose
  lane-trend, runtime-trend, and lane-drift checks for long-run matrix
  regression tracking.
- Latest mutation-governance milestone (current): policy-mode enums are now
  extended to `smoke|nightly|strict|trend-nightly|trend-strict` across
  `circt-mut init`, `circt-mut run`, and `circt-mut report`, including
  stop-on-fail composite trend bundles
  (`formal-regression-matrix-composite-stop-on-fail-trend-{nightly,strict}`)
  so config/CLI governance remains orthogonal when trend windows are enabled.
- Latest mutation-governance milestone (current): native matrix dispatch now
  supports cache-aware lane scheduling parity (`--lane-schedule-policy
  cache-aware`) with deterministic leader-first key spreading and explicit
  scheduling telemetry (`native_matrix_dispatch_schedule_*`) for CI/runtime
  debugging.
- Latest mutation-governance milestone (current): native matrix dispatch lane
  filtering now supports repeatable `--include-lane-regex` and
  `--exclude-lane-regex` arguments (both `--flag value` and `--flag=value`)
  with deterministic OR semantics per filter class and strict per-pattern
  validation, closing another script-parity gap for CI lane slicing.
- Latest mutation-governance milestone (current): native matrix dispatch now
  emits lane-filter accounting telemetry
  (`native_matrix_dispatch_filtered_include`,
  `native_matrix_dispatch_filtered_exclude`) so CI/report consumers can
  attribute lane-selection drops to include/exclude filters without parsing
  rewritten lane manifests.
- Latest mutation-governance milestone (current): matrix results annotation now
  carries lane-level prequalification traceability paths
  (`prequalify_pair_file`, `prequalify_log_file`) alongside numeric
  prequalification counters, enabling report/triage tooling to map per-lane
  metrics back to concrete prequalification artifacts without separate joins.
- Latest mutation-governance milestone (current): `circt-mut report` now
  consumes and re-emits per-lane prequalification traceability fields through
  lane-budget keys (`matrix.lane_budget.lane.<id>.prequalify_pair_file`,
  `.prequalify_log_file`) and aggregate availability counters
  (`matrix.prequalify_results_pair_file_present_lanes`,
  `matrix.prequalify_results_log_file_present_lanes`) for direct policy/trend
  gating on provenance completeness.
- Latest mutation-governance milestone (current): report now emits explicit
  completeness-deficit counters for summary-present lanes missing provenance
  artifacts:
  - `matrix.prequalify_results_summary_present_missing_pair_file_lanes`
  - `matrix.prequalify_results_summary_present_missing_log_file_lanes`
  enabling straightforward strict gates on native prequalification artifact
  hygiene in CI.
- Latest mutation-governance milestone (current): matrix guard/strict policy
  bundles now gate provenance completeness directly by requiring both deficit
  counters above to stay zero, enabling native prequalification artifact
  hygiene enforcement without bespoke CI rule wiring.
- Latest mutation-governance milestone (current): added dedicated matrix
  provenance policy profiles for incremental rollout by lane class:
  - `formal-regression-matrix-provenance-guard`
  - `formal-regression-matrix-provenance-strict`
  with explicit column-presence + deficit-zero gating on prequalify artifact
  provenance metrics, so CI can adopt provenance enforcement independently from
  broader guard/strict mutation quality bundles.
- Latest mutation-governance milestone (current): matrix `policy_mode`
  expansion now appends provenance profiles by default:
  - smoke/nightly/trend-nightly -> `formal-regression-matrix-provenance-guard`
  - strict/trend-strict -> `formal-regression-matrix-provenance-strict`
  so CI defaults (`--policy-mode ...`) enforce prequalify artifact provenance
  presence/hygiene without extra per-job `--policy-profile` wiring.
- Latest mutation-governance milestone (current): `circt-mut run` now forwards
  report policy mode as first-class report options (`--policy-mode` +
  `--policy-stop-on-fail`) instead of eagerly expanding to profile names in
  run-layer plumbing, removing duplicate mapping logic and keeping policy-mode
  behavior sourced from report-layer defaults.
- Latest mutation-governance milestone (current): matrix policy-mode now
  supports provenance-only rollout modes:
  - `provenance-guard`
  - `provenance-strict`
  mapping directly to dedicated provenance policy profiles, enabling staged CI
  adoption of prequalify artifact hygiene gates without composite mutation gate
  bundles.
- Latest mutation-governance milestone (current): matrix policy-mode now
  supports native trend rollout modes:
  - `native-trend-nightly`
  - `native-trend-strict`
  both map to trend composite bundles with native-family contract gating
  and provenance defaults for long-run native matrix governance.
- Latest mutation-governance milestone (current): report history shorthand
  semantics are now strict and deterministic across `circt-mut report` and
  `circt-mut run`:
  - `--history` / `[report] history` and
    `--report-history` / `[run] report_history`
    are mutually exclusive with explicit compare/trend/append selectors,
    preventing ambiguous mixed-source policy/trend baselines in CI.
- Latest mutation-governance milestone (current): matrix trend policy bundles
  now enforce history data-quality minimums (`trend.history_runs_selected >= 2`
  and `trend.numeric_keys >= 1`) through a dedicated profile component
  (`formal-regression-matrix-trend-history-quality`), preventing one-run trend
  snapshots from silently passing nightly/strict trend governance.
- Latest mutation-governance milestone (current): trend-history quality
  contracts are now consistently applied across direct trend profiles
  (`formal-regression-trend`, `formal-regression-matrix-trend*`,
  `formal-regression-matrix-stop-on-fail-trend`,
  `formal-regression-matrix-lane-trend-*`), not only composite trend bundles,
  so one-run history snapshots cannot bypass trend governance via profile
  selection.
- Latest mutation-governance milestone (current): strict matrix trend bundles
  now enforce deeper trend-history evidence (`trend.history_runs_selected >= 3`)
  via `formal-regression-matrix-trend-history-quality-strict`, composed into:
  - `formal-regression-matrix-composite-trend-strict`
  - `formal-regression-matrix-composite-stop-on-fail-trend-strict`
  while nightly trend bundles keep the `>= 2` contract for staged rollout.
- Latest mutation-governance milestone (current): strict matrix trend policy now
  also enforces history-coverage quality for numeric trend keys:
  - `trend.numeric_keys_full_history_pct >= 80`
  - `trend.matrix_core_numeric_keys_full_history_pct = 100`
  where matrix core keys are
  `matrix.detected_mutants_sum`, `matrix.lanes_skip`, and runtime
  (`matrix.runtime_ns_{avg,max,sum}`), reducing false confidence from sparse
  strict trend windows with missing core-key history rows.
- Latest mutation-governance milestone (current): strict matrix trend policy
  now also enforces an explicit core-key full-history count floor:
  - `trend.matrix_core_numeric_keys_full_history >= 5`
  so strict gates fail clearly on missing core key history, even when users
  only inspect percentages.
- Latest mutation-governance milestone (current): direct strict lane trend
  policy now reuses the strict history-quality contract (instead of the generic
  trend-nightly contract), eliminating a policy-selection bypass where
  `formal-regression-matrix-lane-trend-strict` could run with only 2 history
  runs while strict composite profiles required 3.
- Latest mutation-governance milestone (current): strict trend policy now has
  explicit missing-history diagnostics/gating for matrix core keys:
  - `trend.matrix_core_numeric_keys_missing_history`
  - `trend.matrix_core_numeric_keys_missing_history_list`
  with strict gate `missing_history == 0`, making history sparsity failures
  directly actionable in CI logs (not only via aggregate percentages/counts).
- Latest mutation-governance milestone (current): strict direct lane-trend
  profile parity is closed end-to-end with strict composites:
  - direct strict lane trend now requires strict history quality (3-run floor),
  - and strict reports emit explicit missing core-key history diagnostics.
- Latest mutation-governance milestone (current): `circt-mut report` now
  ingests external formal result snapshots via
  `--external-formal-results` / `[report] external_formal_results`, emits
  normalized `external_formal.*` governance metrics (status and summary
  fail-like rollups), and adds
  `formal-regression-matrix-external-formal-guard` policy gating
  (`external_formal.files >= 1`, `external_formal.fail_like_sum == 0`) for
  direct matrix+formal closure checks in mutation CI.
- Latest mutation-governance milestone (current): policy-mode now has a
  first-class external-formal native strict rollout mode:
  - `native-strict-formal`
  and `circt-mut run` now forwards repeatable report external-formal inputs
  via `--report-external-formal-results` / `[run] report_external_formal_results`.
  This closes the last manual wiring gap for native strict matrix governance
  + external formal closure in one mode invocation.
- Latest mutation-governance milestone (current): policy-mode now also supports
  strict non-native external-formal governance:
  - `strict-formal`
  mapping to strict composite + provenance strict + external formal guard
  profiles, enabling strict formal-closure enforcement without native-family
  mode contracts.
- Latest mutation-governance milestone (current): `circt-mut report` now
  supports external-formal out-dir auto-discovery:
  - `--external-formal-out-dir` / `[report] external_formal_out_dir`
  - `circt-mut run` forwarding via
    `--report-external-formal-out-dir` /
    `[run] report_external_formal_out_dir`
  with report telemetry keys (`external_formal.out_dir`,
  `external_formal.files_discovered`) so strict-formal governance can consume
  runner out-directories directly without manual file-list wiring.
- Latest mutation-governance milestone (current): external-formal governance
  now has schema-aware summary coverage for `summary.tsv`:
  - new metrics:
    - `external_formal.summary_tsv_files`
    - `external_formal.summary_tsv_rows`
    - `external_formal.summary_tsv_schema_valid_files`
    - `external_formal.summary_tsv_schema_invalid_files`
    - `external_formal.summary_tsv_parse_errors`
  - new strict profile:
    - `formal-regression-matrix-external-formal-summary-guard`
  - new policy modes for staged rollout:
    - `strict-formal-summary`
    - `native-strict-formal-summary`
  enabling strict matrix+formal closure to gate not just on external-formal
  fail-like counts, but on structured summary-schema integrity.
- Latest mutation-governance milestone (current): summary-schema strict
  governance now also enforces per-row count consistency in `summary.tsv`:
  - new metrics:
    - `external_formal.summary_tsv_consistent_rows`
    - `external_formal.summary_tsv_inconsistent_rows`
  - strict summary guard now requires zero inconsistent rows
  so malformed `total` vs status-count tuples fail policy deterministically.
- Latest mutation-governance milestone (current): strict summary governance now
  has a backward-compatible schema-version contract:
  - new metrics:
    - `external_formal.summary_tsv_schema_version_rows`
    - `external_formal.summary_tsv_schema_version_invalid_rows`
    - `external_formal.summary_tsv_schema_version_min`
    - `external_formal.summary_tsv_schema_version_max`
  - parsing defaults to schema version `1` when `schema_version` column is
    absent; when present, non-numeric values are rejected and policy-gated.
  This allows future versioned summary evolution without regressing current
  runner outputs.
- Latest mutation-governance milestone (current): strict summary governance now
  also checks for duplicate lane rows in `summary.tsv` when suite+mode keys are
  present:
  - new metrics:
    - `external_formal.summary_tsv_duplicate_rows`
    - `external_formal.summary_tsv_unique_rows`
  - strict summary guard now requires zero duplicates, preventing silently
  double-counted external formal lanes from passing policy.
- Latest mutation-governance milestone (current): strict summary governance now
  has an explicit schema-evolution rollout track:
  - new policy modes:
    - `strict-formal-summary-v1`
    - `native-strict-formal-summary-v1`
  - new profile:
    - `formal-regression-matrix-external-formal-summary-v1-guard`
  with strict gate `external_formal.summary_tsv_schema_version_max <= 1`,
  enabling deterministic schema-v1 pinning while keeping existing
  `strict-formal-summary*` modes available for broader compatibility.
- Future iterations should add:
  - concise outcome and planning impact in `PROJECT_PLAN.md`
  - detailed implementation + validation data in `CHANGELOG.md`

### Active Formal Gaps (Near-Term)
- Mutation/report governance closure (next long-term mutation tranche):
  - Composite matrix policy bundles are now available
    (`formal-regression-matrix-nightly|strict`) with dedicated lane-drift
    bundles (`formal-regression-matrix-lane-drift-nightly|strict`) and
    runtime bundles (`formal-regression-matrix-runtime-smoke|nightly|strict|trend`).
    Policy-mode now supports smoke/nightly/strict/trend-* with composite
    profile mapping, and native matrix lane scheduling now supports
    cache-aware key spreading. Remaining migration work: full script parity for
    all lane telemetry/control options and elimination of script fallback paths.
  - Wire matrix `--policy-mode` defaults through formal driver entrypoints that
    still pass explicit legacy `--policy-profile` values.
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
  - Latest closure hardening (current):
    `ImportVerilog` now treats `disable iff` conditions using integral
    truthiness (not just 1-bit operands), and BMC regression coverage now
    includes explicit local-var/`disable iff` abort-vs-no-abort edge tests.
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
6. Current closure hardening update (February 11, 2026):
   - `ImportVerilog` is now tolerant of older slang builds that do not expose
     `CompilationFlags::AllowVirtualIfaceWithOverride`, avoiding unrelated
     build breaks while continuing BMC/LEC closure work.
   - Focused UVM semantic slice remains `pass=3 fail=1` on:
     `16.10--property-local-var-uvm`,
     `16.10--sequence-local-var-uvm`,
     `16.15--property-iff-uvm`,
     `16.15--property-iff-uvm-fail`.
   - The remaining `16.15--property-iff-uvm-fail` status is currently
     attributable to test metadata/content mismatch (same effective property as
     the pass variant), not a newly reproducing local-var/`disable iff`
     importer regression.
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
