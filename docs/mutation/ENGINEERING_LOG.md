# Mutation Engineering Log

## 2026-02-28 (case wildcard expansion + BA/NBA context guards + parity hardening)

- realizations:
  - Wildcard decode confusion coverage was still incomplete after
    `case<->casez` and `case<->casex`; we were missing direct
    `casez<->casex` transitions.
  - Two earlier xrun-vs-circt “fail” rows were not semantic mismatches; they
    were transient `Permission denied` executions while `circt-sim` was being
    relinked in-place by a concurrent build.
  - Parity harnesses must avoid declaration initialization on
    `always_comb`/`always_ff` targets to remain portable across simulators.

- changes made:
  - Added native operators:
    - `CASEZ_TO_CASEX`
    - `CASEX_TO_CASEZ`
  - Integrated these operators in:
    - planner op catalog, site collection, family classification, and apply
      rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`control`, `connect`, `invert`, `inv`,
      `balanced/all`) in `tools/circt-mut/circt-mut.cpp`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/native-create-mutated-casez-to-casex-site-index.test`
    - `test/Tools/native-create-mutated-casex-to-casez-site-index.test`
    - `test/Tools/native-mutation-plan-case-keyword-zx-swap.test`
    - `test/Tools/circt-mut-generate-circt-only-control-mode-case-keyword-zx-ops.test`
  - Hardened assignment-timing site selection to reduce simulator-dependent
    race mutations:
    - skip `BA_TO_NBA`/`NBA_TO_BA` in `always_comb` and `always_latch`
    - skip `NBA_TO_BA` in `initial`
  - Added timing-skip regressions:
    - `test/Tools/native-create-mutated-ba-to-nba-skip-always-comb.test`
    - `test/Tools/native-create-mutated-nba-to-ba-skip-event-initial.test`

- validation:
  - Red/green for new wildcard-case tests: all 4 fail before implementation and
    pass after.
  - Focused lit slice over case keyword + BA/NBA guard tests:
    - `13 passed`
  - Seeded broad parity rerun after BA/NBA guarding (`count=40`, `seed=506`,
    `--modes all`):
    - definitive rerun result: `ok=40`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_casex_parity_after_banba_guard3_1772291633`
  - Seeded wildcard-case parity campaign including all 6 case-keyword ops
    (`count=36`, `seed=601`):
    - deterministic baseline: xrun/circt `COV=100.00`, `SIG=e95c999c`
    - definitive rerun result: `ok=36`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_casezx_parity3_1772291987`
  - Additional all-mode sweep on the same wildcard harness (`count=40`,
    `seed=707`, `--modes all`):
    - result: `ok=40`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_casezx_allmode_parity_1772292117`
  - Additional post-fix broad sweep on the `cov_intro_seeded_casex` harness
    (`count=60`, `seed=912`, `--modes all`):
    - baseline: xrun/circt `COV=95.83`, `SIG=000607ab`
    - result: `ok=60`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_casex_postfix_sweep_1772292260`

## 2026-02-28 (case/casez keyword mutation class + seeded parity campaign)

- realizations:
  - Wildcard decode mistakes (`case` vs `casez`) were missing from native
    control-class mutations.
  - This fault class is lightweight to model textually and integrates well with
    existing deterministic site-index contracts.

- changes made:
  - Added native operators:
    - `CASE_TO_CASEZ`
    - `CASEZ_TO_CASE`
  - Integrated these operators in:
    - planner op catalog, keyword-site collection, family classification, and
      apply rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`control`, `connect`, `invert`, `inv`,
      `balanced/all`) in `tools/circt-mut/circt-mut.cpp`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/native-create-mutated-case-to-casez-site-index.test`
    - `test/Tools/native-create-mutated-casez-to-case-site-index.test`
    - `test/Tools/native-mutation-plan-case-keyword-swap.test`
    - `test/Tools/circt-mut-generate-circt-only-control-mode-case-keyword-ops.test`

- validation:
  - Focused lit slice over new case-keyword tests and adjacent case/control
    coverage: `8 passed`.
  - Seeded xrun-vs-circt parity campaign on a harness containing both `case`
    and `casez` (`count=20`, `seed=404`,
    `--native-ops CASE_TO_CASEZ,CASEZ_TO_CASE`):
    - baseline: xrun/circt `COV=91.67`, `SIG=0165e064`
    - result: `ok=20`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_casekw_parity_1772290530`

## 2026-02-28 (case-item arm swap mutation class + seeded parity campaigns)

- realizations:
  - `if/else` arm swap existed, but `case`-item arm swap did not; that left a
    common decode/control bug class underrepresented.
  - `case` rewrites need conservative structural guards to avoid invalid
    rewrites of nested/procedural item bodies.

- changes made:
  - Added native operator:
    - `CASE_ITEM_SWAP_ARMS`
  - Integrated operator in:
    - planner op catalog, site collection, family classification, and apply
      rewrite path (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`control`, `connect`, `invert`, `inv`,
      `balanced/all`) in `tools/circt-mut/circt-mut.cpp`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/native-create-mutated-case-item-swap-arms-site-index.test`
    - `test/Tools/native-mutation-plan-case-item-swap-arms.test`
    - `test/Tools/circt-mut-generate-circt-only-control-mode-case-item-swap-op.test`

- validation:
  - Focused parity campaign on `cov_intro_seeded_case.sv` with
    `--native-ops CASE_ITEM_SWAP_ARMS` (`count=24`):
    - result: `ok=24`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_case_parity_1772289672`
  - Broader seeded parity campaigns with per-mutant `circt-mut apply` on the
    same harness:
    - `--modes control` (`count=24`, `seed=101`):
      `ok=24`, `mismatch=0`, `fail=0`
    - `--modes all` (`count=24`, `seed=202`):
      `ok=24`, `mismatch=0`, `fail=0`
    - workspace: `/tmp/cov_seeded_case_parity_broad2_1772290014`
  - Extended all-mode sweep on same harness
    (`--modes all`, `count=60`, `seed=303`) with real per-mutant apply:
    - result: `ok=60`, `mismatch=0`, `fail=0`
    - includes `NATIVE_CASE_ITEM_SWAP_ARMS@1` in sampled operator set
    - workspace: `/tmp/cov_seeded_case_parity_postcase_1772290209`

## 2026-02-28 (reset-condition force mutation class + reset-aware parity campaigns)

- realizations:
  - We had reset-condition inversion (`RESET_COND_NEGATE`) but no direct
    reset-condition stuck-at forcing, which left a realistic reset fault class
    underrepresented.
  - Reset-aware seeded parity harnesses are useful to ensure control mutation
    classes affect behavior and coverage signatures, not just syntactic churn.

- changes made:
  - Added native operators:
    - `RESET_COND_TRUE`
    - `RESET_COND_FALSE`
  - Integrated these operators in:
    - planner op catalog, reset-site collection reuse, family classification,
      and apply rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`control`, `connect`, `invert`, `inv`,
      `balanced/all`) in `tools/circt-mut/circt-mut.cpp`
    - primitive cnot polarity sets:
      - `cnot0`: includes `RESET_COND_FALSE`
      - `cnot1`: includes `RESET_COND_TRUE`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/native-create-mutated-reset-cond-true-site-index.test`
    - `test/Tools/native-create-mutated-reset-cond-false-site-index.test`
    - `test/Tools/native-mutation-plan-reset-cond-force.test`
    - `test/Tools/circt-mut-generate-circt-only-control-mode-reset-cond-force-op.test`

- validation:
  - Red-first: all four new tests fail before implementation and pass after.
  - Focused lit slices over reset/control/cnot touched tests:
    `12 passed`.
  - Deterministic reset-aware baseline parity (`cov_intro_seeded_reset.sv`):
    - xrun: `COV=89.58`, `SIG=2b186888`
    - circt: `COV=89.58`, `SIG=2b186888`
  - Control-mode seeded parity campaign (`count=30`, `seed=20260228`):
    - includes `RESET_COND_NEGATE`, `RESET_COND_TRUE`, `RESET_COND_FALSE`
    - result: `ok=30`, `mismatch=0`, `fail=0`
  - Weighted all-mode seeded parity campaign (`count=40`, `seed=20260229`):
    - result: `ok=40`, `mismatch=0`, `fail=0`
  - Extended weighted all-mode seeded parity campaign (`count=60`,
    `seed=20260301`) on the same reset-aware harness:
    - result: `ok=60`, `mismatch=0`, `fail=0`

## 2026-02-28 (cnot polarity tightening + preprocessor-safe div/mul mutations)

- realizations:
  - `cnot0` and `cnot1` in CIRCT-only mode were effectively the same broad
    control set, which reduced semantic distinctness versus Yosys-style
    polarity intent.
  - Weighted parity sweeps surfaced a real native-mutator bug: division-family
    rewrites could mutate `` `timescale 1ns/1ns`` into invalid directives
    (for example `1ns%1ns`, `1ns*1ns`), causing xrun parse failures.

- changes made:
  - Added native control-stuck mux operators:
    - `MUX_FORCE_TRUE` (`cond ? t : t`)
    - `MUX_FORCE_FALSE` (`cond ? f : f`)
  - Integrated these in:
    - planner op catalog, site collection, family classification, and apply
      rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`control`, `connect`, `invert`, `inv`,
      `balanced/all`) in `tools/circt-mut/circt-mut.cpp`
  - Tightened primitive mode semantics:
    - `cnot0` now emits `IF_COND_FALSE`/`MUX_FORCE_FALSE`
    - `cnot1` now emits `IF_COND_TRUE`/`MUX_FORCE_TRUE`
  - Fixed preprocessor mutation bug:
    - directive lines whose first non-space token is backtick are masked from
      mutation-site scanning in `buildCodeMask`.
  - Updated native-op validator allowlist:
    - `utils/run_mutation_mcy_examples.sh` now accepts
      `IF_COND_TRUE|IF_COND_FALSE|MUX_FORCE_TRUE|MUX_FORCE_FALSE`.
  - Added TDD regressions:
    - `test/Tools/native-create-mutated-mux-force-true-site-index.test`
    - `test/Tools/native-create-mutated-mux-force-false-site-index.test`
    - `test/Tools/circt-mut-generate-circt-only-cnot-polarity-distinct.test`
    - `test/Tools/native-create-mutated-div-to-mod-skip-timescale-directive.test`

- validation:
  - Focused lit slice over new and adjacent tests: `6 passed`.
  - Deterministic baseline parity on seeded harness:
    - xrun: `COV=87.50`, `SIG=38962b64`
    - circt: `COV=87.50`, `SIG=38962b64`
  - cnot-focused parity campaign (`count=20`, `seed=20260228`):
    `ok=20`, `mismatch=0`, `fail=0`.
  - weighted all-mode parity campaign (`count=40`, `seed=20260229`):
    - before fix: `ok=37`, `xrun_fail=3` (all due malformed `timescale`)
    - after fix: `ok=40`, `mismatch=0`, `fail=0`.

## 2026-02-28 (empty-sensitivity wait semantics parity fix)

- realizations:
  - A deterministic parity mismatch (`NATIVE_IF_COND_TRUE@2`) was not random;
    circt-sim hit `DELTA_OVERFLOW` at time 0 while xrun completed.
  - The mismatch came from simulator semantics, not mutation generation:
    mutated `always_comb` lowered to `llhd.wait ^bb1` (no delay, no observed).
  - LLHD-correct behavior here is "wait forever" (no wake condition), not
    per-delta re-arm.

- root cause:
  - `LLHDProcessInterpreter::interpretWait` had an empty-sensitivity
    `scheduleNextDelta` re-arm path, creating an artificial zero-time loop.

- fix:
  - Removed delta re-arm fallback for empty-sensitivity waits.
  - Implemented semantic handling via
    `suspendProcessForEvents(procId, SensitivityList())`.
  - Removed dead fallback bookkeeping field
    `emptySensitivityFallbackExecuted`.

- validation:
  - Added regression:
    `test/Tools/circt-sim/always-comb-constant-no-sensitivity.sv`.
  - Full `test/Tools/circt-sim` sweep: `898/898 passed`.
  - Re-ran mutation parity on generated batch (`m1..m23`) with functional
    coverage enabled in xrun:
    `23/23` summary matches in
    `/tmp/mut_parity_broad_1772286797/parity_after_fix.tsv`.

## 2026-02-28 (compound modulo/division assignment mutation class)

- realizations:
  - `%=`/`/=` confusion is a realistic arithmetic implementation bug class in
    sequential datapath/control updates and complements existing binary
    `%`/`/` mutation operators.
  - Lit tests that synthesize SV using shell `printf` require careful `%`
    escaping (`%%%%`) to avoid format-string interpretation side effects.

- changes made:
  - Added native operators:
    - `MOD_EQ_TO_DIV_EQ`
    - `DIV_EQ_TO_MOD_EQ`
  - Integrated operators in:
    - planner op catalog, compound-assignment site detection, family
      classification, and native apply rewrites
      (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`)
      in `tools/circt-mut/circt-mut.cpp`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-mod-assign-ops.test`
    - `test/Tools/native-create-mutated-mod-eq-to-div-eq-site-index.test`
    - `test/Tools/native-create-mutated-div-eq-to-mod-eq-site-index.test`

- validation:
  - Red-first: all new tests fail before implementation and pass after.
  - Focused lit slice across compound-assign families: `13 passed`.
  - Seeded xrun-vs-circt parity campaign on `cov_intro_seeded_modassign`
    variation with safe native-op allowlist:
    - `count=24`, `seed=20260228`
    - result: `match=24`, `mismatch=0`
    - operators observed include both new ops:
      `NATIVE_MOD_EQ_TO_DIV_EQ`, `NATIVE_DIV_EQ_TO_MOD_EQ`
    - workspace: `/tmp/mut_parity_modassign_898860`

## 2026-02-28 (compound arithmetic-shift assignment mutation class)

- realizations:
  - Confusion between logical and arithmetic shift-assignment (`>>=` vs
    `>>>=`) is a realistic sequential fault class and semantically distinct
    from expression-level `>>`/`>>>` swaps.
  - Compound-assignment token matching must explicitly avoid overlap with
    neighboring longer/shorter tokens (`>>=` vs `>>>=`) to keep deterministic
    site-index contracts.
  - For parity campaigns, timeout-guarded safe-op subsets remain important to
    prevent liveness stalls from unrelated control-loop mutations.

- changes made:
  - Added native operators:
    - `SHR_EQ_TO_ASHR_EQ`
    - `ASHR_EQ_TO_SHR_EQ`
  - Integrated these operators in:
    - native op catalog, site collection, family classification, and apply
      rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`)
      in `tools/circt-mut/circt-mut.cpp`
    - native-op token validation in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-ashr-assign-ops.test`
    - `test/Tools/native-create-mutated-shr-eq-to-ashr-eq-site-index.test`
    - `test/Tools/native-create-mutated-ashr-eq-to-shr-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-ashr-eq-skip-ashr-assign.test`

- validation:
  - Red-first: new tests fail before implementation, pass after.
  - Focused lit slice over touched shift/compound/native-op tests:
    `18 passed`.
  - Seeded xrun-vs-circt parity campaign on deterministic `cov_intro_seeded`
    harness with safe-op allowlist including new operators:
    - `count=24`, `seed=20260228`
    - result: `match=24`, `mismatch=0`
    - operators observed include:
      `NATIVE_SHR_EQ_TO_ASHR_EQ` and `NATIVE_ASHR_EQ_TO_SHR_EQ`
    - workspace: `/tmp/mut_parity_shift_740018`

## 2026-02-28 (compound shift-assignment mutation class + seeded parity hardening)

- realizations:
  - `<<=`/`>>=` confusion is a realistic sequential datapath/control fault class
    and was missing from the native mutation operator set.
  - Seeded parity campaigns need per-mutant timeouts and a safe-op subset;
    unrestricted arithmetic sets can mutate loop updates and create liveness
    hangs (for example non-terminating `repeat/for` control).
  - A transient xrun-vs-circt mismatch in this campaign was harness-induced:
    `always @*` sanitizer initialization at time-zero is simulator-scheduling
    sensitive; switching to `always_comb` removed the artifact.

- changes made:
  - Added native operators:
    - `SHL_EQ_TO_SHR_EQ`
    - `SHR_EQ_TO_SHL_EQ`
  - Integrated operators into:
    - native op catalog, site collection, family classification, and apply:
      `tools/circt-mut/NativeMutationPlanner.cpp`
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`):
      `tools/circt-mut/circt-mut.cpp`
    - native-op validator:
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-shift-assign-ops.test`
    - `test/Tools/native-create-mutated-shl-eq-to-shr-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-shl-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-shl-eq-skip-ashr-assign.test`

- validation:
  - Red-first: all four new tests fail before implementation, pass after.
  - Focused lit slices over compound-assign/native-plan/native-backend tests:
    all passing (`22 passed` in touched slice).
  - Seeded xrun-vs-circt parity campaign on deterministic `cov_intro_seeded`
    harness with safe native op subset (including both new ops):
    - `count=20`, `seed=20260228`
    - result: `match=20`, `mismatch=0`
    - campaign workspace: `/tmp/mut_parity_shift_740018`

## 2026-02-28 (compound-assign realism + array-index sentinel parity bug)

- realizations:
  - Compound-assignment confusion for `*=`/`/=` is a realistic arithmetic fault
    class and fits naturally with existing `+=`/`-=` mutations.
  - A real xrun-vs-circt mismatch surfaced under mutation:
    `NATIVE_LT_TO_LE@3` changed `i < 16` to `i <= 16`, and CIRCT interpreted an
    imported sentinel index as a valid in-range access.
  - Root cause was not randomness: deterministic seeded parity reproduced it
    consistently (`79/80` before fix, one stable mismatch).

- changes made:
  - Added native operators:
    - `MUL_EQ_TO_DIV_EQ`
    - `DIV_EQ_TO_MUL_EQ`
  - Integrated new operators in planner + mutator + mode mappings:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `tools/circt-mut/circt-mut.cpp`
    - `utils/mutation_mcy/templates/native_create_mutated.py`
    - `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions for new ops:
    - `test/Tools/native-create-mutated-mul-eq-to-div-eq-site-index.test`
    - `test/Tools/native-create-mutated-div-eq-to-mul-eq-site-index.test`
    - updated `circt-mut-generate-circt-only-arith-mode-compound-assign-ops.test`
  - Fixed circt-sim array OOB sentinel behavior:
    - added `isImportedArrayIndexOutOfBoundsSentinel(...)` in
      `LLHDProcessInterpreter`,
    - applied it to `hw.array_get`, `hw.array_inject`, and `llhd.sig.array_get`
      probe/drive paths.
  - Added circt-sim regression test:
    - `test/Tools/circt-sim/array-get-oob-sentinel-index.sv`

- validation:
  - Focused lit slices for compound-assign and sentinel regression pass.
  - Minimal reproducer before/after:
    - before: `COUNT=2` for out-of-bounds loop (`i <= 16`)
    - after: `COUNT=1` (matches xrun semantics)
  - Previously failing mutant now matches:
    - `NATIVE_LT_TO_LE@3` summary parity restored.
  - Full seeded parity rerun (`count=80`, `seed=31`) now:
    - `matches=80`, `fails=0`.

## 2026-02-28 (modulo/division confusion mutation class)

- realizations:
  - `%` vs `/` confusion is a realistic arithmetic bug class in datapath RTL
    and complements existing `MUL`/`DIV` confusion operators.
  - xrun-vs-circt parity must avoid harness artifacts:
    - `get_coverage()` in xrun requires `-coverage ...` enabled.
    - stimulus driven on the same `posedge` as DUT flops can introduce race
      order differences that look like simulator bugs.
    - uninitialized memories can create X-vs-value noise unrelated to mutation
      semantics.

- changes made:
  - Added native operators:
    - `MOD_TO_DIV`
    - `DIV_TO_MOD`
  - Integrated new operators into CIRCT-only planning:
    - native op catalog (`kNativeMutationOpsAll`)
    - site collection (`collectBinaryMulDivSites` extended to `%`)
    - op-site dispatch (`collectSitesForOp`)
    - arithmetic fault-family classification (`getOpFamily`)
  - Added CIRCT-only mode mapping coverage in `circt-mut generate`:
    - `arith`
    - `invert` / `inv`
    - `balanced` / `all`
  - Added native mutator rewrite support:
    - `utils/mutation_mcy/templates/native_create_mutated.py`
      - `% -> /` for `MOD_TO_DIV`
      - `/ -> %` for `DIV_TO_MOD`
  - Updated public native-op CLI validation list:
    - `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-mod-ops.test`
    - `test/Tools/native-create-mutated-mod-to-div-site-index.test`
    - `test/Tools/native-create-mutated-div-to-mod-site-index.test`

- validation:
  - TDD step: new tests fail before implementation; pass after implementation.
  - Focused lit slices:
    - `circt-mut-generate-circt-only-arith-mode-(div|mod)-ops`
    - `native-create-mutated-(div-to-mul|mul-to-div|mod-to-div|div-to-mod)-site-index`
    - broader touched slice with related arith/native tests (`12 passed`)
  - Seeded parity campaign using generated mutants (`count=24`, `seed=20260228`)
    on a race-free deterministic `cov_intro_seeded`-style bench:
    - includes new ops `NATIVE_MOD_TO_DIV@1`, `NATIVE_DIV_TO_MOD@1`
    - `xrun` (with coverage enabled) vs `circt-sim` summary comparison:
      `ok=24 mismatch=0 fail=0`

## 2026-02-28 (if-condition polarity mutation class)

- realizations:
  - Inverted `if` conditions are a common and realistic control-path bug class
    not captured by pure operator substitutions.
  - Mutation campaigns need explicit liveness caps (`timeout`, `--max-time`,
    `--max-deltas`) because some control mutations intentionally create
    non-terminating behavior (for example clock-toggle logic inversion).
  - Intermittent `Permission denied` on `circt-sim` invocation appears as an
    infra flake; re-running the same command in-place succeeds and should be
    treated as non-semantic noise in parity campaigns.

- changes made:
  - Added native operator `IF_COND_NEGATE`:
    - planner site detection via `if` word-boundary token matching and
      balanced-parenthesis parsing,
    - mutator rewrite from `if (cond)` to `if (!(cond))`,
    - fallback planner support in `native_mutation_plan.py`.
  - Extended CIRCT-only mode mappings (`control`, `connect`, `cnot0/cnot1`,
    `inv`/`invert`, `balanced/all`) to include `IF_COND_NEGATE`.
  - Added regression tests:
    - `native-create-mutated-if-cond-negate-site-index`
    - `native-create-mutated-if-cond-negate-nested-cond`
    - `circt-mut-generate-circt-only-control-mode-if-cond-negate-op`
  - Re-ran seeded xrun-vs-circt campaigns:
    - single-module signature campaign (`24` mutants): `ok=22`,
      no mismatches; remaining failures were bounded liveness/infra cases.
    - `cov_intro_seeded`-style coverage campaign (`24` mutants): no semantic
      mismatches on terminating mutants; `IF_COND_NEGATE` parity matched.

## 2026-02-28 (mux-arm swap mutation class)

- realizations:
  - Swapping ternary mux arms (`sel ? a : b` -> `sel ? b : a`) is a realistic
    control/data bug class and complements assignment-timing faults.
  - Ternary token matching must be assignment-context aware; wildcard case-item
    patterns (`2'b0?`) can otherwise be misclassified as ternary operators.

- changes made:
  - Added native operator `MUX_SWAP_ARMS` to CIRCT-only planner op set.
  - Implemented site detection with:
    - assignment-context gating (`=` or `<=` before the `?` at statement depth),
    - declaration/port/type disqualifier filtering,
    - nested-ternary-aware `?`/`:` pairing.
  - Added mutator rewrite support to swap ternary true/false arms while
    preserving surrounding whitespace slices.
  - Extended CIRCT-only mode mappings (`control`, `connect`, `cnot0/cnot1`,
    `inv`/`invert`, `balanced/all`) to include `MUX_SWAP_ARMS`.
  - Updated fallback native planner utility (`native_mutation_plan.py`) and
    native-op validation in `run_mutation_mcy_examples.sh`.
  - Added regression tests:
    - `native-create-mutated-mux-swap-arms-site-index`
    - `native-create-mutated-mux-swap-arms-nba`
    - `native-create-mutated-mux-skip-case-item`
    - `circt-mut-generate-circt-only-control-mode-mux-swap-op`
  - Added declaration-safety regression coverage for assignment timing:
    - `native-create-mutated-ba-skip-user-type-decl`
    - `circt-mut-generate-circt-only-ba-skip-user-type-decl`
  - Fixed a real mutation bug where `BA_TO_NBA` could rewrite typed
    declarations (`my_t v = ...`) into invalid syntax (`<=` in declarations).
    Added typed-declaration filters to both planner and mutator paths.
  - Re-ran seeded parity campaigns after the fix:
    - control-mode single-module signature campaign: `ok=24 mismatch=0 fail=0`
      (includes `MUX_SWAP_ARMS` mutants),
    - `cov_intro_seeded`-style coverage campaign: no xrun-vs-circt mismatches on
      terminating mutants; remaining failures are liveness/time-limit cases from
      clock-killing mutants (for example `UNARY_BNOT_DROP` on `clk = ~clk`).

## 2026-02-28 (assignment timing faults and seeded parity campaigns)

- realizations:
  - Assignment-style timing mutations (`=` vs `<=`) are a realistic RTL bug
    class and should be modeled explicitly instead of only via operator swaps.
  - Seeded parity campaigns need deterministic initialization boundaries; when a
    signature depends on uninitialized/X-heavy state, xrun-vs-circt mismatches
    can be false alarms for mutation parity.
  - Some constant-flip mutants on clock-generator literals produce liveness
    failures (`both_fail`) in both simulators; these should be classified as
    non-parity failures, not semantic mismatches.
  - Cross-instance scheduling in current `circt-sim` can shift observation
    timing for NBA-fed control signals in seeded harnesses; single-module
    parity harnesses are currently more reliable for mutation equivalence runs.

- changes made:
  - Added native assignment operators:
    - `BA_TO_NBA`
    - `NBA_TO_BA`
  - Added planner and mutator site detection with context guards to avoid:
    - declaration/typedef/port contexts,
    - comparator/assignment-token collisions,
    - continuous-assignment contexts.
  - Added CIRCT-only mode mapping coverage for assignment faults in
    `control`, `connect`, `cnot0/cnot1`, `inv`/`invert`, and `balanced/all`.
  - Added regression tests for assignment site indexing and safety filters:
    - `native-create-mutated-ba-to-nba-site-index`
    - `native-create-mutated-nba-to-ba-site-index`
    - `native-create-mutated-ba-skip-decl-and-comparator`
    - `native-create-mutated-ba-skip-continuous-assign`
    - `circt-mut-generate-circt-only-control-mode-assignment-ops`
  - Re-ran focused lit coverage for new assignment operators: all targeted
    tests pass.
  - Ran multiple seeded mutation parity campaigns:
    - assignment-focused campaign: `ok=24 mismatch=0 fail=0`
    - broader deterministic single-module campaign:
      `ok=37 mismatch=0 fail=11` (`both_fail` liveness mutants)

## 2026-02-27

- realizations:
  - Adding arithmetic mutations (`ADD_TO_SUB`, `SUB_TO_ADD`) increases fault-class coverage for realistic RTL datapath bugs, but only if site detection stays context-aware.
  - `xrun` coverage API methods require functional coverage to be explicitly enabled (`-coverage functional`), otherwise `get_coverage()` is reported as `0.00`.
  - For stable xrun batch runs, each mutant needs isolated coverage output (`-covworkdir`/`-covtest`) or explicit overwrite; otherwise runs fail with existing DB errors.

- surprises:
  - Initial arithmetic site matching was over-inclusive and treated packed-range width math (for example `[W-1:0]`) as mutation targets.
  - Those range-level mutations dominated weighted schedules and reduced semantic diversity despite anti-dominance scoring.

- changes made:
  - Added arithmetic operators in CIRCT-only planner/mutator and mode mappings (`arith`, `inv`, `balanced/all`).
  - Added bracket-context exclusion for arithmetic site detection in both planner and mutator so site indexing remains aligned.
  - Added regression tests for arithmetic site indexing, unary-minus safety, and range-minus exclusion.
  - Re-ran seeded mutation sweeps with `xrun` and `circt`; no mismatches observed in coverage summaries for tested mutants.

## 2026-02-27 (shift operators)

- realizations:
  - Shift-direction swaps (`<<`/`>>`) are a practical, realistic RTL bug class and complement compare/logic/arithmetic fault families.
  - Legacy-cycle assertions based on fixed line numbers are brittle as the native op set grows; tests should assert behavioral properties instead.

- surprises:
  - Local `yosys` wrappers in this environment recurse to themselves, so direct runtime decomp/help probes are unreliable here; static/native implementation work should not depend on invoking it.

- changes made:
  - Added `SHL_TO_SHR` and `SHR_TO_SHL` in planner/mutator with structural guards against `<<<`, `>>>`, `<<=`, and `>>=`.
  - Extended CIRCT-only mode mappings (`arith`, `inv`, `invert`, `balanced/all`) to include shift mutations.
  - Hardened cycle/index tests to avoid depending on an exact global operator count.
  - Ran a shift-focused seeded mutation campaign (24 mutants) through both `xrun` and `circt`; no mismatches observed.

## 2026-02-27 (4-state compare operators)

- realizations:
  - X-propagation handling (`==` vs `===`, `!=` vs `!==`) is a distinct and realistic RTL fault class that should not be conflated with ordinary compare mutations.
  - Comparator token matching must be exact; naive `==` scanning can incorrectly match inside `===` and create semantically wrong mutants.

- changes made:
  - Added native operators `CASEEQ_TO_EQ` and `CASENEQ_TO_NEQ`.
  - Tightened comparator site detection in both planner and mutator so `EQ_TO_NEQ`/`NEQ_TO_EQ` no longer target `===`/`!==` sites.
  - Extended `arith`/`inv`/`invert`/`balanced` mode mappings to include 4-state compare mutations.

## 2026-02-27 (binary XOR-only tightening)

- realizations:
  - Mutating reduction XOR (`^a`) as if it were binary XOR is structurally valid textually but semantically a different fault class, and it skews campaigns toward less realistic operator substitutions.

- changes made:
  - Tightened `XOR_TO_OR` site detection in planner and mutator to binary XOR only.
  - Added guards for XNOR and assignment forms (`^~`, `~^`, `^=`), and ensured site-index parity between planner and mutator.

## 2026-02-27 (constant literal coverage broadening)

- realizations:
  - Many real SV designs use unsized tick constants and hex 1-bit literals (`'0`, `'1`, `1'h0`, `1'h1`) rather than only decimal/binary 1-bit constants.
  - Limiting stuck-at mutations to `1'b*`/`1'd*` leaves practical constant-fault space under-exercised.

- changes made:
  - Extended `CONST0_TO_1`/`CONST1_TO_0` detection and rewrite support to include `1'h0`/`1'h1` and `'0`/`'1`.
  - Refactored constant-flip rewrite logic into a single helper to avoid duplicated token-mapping code.

## 2026-02-27 (signedness cast mutations)

- realizations:
  - Missing or inverted signedness casts are a realistic non-trivial RTL bug class, especially around arithmetic comparisons and overflow behavior.

- changes made:
  - Added native cast operators `SIGNED_TO_UNSIGNED` and `UNSIGNED_TO_SIGNED`.
  - Added boundary-safe cast-call site detection with optional whitespace handling.
  - Integrated cast operators into CIRCT-only `arith`/`inv`/`invert`/`balanced` mode mappings.

## 2026-02-27 (seeded parity campaign hygiene)

- realizations:
  - A full xrun-vs-circt mutant mismatch can be a false alarm when the harness reads uninitialized state (`X`), even with fixed seeds.
  - `$random(seed)` sequence parity was confirmed separately; the mismatch source was testbench/design initialization, not RNG drift.

- changes made:
  - Switched seeded mini parity harnesses to deterministic reset initialization before signature comparison.
  - Re-ran a 24-mutant campaign after reset hygiene; parity result was `ok=24 mismatch=0 fail=0`.

## 2026-02-27 (bitwise operator fault class)

- realizations:
  - Logical (`&&`, `||`) and bitwise (`&`, `|`) operator faults are distinct and both appear in realistic RTL bug patterns.
  - Keeping them separate avoids operator-class blind spots and avoids duplicating existing logical mutation semantics.

- changes made:
  - Added native bitwise operators `BAND_TO_BOR` and `BOR_TO_BAND`.
  - Added binary-site filtering to skip reduction and assignment forms while preserving site-index determinism.
  - Added site-index mutation tests and mode-generation coverage tests for the new bitwise operators.

## 2026-02-27 (unary bitwise inversion fault class)

- realizations:
  - Dropped bitwise inversion (`~expr` accidentally removed) is a common RTL polarity bug and should be modeled separately from logical-negation drops (`!expr`).
  - Unary `~` must be token-aware to avoid mutating reduction and XNOR spellings (`~&`, `~|`, `~^`, `^~`).

- changes made:
  - Added native operator `UNARY_BNOT_DROP`.
  - Implemented site detection and mutation rewriting with guards for reduction/XNOR contexts.
  - Extended CIRCT-only mode mappings (`control`, `inv`, `invert`, `balanced/all`) to include `UNARY_BNOT_DROP`.
  - Added site-index rewrite and mode-generation regression tests.
  - Ran seeded `xrun` vs `circt` parity campaign (`30` mutants including `UNARY_BNOT_DROP`): `ok=30 mismatch=0 fail=0` (with retry guard for transient `Permission denied` relink races).

## 2026-02-27 (multiplication arithmetic fault class)

- realizations:
  - Operand/operator confusions between `+` and `*` are realistic datapath bugs and are semantically distinct from existing add/sub flips.
  - Binary `*` site detection must explicitly exclude wildcard and assignment contexts (`(*)`, `**`, `*=`) to avoid structurally invalid or semantically unrelated mutations.

- changes made:
  - Added native operators `MUL_TO_ADD` and `ADD_TO_MUL`.
  - Added bracket-depth filtering so width/index expressions in `[...]` are not targeted.
  - Integrated multiplication operators into CIRCT-only `arith`, `inv`/`invert`, and `balanced/all` mode mappings.
  - Added mutation rewrite/site-index regression tests plus arith-mode generation coverage tests.
  - Hardened const-site generation regression (`const-unsized-sites`) to avoid brittle dependence on exact global op count growth.
  - Ran seeded parity campaign on a multiplier-bearing mini design (`35` mutants): `ok=35 mismatch=0 fail=0`.

## 2026-02-27 (unary arithmetic sign fault class)

- realizations:
  - Missing unary minus is a realistic arithmetic polarity bug and is semantically distinct from binary operator substitutions.
  - Site detection must reject binary/compound minus contexts (`a-b`, `--`, `->`) to avoid invalid rewrites.

- changes made:
  - Added native operator `UNARY_MINUS_DROP`.
  - Implemented planner and mutator site matching for unary-minus tokens with binary/compound context guards.
  - Integrated `UNARY_MINUS_DROP` into CIRCT-only `arith`, `inv`/`invert`, and `balanced/all` mode mappings.
  - Added site-index and no-op fallback regression tests for unary-minus behavior, plus arith-mode generation test coverage.
  - Ran seeded parity campaign on unary-minus-bearing mini design (`35` mutants): `ok=35 mismatch=0 fail=0`.

## 2026-02-27 (division arithmetic fault class)

- realizations:
  - `/` vs `*` confusion is a realistic datapath bug class and complements the
    existing add/sub/mul operator substitutions.
  - Division site detection must be strict enough to avoid non-expression
    spellings (`/=`, `//`) and keep site-index parity between planner and
    mutator.

- changes made:
  - Added native operators `DIV_TO_MUL` and `MUL_TO_DIV`.
  - Refactored mutator-side mul/div token scanning into one helper to avoid
    duplicated site logic.
  - Integrated division operators into CIRCT-only `arith`, `inv`/`invert`, and
    `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-div-to-mul-site-index`
    - `native-create-mutated-mul-to-div-site-index`
    - `native-create-mutated-div-skip-contexts`
    - `circt-mut-generate-circt-only-arith-mode-div-ops`
  - Ran seeded xrun-vs-circt parity campaign including division/multiplication
    mutations (`40` mutants): `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (signed shift fault class)

- realizations:
  - Confusing logical and arithmetic right shifts (`>>` vs `>>>`) is a
    realistic RTL bug class, especially in signed datapaths.
  - Matching `>>>` requires distinct token handling from `>>` so we avoid
    consuming shift-assignment spellings (`>>>=`) and preserve site-index
    consistency.

- surprises:
  - The local `yosys` executable available in this environment is a
    self-recursing wrapper script, so direct binary decompilation was not
    possible here.

- changes made:
  - Added native operators `SHR_TO_ASHR` and `ASHR_TO_SHR`.
  - Implemented dedicated planner/mutator site detection for arithmetic
    right-shift tokens (`>>>`) with assignment-guard filtering.
  - Integrated the new operators into CIRCT-only `arith`, `inv`/`invert`, and
    `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-shr-to-ashr-site-index`
    - `native-create-mutated-ashr-to-shr-site-index`
    - `native-create-mutated-ashr-skip-assign`
    - `circt-mut-generate-circt-only-arith-mode-ashr-ops`
  - Hardened an existing xcompare generation regression that depended on a
    stale fixed op-count window (`circt-mut-generate-circt-only-xcompare-mode-ops`).
  - Ran seeded xrun-vs-circt parity campaign including shift-class mutations
    (`40` mutants, `SHR_TO_ASHR=6`, `ASHR_TO_SHR=6`): `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (logical vs bitwise confusion fault class)

- realizations:
  - Confusing short-circuit logical operators with bitwise operators
    (`&&`/`||` vs `&`/`|`) is a realistic RTL bug class and semantically
    distinct from pure invert/swap mutations.
  - Logical token matching should avoid assertion-style triple tokens so site
    indexing remains structurally valid.

- changes made:
  - Added native operators:
    - `LAND_TO_BAND`, `LOR_TO_BOR`
    - `BAND_TO_LAND`, `BOR_TO_LOR`
  - Implemented planner-side logical token detection with triple-token guards
    and operand-boundary checks.
  - Implemented mutator-side rewrites for all four operators with matching
    site-index behavior.
  - Integrated the new operators into CIRCT-only `control`, `connect`,
    `inv`/`invert`, and `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-land-to-band-site-index`
    - `native-create-mutated-band-to-land-site-index`
    - `circt-mut-generate-circt-only-control-mode-logbit-ops`
  - Updated brittle default-sequence regression to reflect deterministic op
    ordering after the new operator family was added.
  - Ran seeded xrun-vs-circt parity campaign including logical/bitwise
    confusion mutations (`40` mutants, each new op present 5 times):
    `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (xor/xnor confusion fault class)

- realizations:
  - XOR vs XNOR confusion is a realistic logic fault class and is semantically
    distinct from the existing `XOR_TO_OR` mutation.
  - `XNOR_TO_XOR` must avoid reduction-XNOR forms so binary-site indexing stays
    aligned and structurally valid.

- changes made:
  - Added native operators `XOR_TO_XNOR` and `XNOR_TO_XOR`.
  - Implemented planner-side binary XNOR site detection for `^~` and `~^` with
    operand-boundary checks.
  - Implemented mutator-side rewrites for both directions with site-index
    parity against planner behavior.
  - Integrated XOR/XNOR operators into CIRCT-only `control`, `connect`,
    `inv`/`invert`, and `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-xor-to-xnor-site-index`
    - `native-create-mutated-xnor-to-xor-site-index`
    - `native-create-mutated-xnor-skip-reduction`
    - `circt-mut-generate-circt-only-control-mode-xnor-ops`
  - Updated weighted diversity regression to include the expanded logic-op set.
  - Ran seeded xrun-vs-circt parity campaign including xor/xnor mutations
    (`40` mutants, `XOR_TO_XNOR=8`, `XNOR_TO_XOR=8`):
    `ok=40 mismatch=0 fail=0`.

## 2026-02-28 (relational polarity swap fault class)

- realizations:
  - Opposite-direction relational swaps (`<`↔`>`, `<=`↔`>=`) are a realistic
    fault class and distinct from the existing widen/narrow comparator class
    (`LT_TO_LE`, `LE_TO_LT`, etc.).
  - Several generation regressions were brittle to exact operator-count
    windows; they needed to assert properties rather than fixed cycle lengths.

- changes made:
  - Added native operators:
    - `LT_TO_GT`, `GT_TO_LT`
    - `LE_TO_GE`, `GE_TO_LE`
  - Integrated operators in native planner op catalog and site detection.
  - Integrated operators in CIRCT-only mode mappings for `arith`,
    `inv`/`invert`, and `balanced/all`.
  - Extended native mutator rewrites and refactored standalone `<`/`>` token
    lookup into one helper to avoid duplicated matcher logic.
  - Added/updated regression coverage:
    - `native-create-mutated-lt-to-gt-site-index`
    - `native-create-mutated-gt-to-lt-site-index`
    - `native-create-mutated-le-to-ge-site-index`
    - `native-create-mutated-ge-to-le-site-index`
    - `circt-mut-generate-circt-only-arith-mode-relpol-ops`
  - Hardened brittle schedule-dependent tests:
    - `circt-mut-generate-circt-only-ignore-comments-strings`
    - `circt-mut-generate-circt-only-site-aware-cycle`
    - `circt-mut-generate-circt-only-weighted-fault-class-diversity`
  - Validation:
    - focused relpol lit slice: `5 passed`
    - broader CIRCT-only mutation suite: `83 passed`
    - seeded parity campaign on relational-signature mini design
      (`20` mutants): `ok=20 mismatch=0 fail=0`
    - seeded parity campaign on `cov_intro_seeded` (`12` mutants):
      `ok=12 mismatch=0 fail=0`

## 2026-02-28 (xcompare bidirectional fault class)

- realizations:
  - X-sensitivity bugs are often symmetric in practice: using `==`/`!=` where
    `===`/`!==` was intended (and vice versa) can silently change behavior
    around unknown values.
  - Weighted-applicable and context-priority regressions should assert the
    property they validate, not a single hard-coded operator when multiple
    same-site candidates are valid.

- changes made:
  - Added native operators:
    - `EQ_TO_CASEEQ`
    - `NEQ_TO_CASENEQ`
  - Integrated new operators into:
    - native planner op catalog + comparator site counting
    - CIRCT-only mode mappings (`arith`, `inv`/`invert`, `balanced/all`)
    - native mutator rewrite dispatch
  - Added regression tests:
    - `native-create-mutated-eq-to-caseeq-site-index`
    - `native-create-mutated-neq-to-caseneq-site-index`
    - `circt-mut-generate-circt-only-xcompare-bidir-ops`
  - Hardened schedule-sensitive tests to allow either applicable eq-site
    mutation when appropriate:
    - `circt-mut-generate-circt-only-weighted-applicable-only`
    - `circt-mut-generate-circt-only-weighted-context-priority`

- validation:
  - focused new-op lit slice: `3 passed`
  - broader CIRCT-only mutation suite: `86 passed`
  - seeded `cov_intro_seeded` parity campaign (`16` mutants):
    `ok=16 mismatch=0 fail=0`
  - seeded xcompare-signature parity campaign (`24` arith mutants):
    `ok=24 mismatch=0 fail=0`
  - direct-op parity checks:
    - `NATIVE_EQ_TO_CASEEQ@1`: `circt_sig==xrun_sig`
    - `NATIVE_NEQ_TO_CASENEQ@1`: `circt_sig==xrun_sig`

## 2026-02-28 (edge-polarity fault class + native-op validator alignment)

- realizations:
  - Event-control edge polarity (`posedge` vs `negedge`) is a realistic and
    semantically distinct control/timing fault class (clock/reset sensitivity
    bugs).
  - `run_mutation_mcy_examples.sh` had drifted to a legacy native-op allowlist;
    the validator rejected many supported native operators and needed to be
    synchronized with current CIRCT-only operator coverage.
  - Xcelium/xrun does not enable covergroup methods by default; parity harnesses
    should use explicit signatures/counters unless functional coverage is
    explicitly enabled.

- changes made:
  - Added native operators:
    - `POSEDGE_TO_NEGEDGE`
    - `NEGEDGE_TO_POSEDGE`
  - Implemented planner-side keyword site detection with identifier-boundary
    checks and integrated both operators into timing-family classification.
  - Implemented mutator-side rewrites and site-index behavior for both
    operators.
  - Integrated operators into CIRCT-only mode mappings:
    - `control`, `connect`, `inv`/`invert`, `balanced/all`, and primitive
      `cnot0/cnot1`.
  - Extended Python fallback planner (`native_mutation_plan.py`) with matching
    operator support and keyword site counting.
  - Added/updated regression tests:
    - `native-create-mutated-posedge-to-negedge-site-index`
    - `native-create-mutated-negedge-to-posedge-site-index`
    - `native-mutation-plan-edge-polarity`
    - `circt-mut-generate-circt-only-control-mode-edge-polarity-ops`
    - `run-mutation-mcy-examples-native-mutation-ops-edge-polarity-pass`
    - updated `circt-mut-generate-circt-only-weighted-fault-class-diversity`
      to include edge-polarity operators in control-family coverage checks.
  - Synchronized `run_mutation_mcy_examples.sh` native-op token validation with
    the full native CIRCT-only operator set.

- validation:
  - focused lit slice for new edge operators and script validation: `7 passed`
  - targeted xrun-vs-circt parity checks on fixed-seed `cov_intro_seeded`
    harness:
    - baseline: `SIG=2921849304`, `COV_TOTAL=84.38` (match)
    - `NATIVE_POSEDGE_TO_NEGEDGE@1`: match
    - `NATIVE_NEGEDGE_TO_POSEDGE@1`: match
    - `NATIVE_IF_COND_NEGATE@1`: match
  - seeded mini-campaign (`10` balanced mutants): `ok=10 mismatch=0 fail=0`.

## 2026-02-28 (xrun/circt mismatch root cause: uninitialized 4-state arrays)

- realizations:
  - The deterministic xrun/circt delta on seeded mutants was not RNG-related:
    it came from uninitialized `logic` memory semantics.
  - In `--ir-llhd` output, uninitialized `moore.variable : <uarray<... x lN>>`
    was lowered to a zeroed `llhd.sig` initializer instead of all-`X`.
  - This made `==/!=`-based control mutations diverge whenever reads touched
    uninitialized memory (xrun saw `X`, CIRCT saw `0`).
  - One batch-sweep mismatch set was a false positive caused by transient
    `Permission denied` while binaries were being updated concurrently; direct
    per-mutant rechecks matched.

- changes made:
  - Added regression test:
    - `test/Tools/circt-sim/uninitialized-logic-memory-reads-x.sv`
  - Implemented Moore-to-core fix for uninitialized fixed arrays of 4-state
    elements:
    - added `createAllXArrayOfFourStateValue(...)`
    - added compact integer-pattern replication helper
      `repeatIntegerPattern(...)`
    - wired `VariableOpConversion` fallback init path to use all-`X` for
      fixed-size arrays with 4-state leaf elements before generic zero init.

- validation:
  - TDD:
    - new lit test fails before fix (observed `RES r=0 ... hit_neq=1`),
      passes after fix (`RES r=x ... hit_neq=0`).
  - focused stability checks:
    - `class-null-compare.sv`, `readmemh-basic.sv`,
      `syscall-random-unseeded-process-stability.sv` all pass.
  - minimized repro:
    - xrun: `RES r=x hit_eq=0 hit_neq=0 hit_not_eq=0`
    - circt (post-fix): same.
  - seeded mutant parity rechecks:
    - previously divergent `NATIVE_EQ_TO_NEQ@1` and
      `NATIVE_IF_COND_NEGATE@3` now match xrun exactly.
    - previously flagged status mismatches (`51/52/55/56`) rechecked as
      xrun/circt matches (earlier failures were transient execution races).

## 2026-02-28 (inc/dec fault class + seeded parity triage)

- realizations:
  - Increment/decrement operator faults (`++`/`--`) are realistic counter/state
    bugs and were missing from the native operator set.
  - Some planner tests were brittle because they implicitly depended on full
    operator catalog length and default ordering instead of constraining
    operator scope in-test.
  - One xrun-vs-circt mutant mismatch on `cov_intro_seeded` was due to an
    intentional race introduced by `NBA_TO_BA` in clock-edge stimulus
    (`we <=` -> `we =`), not a circt scheduling bug.

- changes made:
  - Added native operators:
    - `INC_TO_DEC`
    - `DEC_TO_INC`
  - Integrated these operators into:
    - native planner catalog/site counting/family classification
    - CIRCT-only mode mappings (`arith`, `inv`/`invert`, `balanced/all`)
    - native mutator rewrite dispatch (`native_create_mutated.py`)
    - Python native-plan fallback operator set and site counting
  - Added regression tests:
    - `circt-mut-generate-circt-only-arith-mode-incdec-ops`
    - `native-create-mutated-inc-to-dec-site-index`
    - `native-create-mutated-dec-to-inc-site-index`
  - Stabilized planner tests by explicitly constraining ops where needed:
    - `native-mutation-plan-site-aware-cycle`
    - `native-mutation-plan-le-assignment-order`
    - `circt-mut-generate-circt-only-weighted-context-priority`
    - adjusted `circt-mut-generate-circt-only-inv-mode-unary-bnot-op` count
      for expanded operator catalog.

- validation:
  - focused lit slice for touched tests: pass
  - broader CIRCT-only mutation suite slice (`generate/native-plan/create-mutated`):
    `76 passed`
  - seeded `cov_intro_seeded` parity sweep (`40` mutants):
    - overall: `39 match / 1 diff`
    - diff: `NATIVE_NBA_TO_BA@3` with race-sensitive clock-edge blocking write
      in stimulus (`xrun rdata=252`, `circt rdata=56`)
    - excluding race-prone BA/NBA swaps: `38 match / 0 diff`
  - seeded `incdec_seeded` arith sweep (`20` mutants, includes inc/dec ops):
    `20 match / 0 diff`.

## 2026-02-28 (MooreToCore ref-arg store fallback deletion safety check)

- realizations:
  - The removed `AssignOpConversion` fallback in
    `lib/Conversion/MooreToCore/MooreToCore.cpp` was forcing `llvm.store` for
    *all* `llhd.ref` function block args, including nonblocking assignments.
  - That fallback contradicts current interpreter behavior, which now tracks
    ref-argument signal provenance across function/task boundaries via
    `refBlockArgSources` and related ref mapping paths.
  - The prior mismatch was semantic, not random: task NBA through ref params was
    being lowered with blocking-store behavior.

- changes made:
  - Kept the fallback removal in MooreToCore (no replacement path added there).
  - Added/kept regression to lock expected task NBA ordering:
    - `test/Tools/circt-sim/task-nonblocking-assign-in-task-order.sv`

- validation:
  - Targeted lit tests pass with current built tools:
    - `test/Conversion/MooreToCore/ref-param-store.mlir`
    - `test/Conversion/MooreToCore/interface-timing-after-inlining.sv`
    - `test/Tools/circt-sim/task-nonblocking-assign-in-task-order.sv`
    - `test/Tools/circt-sim/coverage-event-sampling-order-negedge.sv`
  - Deterministic seeded parity repro rerun:
    - xrun and circt both report:
      `SUMMARY addr_unique=12 we_unique=2 cov_total=87.50 sig=876`
  - Full rebuild/suite execution is currently blocked by concurrent dirty-file
    compile failures outside MooreToCore (`ImportVerilog` and
    `LLHDProcessInterpreter`), so safety claim is bounded to targeted semantics
    and parity checks above.

## 2026-02-28 (reset-polarity fault class in native mutation planner)

- realizations:
  - A reset polarity bug is a high-frequency, realistic control fault, but the
    native operator catalog only had generic `IF_COND_NEGATE`.
  - We need reset-focused site filtering (identifiers like `rst*` / `*reset*`)
    to avoid spending control mutations on unrelated conditions.
  - During parity sweeps, `xrun -R` with source files can silently ignore HDL
    inputs and reuse old snapshots; for reproducible mutant comparison we must
    compile-run without `-R`.

- changes made:
  - Added new native operator: `RESET_COND_NEGATE`.
  - Integrated operator into CIRCT native planner and mode catalogs:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `tools/circt-mut/circt-mut.cpp`
  - Added reset-aware condition scanning in planner:
    - detects reset-like identifiers in `if (...)` conditions, including escaped
      identifiers.
  - Added reset-aware mutation rewrite in native mutator template:
    - `utils/mutation_mcy/templates/native_create_mutated.py`
  - Updated Python native plan helper parity:
    - `utils/mutation_mcy/lib/native_mutation_plan.py`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-control-mode-reset-cond-negate-op.test`
    - `test/Tools/native-create-mutated-reset-cond-negate-site-index.test`

- validation:
  - TDD: new tests fail before implementation, pass after.
  - Rebuilt and tested:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
    - `llvm-lit -sv` slices for control-mode generation, native mutation plan,
      and native create-mutated tests (`124 passed` in broader touched slice).
  - Seeded campaign (`cov_intro_seeded.sv`, `count=24`, `seed=20260228`,
    mode=`balanced`) with per-mutant xrun/circt comparison:
    - semantic matches: all mutants after rerunning transient tool errors.
    - transient non-matches were infra only (`circt-sim` temporary
      `Permission denied` while binaries changed), not semantic divergence.

## 2026-02-28 (remove duplicated native planner implementation)

- realizations:
  - Native mutation planning logic existed in two places:
    C++ (`circt-mut` / `NativeMutationPlanner`) and Python
    (`utils/mutation_mcy/lib/native_mutation_plan.py`).
  - Keeping both planners in sync is brittle and caused duplicate maintenance
    work for each new mutation operator and site contract update.
  - The apply path remains Python (`native_create_mutated.py`), which is fine as
    long as planning has a single source of truth.

- changes made:
  - Added `circt-mut generate` option support for operator allowlisting:
    - `--native-op OP` (repeatable)
    - `--native-ops CSV`
  - Implemented allowlist filtering in CIRCT-only generation with deterministic
    ordering compatibility (applicable ops first, non-applicable requested ops
    preserved afterward).
  - Routed `--native-op(s)` runs to CIRCT-only planning even when
    `CIRCT_MUT_ALLOW_THIRD_PARTY=1`.
  - Wired allowlist into generation cache key material (`native_ops=`).
  - Removed Python planner implementation file:
    - deleted `utils/mutation_mcy/lib/native_mutation_plan.py`
  - Removed shell fallback planner duplication:
    - `utils/mutation_mcy/lib/native_mutation_plan.sh` now delegates directly to
      `circt-mut generate --native-ops ...` with
      `CIRCT_MUT_ALLOW_THIRD_PARTY=0`.
  - Updated MCY module docs:
    - `utils/mutation_mcy/README.md`
  - Updated planner regression tests to use `circt-mut generate` directly:
    - `test/Tools/native-mutation-plan-*.test`
  - Added invalid-allowlist regression:
    - `test/Tools/circt-mut-generate-circt-only-native-ops-invalid.test`
  - Updated native-backend runner stub test to accept `generate` in addition to
    `cover`:
    - `test/Tools/run-mutation-mcy-examples-native-mutation-plan-safe-ops-pass.test`

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - Regression slice:
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='native-mutation-plan-|circt-mut-generate-circt-only-native-ops-invalid|run-mutation-mcy-examples-native-mutation-plan-safe-ops-pass'` (`10 passed`)
  - Additional generate-option slice:
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='circt-mut-generate-circt-only-(native-ops-invalid|unsupported-options|modes-basic|mode-counts-basic|mode-weights-basic|profiles-basic)'` (`6 passed`)
  - Native MCY backend slice (fake `--circt-mut` harnesses):
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='run-mutation-mcy-examples-native-(mutation-ops|mutation-op-filter|mutation-plan-safe-ops|mutation-seed-order|noop-fallback|backend-no-yosys|real-wrapper)'` (`14 passed`)

## 2026-02-28 (parity campaign follow-up: BA->NBA root cause fixed)

- realizations:
  - Control-mode campaigns with seeded `cov_intro` surfaced deterministic
    xrun/circt mismatches for `NATIVE_BA_TO_NBA` sites.
  - The divergences were not RNG-related; they traced to lowering semantics.
  - A second mismatch class remained for `NATIVE_NEGEDGE_TO_POSEDGE`, but that
    class is race-prone (same-edge blocking drives), so cross-simulator output
    depends on scheduler ordering.

- changes made:
  - Added red parity reproducer test:
    - `test/Tools/circt-sim/mixed-ba-nba-shadow-global-visibility.sv`
  - Fixed mixed BA/NBA shadow rewrite bug in:
    - `lib/Dialect/Moore/Transforms/SimplifyProcedures.cpp`
    - Skip shadowing when a global has any nonblocking assignment users.
  - Added pass regression for mixed BA/NBA shadow behavior:
    - `test/Dialect/Moore/simplify-procedures.mlir` (`@MixedBANBA`).

- validation:
  - `utils/ninja-with-lock.sh -C build_test circt-opt circt-verilog`
  - `build_test/bin/llvm-lit -sv test/Dialect/Moore/simplify-procedures.mlir test/Tools/circt-sim/mixed-ba-nba-shadow-global-visibility.sv`
  - Reran all 80 mutants with updated circt and compared to stored xrun
    summaries:
    - `matches=78`, `fails=2`
    - both residual failures are duplicate `NATIVE_NEGEDGE_TO_POSEDGE@2`.

- next mutation-planner implication:
  - Treat edge-polarity swaps on synchronization waits as race-sensitive in
    parity mode (or classify as unstable) to avoid false bug attribution.

## 2026-02-28 (edge-polarity planning/apply alignment + race-aware filtering)

- realizations:
  - `POSEDGE_TO_NEGEDGE` / `NEGEDGE_TO_POSEDGE` previously matched raw keyword
    tokens, so planner could mutate procedural waits in `initial` blocks.
  - This produced parity noise from race-prone same-edge scheduling, not design
    logic defects.
  - Planner/apply drift was present: planner could classify edge sites as
    inapplicable while Python apply still rewrote a fallback site for unsuffixed
    ops.

- changes made:
  - Native planner edge-site collection is now context-aware:
    - only `always* @(...)` sensitivity edge keywords are eligible.
  - Added race-aware target-edge filter:
    - skip edge swaps that would move an `always*` sensitivity edge onto a
      non-`always` event-control edge of the same signal.
  - Synced Python mutation apply logic with planner:
    - edge-site selection uses the same `always*` sensitivity filter.
    - added same target-edge non-`always` conflict filter, so unsuffixed edge
      ops become no-op fallback when planner found no safe site.
  - Files:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `utils/mutation_mcy/templates/native_create_mutated.py`
  - Added TDD regressions:
    - `test/Tools/native-mutation-plan-edge-polarity-always-sensitivity-only.test`
    - `test/Tools/native-mutation-plan-edge-polarity-avoid-target-wait-race.test`
    - `test/Tools/native-create-mutated-negedge-to-posedge-skip-procedural-wait.test`

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - Focused lit slices (edge planner/apply + reset-edge + control-mode edge
    generation):
    - `12 passed`
  - Seeded parity reruns on `cov_intro_seeded`-style design (`count=40`,
    `seed=41`, mode=`control`):
    - before alignment: `matches=39`, `fails=1` (`NATIVE_POSEDGE_TO_NEGEDGE@1`)
    - after planner/apply alignment: `matches=40`, `fails=0`
  - Verified unsuffixed race-pruned edge mutants now produce explicit
    no-op fallback marker in mutated source:
    - `// native_mutation_noop_fallback NATIVE_POSEDGE_TO_NEGEDGE`

## 2026-02-28 (native apply migration: remove Python mutator dependency)

- realizations:
  - Native mutation planning was C++, but apply remained in
    `native_create_mutated.py`, creating a split implementation surface.
  - Keeping planner/apply in one codepath is higher-value than maintaining a
    Python template copy in MCY worker glue.

- changes made:
  - Added native C++ apply API in planner implementation:
    - `tools/circt-mut/NativeMutationPlanner.h`
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - new exported entry: `applyNativeMutationLabel(...)`
  - Added `circt-mut apply` subcommand (`-i/-o/-d`) in:
    - `tools/circt-mut/circt-mut.cpp`
  - Migrated native create-mutated lit tests from Python invocation to
    `circt-mut apply`:
    - all `test/Tools/native-create-mutated-*.test`
    - added `test/Tools/circt-mut-apply-basic.test`
  - Updated MCY worker native backend to use a generated shell wrapper that
    calls `circt-mut apply` (no Python template copy):
    - `utils/mutation_mcy/lib/worker.sh`
  - Removed Python mutator template:
    - deleted `utils/mutation_mcy/templates/native_create_mutated.py`
  - Updated runner/module docs:
    - `utils/mutation_mcy/README.md`
    - `utils/run_mutation_mcy_examples.sh`
  - Tightened native real harness args validation to report shell-quoting
    errors before smoke-mode gating:
    - `utils/run_mutation_mcy_examples.sh`

- validation:
  - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - `build_test/bin/llvm-lit -sv test/Tools/circt-mut-apply-basic.test`
  - `build_test/bin/llvm-lit -sv test/Tools --filter='native-create-mutated'`
    (`76 passed`)
  - `build_test/bin/llvm-lit -sv test/Tools --filter='(circt-mut-apply-basic|native-create-mutated|run-mutation-mcy-examples-native)'`
    (`111 passed`)

## 2026-02-28 (native xor compound-assign ops + parity bug root cause/fix)

- realizations:
  - Native control-mode compound assignment mutations were missing a realistic
    xor-assignment fault class (`^=` confusion with `|=` / `&=`).
  - Seeded xrun-vs-circt parity on xor-assign mutants exposed a CIRCT bug:
    some mutants diverged with `xrun` showing concrete values while CIRCT
    printed stale/incorrect values.
  - Root cause was not mutation planning; it was procedure simplification:
    `SimplifyProcedures` cached module globals into local shadows once per
    procedure entry, which is invalid across `wait_*` suspension points.

- changes made:
  - Added native mutation ops:
    - `BXOR_EQ_TO_BOR_EQ`
    - `BXOR_EQ_TO_BAND_EQ`
  - Planner/apply integration updates:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
      - op catalog, site collection, family classification, apply rewrite.
    - `tools/circt-mut/circt-mut.cpp`
      - mode operator sets include new xor compound-assign ops.
  - Added TDD lit tests for new ops:
    - `test/Tools/circt-mut-generate-circt-only-control-mode-compound-xor-assign-ops.test`
    - `test/Tools/native-create-mutated-bxor-eq-to-bor-eq-site-index.test`
    - `test/Tools/native-create-mutated-bxor-eq-to-band-eq-site-index.test`
  - Fixed CIRCT bug causing stale globals across waits:
    - `lib/Dialect/Moore/Transforms/SimplifyProcedures.cpp`
    - Skip global->local shadow rewriting for procedures containing suspension
      points (`wait_event`, `wait_condition`, `wait_delay`, `wait_fork`).
  - Added regression test for the stale-shadow bug:
    - `test/Tools/circt-sim/blocking-compound-assign-shadow-refresh.sv`

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
    - `utils/ninja-with-lock.sh -C build_test circt-verilog`
  - Red-green tests:
    - new xor-op lit tests failed before implementation and passed after.
    - stale-shadow regression failed before fix (`SUMMARY shadow=0`) and passed
      after fix (`SUMMARY shadow=7`).
  - Focused lit slice:
    - 12 targeted tests passed:
      - new xor-op generation/apply tests
      - prior compound assign tests
      - new circt-sim regression
      - `test/Dialect/Moore/simplify-procedures.mlir`
  - Seeded parity runs:
    - xor-assign campaign before fix: `matches=5`, `mismatches=7`
    - xor-assign campaign after fix: `matches=12`, `mismatches=0`
    - control-mode campaign on `cov_intro_seeded` class example: all compared
      mutants matched (`xrun_rc=0`, `circt_rc=0`, identical summary lines).

## 2026-02-28 (native bitwise compound assign to xor faults)

- realizations:
  - We had `^=`-to-(`|=`/`&=`) faults but not the reverse class.
  - Missing reverse direction leaves a realistic bug family uncovered:
    accidental xor update in masking/merge logic.

- changes made:
  - Added native mutation ops:
    - `BAND_EQ_TO_BXOR_EQ` (`&=` -> `^=`)
    - `BOR_EQ_TO_BXOR_EQ` (`|=` -> `^=`)
  - Planner/apply integration:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
      - op catalog, site collection, family mapping, apply rewrites.
    - `tools/circt-mut/circt-mut.cpp`
      - included new ops in CIRCT-only control/invert/connect/balanced/all mode
        allowlists.

- TDD coverage added:
  - `test/Tools/circt-mut-generate-circt-only-control-mode-compound-bitwise-xor-assign-ops.test`
  - `test/Tools/native-create-mutated-band-eq-to-bxor-eq-site-index.test`
  - `test/Tools/native-create-mutated-bor-eq-to-bxor-eq-site-index.test`

- validation:
  - red->green flow:
    - new tests failed pre-implementation (unsupported op + apply no-op fallback)
    - passed after implementation.
  - focused lit slice:
    - 9 bitwise/compound related tests passed.
  - deterministic parity campaign (xrun vs circt) on seeded benchmark using
    only new ops (`count=12`, `seed=20260228`):
    - `matches=12`, `mismatches=0`
    - artifacts: `/tmp/mut_parity_bitwise_to_xor_1772283986`.

## 2026-02-28 (ASSIGN_RHS_NEGATE + assignment family classification cleanup)

- realizations:
  - Assignment-RHS mutations were partially misclassified at family level:
    `ASSIGN_RHS_PLUS_ONE` / `ASSIGN_RHS_MINUS_ONE` were arithmetic but still
    grouped under `connect`.
  - This weakens weighted/all-mode allocation semantics and makes mutation
    reports less representative of actual fault classes.
  - During parity reruns, one full-batch failure was harness-induced (`xrun -R`
    with source files), not a simulator mismatch.

- changes made:
  - Added native operator:
    - `ASSIGN_RHS_NEGATE` (`rhs` -> `-(rhs)`).
  - Refactored assignment-RHS family classification:
    - const replacements (`TO_CONST0/1`) -> `connect`
    - invert replacement (`INVERT`) -> `logic`
    - arithmetic replacements (`PLUS_ONE`, `MINUS_ONE`, `NEGATE`) -> `arithmetic`
  - Integrated new op into CIRCT-only mode allowlists where arithmetic
    assignment mutations belong:
    - `arith`, `invert`, `balanced`, `all`.
  - Updated native-op token validation in:
    - `utils/run_mutation_mcy_examples.sh`.
  - Added TDD tests:
    - `test/Tools/native-create-mutated-assign-rhs-negate-site-index.test`
    - `test/Tools/native-mutation-plan-assign-rhs-negate-force.test`
  - Extended existing arith-mode generation coverage check:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-assign-rhs-plus-one-op.test`
      now also checks `NATIVE_ASSIGN_RHS_NEGATE`.

- validation:
  - Focused lit slice:
    - `build_test/bin/llvm-lit -sv test/Tools/native-create-mutated-assign-rhs-negate-site-index.test test/Tools/native-mutation-plan-assign-rhs-negate-force.test test/Tools/circt-mut-generate-circt-only-arith-mode-assign-rhs-plus-one-op.test test/Tools/circt-mut-generate-circt-only-weighted-fault-class-diversity.test test/Tools/circt-mut-generate-circt-only-weighted-context-priority.test`
    - result: `5 passed`.
  - Seeded parity campaigns (`xrun` vs `circt`) with negate-enabled binary:
    - `/tmp/cov_seeded_rhsconst_allmode_negate_1772295675`
      - `ok=80 mismatch=0 fail=0 negate_mutants=1`
    - `/tmp/cov_intro_seeded_allmode_negate_1772296033`
      - `ok=80 mismatch=0 fail=0 negate_mutants=1`
  - Harness correction note:
    - invalid `xrun -R` usage in one aborted rerun produced `xrun_fail` rows;
      rerun without `-R` restored valid parity measurements.

## 2026-02-28 (ASSIGN_RHS_SHL_ONE / ASSIGN_RHS_SHR_ONE)

- realizations:
  - Assignment-RHS arithmetic mutations still under-covered a common scaling bug
    class: incorrect single-bit shifts in RHS update expressions.
  - This class is realistic for datapath/control arithmetic and distinct from
    global operator swaps because it scopes to assignment RHS sites.

- changes made:
  - Added native assignment-RHS mutation operators:
    - `ASSIGN_RHS_SHL_ONE` (`rhs` -> `(rhs << 1'b1)`)
    - `ASSIGN_RHS_SHR_ONE` (`rhs` -> `(rhs >> 1'b1)`)
  - Wired through planner/apply and fault-family classification:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - included in assignment-RHS op detection and `arithmetic` family mapping.
  - Included both ops in CIRCT-only mode allowlists where arithmetic
    assignment RHS mutations are selected:
    - `tools/circt-mut/circt-mut.cpp` (`arith`, `invert`, `balanced`, `all`)
  - Updated native-op token validation:
    - `utils/run_mutation_mcy_examples.sh`
  - Added TDD tests:
    - `test/Tools/native-create-mutated-assign-rhs-shl-one-site-index.test`
    - `test/Tools/native-create-mutated-assign-rhs-shr-one-site-index.test`
    - `test/Tools/native-mutation-plan-assign-rhs-shl-one-force.test`
    - `test/Tools/native-mutation-plan-assign-rhs-shr-one-force.test`
  - Extended arith-mode generation coverage test:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-assign-rhs-plus-one-op.test`
      now checks both shift ops.

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - Focused lit slice:
    - 9 tests passed across assign-rhs negate/shift apply+plan and weighted mode
      coverage checks.
  - Seeded parity campaign (`xrun` vs `circt`) restricted to new ops:
    - `/tmp/cov_intro_seeded_assign_rhs_shift_parity_1772296371`
    - result: `ok=12 mismatch=0 fail=0` (`shl_mutants=6`, `shr_mutants=6`).
  - Additional seeded all-mode regression sweep after integrating both shift ops:
    - `/tmp/cov_intro_seeded_allmode_post_shift_1772296444`
    - initial result: `ok=87 mismatch=0 fail=3` (all 3 were transient
      `circt-sim: Permission denied` execution failures).
    - targeted recheck of failed IDs (`17/18/19`) with the same mutants:
      all rechecked `ok` with matching `COV` and `RESULT` vs xrun.

## 2026-02-28 (assignment-RHS expression-site coverage expansion)

- realizations:
  - Assignment-RHS operators were previously restricted to RHS values that are
    single identifiers, which misses realistic expression-level bugs
    (`a+b`, `c^d`, ternary and parenthesized forms).
  - Extending site eligibility to full RHS expressions increases mutation
    functional-space coverage without adding new operator tokens.

- changes made:
  - Relaxed assignment-RHS site matching to accept full RHS spans up to the
    statement semicolon (while retaining existing declarative/timing guards).
  - Updated assignment-RHS replacement construction to preserve precedence for
    binary arithmetic/shift mutations by parenthesizing non-identifier RHS
    expressions before applying `+1/-1/<<1/>>1`.
  - Files:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
  - Added regression:
    - `test/Tools/native-create-mutated-assign-rhs-plus-one-expression-site-index.test`
    - verifies `c ^ d` is mutated as `((c ^ d) + 1'b1)` (not precedence-broken).

- validation:
  - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - focused assign-RHS lit slice:
    - filter: `native-create-mutated-assign-rhs|native-mutation-plan-assign-rhs|circt-mut-generate-circt-only-arith-mode-assign-rhs-plus-one-op|circt-mut-generate-circt-only-connect-mode-assign-rhs-const-ops`
    - result: `19 passed`.
  - seeded parity campaign on deterministic `cov_intro_seeded` with updated
    expression-capable assignment-RHS matching:
    - `/tmp/cov_intro_seeded_allmode_exprrhs_1772296907`
    - result: `ok=100 mismatch=0 fail=0`, including assignment-RHS mutants now
      reaching deeper site indices (`@4`) with xrun/circt agreement.
  - seeded parity campaign on deterministic `cov_intro_seeded_rhsconst`:
    - `/tmp/cov_seeded_rhsconst_allmode_exprrhs_recheck_1772297263`
    - initial run: `ok=78 mismatch=0 fail=2` (both fails were transient
      `Permission denied` launching `build_test/bin/circt-verilog`).
    - targeted recheck of failed IDs (`28/29`) after relink window:
      both rechecked `ok` with matching `COV`/`SIG` vs xrun.
