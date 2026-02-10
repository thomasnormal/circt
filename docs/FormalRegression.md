# Formal Regression Harness

This document describes the formal regression harness used to run SVA BMC and
LEC suites and record baselines.

## Quick Start

Run all BMC/LEC suites with default paths:

```bash
utils/run_formal_all.sh
```

Write logs to a custom directory:

```bash
utils/run_formal_all.sh --out-dir /tmp/formal-results
```

Accept LEC mismatches that are diagnosed as `XPROP_ONLY`:

```bash
utils/run_formal_all.sh --lec-accept-xprop-only
```

Run strict OpenTitan LEC audit (non-optimistic X semantics):

```bash
utils/run_formal_all.sh --with-opentitan-lec-strict --opentitan ~/opentitan --include-lane-regex '^opentitan/LEC_STRICT$'
```

Enable strict-lane unknown-source dumps for X-prop triage:

```bash
utils/run_formal_all.sh --with-opentitan-lec-strict --opentitan ~/opentitan --opentitan-lec-strict-dump-unknown-sources --include-lane-regex '^opentitan/LEC_STRICT$'
```

Update baselines (updates `utils/formal-baselines.tsv` and the baseline table
in `PROJECT_PLAN.md`):

```bash
utils/run_formal_all.sh --update-baselines
```

Fail if the latest baseline differs:

```bash
utils/run_formal_all.sh --fail-on-diff
```

Enable strict baseline-aware quality gates:

```bash
utils/run_formal_all.sh --strict-gate
```

`--strict-gate` now enforces:
- fail/error non-regression
- xpass non-regression
- pass-rate non-regression
- fail-like case-ID non-regression (`failure_cases` baseline telemetry)
- OpenTitan `E2E_MODE_DIFF` `strict_only_fail` non-regression
- OpenTitan `E2E_MODE_DIFF` `status_diff` non-regression

Use a trailing baseline window for trend-aware strict gates:

```bash
utils/run_formal_all.sh --strict-gate --baseline-window 7
```

Restrict baseline history to a time window (in days from latest suite baseline):

```bash
utils/run_formal_all.sh --strict-gate --baseline-window 7 --baseline-window-days 30
```

Select OpenTitan E2E LEC mode explicitly (default remains x-optimistic):

```bash
utils/run_formal_all.sh --with-opentitan-e2e --opentitan ~/opentitan --opentitan-e2e-lec-strict-x --include-lane-regex '^opentitan/E2E$'
```

Run both OpenTitan E2E lanes side-by-side (default parity + strict audit):

```bash
utils/run_formal_all.sh --with-opentitan-e2e --with-opentitan-e2e-strict --opentitan ~/opentitan --include-lane-regex '^opentitan/(E2E|E2E_STRICT)$'
```

`opentitan/E2E_STRICT` emits case-level rows in
`<out-dir>/opentitan-e2e-strict-results.txt` for expected-failure case gates.
When both lanes run, formal-all also emits a normalized mode-diff artifact:
`<out-dir>/opentitan-e2e-mode-diff.tsv` and fail-like cases in
`<out-dir>/opentitan-e2e-mode-diff-results.txt` (`mode=E2E_MODE_DIFF`).
Classification counts are also exported to
`<out-dir>/opentitan-e2e-mode-diff-metrics.tsv` and embedded in the
`E2E_MODE_DIFF` summary string (`strict_only_fail`, `same_status`, etc.).
Use a targeted gate without enabling all strict-gate checks:

```bash
utils/run_formal_all.sh --with-opentitan-e2e --with-opentitan-e2e-strict --opentitan ~/opentitan --fail-on-new-e2e-mode-diff-strict-only-fail --baseline-file /tmp/formal-baselines.tsv
```

```bash
utils/run_formal_all.sh --with-opentitan-e2e --with-opentitan-e2e-strict --opentitan ~/opentitan --fail-on-new-e2e-mode-diff-status-diff --baseline-file /tmp/formal-baselines.tsv
```

## Mutation Coverage Harness (Certitude-Style Classification)

Use the mutation harness to classify injected faults into:
`not_activated`, `not_propagated`, `propagated_not_detected`, and `detected`.

Preferred frontend: `circt-mut` (`init`, `run`, `report`, `cover`, `matrix`,
`generate`).
Legacy `utils/run_mutation_*.sh` entrypoints remain supported.

Bootstrap a project template:

```bash
circt-mut init --project-dir /path/to/mut-campaign
```

This creates:
- `circt-mut.toml` with baseline cover/matrix settings
- `tests.tsv` template
- `lanes.tsv` template

Run from project config:

```bash
circt-mut run --project-dir /path/to/mut-campaign --mode all
```

`circt-mut run` reads `circt-mut.toml` and dispatches the native
preflight-backed `cover` and/or `matrix` flows (`--mode cover|matrix|all`).
For `[cover]`, config now supports either:
- `mutations_file = "..."`, or
- `generate_mutations = <N>` with generator controls
  (`mutations_modes`, `mutations_mode_counts`, `mutations_mode_weights`,
  `mutations_profiles`,
  `mutations_cfg`, `mutations_select`, `mutations_yosys`, `mutations_seed`,
  `mutations_top`).
Those modes are mutually exclusive and validated natively before script
dispatch.
`circt-mut run` now also accepts strict boolean config toggles
(`1|0|true|false|yes|no|on|off`) for:
- cover/matrix gate toggles:
  `resume`, `skip_baseline`, `fail_on_undetected`, `fail_on_errors`,
  `stop_on_fail`.
- cover formal toggles:
  `formal_global_propagate_assume_known_inputs`,
  `formal_global_propagate_accept_xprop_only`,
  `formal_global_propagate_bmc_run_smtlib`,
  `formal_global_propagate_bmc_assume_known_inputs`.

Aggregate campaign results:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --out /tmp/mutation-report.tsv
```

`circt-mut report` emits normalized key/value summary output for cover and
matrix artifacts, with optional file output via `--out`.
It now includes aggregated formal-global-filter telemetry useful for BMC/LEC
triage and performance tracking:
- global-filter timeout/unknown counters
- chained filter fallback/disagreement counters
- global-filter runtime/run counters
- BMC original-design cache counters/runtime

`circt-mut report` also supports baseline comparison:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --compare /path/to/baseline-report.tsv \
  --out /tmp/mutation-report-with-diff.tsv
```

Comparison emits per-metric numeric diffs for overlapping keys:
- `diff.<metric>.delta`
- `diff.<metric>.pct_change`
and summary counts (`diff.overlap_keys`, `diff.numeric_overlap_keys`,
`diff.exact_changed_keys`, `diff.added_keys`, `diff.missing_keys`).

It also supports history-based baseline selection and snapshot appends:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --compare-history-latest /tmp/mutation-history.tsv
```

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --append-history /tmp/mutation-history.tsv
```

History file schema:
- `run_id<TAB>timestamp_utc<TAB>key<TAB>value`

Trend summaries can be computed from history windows:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --trend-history /tmp/mutation-history.tsv \
  --trend-window 10
```

Comparison can also enforce numeric delta gates:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --compare-history-latest /tmp/mutation-history.tsv \
  --fail-if-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-delta-lt cover.detected_mutants=0 \
  --trend-history /tmp/mutation-history.tsv \
  --trend-window 10 \
  --fail-if-trend-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-trend-delta-lt cover.detected_mutants=0 \
  --append-history /tmp/mutation-history.tsv \
  --out /tmp/mutation-report-with-gates.tsv
```

Gate mode emits:
- `compare.gate_rules_total`
- `compare.gate_failure_count`
- `compare.gate_status` (`pass`/`fail`)
- `compare.gate_failure_<n>` rows for failing rules.
Gate rules require `--compare` or `--compare-history-latest` with numeric
baseline values for the gated keys.
Trend gate mode emits:
- `trend.gate_rules_total`
- `trend.gate_failure_count`
- `trend.gate_status` (`pass`/`fail`)
- `trend.gate_failure_<n>` rows for failing rules.
Trend gates require `--trend-history` with numeric history values for
the gated keys.
Built-in profiles can be used to apply standard gate sets:
- `--policy-profile formal-regression-basic`
- `--policy-profile formal-regression-trend`
These profiles pre-populate compare/trend gates for formal regression metrics:
`cover.detected_mutants`, `cover.global_filter_timeout_mutants`,
`cover.global_filter_lec_unknown_mutants`, and
`cover.global_filter_bmc_unknown_mutants`.
Gate failures return process exit code `2`.

Basic usage:

```bash
circt-mut cover \
  --design /path/to/design.il \
  --mutations-file /path/to/mutations.txt \
  --tests-manifest /path/to/tests.tsv \
  --formal-activate-cmd "your_activation_check_cmd" \
  --formal-propagate-cmd "your_propagation_check_cmd" \
  --coverage-threshold 90
```

Auto-generate mutations (instead of `--mutations-file`):

```bash
circt-mut cover \
  --design /path/to/design.il \
  --tests-manifest /path/to/tests.tsv \
  --generate-mutations 1000 \
  --mutations-modes arith,control \
  --mutations-yosys yosys \
  --mutations-seed 1
```

## Workflow Comparison: CIRCT vs MCY vs Certitude

The commands below provide practical mapping between this harness, MCY, and a
Certitude-style commercial flow.

0. Bootstrap campaign project

`circt-mut`:

```bash
circt-mut init --project-dir /path/to/mut-campaign
```

Equivalent MCY flow:

```bash
cd /path/to/mcy_project
mcy init
```

Equivalent Certitude-style flow (schematic):

```bash
certitude_init -out /path/to/mut-campaign
```

1. Single campaign

`circt-mut cover`:

```bash
circt-mut cover \
  --design /path/to/design.il \
  --mutations-file /path/to/mutations.txt \
  --tests-manifest /path/to/tests.tsv \
  --formal-global-propagate-circt-chain auto \
  --work-dir /tmp/mutation-cover
```

Equivalent MCY flow:

```bash
cd /path/to/mcy_project
mcy init
mcy run -j8
```

Equivalent Certitude-style flow (schematic; exact CLI depends on deployment):

```bash
certitude_run \
  -rtl /path/to/filelist.f \
  -tb /path/to/testlist.tcl \
  -fault_model rtl_mutation \
  -out /tmp/certitude-run
```

2. Increase mutation volume / refresh generated set

`circt-mut cover`:

```bash
circt-mut cover \
  --design /path/to/design.il \
  --tests-manifest /path/to/tests.tsv \
  --generate-mutations 1000 \
  --mutations-modes arith,control \
  --mutations-yosys yosys \
  --mutations-seed 1
```

Equivalent MCY flow (`size 1000` in `config.mcy`):

```bash
cd /path/to/mcy_project
mcy reset
mcy run -j8
```

Equivalent Certitude-style flow (schematic):

```bash
certitude_run \
  -rtl /path/to/filelist.f \
  -tb /path/to/testlist.tcl \
  -num_faults 1000 \
  -fault_scope arith,control \
  -out /tmp/certitude-run
```

3. Matrix/lane orchestration

`circt-mut matrix`:

```bash
circt-mut matrix \
  --lanes-tsv /path/to/lanes.tsv \
  --out-dir /tmp/mutation-matrix \
  --default-mutations-yosys yosys \
  --default-formal-global-propagate-circt-lec \
  --default-formal-global-propagate-circt-bmc \
  --default-formal-global-propagate-circt-chain auto
```

Tool discovery for built-in global filters:
- If circt-lec/circt-bmc path options are omitted or set to `auto`,
  `run_mutation_cover.sh` resolves tools from install-tree sibling `bin/`
  (for `<prefix>/share/circt/utils` script layouts), then `PATH`, then
  `<circt-root>/build/bin`.
- Explicit paths still take precedence.

Mutant materialization defaults to the in-repo
`utils/create_mutated_yosys.sh` helper. Use `--create-mutated-script` only
when overriding this behavior (for example to force an MCY-site wrapper).

Equivalent MCY flow (one MCY project per lane):

```bash
for lane_dir in /path/to/lanes/*; do
  (
    cd "$lane_dir"
    mcy init
    mcy run -j8
  )
done
```

Equivalent Certitude-style flow (schematic):

```bash
for lane in lane_svtests lane_verilator; do
  certitude_run -config "/path/to/${lane}.cfg" -out "/tmp/${lane}"
done
```

4. Aggregate/report campaign outcomes

`circt-mut report`:

```bash
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --out /tmp/mutation-report.tsv
```

Equivalent MCY flow (schematic):

```bash
cd /path/to/mcy_project
mcy status
# plus site-specific aggregation for multi-project/multi-lane campaigns.
```

Equivalent Certitude-style flow (schematic):

```bash
certitude_report -in /tmp/certitude-run -out /tmp/certitude-report
```

Execution controls:
- `--jobs <N>`: run per-mutant execution with up to `N` local workers.
- `--resume`: reuse existing per-mutant artifacts in `<work-dir>/mutations/*`
  and rebuild global reports without re-running completed mutants.
- `--reuse-pair-file <path>`: reuse prior `pair_qualification.tsv`
  activation/propagation rows so iterative testbench-improvement runs can skip
  repeated formal pre-qualification work.
- `--reuse-summary-file <path>`: reuse prior `summary.tsv` `detected_by_test`
  as per-mutant test-order hints to reduce detection reruns.
- `--reuse-compat-mode off|warn|strict`:
  - `off`: skip compatibility checks for reuse files
  - `warn` (default): verify sidecars when present, disable reuse on mismatch,
    and allow legacy reuse files without sidecars
  - `strict`: require valid compatible sidecars for reuse inputs
- `--reuse-manifest-file <path>`: write run compatibility manifest
  (default `<work-dir>/reuse_manifest.json`)
- `--reuse-cache-dir <path>`: content-addressed compatibility cache root for
  automatic reuse discovery/publish. For generated-mutation runs, this also
  enables cached mutation-list generation under
  `<reuse-cache-dir>/generated_mutations`.
- `--reuse-cache-mode off|read|read-write`: cache read/write policy
  (default `read-write` when cache dir is provided).
- `circt-mut generate` migration note:
  - core mutation generation now runs natively in `circt-mut` for standard
    generation options (mode/profile/mode-count/cfg/select/top-up).
  - generated-mutation caching (`--cache-dir`) is now supported in the native
    generate path with compatibility telemetry (`Mutation cache status`,
    `saved_runtime_ns`, lock wait/contended counters).
- `circt-mut cover` migration note:
  - built-in global-filter tool options (`--formal-global-propagate-circt-lec`
    / `--formal-global-propagate-circt-bmc`) are now pre-resolved natively,
    including bare `auto` forms, before script dispatch.
  - explicit built-in Z3 options are pre-resolved natively:
    `--formal-global-propagate-z3`, `--formal-global-propagate-bmc-z3`.
  - mutation-generation tool option `--mutations-yosys` is now pre-resolved
    natively for fast-fail diagnostics before dispatch.
  - cover mutation source consistency is now validated natively:
    - exactly one of `--mutations-file` or `--generate-mutations` is required.
    - conflicting or missing source configuration fails fast.
  - generated-mutation allocation controls are now validated natively for
    cover dispatch:
    - `--mutations-modes` entries must be known built-in mode/family names
      (`inv`, `const0`, `const1`, `cnot0`, `cnot1`,
      `arith`, `control`, `balanced`, `all`,
      `stuck`, `invert`, `connect`).
    - `--mutations-profiles` entries must be known built-in profile names
      (`arith-depth`, `control-depth`, `balanced-depth`, `fault-basic`,
      `fault-stuck`, `fault-connect`, `cover`, `none`).
    - `--generate-mutations` must be a positive integer.
    - `--mutations-seed` must be a non-negative integer.
    - `--mutations-mode-counts` / `--mutations-mode-weights` entries must be
      `NAME=VALUE` with positive integers.
    - `--mutations-mode-counts` / `--mutations-mode-weights` mode names must
      be known built-in mode/family names.
    - `--mutations-mode-counts` and `--mutations-mode-weights` are mutually
      exclusive.
    - when mode-counts are used, their total must match
      `--generate-mutations`.
  - native preflight now validates chain-mode values, injects missing built-in
    LEC/BMC tools for `--formal-global-propagate-circt-chain`, and rejects
    conflicting non-chain global-filter mode combinations before dispatch.
  - unresolved tool paths fail fast in `circt-mut` with direct diagnostics
    instead of deferred shell-script setup failures.
  - cover formal numeric/cache controls are now validated natively:
    `--formal-global-propagate-timeout-seconds`,
    `--formal-global-propagate-lec-timeout-seconds`,
    `--formal-global-propagate-bmc-timeout-seconds`,
    `--formal-global-propagate-bmc-bound`,
    `--formal-global-propagate-bmc-ignore-asserts-until`,
    `--bmc-orig-cache-max-entries`, `--bmc-orig-cache-max-bytes`,
    `--bmc-orig-cache-max-age-seconds`,
    `--bmc-orig-cache-eviction-policy`.
  - when any non-zero timeout is configured with an active cover global filter
    mode, `circt-mut` now also fail-fast validates `timeout` availability from
    the current `PATH`.
- `circt-mut matrix` migration note:
  - default built-in global-filter options are now pre-resolved/validated
    natively (`--default-formal-global-propagate-circt-lec`,
    `--default-formal-global-propagate-circt-bmc`,
    `--default-formal-global-propagate-circt-chain`) before script dispatch.
  - default generated-mutation Yosys option is now pre-resolved natively:
    `--default-mutations-yosys`.
  - default generated-mutation seed is now configurable/validated natively:
    `--default-mutations-seed`.
  - generated lanes in `--lanes-tsv` now get native preflight validation for
    lane `mutations_yosys` executables (with default/`yosys` fallback).
  - generated lanes now also get native allocation preflight for effective
    lane/default generation controls:
    - `generate_count` must be a positive integer.
    - `mutations_seed` must be a non-negative integer (defaults to `1` when
      unset).
    - effective `mutations_modes` entries must be known built-in mode/family
      names.
    - effective `mutations_mode_counts` / `mutations_mode_weights` entries
      must be `NAME=VALUE` with positive integers.
    - effective `mutations_mode_counts` / `mutations_mode_weights` mode names
      must be known built-in mode/family names.
    - effective count/weight controls are mutually exclusive.
    - effective `mutations_mode_counts` totals must match `generate_count`.
  - lane mutation source consistency is now validated natively:
    - static lanes require `mutations_file` path with unset `generate_count`.
    - generated lanes require `mutations_file=-` with positive
      `generate_count`.
    - conflicting or missing source combinations fail fast before dispatch.
  - lane-level formal tool fields in `--lanes-tsv` now get native preflight
    validation with effective default fallback:
    `global_propagate_circt_lec`, `global_propagate_circt_bmc`,
    `global_propagate_z3`, `global_propagate_bmc_z3`.
  - lane timeout/cache/gate override fields are now validated natively:
    `global_propagate_timeout_seconds`,
    `global_propagate_lec_timeout_seconds`,
    `global_propagate_bmc_timeout_seconds`, `global_propagate_bmc_bound`,
    `global_propagate_bmc_ignore_asserts_until`,
    `bmc_orig_cache_max_entries`, `bmc_orig_cache_max_bytes`,
    `bmc_orig_cache_max_age_seconds`, `bmc_orig_cache_eviction_policy`,
    `skip_baseline`, `fail_on_undetected`, `fail_on_errors`.
  - lane boolean formal fields now get native validation with
    `1|0|true|false|yes|no|-` forms:
    `global_propagate_assume_known_inputs`,
    `global_propagate_accept_xprop_only`,
    `global_propagate_bmc_run_smtlib`,
    `global_propagate_bmc_assume_known_inputs`.
  - matrix default numeric/cache controls are also validated natively:
    `--default-formal-global-propagate-timeout-seconds`,
    `--default-formal-global-propagate-lec-timeout-seconds`,
    `--default-formal-global-propagate-bmc-timeout-seconds`,
    `--default-formal-global-propagate-bmc-bound`,
    `--default-formal-global-propagate-bmc-ignore-asserts-until`,
    `--default-bmc-orig-cache-max-entries`,
    `--default-bmc-orig-cache-max-bytes`,
    `--default-bmc-orig-cache-max-age-seconds`,
    `--default-bmc-orig-cache-eviction-policy`.
  - matrix default mutation allocation options are now validated natively:
    `--default-mutations-modes`,
    `--default-mutations-mode-counts`,
    `--default-mutations-mode-weights` (syntax/value/mode-name checks plus
    mutual-exclusion conflict detection).
  - matrix default/lane generated-mutation profile names are now validated
    natively:
    `--default-mutations-profiles` and lane `mutations_profiles` must resolve
    to known built-in profile names.
  - when any effective matrix timeout is non-zero (default or lane override)
    with an active effective global filter mode, `circt-mut` now fail-fast
    validates `timeout` availability from the current `PATH`.
  - explicit default Z3 options are pre-resolved natively:
    `--default-formal-global-propagate-z3`,
    `--default-formal-global-propagate-bmc-z3`.
  - chain mode auto-injects default built-in LEC/BMC tools when omitted, and
    conflicting non-chain default global-filter modes are rejected early.
- `circt-mut generate` migration note:
  - native generate now fail-fast resolves `--yosys` executable before
    execution and cache-key generation.
  - native generate now also validates `--mode`, `--mode-count(s)`, and
    `--mode-weight(s)` names against built-in mode/family names.
  - `utils/generate_mutations_yosys.sh` now applies the same mode-name
    validation for `--mode` / `--modes`, `--mode-count(s)`, and
    `--mode-weight(s)`.
- `--bmc-orig-cache-max-entries <n>`: cap differential-BMC original-design
  cache entries (`0` disables limit, default `0`).
- `--bmc-orig-cache-max-bytes <n>`: cap differential-BMC original-design cache
  byte footprint (`0` disables limit, default `0`).
- `--bmc-orig-cache-max-age-seconds <n>`: cap differential-BMC
  original-design cache entry age (`0` disables limit, default `0`).
- `--bmc-orig-cache-eviction-policy lru|fifo|cost-lru`: eviction mode for
  count/byte bounded differential-BMC original cache (`lru` default).
  - `lru`: evict least-recently-used entries by access time.
  - `fifo`: evict oldest entries by creation/update order.
  - `cost-lru`: evict by highest `age/runtime` score, preserving expensive
    cached original-BMC results longer under pressure.
- `--mutations-modes <csv>`: pass-through mode mix for auto-generation
  (`generate_mutations_yosys.sh --modes`), useful for arithmetic/control
  operator-family mixes. Supports concrete Yosys modes
  (`inv,const0,const1,cnot0,cnot1`) and family aliases
  (`arith,control,balanced,all,stuck,invert,connect`), where family counts are
  deterministically split across their concrete modes. Built-in fault-model
  alias mapping:
  - `stuck` -> `const0,const1`
  - `invert` -> `inv`
  - `connect` -> `cnot0,cnot1`
  When total count is not divisible across top-level mode groups or within
  family-mode concrete expansion, remainder assignments are deterministic but
  seed-rotated by `--mutations-seed`.
- `--mutations-mode-counts <csv>`: explicit mode allocation for
  auto-generation (`generate_mutations_yosys.sh --mode-counts`), e.g.
  `arith=700,control=300` (sum must match `--generate-mutations`).
- `--mutations-mode-weights <csv>`: weighted mode allocation for
  auto-generation (`generate_mutations_yosys.sh --mode-weights`), e.g.
  `arith=3,control=1`; normalized to `--generate-mutations` with
  deterministic seed-rotated remainder assignment.
- `--mutations-profiles <csv>`: pass-through named profile presets for
  auto-generation (`generate_mutations_yosys.sh --profiles`), e.g.
  `arith-depth`, `control-depth`, `balanced-depth`, `fault-basic`,
  `fault-stuck`, `fault-connect`, `cover`.
- `--mutations-cfg <csv>`: pass-through mutate config entries for
  auto-generation (`generate_mutations_yosys.sh --cfgs`), e.g. mutation
  weight tuning such as `weight_pq_w=2,weight_cover=5`.
- `--mutations-select <csv>`: pass-through mutate select expressions for
  auto-generation (`generate_mutations_yosys.sh --selects`) to constrain
  mutation targets.
- `--formal-global-propagate-cmd <cmd>`: per-mutant formal propagation filter
  run once before per-test qualification/detection. Mutants proven
  `NOT_PROPAGATED` are classified as `not_propagated` without running tests.
- `--formal-global-propagate-timeout-seconds <n>`: wall-time cap (seconds) for
  per-mutant global formal filters (`0` disables). Timeout outcomes are
  classified conservatively as `propagated` (no pruning).
- `--formal-global-propagate-lec-timeout-seconds <n>`: wall-time override for
  built-in `circt-lec` global filtering. Defaults to
  `--formal-global-propagate-timeout-seconds`.
- `--formal-global-propagate-bmc-timeout-seconds <n>`: wall-time override for
  built-in differential `circt-bmc` global filtering. Defaults to
  `--formal-global-propagate-timeout-seconds`.
- Built-in circt-lec global filter (mutually exclusive with command mode):
  - `--formal-global-propagate-circt-lec [path]`
  - `--formal-global-propagate-circt-lec-args "<args>"`
  - `--formal-global-propagate-c1 <module>` / `--formal-global-propagate-c2 <module>`
  - `--formal-global-propagate-z3 <path>`
  - `--formal-global-propagate-assume-known-inputs`
  - `--formal-global-propagate-accept-xprop-only`
  This mode classifies by `circt-lec` output tokens:
  - `LEC_RESULT=EQ` => `not_propagated`
  - `LEC_RESULT=NEQ|UNKNOWN` => `propagated` (conservative fallback)
- Built-in differential circt-bmc global filter (mutually exclusive with other
  global filter modes):
  - `--formal-global-propagate-circt-bmc [path]`
  - `--formal-global-propagate-circt-bmc-args "<args>"`
  - `--formal-global-propagate-bmc-bound <n>`
  - `--formal-global-propagate-bmc-module <name>`
  - `--formal-global-propagate-bmc-run-smtlib`
  - `--formal-global-propagate-bmc-z3 <path>`
  - `--formal-global-propagate-bmc-assume-known-inputs`
  - `--formal-global-propagate-bmc-ignore-asserts-until <n>`
  This mode compares BMC outcomes between original and mutant:
  - same `BMC_RESULT` (`SAT`/`UNSAT`) => `not_propagated`
  - different `BMC_RESULT` or any `UNKNOWN` => `propagated`
  - per-run optimization: original-design BMC outcomes are cached in
    `<work_dir>/.global_bmc_orig_cache` keyed by resolved BMC command,
    original-design path, and original-design SHA-256, reducing repeated
    original-design solver invocations across mutants while preventing stale
    hits when design contents change at the same path.
  - cross-run optimization (when `--reuse-cache-dir` is enabled): original
    BMC cache entries are hydrated from and published to
    `<reuse-cache-dir>/global_bmc_orig_cache`.
  - cache bounds can be applied by entry count, byte footprint, and entry age.
- Built-in chained circt-lec/circt-bmc global filter:
  - `--formal-global-propagate-circt-chain lec-then-bmc|bmc-then-lec|consensus|auto`
  - if circt-lec/circt-bmc paths are omitted, both are auto-resolved.
  - `lec-then-bmc`: use LEC first and fall back to differential BMC when LEC
    returns `UNKNOWN` or an error.
  - `bmc-then-lec`: use differential BMC first and fall back to LEC when BMC
    returns `UNKNOWN` or an error.
  - `consensus`: run both LEC and differential BMC and classify
    `not_propagated` only when both agree on non-propagation. If one engine
    errors while the other reports non-propagation, classify conservatively as
    `propagated` (do not prune).
  - `auto`: run LEC and differential BMC in parallel with consensus-safe
    classification (same pruning semantics as `consensus` with lower wall-time).
    When either engine returns a decisive propagated result (`NEQ`, `UNKNOWN`,
    or differential mismatch), auto may short-circuit and cancel the peer
    engine.

`tests.tsv` format (tab-separated):

```text
test_id    run_cmd    result_file    kill_pattern    survive_pattern
```

Example row:

```text
smoke_1    bash /abs/path/run_test.sh smoke_1    result.txt    ^DETECTED$    ^SURVIVED$
```

Generated artifacts (default under `./mutation-cover-results`):
- `summary.tsv`: mutant-level final class and detection provenance
- `pair_qualification.tsv`: activation/propagation status per test-mutant pair
  (includes synthetic `test_id=-` rows for per-mutant global filter outcomes).
- `results.tsv`: detection outcomes for executed pairs
- `metrics.tsv`: gate-oriented aggregate metrics (coverage, bucket counts, errors)
- `summary.json`: machine-readable gate/coverage summary for CI trend ingestion
- `improvement.tsv`: actionable bucket-to-remediation mapping
- hint metrics:
  - `hinted_mutants`: mutants with a usable prior `detected_by_test` hint
  - `hint_hits`: hinted mutants detected by the hinted test
- reuse metrics:
  - `reused_pairs`: reused per-(test,mutant) activation/propagation pairs.
  - `reused_global_filters`: reused per-mutant global filter outcomes from
    prior `pair_qualification.tsv` rows (`test_id=-`).
- chain-filter telemetry metrics:
  - `global_filter_lec_unknown_mutants`: count of mutants where built-in
    global filtering observed LEC `UNKNOWN`.
  - `global_filter_bmc_unknown_mutants`: count of mutants where built-in
    global filtering observed BMC `UNKNOWN`.
  - `global_filter_timeout_mutants`: count of mutants where any built-in or
    command global filter timed out.
  - `global_filter_lec_timeout_mutants`: count of mutants where LEC global
    filtering timed out.
  - `global_filter_bmc_timeout_mutants`: count of mutants where differential
    BMC global filtering timed out.
  - `global_filter_lec_runtime_ns`: summed wall-clock runtime for built-in
    LEC global-filter invocations across mutants.
  - `global_filter_bmc_runtime_ns`: summed wall-clock runtime for built-in
    differential BMC global-filter invocations across mutants.
  - `global_filter_cmd_runtime_ns`: summed wall-clock runtime for
    command-mode global-filter invocations (`--formal-global-propagate-cmd`).
  - `global_filter_lec_runs`: number of built-in LEC global-filter
    invocations across mutants.
  - `global_filter_bmc_runs`: number of built-in differential BMC
    global-filter invocations across mutants.
  - `global_filter_cmd_runs`: number of command-mode global-filter
    invocations across mutants.
  - `generated_mutations_enabled`: whether this run used built-in mutation-list
    generation (`1`/`0`).
  - `generated_mutations_cache_hit`: whether generated-mutation cache lookup
    resolved from cache (`1`/`0`).
  - `generated_mutations_cache_miss`: whether generated-mutation cache was
    enabled but required fresh synthesis (`1`/`0`).
  - `generated_mutations_cache_status`: generated-mutation cache result
    (`hit`, `miss`, or `disabled`).
  - `generated_mutations_runtime_ns`: wall-clock runtime for built-in mutation
    generation in this run.
  - `generated_mutations_cache_saved_runtime_ns`: estimated runtime avoided by
    cache reuse (from cached generation metadata, `0` on miss/disabled).
  - `generated_mutations_cache_lock_wait_ns`: lock acquisition wait time for
    generated-mutation cache access.
  - `generated_mutations_cache_lock_contended`: whether generated-mutation
    cache locking observed contention (`1`/`0`).
  - `chain_lec_unknown_fallbacks`: count of mutants where chained mode fell
    back from LEC `UNKNOWN` to BMC.
  - `chain_bmc_resolved_not_propagated_mutants`: count of mutants classified as
    `not_propagated` by chained-mode BMC fallback.
  - `chain_bmc_resolved_propagated_mutants`: count of mutants classified as
    `propagated` by chained-mode BMC fallback.
  - `chain_bmc_unknown_fallbacks`: count of mutants where chained mode fell
    back from BMC `UNKNOWN` to LEC.
  - `chain_lec_resolved_not_propagated_mutants`: count of mutants classified as
    `not_propagated` by chained-mode LEC fallback.
  - `chain_lec_resolved_propagated_mutants`: count of mutants classified as
    `propagated` by chained-mode LEC fallback.
  - `chain_lec_error_fallbacks`: count of mutants where chained mode observed
    a LEC error and still produced conservative `propagated` classification.
  - `chain_bmc_error_fallbacks`: count of mutants where chained mode observed
    a BMC error and still produced conservative `propagated` classification.
  - `chain_consensus_not_propagated_mutants`: count of mutants classified
    `not_propagated` under consensus mode.
  - `chain_consensus_disagreement_mutants`: count of mutants where LEC/BMC
    disagree in consensus mode (`eq+different` or `neq+equal`).
  - `chain_consensus_error_mutants`: count of consensus-mode mutants that
    resolved to `error`.
  - `chain_auto_parallel_mutants`: count of mutants classified using
    auto-mode parallel LEC/BMC execution.
  - `chain_auto_short_circuit_mutants`: count of auto-mode mutants where
    execution short-circuited after one engine already proved propagation.
  - `bmc_orig_cache_hit_mutants`: count of mutants using cached original-design
    BMC results in built-in differential BMC mode.
  - `bmc_orig_cache_miss_mutants`: count of mutants that required a fresh
    original-design BMC run in built-in differential BMC mode.
  - `bmc_orig_cache_saved_runtime_ns`: estimated original-BMC runtime
    nanoseconds avoided via cache hits (sum of cached original runtimes reused).
  - `bmc_orig_cache_miss_runtime_ns`: original-BMC runtime nanoseconds spent on
    cache misses in this run.
  - `bmc_orig_cache_write_status`: write status for cross-run original-BMC
    cache publication (`disabled|read_only|written|write_error`).
  - `bmc_orig_cache_max_entries` / `bmc_orig_cache_max_bytes` /
    `bmc_orig_cache_max_age_seconds`: configured cache limits.
  - `bmc_orig_cache_eviction_policy`: configured count/byte cache eviction
    policy (`lru`, `fifo`, or `cost-lru`).
  - `bmc_orig_cache_entries` / `bmc_orig_cache_bytes`: post-run local cache
    footprint.
  - `bmc_orig_cache_pruned_entries` / `bmc_orig_cache_pruned_bytes`: local
    cache entries/bytes evicted to satisfy limits.
  - `bmc_orig_cache_pruned_age_entries` / `bmc_orig_cache_pruned_age_bytes`:
    local cache entries/bytes evicted specifically by max-age policy.
  - `bmc_orig_cache_persist_entries` / `bmc_orig_cache_persist_bytes`: post-run
    persisted cache footprint.
  - `bmc_orig_cache_persist_pruned_entries` /
    `bmc_orig_cache_persist_pruned_bytes`: persisted cache evictions under
    limits.
  - `bmc_orig_cache_persist_pruned_age_entries` /
    `bmc_orig_cache_persist_pruned_age_bytes`: persisted-cache evictions
    specifically by max-age policy.
- reuse compatibility sidecars:
  - `<summary.tsv>.manifest.json`
  - `<pair_qualification.tsv>.manifest.json`
  These capture the compatibility hash used for future guarded reuse.
- cache entry layout (when `--reuse-cache-dir` is enabled):
  - `<cache>/<compat_hash>/summary.tsv`
  - `<cache>/<compat_hash>/pair_qualification.tsv`
  - corresponding `*.manifest.json` sidecars

Gate behavior:
- `--coverage-threshold <pct>`: fails with exit code `2` when detected/relevant falls below threshold.
- `--fail-on-undetected`: fails with exit code `3` if any `propagated_not_detected` mutants remain.
- `--fail-on-errors`: fails with exit code `1` when formal/test infrastructure errors occur.

### Mutation Lane Matrix Runner

Run multiple mutation lanes from a single TSV:

```bash
utils/run_mutation_matrix.sh --lanes-tsv /path/to/lanes.tsv --out-dir /tmp/mutation-matrix
```

Matrix `results.tsv` columns:

```text
lane_id    status    exit_code    coverage_percent    gate_status    lane_dir    metrics_file    summary_json    generated_mutations_cache_status    generated_mutations_cache_hit    generated_mutations_cache_miss    generated_mutations_cache_saved_runtime_ns    generated_mutations_cache_lock_wait_ns    generated_mutations_cache_lock_contended
```

Matrix gate summary artifact:

```text
<out-dir>/gate_summary.tsv
gate_status    count
```

Execution controls:
- `--lane-jobs <N>`: run up to `N` lanes concurrently.
- `--lane-schedule-policy fifo|cache-aware`: lane dispatch strategy.
  `cache-aware` schedules one lane per generated-cache key before same-key
  followers to reduce lock contention when `--reuse-cache-dir` is enabled.
- `--jobs-per-lane <N>`: per-lane mutant worker count passed through to
  `run_mutation_cover.sh`.
- `--gate-summary-file <path>`: write matrix gate-status counts
  (default `<out-dir>/gate_summary.tsv`).
- `--skip-baseline`: pass-through `run_mutation_cover.sh --skip-baseline`
  for all lanes.
- `--fail-on-undetected`: pass-through
  `run_mutation_cover.sh --fail-on-undetected` for all lanes.
- `--fail-on-errors`: pass-through
  `run_mutation_cover.sh --fail-on-errors` for all lanes.
- `--default-reuse-pair-file <path>`: default
  `run_mutation_cover.sh --reuse-pair-file` for lanes that do not set a
  lane-specific reuse file.
- `--default-reuse-summary-file <path>`: default
  `run_mutation_cover.sh --reuse-summary-file` for lanes that do not set a
  lane-specific summary hint file.
- `--default-mutations-modes <csv>`: default
  `run_mutation_cover.sh --mutations-modes` for generated-mutation lanes.
- `--default-mutations-mode-counts <csv>`: default
  `run_mutation_cover.sh --mutations-mode-counts` for generated-mutation lanes.
- `--default-mutations-mode-weights <csv>`: default
  `run_mutation_cover.sh --mutations-mode-weights` for generated-mutation lanes.
- `--default-mutations-profiles <csv>`: default
  `run_mutation_cover.sh --mutations-profiles` for generated-mutation lanes.
- `--default-mutations-cfg <csv>`: default
  `run_mutation_cover.sh --mutations-cfg` for generated-mutation lanes.
- `--default-mutations-select <csv>`: default
  `run_mutation_cover.sh --mutations-select` for generated-mutation lanes.
- `--default-mutations-seed <n>`: default
  `run_mutation_cover.sh --mutations-seed` for generated-mutation lanes.
- `--default-mutations-yosys <path>`: default
  `run_mutation_cover.sh --mutations-yosys` for generated-mutation lanes.
- `--default-formal-global-propagate-cmd <cmd>`: default
  `run_mutation_cover.sh --formal-global-propagate-cmd` for lanes without a
  lane-specific global filter command.
- `--default-formal-global-propagate-timeout-seconds <n>`: default
  `run_mutation_cover.sh --formal-global-propagate-timeout-seconds` for lanes
  without a lane-specific global filter timeout.
- `--default-formal-global-propagate-lec-timeout-seconds <n>`: default
  `run_mutation_cover.sh --formal-global-propagate-lec-timeout-seconds` for
  lanes without a lane-specific LEC timeout override.
- `--default-formal-global-propagate-bmc-timeout-seconds <n>`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-timeout-seconds` for
  lanes without a lane-specific BMC timeout override.
- `--default-formal-global-propagate-circt-lec [path]`: default
  `run_mutation_cover.sh --formal-global-propagate-circt-lec` for lanes
  without a lane-specific circt-lec global filter path.
- `--default-formal-global-propagate-circt-lec-args <args>`: default
  `run_mutation_cover.sh --formal-global-propagate-circt-lec-args` for lanes
  without lane-specific circt-lec args.
- `--default-formal-global-propagate-c1 <name>`: default
  `run_mutation_cover.sh --formal-global-propagate-c1`.
- `--default-formal-global-propagate-c2 <name>`: default
  `run_mutation_cover.sh --formal-global-propagate-c2`.
- `--default-formal-global-propagate-z3 <path>`: default
  `run_mutation_cover.sh --formal-global-propagate-z3`.
- `--default-formal-global-propagate-assume-known-inputs`: default
  `run_mutation_cover.sh --formal-global-propagate-assume-known-inputs`.
- `--default-formal-global-propagate-accept-xprop-only`: default
  `run_mutation_cover.sh --formal-global-propagate-accept-xprop-only`.
- `--default-formal-global-propagate-circt-bmc [path]`: default
  `run_mutation_cover.sh --formal-global-propagate-circt-bmc` for lanes
  without a lane-specific circt-bmc global filter path.
- `--default-formal-global-propagate-circt-chain <mode>`: default
  `run_mutation_cover.sh --formal-global-propagate-circt-chain` for lanes
  without a lane-specific chain mode (`lec-then-bmc|bmc-then-lec|consensus|auto`).
- `--default-formal-global-propagate-circt-bmc-args <args>`: default
  `run_mutation_cover.sh --formal-global-propagate-circt-bmc-args` for lanes
  without lane-specific bmc args.
- `--default-formal-global-propagate-bmc-bound <n>`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-bound`.
- `--default-formal-global-propagate-bmc-module <name>`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-module`.
- `--default-formal-global-propagate-bmc-run-smtlib`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-run-smtlib`.
- `--default-formal-global-propagate-bmc-z3 <path>`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-z3`.
- `--default-formal-global-propagate-bmc-assume-known-inputs`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-assume-known-inputs`.
- `--default-formal-global-propagate-bmc-ignore-asserts-until <n>`: default
  `run_mutation_cover.sh --formal-global-propagate-bmc-ignore-asserts-until`.
- `--default-bmc-orig-cache-max-entries <n>`: default
  `run_mutation_cover.sh --bmc-orig-cache-max-entries`.
- `--default-bmc-orig-cache-max-bytes <n>`: default
  `run_mutation_cover.sh --bmc-orig-cache-max-bytes`.
- `--default-bmc-orig-cache-max-age-seconds <n>`: default
  `run_mutation_cover.sh --bmc-orig-cache-max-age-seconds`.
- `--default-bmc-orig-cache-eviction-policy lru|fifo|cost-lru`: default
  `run_mutation_cover.sh --bmc-orig-cache-eviction-policy`.
- `--reuse-cache-dir <path>`: pass-through
  `run_mutation_cover.sh --reuse-cache-dir` for matrix lanes; this also enables
  shared generated-mutation cache reuse across lanes.
  Generated-mutation cache writes are coordinated by per-key locking in
  `generate_mutations_yosys.sh`, preventing duplicate generation work when
  multiple lanes request the same cache key concurrently.
- `--reuse-compat-mode off|warn|strict`: pass-through reuse compatibility
  policy for each lane's `run_mutation_cover.sh` invocation.
- `--include-lane-regex <regex>`: run only lane IDs matching any provided
  ERE selector (repeatable).
- `--exclude-lane-regex <regex>`: skip lane IDs matching any provided ERE
  selector (repeatable, applied after include selectors).
- `--stop-on-fail` is supported with `--lane-jobs=1` (deterministic fail-fast).

Lane TSV schema (tab-separated):

```text
lane_id    design    mutations_file    tests_manifest    activate_cmd    propagate_cmd    coverage_threshold    [generate_count]    [mutations_top]    [mutations_seed]    [mutations_yosys]    [reuse_pair_file]    [reuse_summary_file]    [mutations_modes]    [global_propagate_cmd]    [global_propagate_circt_lec]    [global_propagate_circt_bmc]    [global_propagate_bmc_args]    [global_propagate_bmc_bound]    [global_propagate_bmc_module]    [global_propagate_bmc_run_smtlib]    [global_propagate_bmc_z3]    [global_propagate_bmc_assume_known_inputs]    [global_propagate_bmc_ignore_asserts_until]    [global_propagate_circt_lec_args]    [global_propagate_c1]    [global_propagate_c2]    [global_propagate_z3]    [global_propagate_assume_known_inputs]    [global_propagate_accept_xprop_only]    [mutations_cfg]    [mutations_select]    [mutations_profiles]    [mutations_mode_counts]    [global_propagate_circt_chain]    [bmc_orig_cache_max_entries]    [bmc_orig_cache_max_bytes]    [bmc_orig_cache_max_age_seconds]    [bmc_orig_cache_eviction_policy]    [skip_baseline]    [fail_on_undetected]    [fail_on_errors]    [global_propagate_timeout_seconds]    [global_propagate_lec_timeout_seconds]    [global_propagate_bmc_timeout_seconds]    [mutations_mode_weights]
```

Notes:
- Use `-` for `activate_cmd` / `propagate_cmd` to disable that stage.
- Use `-` for `coverage_threshold` to skip threshold gating per lane.
- For auto-generation lanes, set `mutations_file` to `-` and provide
  `generate_count` (plus optional top/seed/yosys columns).
- `reuse_pair_file` (optional) overrides `--default-reuse-pair-file` for a
  specific lane.
- `reuse_summary_file` (optional) overrides
  `--default-reuse-summary-file` for a specific lane.
- `mutations_modes` (optional) overrides `--default-mutations-modes` for a
  generated-mutation lane (same concrete/family mode semantics as
  `--mutations-modes`).
- `mutations_cfg` (optional) overrides `--default-mutations-cfg` for a
  generated-mutation lane.
- `mutations_select` (optional) overrides `--default-mutations-select` for a
  generated-mutation lane.
- `mutations_profiles` (optional) overrides `--default-mutations-profiles` for
  a generated-mutation lane.
- `mutations_mode_counts` (optional) overrides
  `--default-mutations-mode-counts` for a generated-mutation lane.
- `mutations_mode_weights` (optional) overrides
  `--default-mutations-mode-weights` for a generated-mutation lane.
- `mutations_yosys` (optional) overrides `--default-mutations-yosys` for a
  generated-mutation lane.
- `global_propagate_cmd` (optional) overrides
  `--default-formal-global-propagate-cmd` for a specific lane.
- `global_propagate_timeout_seconds` (optional) overrides
  `--default-formal-global-propagate-timeout-seconds` for a specific lane.
- `global_propagate_lec_timeout_seconds` (optional) overrides
  `--default-formal-global-propagate-lec-timeout-seconds` for a specific lane.
- `global_propagate_bmc_timeout_seconds` (optional) overrides
  `--default-formal-global-propagate-bmc-timeout-seconds` for a specific lane.
- `global_propagate_circt_lec` (optional) overrides
  `--default-formal-global-propagate-circt-lec` for a specific lane.
- `global_propagate_circt_lec_args` (optional) overrides
  `--default-formal-global-propagate-circt-lec-args` for a specific lane.
- `global_propagate_c1` (optional) overrides
  `--default-formal-global-propagate-c1` for a specific lane.
- `global_propagate_c2` (optional) overrides
  `--default-formal-global-propagate-c2` for a specific lane.
- `global_propagate_z3` (optional) overrides
  `--default-formal-global-propagate-z3` for a specific lane.
- `global_propagate_assume_known_inputs` (optional truthy flag:
  `1|true|yes`) enables
  `--formal-global-propagate-assume-known-inputs`.
- `global_propagate_accept_xprop_only` (optional truthy flag:
  `1|true|yes`) enables
  `--formal-global-propagate-accept-xprop-only`.
- `global_propagate_circt_bmc` (optional) overrides
  `--default-formal-global-propagate-circt-bmc` for a specific lane.
- `global_propagate_circt_chain` (optional) overrides
  `--default-formal-global-propagate-circt-chain` for a specific lane.
- `global_propagate_bmc_args` (optional) overrides
  `--default-formal-global-propagate-circt-bmc-args` for a specific lane.
- `skip_baseline` (optional) overrides matrix `--skip-baseline` for a lane.
- `fail_on_undetected` (optional) overrides matrix `--fail-on-undetected` for
  a lane.
- `fail_on_errors` (optional) overrides matrix `--fail-on-errors` for a lane.
  For these three booleans, accepted values are `1|0|true|false|yes|no|-`.
- `global_propagate_bmc_bound` (optional) overrides
  `--default-formal-global-propagate-bmc-bound` for a specific lane.
- `global_propagate_bmc_module` (optional) overrides
  `--default-formal-global-propagate-bmc-module` for a specific lane.
- `global_propagate_bmc_run_smtlib` (optional truthy flag:
  `1|true|yes`) enables `--formal-global-propagate-bmc-run-smtlib`.
- `global_propagate_bmc_z3` (optional) overrides
  `--default-formal-global-propagate-bmc-z3` for a specific lane.
- `global_propagate_bmc_assume_known_inputs` (optional truthy flag:
  `1|true|yes`) enables
  `--formal-global-propagate-bmc-assume-known-inputs`.
- `global_propagate_bmc_ignore_asserts_until` (optional) overrides
  `--default-formal-global-propagate-bmc-ignore-asserts-until` for a specific
  lane.
- `bmc_orig_cache_max_entries` (optional) overrides
  `--default-bmc-orig-cache-max-entries` for a specific lane.
- `bmc_orig_cache_max_bytes` (optional) overrides
  `--default-bmc-orig-cache-max-bytes` for a specific lane.
- `bmc_orig_cache_max_age_seconds` (optional) overrides
  `--default-bmc-orig-cache-max-age-seconds` for a specific lane.
- `bmc_orig_cache_eviction_policy` (optional) overrides
  `--default-bmc-orig-cache-eviction-policy` for a specific lane.
- `global_propagate_cmd` is mutually exclusive with built-in global filter
  modes.
- `global_propagate_circt_chain` requires both
  `global_propagate_circt_lec` and `global_propagate_circt_bmc` (lane value
  or inherited defaults).

Shard/route a run to selected lane IDs:

```bash
utils/run_formal_all.sh --include-lane-regex '^sv-tests/(BMC|LEC)$'
utils/run_formal_all.sh --include-lane-regex '^avip/.*/compile$' --exclude-lane-regex '^avip/spi_avip/compile$'
```

Enable persistent lane checkpoints and resume interrupted matrix runs:

```bash
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --resume-from-lane-state
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --reset-lane-state
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --reset-lane-state --merge-lane-state-tsv /tmp/formal-lanes-worker-a.tsv --merge-lane-state-tsv /tmp/formal-lanes-worker-b.tsv
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-hmac-key-file /secrets/lane-state.key --lane-state-hmac-key-id ci-lane-key-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-hmac-keyring-tsv /secrets/lane-state-keyring.tsv --lane-state-hmac-keyring-sha256 <sha256> --lane-state-hmac-key-id ci-lane-key-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-public-key-file /secrets/lane-state-ed25519-public.pem --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-cert-subject-regex 'CN=lane-state-ed25519' --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-crl-refresh-cmd '/usr/local/bin/refresh-lane-crl.sh' --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-crl-refresh-cmd '/usr/local/bin/refresh-lane-crl.sh' --lane-state-manifest-ed25519-crl-refresh-retries 3 --lane-state-manifest-ed25519-crl-refresh-delay-secs 10 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-crl-refresh-cmd '/usr/local/bin/refresh-lane-crl.sh' --lane-state-manifest-ed25519-crl-refresh-retries 3 --lane-state-manifest-ed25519-crl-refresh-delay-secs 10 --lane-state-manifest-ed25519-crl-refresh-timeout-secs 30 --lane-state-manifest-ed25519-crl-refresh-jitter-secs 5 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-refresh-cmd '/usr/local/bin/refresh-lane-ocsp.sh' --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-refresh-cmd '/usr/local/bin/refresh-lane-ocsp.sh' --lane-state-manifest-ed25519-ocsp-refresh-retries 3 --lane-state-manifest-ed25519-ocsp-refresh-delay-secs 10 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-refresh-cmd '/usr/local/bin/refresh-lane-ocsp.sh' --lane-state-manifest-ed25519-ocsp-refresh-retries 3 --lane-state-manifest-ed25519-ocsp-refresh-delay-secs 10 --lane-state-manifest-ed25519-ocsp-refresh-timeout-secs 30 --lane-state-manifest-ed25519-ocsp-refresh-jitter-secs 5 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-response-sha256 <sha256> --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-max-age-secs 3600 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-require-next-update --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-responder-id-regex 'lane-state-ed25519' --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-responder-cert-file /secrets/lane-state-ed25519-ocsp-responder.pem --lane-state-manifest-ed25519-ocsp-responder-cert-sha256 <sha256> --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-issuer-cert-file /secrets/lane-state-ed25519-ocsp-issuer.pem --lane-state-manifest-ed25519-ocsp-responder-cert-file /secrets/lane-state-ed25519-ocsp-responder.pem --lane-state-manifest-ed25519-ocsp-responder-cert-sha256 <sha256> --lane-state-manifest-ed25519-ocsp-require-responder-ocsp-signing --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-keyring-sha256 <sha256> --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-ocsp-responder-cert-file /secrets/lane-state-ed25519-ocsp-responder.pem --lane-state-manifest-ed25519-ocsp-responder-cert-sha256 <sha256> --lane-state-manifest-ed25519-ocsp-require-responder-aki-match-ca-ski --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia --lane-state-manifest-ed25519-refresh-policy-profiles-json /secrets/lane-state-refresh-policies.json --lane-state-manifest-ed25519-refresh-policy-profile prod_strict --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia --lane-state-manifest-ed25519-refresh-policy-profiles-json /secrets/lane-state-refresh-policies.json --lane-state-manifest-ed25519-refresh-policy-profile prod_strict --lane-state-manifest-ed25519-refresh-policy-profiles-sha256 <sha256> --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia --lane-state-manifest-ed25519-refresh-policy-profiles-json /secrets/lane-state-refresh-policies.json --lane-state-manifest-ed25519-refresh-policy-profile prod_strict --lane-state-manifest-ed25519-refresh-policy-profiles-sha256 <sha256> --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json /secrets/lane-state-refresh-policies.manifest.json --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file /secrets/lane-state-refresh-policy-manifest-ed25519-public.pem --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id profile-key-1 --lane-state-manifest-ed25519-key-id ci-ed25519-1
utils/run_formal_all.sh --lane-state-tsv /tmp/formal-lanes.tsv --lane-state-manifest-ed25519-private-key-file /secrets/lane-state-ed25519-private.pem --lane-state-manifest-ed25519-keyring-tsv /secrets/lane-state-ed25519-keyring.tsv --lane-state-manifest-ed25519-ca-file /secrets/lane-state-ed25519-ca.pem --lane-state-manifest-ed25519-crl-file /secrets/lane-state-ed25519.crl.pem --lane-state-manifest-ed25519-ocsp-response-file /secrets/lane-state-ed25519.ocsp.der --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia --lane-state-manifest-ed25519-refresh-policy-profiles-json /secrets/lane-state-refresh-policies.json --lane-state-manifest-ed25519-refresh-policy-profile prod_strict --lane-state-manifest-ed25519-refresh-policy-profiles-sha256 <sha256> --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json /secrets/lane-state-refresh-policies.manifest.json --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv /secrets/lane-state-refresh-policy-signers.tsv --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256 <sha256> --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file /secrets/lane-state-refresh-policy-signers-ca.pem --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id profile-key-1 --lane-state-manifest-ed25519-key-id ci-ed25519-1
```

Lane-state semantics:
- `--lane-state-tsv` persists per-lane counters (`total/pass/fail/xfail/xpass/error/skip`).
- `--resume-from-lane-state` reuses matching lane rows and skips re-running those lanes.
- `--reset-lane-state` truncates the lane-state file before the run.
- `--merge-lane-state-tsv` merges rows from additional lane-state files
  (repeatable). Merge conflicts on lane-id + incompatible `config_hash` fail
  fast with a diagnostic.
- Resume enforces a lane-state configuration fingerprint; mismatched options or
  tool context fail fast with a diagnostic and require `--reset-lane-state`.
- Resume also enforces lane-state compatibility policy versioning
  (`compat_policy_version`) to prevent unsafe replay across format/semantic
  policy shifts.
- `--lane-state-hmac-key-file` signs lane-state files via
  `<lane-state>.manifest.json` and verifies signatures during resume/merge.
- When `--lane-state-hmac-key-id` is set, manifest key-id must match on
  resume/merge (prevents accidental key drift).
- `--lane-state-hmac-keyring-tsv` resolves keys by `hmac_key_id` from a keyring
  row format:
  - `<hmac_key_id>\t<key_file_path>\t[not_before]\t[not_after]\t[status]\t[key_sha256]`
- Keyring `not_before` / `not_after` fields (when set) are enforced against
  lane-state manifest `generated_at_utc`.
- `--lane-state-hmac-keyring-sha256` can pin exact keyring content hash.
- `--lane-state-hmac-key-file` and `--lane-state-hmac-keyring-tsv` are mutually
  exclusive.
- `--lane-state-manifest-ed25519-private-key-file` +
  `--lane-state-manifest-ed25519-public-key-file` enable asymmetric lane-state
  manifest signing/verification.
- `--lane-state-manifest-ed25519-keyring-tsv` supports key-id based public-key
  trust-store resolution (`key_id`, `public_key_file_path`, optional
  `not_before`, `not_after`, `status`, `key_sha256`, `cert_file_path`,
  `cert_sha256`).
- `--lane-state-manifest-ed25519-keyring-sha256` can pin exact Ed25519 keyring
  content hash.
- `--lane-state-manifest-ed25519-ca-file` enables certificate-chain validation
  for keyring entries that provide `cert_file_path`.
- `--lane-state-manifest-ed25519-crl-file` enables certificate revocation checks
  (`openssl verify -crl_check`) and requires
  `--lane-state-manifest-ed25519-ca-file`.
- `--lane-state-manifest-ed25519-crl-refresh-cmd` runs a command before keyring
  verification to refresh the CRL artifact path.
- `--lane-state-manifest-ed25519-crl-refresh-uri` enables built-in CRL fetch
  (`file://`, `http://`, `https://`) without wrapper scripts.
  - URI mode requires `--lane-state-manifest-ed25519-crl-refresh-metadata-file`.
  - URI mode is mutually exclusive with
    `--lane-state-manifest-ed25519-crl-refresh-cmd`.
  - For `https://` URIs, metadata now attempts to capture the full observed peer
    certificate chain as `cert_chain_sha256` (leaf-first when available), while
    preserving `tls_peer_sha256` compatibility.
- `--lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp` resolves
  the CRL refresh URI from the selected key certificate
  `CRL Distribution Points` extension (keyring mode).
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-json` +
    `--lane-state-manifest-ed25519-refresh-policy-profile` load shared CRL/OCSP
    auto-URI defaults from a JSON profile registry.
    - schema: `{"schema_version":1,"profiles":{...}}`
    - profile keys:
      - shared:
        `auto_uri_policy`, `auto_uri_allowed_schemes`,
        `refresh_retries`, `refresh_delay_secs`,
        `refresh_timeout_secs`, `refresh_jitter_secs`,
        `refresh_metadata_require_transport`,
        `refresh_metadata_require_status`,
        `refresh_metadata_require_uri_regex`,
        `refresh_metadata_require_tls_peer_sha256`,
        `refresh_metadata_require_cert_chain_sha256`,
        `refresh_metadata_require_artifact_sha256`,
        `refresh_metadata_require_cert_chain_length_min`,
        `refresh_metadata_require_ca_cert_in_cert_chain`,
        `refresh_metadata_require_tls_peer_in_cert_chain`,
        `refresh_metadata_max_age_secs`,
        `refresh_metadata_max_future_skew_secs`
      - per-artifact (`crl.{...}`, `ocsp.{...}`) may override:
        `auto_uri_policy`, `auto_uri_allowed_schemes`,
        `refresh_retries`, `refresh_delay_secs`,
        `refresh_timeout_secs`, `refresh_jitter_secs`,
        `refresh_metadata_require_transport`,
        `refresh_metadata_require_status`,
        `refresh_metadata_require_uri_regex`,
        `refresh_metadata_require_tls_peer_sha256`,
        `refresh_metadata_require_cert_chain_sha256`,
        `refresh_metadata_require_artifact_sha256`,
        `refresh_metadata_require_cert_chain_length_min`,
        `refresh_metadata_require_ca_cert_in_cert_chain`,
        `refresh_metadata_require_tls_peer_in_cert_chain`,
        `refresh_metadata_max_age_secs`,
        `refresh_metadata_max_future_skew_secs`
  - Effective precedence for policy/scheme defaults:
    - per-artifact CLI override
    - shared CLI override
    - per-artifact profile value
    - shared profile value
    - built-in default
  - Effective precedence for profile-driven refresh execution controls
    (`...refresh_retries`, `...refresh_delay_secs`,
    `...refresh_timeout_secs`, `...refresh_jitter_secs`):
    - explicit per-artifact CLI flag
    - per-artifact profile value
    - shared profile value
    - built-in default
  - Effective precedence for profile-driven refresh metadata chain-membership
    defaults (`...require_ca_cert_in_cert_chain`,
    `...require_tls_peer_in_cert_chain`):
    - explicit per-artifact CLI flag
    - per-artifact profile value
    - shared profile value
    - built-in default
  - Effective precedence for profile-driven refresh metadata match defaults
    (`...require_transport`, `...require_status`,
    `...require_uri_regex`, `...require_tls_peer_sha256`,
    `...require_cert_chain_sha256`, `...require_artifact_sha256`,
    `...require_cert_chain_length_min`):
    - explicit per-artifact CLI flag
    - per-artifact profile value
    - shared profile value
    - built-in default
  - Effective precedence for profile-driven refresh metadata freshness defaults
    (`...max_age_secs`, `...max_future_skew_secs`):
    - explicit per-artifact CLI flag
    - per-artifact profile value
    - shared profile value
    - built-in default
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-sha256` optionally
    pins exact profile-registry content hash for integrity checks.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json`
    enables signed profile-registry manifest verification.
    - Requires
      `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file`
      or
      `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv`.
    - Public-key-file and keyring modes are mutually exclusive.
    - Manifest schema is strict with `schema_version: 1`, `profiles_sha256`,
      `signature_mode: "ed25519"`, optional `generated_at_utc`, optional
      `key_id`, and `signature_ed25519_base64`.
    - Signed payload is canonical JSON of manifest fields excluding
      `signature_ed25519_base64` (sorted keys, compact separators).
    - Manifest `profiles_sha256` must match the effective profile-registry
      content hash.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv`
    resolves signer key material from a keyring TSV by manifest `key_id`.
    - row format:
      `<key_id>\t<public_key_file_path>\t[not_before]\t[not_after]\t[status]\t[key_sha256]\t[cert_file_path]\t[cert_sha256]`
    - optional status values: `active` or `revoked`
    - optional validity window fields use `YYYY-MM-DD` and are checked against
      manifest `generated_at_utc` when present.
    - optional cert fields support signer cert anchoring and cert SHA pinning.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256`
    optionally pins exact signer-keyring content hash.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file`
    optionally enforces signer certificate verification for keyring rows with
    signer certs:
    - requires keyring mode
    - requires selected signer key row to provide `cert_file_path`
    - validates signer cert chain via `openssl verify -CAfile`.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file`
    optionally enables signer-cert CRL revocation + freshness checks in profile
    manifest keyring mode:
    - requires keyring mode
    - requires `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file`
    - requires selected signer key row to provide `cert_file_path`.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file`
    optionally enables signer-cert OCSP status checks in profile manifest
    keyring mode:
    - requires keyring mode
    - requires `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file`
    - requires selected signer key row to provide `cert_file_path`.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256`
    optionally pins exact signer-cert OCSP response content hash.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs`
    rejects OCSP responses whose `thisUpdate` age exceeds N seconds.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-next-update`
    enforces presence of OCSP `nextUpdate`.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file`
    optionally pins signer OCSP responder identity with explicit responder cert:
    - requires signer keyring OCSP mode
    - responder cert is CA-verified
    - when configured, OCSP verification uses `-verify_other` + responder cert.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-issuer-cert-file`
    optionally sets explicit issuer cert for signer OCSP `-issuer` selection
    (defaults to manifest keyring CA cert).
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256`
    optionally pins responder cert content hash.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-ocsp-signing`
    requires responder cert EKU to include `OCSP Signing`.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-aki-match-ca-ski`
    requires responder cert AKI keyid to match manifest keyring CA cert SKI.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex`
    optionally constrains OCSP `Responder Id` with a regex.
  - `--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id`
    optionally pins the expected manifest signer `key_id`.
  - `--lane-state-manifest-ed25519-refresh-auto-uri-policy` sets a shared
    default policy for both CRL and OCSP auto-URI modes.
    - Specific flags
      (`--lane-state-manifest-ed25519-crl-refresh-auto-uri-policy`,
      `--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy`)
      override the shared default.
  - `--lane-state-manifest-ed25519-refresh-auto-uri-allowed-schemes` sets a
    shared default allowed-scheme set (comma-separated subset of
    `file,http,https`) for both CRL and OCSP auto-URI modes.
    - Specific flags
      (`--lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes`,
      `--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes`)
      override the shared default.
  - `--lane-state-manifest-ed25519-crl-refresh-auto-uri-policy` controls URI
    selection when multiple usable CDP URIs exist:
    - `first` (default)
    - `last`
    - `require_single` (fail when more than one usable URI is present)
  - `--lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes`
    restricts cert-discovered CRL URI schemes to a comma-separated subset of
    `file,http,https`.
  - Auto-URI mode requires
    `--lane-state-manifest-ed25519-crl-refresh-metadata-file`.
  - Auto-URI mode is mutually exclusive with
    `--lane-state-manifest-ed25519-crl-refresh-cmd` and
    `--lane-state-manifest-ed25519-crl-refresh-uri`.
- `--lane-state-manifest-ed25519-crl-refresh-retries` and
  `--lane-state-manifest-ed25519-crl-refresh-delay-secs` control retry/backoff
  for CRL refresh execution (command, URI, or auto-URI mode).
- `--lane-state-manifest-ed25519-crl-refresh-timeout-secs` bounds each CRL
  refresh attempt wall-clock time.
- `--lane-state-manifest-ed25519-crl-refresh-jitter-secs` adds randomized
  retry delay (`0..N` seconds) for CRL refresh retries.
- `--lane-state-manifest-ed25519-crl-refresh-metadata-file` accepts a JSON
  sidecar path produced by refresh automation and embeds it in signed CRL
  refresh provenance.
  - Metadata schema is strict (`schema_version: 1`) with required fields:
    `source`, `transport`, `uri`, `fetched_at_utc`, `status`.
  - Optional fields: `http_status`, `tls_peer_sha256`, `cert_chain_sha256`,
    `error`.
    - `http_status` is required when `transport` is `http` or `https`.
    - `tls_peer_sha256` and non-empty `cert_chain_sha256` are required when
      `transport` is `https`.
  - Optional policy gates:
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-transport`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-status`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-uri-regex`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-sha256`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-sha256`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-ca-cert-in-cert-chain`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-in-cert-chain`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-length-min`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-max-age-secs`
    - `--lane-state-manifest-ed25519-crl-refresh-metadata-max-future-skew-secs`
- CRL mode enforces freshness from CRL `nextUpdate`; stale CRLs are rejected
  before certificate verification.
- `--lane-state-manifest-ed25519-ocsp-response-file` enables OCSP revocation
  checks against a pinned DER response (`openssl ocsp -respin`) and requires
  `--lane-state-manifest-ed25519-ca-file`.
- `--lane-state-manifest-ed25519-ocsp-refresh-cmd` runs a command before
  keyring verification to refresh the OCSP response artifact path.
- `--lane-state-manifest-ed25519-ocsp-refresh-uri` enables built-in OCSP
  response fetch (`file://`, `http://`, `https://`) without wrapper scripts.
  - URI mode requires `--lane-state-manifest-ed25519-ocsp-refresh-metadata-file`.
  - URI mode is mutually exclusive with
    `--lane-state-manifest-ed25519-ocsp-refresh-cmd`.
  - For `https://` URIs, metadata now attempts to capture the full observed peer
    certificate chain as `cert_chain_sha256` (leaf-first when available), while
    preserving `tls_peer_sha256` compatibility.
- `--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia` resolves
  the OCSP refresh URI from the selected key certificate
  `Authority Information Access` extension (keyring mode).
  - `--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy` controls URI
    selection when multiple usable AIA OCSP URIs exist:
    - `first` (default)
    - `last`
    - `require_single` (fail when more than one usable URI is present)
  - `--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes`
    restricts cert-discovered OCSP URI schemes to a comma-separated subset of
    `file,http,https`.
  - Auto-URI mode requires
    `--lane-state-manifest-ed25519-ocsp-refresh-metadata-file`.
  - Auto-URI mode is mutually exclusive with
    `--lane-state-manifest-ed25519-ocsp-refresh-cmd` and
    `--lane-state-manifest-ed25519-ocsp-refresh-uri`.
- `--lane-state-manifest-ed25519-ocsp-refresh-retries` and
  `--lane-state-manifest-ed25519-ocsp-refresh-delay-secs` control retry/backoff
  for OCSP refresh execution (command, URI, or auto-URI mode).
- `--lane-state-manifest-ed25519-ocsp-refresh-timeout-secs` bounds each OCSP
  refresh attempt wall-clock time.
- `--lane-state-manifest-ed25519-ocsp-refresh-jitter-secs` adds randomized
  retry delay (`0..N` seconds) for OCSP refresh retries.
- `--lane-state-manifest-ed25519-ocsp-refresh-metadata-file` accepts a JSON
  sidecar path produced by refresh automation and embeds it in signed OCSP
  refresh provenance.
- OCSP has matching policy gates:
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-transport`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-status`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-uri-regex`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-sha256`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-sha256`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-ca-cert-in-cert-chain`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-in-cert-chain`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-length-min`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-max-age-secs`
  - `--lane-state-manifest-ed25519-ocsp-refresh-metadata-max-future-skew-secs`
  - `...require-ca-cert-in-cert-chain` requires configured CA-cert digest
    membership in `cert_chain_sha256`.
  - `...require-tls-peer-in-cert-chain` is an opt-in strict linkage gate that
    requires `https` transport and requires `tls_peer_sha256` membership in
    `cert_chain_sha256`.
- Refresh hooks receive
  `LANE_STATE_MANIFEST_ED25519_REFRESH_METADATA_FILE=<configured-sidecar-path>`
  so wrappers can write metadata deterministically before returning.
- When Ed25519 manifest signing is enabled, CRL/OCSP refresh hooks now emit
  signed refresh provenance into `<lane-state>.manifest.json` (attempt timeline,
  timeout/failure markers, and observed artifact SHA256 per attempt).
- `--lane-state-manifest-ed25519-ocsp-response-sha256` optionally pins exact
  OCSP response content by SHA256.
- `--lane-state-manifest-ed25519-ocsp-max-age-secs` bounds OCSP response age
  from `This Update` (default when OCSP mode is enabled: 604800 seconds).
- `--lane-state-manifest-ed25519-ocsp-require-next-update` requires OCSP
  responses to carry a `Next Update` field.
- `--lane-state-manifest-ed25519-ocsp-responder-id-regex` enforces responder
  identity policy by matching OCSP `Responder Id`.
- `--lane-state-manifest-ed25519-ocsp-responder-cert-file` optionally pins OCSP
  signer verification to a specific responder certificate.
- `--lane-state-manifest-ed25519-ocsp-issuer-cert-file` optionally sets the
  certificate passed to OCSP `--issuer` (defaults to
  `--lane-state-manifest-ed25519-ca-file`).
- `--lane-state-manifest-ed25519-ocsp-responder-cert-sha256` optionally pins
  exact responder certificate bytes by SHA256.
- `--lane-state-manifest-ed25519-ocsp-require-responder-ocsp-signing` requires
  the pinned responder cert to carry `Extended Key Usage: OCSP Signing`.
- `--lane-state-manifest-ed25519-ocsp-require-responder-aki-match-ca-ski`
  requires responder cert `Authority Key Identifier` keyid to match CA cert
  `Subject Key Identifier`.
- `--lane-state-manifest-ed25519-cert-subject-regex` enforces certificate
  identity constraints for selected keyring entries.
- `--lane-state-manifest-ed25519-key-id` pins manifest `ed25519_key_id`.
- Ed25519 mode is mutually exclusive with lane-state HMAC signing modes.

Inspect and validate lane-state artifacts (single or federated files):

```bash
python3 utils/inspect_formal_lane_state.py /tmp/formal-lanes.tsv --print-lanes
python3 utils/inspect_formal_lane_state.py /tmp/formal-lanes-worker-a.tsv /tmp/formal-lanes-worker-b.tsv --require-config-hash --require-compat-policy-version --expect-compat-policy-version 1 --require-single-config-hash --require-lane sv-tests/BMC --json-out /tmp/formal-lanes-summary.json
```

Inspector semantics:
- Validates lane-state row shape and field types using the same schema rules as
  `run_formal_all.sh`.
- Merges duplicate lane rows with the same precedence policy used by the
  harness (`config_hash` compatibility + `updated_at_utc` tie-break behavior).
- Supports CI gating for required lane coverage and hash policy:
  - `--require-compat-policy-version`
  - `--expect-compat-policy-version <v>`
  - `--require-config-hash`
  - `--require-single-config-hash`
  - `--require-lane <lane_id>` (repeatable)

Lane-id format:
- `sv-tests/BMC`
- `sv-tests/LEC`
- `verilator-verification/BMC`
- `verilator-verification/LEC`
- `yosys/tests/sva/BMC`
- `yosys/tests/sva/LEC`
- `opentitan/LEC`
- `opentitan/LEC_STRICT`
- `avip/<name>/compile`

Fail only on specific gate classes:

```bash
utils/run_formal_all.sh --fail-on-new-xpass
utils/run_formal_all.sh --fail-on-passrate-regression
utils/run_formal_all.sh --fail-on-new-failure-cases
utils/run_formal_all.sh --fail-on-new-e2e-mode-diff-strict-only-fail
utils/run_formal_all.sh --fail-on-new-e2e-mode-diff-status-diff
```

Gate on known expected-failure budgets per suite/mode:

```bash
utils/run_formal_all.sh --expected-failures-file utils/formal-expected-failures.tsv --fail-on-unexpected-failures
```

Refresh expected-failure budgets from the current run:

```bash
utils/run_formal_all.sh --refresh-expected-failures-file utils/formal-expected-failures.tsv
```

Refresh expected-failure budgets for a scoped subset:

```bash
utils/run_formal_all.sh --refresh-expected-failures-file utils/formal-expected-failures.tsv --refresh-expected-failures-include-suite-regex '^sv-tests$' --refresh-expected-failures-include-mode-regex '^LEC$'
```

Prune stale expected-failure budget rows:

```bash
utils/run_formal_all.sh --expected-failures-file utils/formal-expected-failures.tsv --prune-expected-failures-file utils/formal-expected-failures.tsv --prune-expected-failures-drop-unused
```

Preview refresh/prune changes without rewriting expectation files:

```bash
utils/run_formal_all.sh --expected-failures-file utils/formal-expected-failures.tsv --prune-expected-failures-file utils/formal-expected-failures.tsv --expectations-dry-run
```

Emit machine-readable dry-run operation summaries:

```bash
utils/run_formal_all.sh --refresh-expected-failures-file utils/formal-expected-failures.tsv --expectations-dry-run --expectations-dry-run-report-jsonl /tmp/formal-dryrun.jsonl
```

Gate on per-test expected failure cases with expiry policy:

```bash
utils/run_formal_all.sh --expected-failure-cases-file utils/formal-expected-failure-cases.tsv --fail-on-unexpected-failure-cases --fail-on-expired-expected-failure-cases
```

Refresh expected failure cases from current observed fail-like rows:

```bash
utils/run_formal_all.sh --refresh-expected-failure-cases-file utils/formal-expected-failure-cases.tsv --refresh-expected-failure-cases-default-expires-on 2099-12-31
```

Refresh expected failure cases while collapsing per-status rows to `status=ANY`:

```bash
utils/run_formal_all.sh --refresh-expected-failure-cases-file utils/formal-expected-failure-cases.tsv --refresh-expected-failure-cases-collapse-status-any
```

Refresh expected failure cases with scoped filters:

```bash
utils/run_formal_all.sh --refresh-expected-failure-cases-file utils/formal-expected-failure-cases.tsv --refresh-expected-failure-cases-include-suite-regex '^sv-tests$' --refresh-expected-failure-cases-include-mode-regex '^BMC$' --refresh-expected-failure-cases-include-status-regex '^XFAIL$' --refresh-expected-failure-cases-include-id-regex '^__aggregate__$'
```

Prune stale expected-case rows in-place:

```bash
utils/run_formal_all.sh --expected-failure-cases-file utils/formal-expected-failure-cases.tsv --prune-expected-failure-cases-file utils/formal-expected-failure-cases.tsv --prune-expected-failure-cases-drop-unmatched --prune-expected-failure-cases-drop-expired
```

Run formal suites on a fixed cadence (6-hour interval example):

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 -- --with-opentitan --opentitan ~/opentitan --with-avip --avip-glob "~/mbit/*avip*"
```

The cadence runner fails fast on the first failing iteration and keeps
per-iteration artifacts under `formal-cadence-results-YYYYMMDD/`.

Limit retained run directories to cap disk usage:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --retain-runs 12 -- --with-opentitan --opentitan ~/opentitan
```

Prune stale run directories by age:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --retain-hours 72 -- --with-opentitan --opentitan ~/opentitan
```

Invoke an executable hook when a cadence iteration fails:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --on-fail-hook ./notify_failure.sh -- --with-opentitan --opentitan ~/opentitan
```

Fail hook arguments:
`<iteration> <exit_code> <run_dir> <out_root> <cadence_log> <cadence_state>`.

Send native email notifications on failure:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --on-fail-email formal-alerts@example.com --sendmail-path /usr/sbin/sendmail --email-subject-prefix "[formal-cadence]" -- --with-opentitan --opentitan ~/opentitan
```

Post failure events directly to an HTTP webhook:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --on-fail-webhook https://hooks.example/formal -- --with-opentitan --opentitan ~/opentitan
```

Use a JSON file for per-endpoint webhook policy overrides:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --on-fail-webhook-config ./webhooks.json -- --with-opentitan --opentitan ~/opentitan
```

Webhook options:

- `--on-fail-webhook <url>` can be specified multiple times.
- `--on-fail-webhook-config <file>` loads webhook endpoints with per-endpoint
  retry/backoff/timeout overrides.
- `--webhook-fanout-mode <sequential|parallel>` controls endpoint dispatch.
- `--webhook-max-parallel <n>` bounds concurrent webhook posts in parallel mode.
- `--webhook-retries <n>` controls retries per endpoint.
- `--webhook-backoff-mode <fixed|exponential>` controls retry timing policy.
- `--webhook-backoff-secs <n>` controls retry backoff delay.
- `--webhook-backoff-max-secs <n>` caps retry backoff delay.
- `--webhook-jitter-secs <n>` adds random jitter to retry delay.
- `--webhook-timeout-secs <n>` controls per-request timeout.

Email options:

- `--on-fail-email <addr>` can be specified multiple times.
- `--sendmail-path <path>` selects the sendmail binary/command.
- `--email-subject-prefix <text>` controls email subject prefix.

## Inputs and Suites

The harness wraps existing suite runners:

- `utils/run_sv_tests_circt_bmc.sh`
- `utils/run_sv_tests_circt_lec.sh`
- `utils/run_verilator_verification_circt_bmc.sh`
- `utils/run_verilator_verification_circt_lec.sh`
- `utils/run_yosys_sva_circt_bmc.sh`
- `utils/run_yosys_sva_circt_lec.sh`

Default paths (override with flags):

- `~/sv-tests`
- `~/verilator-verification`
- `~/yosys/tests/sva`

## Optional Runs

OpenTitan and AVIP runs are optional:

```bash
utils/run_formal_all.sh --with-opentitan --opentitan ~/opentitan
utils/run_formal_all.sh --with-avip --avip-glob "~/mbit/*avip*"
utils/run_formal_all.sh --with-avip --avip-glob "~/mbit/*avip*" --circt-verilog /abs/path/to/circt-verilog
utils/run_formal_all.sh --with-opentitan --opentitan ~/opentitan --circt-verilog-opentitan /abs/path/to/circt-verilog --with-avip --avip-glob "~/mbit/*avip*" --circt-verilog-avip /abs/path/to/circt-verilog
```

These are not formal checks but are tracked here to keep external testing
cadence consistent with the project plan.

OpenTitan E2E LEC-mode controls:

```bash
utils/run_opentitan_formal_e2e.sh --skip-sim --skip-verilog --lec-x-optimistic
utils/run_opentitan_formal_e2e.sh --skip-sim --skip-verilog --lec-strict-x
utils/run_opentitan_formal_e2e.sh --skip-sim --skip-verilog --lec-assume-known-inputs
```

## Outputs

Each run writes:

- `<out-dir>/*.log` per suite
- `<out-dir>/summary.tsv` machine-readable summary
- `<out-dir>/summary.txt` human-readable summary
- `<out-dir>/summary.json` machine-readable JSON summary (override path via
  `--json-summary`)
- `<out-dir>/expected-failures-summary.tsv` expected-failure budget comparison
  (when `--expected-failures-file` is used)
- `<out-dir>/expected-failure-cases-summary.tsv` expected per-test failure case
  matching summary (when `--expected-failure-cases-file` is used)
- `<out-dir>/unexpected-failure-cases.tsv` observed fail-like cases not covered
  by expected failure cases (when `--expected-failure-cases-file` is used)
- `<out-dir>/opentitan-lec-results.txt` per-implementation OpenTitan LEC case
  rows (when `--with-opentitan` is used)
- `<out-dir>/opentitan-lec-strict-results.txt` per-implementation strict
  OpenTitan LEC case rows (when `--with-opentitan-lec-strict` is used)
- `<out-dir>/avip-results.txt` per-AVIP compile case rows (when `--with-avip`
  is used)
- Harnesses treat `BMC_RESULT=SAT|UNSAT|UNKNOWN` and
  `LEC_RESULT=EQ|NEQ|UNKNOWN` tokens as the source of truth for pass/fail
  classification when not in smoke mode.
  - OpenTitan LEC case artifacts may include a diagnostic tag suffix
    (for example `#XPROP_ONLY`) to preserve mismatch class in case-level
    gating artifacts.
  - OpenTitan LEC artifact paths are deterministic in formal-all runs:
    - default lane: `<out-dir>/opentitan-lec-work/<impl>`
    - strict lane: `<out-dir>/opentitan-lec-strict-work/<impl>`
  - OpenTitan E2E LEC artifacts are deterministic under:
    `<out-dir>/opentitan-formal-e2e/lec-workdir/<impl>`.

JSON summary schema:

- `utils/formal-summary-schema.json`

Baselines:

- `utils/formal-baselines.tsv` (latest baselines per suite/mode)
  - columns: `date suite mode total pass fail xfail xpass error skip pass_rate result failure_cases`
  - legacy baseline files without `failure_cases` remain accepted; case-ID
    strict-gate checks are skipped for those legacy rows.
- `PROJECT_PLAN.md` baseline table (updated when `--update-baselines` is used)

Expected-failure budget file:

- `--expected-failures-file` expects TSV with header:
  - `suite	mode	expected_fail	expected_error	notes`
- `--expectations-dry-run` previews expectation refresh/prune updates without
  rewriting expectation files.
- `--expectations-dry-run-report-jsonl <file>` appends JSON Lines dry-run
  operation summaries (`operation`, `target_file`, row-count metadata).
  Use `--expectations-dry-run-report-max-sample-rows <N>` to control embedded
  row samples (`output_rows_sample`, `dropped_rows_sample`) per operation.
  Use `--expectations-dry-run-report-hmac-key-file <file>` to emit
  `payload_hmac_sha256` in `run_end` for authenticated digest verification.
  Use `--expectations-dry-run-report-hmac-key-id <id>` to emit an explicit
  key identifier (`hmac_key_id`) in `run_meta`/`run_end`.
  The first row per run is `operation=run_meta` with `schema_version=1` and
  `run_id`, operation rows carry the same `run_id`, and the final row is
  `operation=run_end` with `exit_code`, `row_count`, and `payload_sha256`
  (plus optional `payload_hmac_sha256`).
- `python3 utils/verify_formal_dryrun_report.py <file>` verifies JSONL run
  envelopes and checks `run_end.row_count`/`run_end.payload_sha256`.
  Use `--allow-legacy-prefix` when the file contains older pre-enveloped rows
  before the first `run_meta`.
  Use `--hmac-key-file <file>` to verify `payload_hmac_sha256`.
  Use `--hmac-keyring-tsv <file>` to resolve HMAC keys by `hmac_key_id`
  (`<hmac_key_id>\t<key_file_path>\t[not_before]\t[not_after]\t[status]\t[key_sha256]`
  per row; optional date columns use `YYYY-MM-DD`; `status` is
  `active|revoked`; `key_sha256` pins key-file content; cannot be combined with
  `--hmac-key-file`).
  Use `--hmac-keyring-sha256 <hex>` to pin the exact keyring content hash.
  Use `--hmac-keyring-manifest-json <file>` to validate signed keyring
  metadata (`keyring_sha256`, `signer_id`, optional `expires_on`).
  Manifest signature modes:
  - HMAC mode:
    - `--hmac-keyring-manifest-hmac-key-file <file>`
    - manifest field: `signature_hmac_sha256`
  - Ed25519 mode:
    - `--hmac-keyring-manifest-ed25519-public-key-file <file>`
    - manifest field: `signature_ed25519_base64`
  - Ed25519 signer-keyring mode:
    - `--hmac-keyring-manifest-ed25519-keyring-tsv <file>`
    - optional `--hmac-keyring-manifest-ed25519-keyring-sha256 <hex>`
    - keyring rows:
      `<signer_id>\t<public_key_file_path>\t[not_before]\t[not_after]\t[status]\t[key_sha256]`
    - manifest `signer_id` must resolve to an active signer keyring row
  Manifest signer-source flags are mutually exclusive.
  Use `--expected-keyring-signer-id <id>` to pin manifest signer identity.
  Use `--expected-hmac-key-id <id>` to enforce `hmac_key_id` match.
- Missing suite/mode rows default to `expected_fail=0 expected_error=0`.
- `--fail-on-unused-expected-failures` fails when expected-failures rows do not
  match any suite/mode in current run results.
- `--prune-expected-failures-file <file>` rewrites expected-failures rows after
  summary matching.
- `--prune-expected-failures-drop-unused` drops rows not present in current run
  summary. If prune is enabled and no explicit prune policy is set, this policy
  is enabled by default.
- `--refresh-expected-failures-file <file>` rewrites expected-failures rows from
  current summary fail/error counts (preserves existing notes by suite/mode).
- `--refresh-expected-failures-include-suite-regex <regex>` keeps only matching
  suite rows during budget refresh.
- `--refresh-expected-failures-include-mode-regex <regex>` keeps only matching
  mode rows during budget refresh.

Expected-failure cases file:

- `--expected-failure-cases-file` expects TSV with required columns:
  - `suite	mode	id`
- Optional columns:
  - `id_kind` (`base`, `base_diag`, `path`, `aggregate`, `base_regex`,
    `base_diag_regex`, or `path_regex`,
    default: `base`)
  - `status` (`ANY`, `FAIL`, `ERROR`, `XFAIL`, `XPASS`, `EFAIL`; default: `ANY`)
  - `expires_on` (`YYYY-MM-DD`)
  - `reason`
- `id_kind=aggregate` matches the synthetic aggregate failing case for
  suite/mode lanes that do not emit per-test result rows (`id=__aggregate__`).
- `id_kind=base_diag` matches `<base>#<DIAG>` where `<DIAG>` is parsed from a
  trailing diagnostic suffix in observed artifact paths (for example
  `aes_sbox_canright#XPROP_ONLY`).
- `id_kind=base_regex` / `base_diag_regex` / `path_regex` treat `id` as a
  Python regular expression matched against observed `base` / `base_diag` /
  `path`, respectively.
  Use `path_regex` for full-path matching; prefer `base_diag` for strict
  OpenTitan diagnostic class matching.
- `--fail-on-unexpected-failure-cases` fails if observed fail-like test cases are
  not matched by the expected-cases file.
- `--fail-on-expired-expected-failure-cases` fails if any expected case is past
  its `expires_on` date.
- `--fail-on-unmatched-expected-failure-cases` fails when expected-case rows
  have no observed match (stale expectation cleanup gate).
- `--prune-expected-failure-cases-file <file>` rewrites expected-case rows
  after matching against observed failures.
- `--prune-expected-failure-cases-drop-unmatched` drops rows with
  `matched_count=0`.
- `--prune-expected-failure-cases-drop-expired` drops rows with `expired=yes`.
  If prune is enabled and no explicit drop policy is set, both unmatched and
  expired drops are enabled by default.
- `--refresh-expected-failure-cases-file <file>` rewrites expected-case rows
  from current observed fail-like cases.
- `--refresh-expected-failure-cases-default-expires-on <YYYY-MM-DD>` sets
  default expiry for newly added refreshed case rows.
- `--refresh-expected-failure-cases-collapse-status-any` collapses refreshed
  rows to one `status=ANY` row per `(suite, mode, id_kind, id)`, preserving
  metadata from matching existing `ANY` rows first, then exact-status rows.
- `--refresh-expected-failure-cases-include-suite-regex <regex>` keeps only
  matching suite rows during case refresh.
- `--refresh-expected-failure-cases-include-mode-regex <regex>` keeps only
  matching mode rows during case refresh.
- `--refresh-expected-failure-cases-include-status-regex <regex>` keeps only
  matching status rows during case refresh.
- `--refresh-expected-failure-cases-include-id-regex <regex>` keeps only
  matching case IDs during case refresh.

## Notes

- The harness only records and compares summaries; detailed failures must be
  read from the per-suite logs and result files.
- When adding a new formal feature or fixing a bug, add both a minimal MLIR
  regression and an end-to-end SV test where possible.
- `--rising-clocks-only` rejects negedge/edge-triggered properties; keep it
  disabled when running suites that include negedge/edge assertions.
