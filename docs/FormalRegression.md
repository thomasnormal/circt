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

Use a trailing baseline window for trend-aware strict gates:

```bash
utils/run_formal_all.sh --strict-gate --baseline-window 7
```

Restrict baseline history to a time window (in days from latest suite baseline):

```bash
utils/run_formal_all.sh --strict-gate --baseline-window 7 --baseline-window-days 30
```

Fail only on specific gate classes:

```bash
utils/run_formal_all.sh --fail-on-new-xpass
utils/run_formal_all.sh --fail-on-passrate-regression
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
- `<out-dir>/avip-results.txt` per-AVIP compile case rows (when `--with-avip`
  is used)
- Harnesses treat `BMC_RESULT=SAT|UNSAT|UNKNOWN` and
  `LEC_RESULT=EQ|NEQ|UNKNOWN` tokens as the source of truth for pass/fail
  classification when not in smoke mode.

JSON summary schema:

- `utils/formal-summary-schema.json`

Baselines:

- `utils/formal-baselines.tsv` (latest baselines per suite/mode)
  - columns: `date suite mode total pass fail xfail xpass error skip pass_rate result`
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
  The first row per run is `operation=run_meta` with `schema_version=1` and
  `run_id`, operation rows carry the same `run_id`, and the final row is
  `operation=run_end` with `exit_code`, `row_count`, and `payload_sha256`
  (plus optional `payload_hmac_sha256`).
- `python3 utils/verify_formal_dryrun_report.py <file>` verifies JSONL run
  envelopes and checks `run_end.row_count`/`run_end.payload_sha256`.
  Use `--allow-legacy-prefix` when the file contains older pre-enveloped rows
  before the first `run_meta`.
  Use `--hmac-key-file <file>` to verify `payload_hmac_sha256`.
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
  - `id_kind` (`base`, `path`, or `aggregate`, default: `base`)
  - `status` (`ANY`, `FAIL`, `ERROR`, `XFAIL`, `XPASS`, `EFAIL`; default: `ANY`)
  - `expires_on` (`YYYY-MM-DD`)
  - `reason`
- `id_kind=aggregate` matches the synthetic aggregate failing case for
  suite/mode lanes that do not emit per-test result rows (`id=__aggregate__`).
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
