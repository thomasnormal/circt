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

Fail only on specific gate classes:

```bash
utils/run_formal_all.sh --fail-on-new-xpass
utils/run_formal_all.sh --fail-on-passrate-regression
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

Invoke an executable hook when a cadence iteration fails:

```bash
utils/run_formal_cadence.sh --interval-secs 21600 --iterations 0 --on-fail-hook ./notify_failure.sh -- --with-opentitan --opentitan ~/opentitan
```

Fail hook arguments:
`<iteration> <exit_code> <run_dir> <out_root> <cadence_log> <cadence_state>`.

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
- Harnesses treat `BMC_RESULT=SAT|UNSAT|UNKNOWN` and
  `LEC_RESULT=EQ|NEQ|UNKNOWN` tokens as the source of truth for pass/fail
  classification when not in smoke mode.

JSON summary schema:

- `utils/formal-summary-schema.json`

Baselines:

- `utils/formal-baselines.tsv` (latest baselines per suite/mode)
  - columns: `date suite mode total pass fail xfail xpass error skip pass_rate result`
- `PROJECT_PLAN.md` baseline table (updated when `--update-baselines` is used)

## Notes

- The harness only records and compares summaries; detailed failures must be
  read from the per-suite logs and result files.
- When adding a new formal feature or fixing a bug, add both a minimal MLIR
  regression and an end-to-end SV test where possible.
- `--rising-clocks-only` rejects negedge/edge-triggered properties; keep it
  disabled when running suites that include negedge/edge assertions.
