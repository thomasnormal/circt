# Simulation Runner Contract

Last updated: February 20, 2026

## Scope
This document defines the stable integration contract for simulation workflows:

- `utils/run_avip_circt_sim.sh`
- `utils/run_sv_tests_circt_sim.sh`

The contract targets automation and regression harnesses, not internal
implementation details.

## Stable Entrypoints
1. AVIP matrix runner:
   - `run_avip_circt_sim.sh [out_dir]`
2. sv-tests simulation runner:
   - `run_sv_tests_circt_sim.sh [sv_tests_dir]`

Both scripts are configured primarily through environment variables.

## AVIP Matrix Runner Contract
Given `run_avip_circt_sim.sh <OUT_DIR>`, the runner must emit:

1. `<OUT_DIR>/matrix.tsv`
2. `<OUT_DIR>/meta.txt`

### `matrix.tsv` base schema
Tab-separated header (in order):

1. `avip`
2. `seed`
3. `compile_status`
4. `compile_sec`
5. `sim_status`
6. `sim_exit`
7. `sim_sec`
8. `sim_time_fs`
9. `uvm_fatal`
10. `uvm_error`
11. `cov_1_pct`
12. `cov_2_pct`
13. `peak_rss_kb`
14. `compile_log`
15. `sim_log`

### `meta.txt` required keys
`meta.txt` is `key=value` text and must include at minimum:

1. `script_path`
2. `avips`
3. `seeds`
4. `sim_timeout_hard`
5. `circt_sim_mode`
6. `circt_sim_extra_args`
7. `fail_on_functional_gate`
8. `fail_on_coverage_baseline`

Additional keys are allowed.

## sv-tests Simulation Runner Contract
`run_sv_tests_circt_sim.sh` requires an input root containing `tests/`.

Failure contract:
1. if the provided root is missing `tests/`, script exits non-zero.
2. stderr includes: `sv-tests directory not found: <path>`.

## Exit Status Contract
1. `0`: run completed with active gates/policies satisfied.
2. Non-zero: usage/configuration error, tool/runtime failure, or gate failure.

## Compatibility Rules
1. Additive fields/keys are allowed.
2. Removing or renaming required artifacts/columns/keys is a breaking change.
3. Breaking changes require:
   - contract doc update,
   - golden test update,
   - migration note for downstream automation.

## Contract Verification
Golden contract coverage is provided by:

- `test/Tools/run-avip-circt-sim-contract-golden.test`
