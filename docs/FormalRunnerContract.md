# Formal Runner Contract

Last updated: February 20, 2026

## Scope
This document defines the stable integration contract for the formal workflow
entrypoints used by automation and CI.

Primary orchestrator:
- `utils/run_formal_all.sh`

Suite/target runners invoked by the orchestrator:
- `utils/run_sv_tests_circt_bmc.sh`
- `utils/run_sv_tests_circt_lec.sh`
- `utils/run_verilator_verification_circt_bmc.sh`
- `utils/run_verilator_verification_circt_lec.sh`
- `utils/run_yosys_sva_circt_bmc.sh`
- `utils/run_yosys_sva_circt_lec.sh`
- `utils/run_opentitan_circt_bmc.py`
- `utils/run_opentitan_circt_lec.py`
- `utils/run_opentitan_connectivity_circt_bmc.py`
- `utils/run_opentitan_connectivity_circt_lec.py`
- `utils/run_opentitan_fpv_circt_bmc.py`
- `utils/run_opentitan_fpv_circt_lec.py`
- `utils/run_pairwise_circt_bmc.py`

## Stable Orchestrator Surface
`run_formal_all.sh` must provide:
1. A stable `--help` surface (`run_formal_all.sh [options]`).
2. Stable lane/suite selection and output routing via explicit flags:
   - `--out-dir`
   - `--sv-tests`
   - `--verilator`
   - `--yosys`
3. Baseline lifecycle controls:
   - `--baseline-file`
   - `--update-baselines`
   - `--fail-on-diff`

## Required Orchestrator Artifacts
For any successful orchestrator run (`exit 0`) with `--out-dir <DIR>`:
1. `<DIR>/summary.tsv`
2. `<DIR>/summary.json`
3. `<DIR>/summary.txt`

### `summary.tsv` base schema
The base summary table header must include these columns in order:
1. `suite`
2. `mode`
3. `total`
4. `pass`
5. `fail`
6. `xfail`
7. `xpass`
8. `error`
9. `skip`
10. `summary`

### `summary.json` base schema
Top-level keys:
1. `date` (`YYYY-MM-DD`)
2. `git_sha` (hex revision)
3. `rows` (array)

## Baseline Contract
When `--update-baselines` is enabled and run succeeds:
1. baseline file is created/updated at `--baseline-file`.
2. baseline header must include (at minimum) the leading core fields:
   - `date`, `suite`, `mode`, `total`, `pass`, `fail`, `xfail`, `xpass`,
     `error`, `skip`, `pass_rate`, `result`
3. additional columns are allowed and treated as extensible diagnostics.

## OpenTitan Helper CLI Contract
The following helper scripts must expose parseable `--help` output:
1. `run_opentitan_circt_bmc.py`
2. `run_opentitan_circt_lec.py`

At minimum, each helper must keep:
1. `--opentitan-root`
2. `--workdir`
3. `--results-file`

Additionally:
1. BMC helper keeps `--case-policy-file`.
2. LEC helper keeps `--resolved-contracts-file`.

## Exit Status Contract
1. `0`: orchestrator completed and active gates passed.
2. Non-zero: usage errors, tool failures, or gate regressions.

## Compatibility Rules
1. Additive flags/diagnostic columns are allowed.
2. Removing or renaming required flags/artifacts/keys is a breaking change.
3. Breaking changes require:
   - contract doc update,
   - golden test update,
   - migration plan for automation consumers.

## Contract Verification
Golden contract coverage is provided by:
- `test/Tools/run-formal-all-contract-golden.test`
