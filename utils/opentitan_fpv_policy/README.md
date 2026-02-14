# OpenTitan FPV BMC Policy Artifacts

This directory holds repo-managed policy inputs for OpenTitan FPV BMC runs.

## Task-Profile Presets

- `task_profile_status_presets.tsv`
  - canonical task-profile assertion status policy presets
  - schema: `task_profile`, `required_statuses`, `forbidden_statuses`

## Baseline Workflow

Use `utils/run_opentitan_fpv_bmc_policy_workflow.sh` to materialize and enforce
policy baselines:

- update baselines:
  - `utils/run_opentitan_fpv_bmc_policy_workflow.sh update ...`
- check drift:
  - `utils/run_opentitan_fpv_bmc_policy_workflow.sh check ...`
- per-cohort baseline prefixing:
  - `utils/run_opentitan_fpv_bmc_policy_workflow.sh --baseline-prefix opentitan-fpv-bmc-prim update ...`
- verilog frontend cache controls (forwarded to OpenTitan FPV BMC lane):
  - `--opentitan-fpv-bmc-verilog-cache-mode off|read|readwrite|auto`
  - `--opentitan-fpv-bmc-verilog-cache-dir <dir>`
  - `--opentitan-fpv-bmc-verilog-cache-dir` requires mode.
- optional check-only launch reason-event budget controls:
  - `--check-bmc-launch-reason-key-allowlist-file <file>`
  - `--check-lec-launch-reason-key-allowlist-file <file>`
  - `--check-max-bmc-launch-reason-event-rows <N>`
  - `--check-max-lec-launch-reason-event-rows <N>`
  - `--check-fail-on-any-bmc-launch-reason-events`
  - `--check-fail-on-any-lec-launch-reason-events`

By default, baselines are written under:

- `utils/opentitan_fpv_policy/baselines/`
  - `opentitan-fpv-bmc-fpv-summary-baseline.tsv`
  - `opentitan-fpv-bmc-assertion-results-baseline.tsv`
  - `opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline.tsv`

With `--baseline-prefix <name>`, artifacts become:

- `<name>-fpv-summary-baseline.tsv`
- `<name>-assertion-results-baseline.tsv`
- `<name>-assertion-status-policy-grouped-violations-baseline.tsv`

## Cohort Profile Packs

- `profile_packs.tsv` defines checked-in OpenTitan FPV cohort packs
  (`prim_all`, `ip_all`, `sec_cm_all`) with:
  - fpv cfg source (`fpv_cfg`)
  - baseline prefix (`baseline_prefix`)
  - optional target-selection controls (`select_cfgs`, `target_filter`,
    `allow_unfiltered`, `max_targets`)
  - optional OpenTitan FPV BMC verilog cache controls
    (`verilog_cache_mode`, `verilog_cache_dir`)
  - optional check-only launch reason-event budget controls:
    - `check_bmc_launch_reason_key_allowlist_file`
    - `check_lec_launch_reason_key_allowlist_file`
    - `check_max_bmc_launch_reason_event_rows`
    - `check_max_lec_launch_reason_event_rows`
    - `check_fail_on_any_bmc_launch_reason_events`
    - `check_fail_on_any_lec_launch_reason_events`
  - current defaults are canary seeds (one representative target per cohort)
    for deterministic baseline bootstrap.

Use `utils/run_opentitan_fpv_bmc_policy_profiles.sh` to run packs:

- update all packs:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh update --opentitan-root ~/opentitan`
- check all packs:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh check --opentitan-root ~/opentitan`
- check all packs without strict-gate dependency on global
  `utils/formal-baselines.tsv`:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh --no-strict-gate check --opentitan-root ~/opentitan`
- check one pack:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh --profile prim_all check --opentitan-root ~/opentitan`
- set workflow-level verilog cache defaults for all selected packs:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh check --opentitan-root ~/opentitan --workflow-verilog-cache-mode read --workflow-verilog-cache-dir /tmp/circt-fpv-cache`
  - per-pack TSV values (`verilog_cache_mode`, `verilog_cache_dir`) override workflow defaults.
- set workflow-level check-only launch reason-event budget defaults:
  - `utils/run_opentitan_fpv_bmc_policy_profiles.sh check --opentitan-root ~/opentitan --workflow-check-bmc-launch-reason-key-allowlist-file utils/opentitan_fpv_policy/bmc_launch_reason_key_allowlist.txt --workflow-check-max-bmc-launch-reason-event-rows 0 --workflow-check-fail-on-any-bmc-launch-reason-events`
  - per-pack TSV values (`check_*`) override workflow defaults.

## Drift Triage

- See `utils/opentitan_fpv_policy/DRIFT_TRIAGE.md` for update-vs-allowlist
  decision policy and review flow.
