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

## Drift Triage

- See `utils/opentitan_fpv_policy/DRIFT_TRIAGE.md` for update-vs-allowlist
  decision policy and review flow.
