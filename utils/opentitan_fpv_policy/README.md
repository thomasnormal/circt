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

By default, baselines are written under:

- `utils/opentitan_fpv_policy/baselines/`
  - `opentitan-fpv-bmc-fpv-summary-baseline.tsv`
  - `opentitan-fpv-bmc-assertion-results-baseline.tsv`
  - `opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline.tsv`
