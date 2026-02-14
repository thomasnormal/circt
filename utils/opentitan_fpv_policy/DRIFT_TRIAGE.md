# OpenTitan FPV BMC Drift Triage Playbook

This playbook governs drift outcomes from canonical OpenTitan FPV BMC policy
baselines.

## Inputs

- Baseline artifacts:
  - `<prefix>-fpv-summary-baseline.tsv`
  - `<prefix>-assertion-results-baseline.tsv`
  - `<prefix>-assertion-status-policy-grouped-violations-baseline.tsv`
- Current run artifacts under `--out-dir`.
- Drift outputs emitted by `run_formal_all.sh` in FPV BMC lane.

## Decision Rule

1. **Expected design/tool improvement**:
   - update baselines using workflow `update` mode.
2. **Temporary instability / known rollout**:
   - add narrow allowlist entries with explicit removal follow-up.
3. **Unexpected regression**:
   - fail check mode, do not update baseline, open fix issue.

Do not combine broad allowlists with baseline updates in the same change unless
the change is strictly a migration with a documented sunset.

## Update Procedure

1. Run:
   - `utils/run_opentitan_fpv_bmc_policy_profiles.sh update --opentitan-root <ot-root>`
2. Inspect drift deltas and summarize rationale in commit message.
3. Commit updated baseline artifacts and changelog entry together.

## Check Procedure

1. Run:
   - `utils/run_opentitan_fpv_bmc_policy_profiles.sh check --opentitan-root <ot-root>`
2. If failure:
   - inspect per-assertion drift first,
   - then grouped status-policy drift,
   - then summary drift.
3. Classify each drift as update/allowlist/fix using the decision rule.

## Allowlist Hygiene

- Prefer row-level allowlists over target-level allowlists.
- Include owner + expiry context in allowlist change description.
- Remove stale allowlists when baseline is refreshed.
