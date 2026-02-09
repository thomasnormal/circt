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

## Mutation Coverage Harness (Certitude-Style Classification)

Use the mutation harness to classify injected faults into:
`not_activated`, `not_propagated`, `propagated_not_detected`, and `detected`.

Basic usage:

```bash
utils/run_mutation_cover.sh \
  --design /path/to/design.il \
  --mutations-file /path/to/mutations.txt \
  --tests-manifest /path/to/tests.tsv \
  --formal-activate-cmd "your_activation_check_cmd" \
  --formal-propagate-cmd "your_propagation_check_cmd" \
  --coverage-threshold 90
```

Auto-generate mutations (instead of `--mutations-file`):

```bash
utils/run_mutation_cover.sh \
  --design /path/to/design.il \
  --tests-manifest /path/to/tests.tsv \
  --generate-mutations 1000 \
  --mutations-yosys yosys \
  --mutations-seed 1
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
- `results.tsv`: detection outcomes for executed pairs
- `metrics.tsv`: gate-oriented aggregate metrics (coverage, bucket counts, errors)
- `summary.json`: machine-readable gate/coverage summary for CI trend ingestion
- `improvement.tsv`: actionable bucket-to-remediation mapping
- hint metrics:
  - `hinted_mutants`: mutants with a usable prior `detected_by_test` hint
  - `hint_hits`: hinted mutants detected by the hinted test

Gate behavior:
- `--coverage-threshold <pct>`: fails with exit code `2` when detected/relevant falls below threshold.
- `--fail-on-undetected`: fails with exit code `3` if any `propagated_not_detected` mutants remain.
- `--fail-on-errors`: fails with exit code `1` when formal/test infrastructure errors occur.

### Mutation Lane Matrix Runner

Run multiple mutation lanes from a single TSV:

```bash
utils/run_mutation_matrix.sh --lanes-tsv /path/to/lanes.tsv --out-dir /tmp/mutation-matrix
```

Execution controls:
- `--lane-jobs <N>`: run up to `N` lanes concurrently.
- `--jobs-per-lane <N>`: per-lane mutant worker count passed through to
  `run_mutation_cover.sh`.
- `--default-reuse-pair-file <path>`: default
  `run_mutation_cover.sh --reuse-pair-file` for lanes that do not set a
  lane-specific reuse file.
- `--stop-on-fail` is supported with `--lane-jobs=1` (deterministic fail-fast).

Lane TSV schema (tab-separated):

```text
lane_id    design    mutations_file    tests_manifest    activate_cmd    propagate_cmd    coverage_threshold    [generate_count]    [mutations_top]    [mutations_seed]    [mutations_yosys]    [reuse_pair_file]
```

Notes:
- Use `-` for `activate_cmd` / `propagate_cmd` to disable that stage.
- Use `-` for `coverage_threshold` to skip threshold gating per lane.
- For auto-generation lanes, set `mutations_file` to `-` and provide
  `generate_count` (plus optional top/seed/yosys columns).
- `reuse_pair_file` (optional) overrides `--default-reuse-pair-file` for a
  specific lane.

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
- `avip/<name>/compile`

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
