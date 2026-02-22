# Mutation Runner Contract (MCY Examples)

Last updated: February 20, 2026

## Scope
This document defines the stable contract for the MCY example mutation runner:

- `utils/run_mutation_mcy_examples.sh`
- `utils/run_mutation_mcy_examples_api.sh`
- `utils/run_mutation_mcy_examples_native_real.sh`

The contract is for automation and CI integrations, not internal implementation
details.

## Stable Entrypoints
1. Base runner:
   - `run_mutation_mcy_examples.sh [options]`
2. Profile API wrapper:
   - `run_mutation_mcy_examples_api.sh <profile> [options]`
   - Supported profiles:
     - `default`: forwards options unchanged to base runner.
     - `native-real`: enforces native backend + real harness policy.
3. Native-real convenience wrapper:
   - `run_mutation_mcy_examples_native_real.sh [options]`
   - Delegates to API profile `native-real`.

## Native-Real Profile Policy Contract
`run_mutation_mcy_examples_api.sh native-real` must:

1. Inject:
   - `--mutations-backend native`
   - `--require-native-backend`
   - `--native-tests-mode real`
   - `--native-real-tests-strict`
   - `--fail-on-native-noop-fallback`
2. Reject conflicting user options:
   - `--smoke`
   - `--yosys`
   - `--mutations-backend`
   - `--native-tests-mode`
3. Exit with code `2` for profile/argument contract violations.

## Artifact Contract
Given `--out-dir <DIR>`, the runner emits:

1. `<DIR>/summary.tsv`
2. `<DIR>/retry-summary.tsv`
3. `<DIR>/retry-reason-summary.tsv`
4. `<DIR>/summary.schema-version`
5. `<DIR>/summary.schema-contract`
6. `<DIR>/retry-reason-summary.schema-version`
7. `<DIR>/retry-reason-summary.schema-contract`

### `summary.tsv` schema
Columns (tab-separated, in order):

1. `example`
2. `status`
3. `exit_code`
4. `detected`
5. `relevant`
6. `coverage_percent`
7. `errors`
8. `policy_fingerprint`

### `retry-summary.tsv` schema
Columns (tab-separated, in order):

1. `example`
2. `retry_attempts`
3. `retries_used`

### `retry-reason-summary.tsv` schema
Columns (tab-separated, in order):

1. `retry_reason`
2. `retries`

## Schema Metadata Contract
Current expected schema versions:

1. `summary.schema-version`: `v2`
2. `retry-reason-summary.schema-version`: `v1`

`*.schema-contract` files are required and contain opaque contract
fingerprints.

## Exit Status Contract
1. `0`: run completed and all active gates/policies passed.
2. Non-zero: usage error, tool/runtime failure, or gate/policy regression.
3. Usage/profile validation failures return code `2`.

## Compatibility Rules
1. Additive schema changes must bump corresponding schema-version.
2. Breaking schema changes require:
   - contract doc update,
   - migration path (or explicit break announcement),
   - golden test updates.
3. Wrapper profile semantics must remain behavior-preserving unless explicitly
documented as a contract change.

## Contract Verification
Golden contract coverage is provided by:

- `test/Tools/run-mutation-mcy-examples-api-contract-golden.test`
