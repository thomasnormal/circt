# Formal Results Schema

Last updated: March 1, 2026

## Scope
This document defines the machine-readable JSONL contract used by CIRCT formal
workflows (BMC/LEC/connectivity LEC) for baseline capture, drift reporting, and
dashboard aggregation.

Primary producers/consumers in this branch:
- Producers: formal runners and wrappers writing `FORMAL_RESULTS_JSONL_OUT`.
- Validators: `utils/formal/validate_formal_results_schema.py`.
- Aggregators: `utils/formal/build_formal_dashboard_inputs.py`.
- Baseline capture gate: `utils/formal/capture_formal_baseline.py`.

## Schema Version
1. `schema_version` is required in every row.
2. Current version is `1`.
3. Rows with any other version are invalid.

## Required Fields (v1)
Each JSONL row must be an object containing all fields below:
1. `schema_version` (`int`, must be `1`)
2. `suite` (`string`, non-empty)
3. `mode` (`string`, see enum)
4. `case_id` (`string`, non-empty)
5. `case_path` (`string`, non-empty)
6. `status` (`string`, see enum)
7. `reason_code` (`string`, conditional non-empty)
8. `stage` (`string`, see enum)
9. `solver` (`string`, may be empty in non-strict mode)
10. `solver_time_ms` (`null` or non-negative integer)
11. `frontend_time_ms` (`null` or non-negative integer)
12. `log_path` (`string`)
13. `artifact_dir` (`string`)

## Enums (v1)
### `mode`
1. `BMC`
2. `LEC`
3. `CONNECTIVITY_LEC`

### `status`
1. `PASS`
2. `FAIL`
3. `ERROR`
4. `TIMEOUT`
5. `UNKNOWN`
6. `SKIP`
7. `XFAIL`
8. `XPASS`

### `stage`
1. `frontend`
2. `lowering`
3. `solver`
4. `result`
5. `postprocess`

## Normalization and Semantic Rules
1. String fields are trimmed before validation.
2. Enum fields are case-normalized (`mode`/`status` uppercase, `stage` lowercase).
3. `reason_code` may be empty only when `status` is `PASS` or `UNKNOWN`.
4. Timing fields must be either `null` or non-negative integers.

## Strict Contract Mode
Strict mode is optional and currently exposed by:
- `validate_formal_results_schema.py --strict-contract`
- `capture_formal_baseline.py --validate-results-schema-strict-contract`

When enabled, these additional invariants are required:
1. Rows must be sorted by `(case_id, status, case_path)`.
2. `solver` must be non-empty when `stage == "solver"`.

## Versioning Policy
1. Additive non-breaking fields:
   - keep `schema_version` unchanged,
   - update this document and relevant consumers.
2. Enum changes or required-field changes:
   - increment `schema_version`,
   - update validators/aggregators and tests,
   - keep one release of backward parser compatibility when practical.
3. Semantic contract changes (for example strict-mode behavior):
   - require explicit CLI opt-in or a version bump.

## Reason Code Taxonomy Reference
Canonical reason-code families are tracked in
`docs/BMC_LEC_CLEANUP_REFACTOR_PLAN.md` (Section 27) and should be treated as
the normative source until extracted to a dedicated taxonomy document.
