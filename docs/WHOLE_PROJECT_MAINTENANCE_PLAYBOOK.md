# Whole-Project Maintenance Playbook

Last updated: February 20, 2026

Operational playbook for baseline updates, schema migrations, and refactor-safe
maintenance across mutation/formal/simulation workflows.

## 1. Baseline Update Workflow

### Mutation (`run_mutation_mcy_examples.sh` family)

1. Run candidate suite with `--fail-on-diff` against existing baseline.
2. If change is intentional, rerun with `--update-baseline` (and policy flags as needed).
3. Regenerate/validate baseline schema sidecars when required.
4. Run focused lit filter: `run-mutation-mcy-examples-.*`.
5. Record baseline rationale in `CHANGELOG.md`.

### Formal (BMC/LEC)

1. Execute suite with drift gates enabled against current baseline.
2. Update baseline only after confirming no accidental solver/toolchain regressions.
3. Re-run focused formal lit tests + one representative external suite sample.
4. Update relevant formal contract doc if output schema changed.

### Unified Orchestrator

1. Run `--profile smoke --engine both --dry-run` first.
2. Run bounded lane execution (`--jobs`, sharding as needed) on target suites.
3. If output fields/semantics changed, update orchestrator contract tests before baseline refresh.

## 2. Schema Migration Workflow

1. Bump schema version only for incompatible format/semantic changes.
2. Keep old readers or provide migration path for one transition window.
3. Update contract docs and example headers.
4. Add migration/compatibility lit tests.
5. Document migration notes in changelog.

## 3. Adapter Catalog Maintenance (Unified Orchestrator)

1. Add/modify entries in `docs/unified_regression_adapter_catalog.tsv`.
2. Add matching manifest example in `docs/unified_regression_manifest.tsv`.
3. Add/adjust orchestrator lit test:
   - adapter pass case,
   - missing-entry failure case when relevant.
4. Validate with:
   - `run-regression-unified-.*` lit filter,
   - one local dry-run command for the changed adapter.

## 4. New Suite Onboarding Checklist

1. Choose adapter id and define circt/xcelium command prefixes.
2. Add manifest row with `suite_root` and `adapter_args`.
3. Run smoke dry-run first (`--dry-run`).
4. Run with `--jobs 1` then scale concurrency.
5. Optional: enable sharding for longer suites.

## 5. Refactor Slice Checklist (Per PR)

1. One behavior-preserving slice.
2. Focused validation commands included in changelog entry.
3. `docs/WHOLE_PROJECT_REFACTOR_TODO.md` updated in same change.
4. Contract docs updated when output/CLI semantics change.

## 6. Suggested Validation Cadence

1. Per slice: focused lit tests for touched domain.
2. Daily/regular:
   - mutation smoke or native-real sample,
   - formal sample (sv-tests or verilator-verification subset),
   - orchestrator smoke (`circt` + `xcelium` where available).
3. Before merge train: run orchestrator smoke on selected fleet subset.

## 7. Failure Triage Order

1. Contract/schema mismatch failures.
2. Tool resolution/environment failures.
3. Deterministic logic regressions (retry/resume/shard/adapter routing).
4. External suite-specific failures.

Use `docs/WHOLE_PROJECT_OWNERSHIP_MAP.md` to route incident ownership.
