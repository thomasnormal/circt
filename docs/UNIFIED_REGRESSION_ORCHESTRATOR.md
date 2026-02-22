# Unified Regression Orchestrator

`utils/run_regression_unified.sh` is a single entrypoint for multi-suite runs
across CIRCT and Xcelium baselines.

## Quick Start

1. Edit `docs/unified_regression_manifest.tsv` for your local suite roots.
2. Run a dry-run smoke plan:
   `utils/run_regression_unified.sh --profile smoke --engine both --dry-run`
3. Run selected suites:
   `utils/run_regression_unified.sh --profile nightly --engine circt --suite-regex 'sv_tests|opentitan'`

## Manifest Schema

Base TSV columns:

1. `suite_id`
2. `profiles` (`smoke,nightly,full,all`)
3. `circt_cmd`
4. `xcelium_cmd`

Optional extension columns:

5. `suite_root`
6. `circt_adapter`
7. `xcelium_adapter`
8. `adapter_args`

Notes:

- Use `-` in a command column to mark that lane command should be adapter-resolved.
- Use `--engine both` to run paired CIRCT/Xcelium lanes from the same manifest.
- `xcelium_adapter` defaults to `circt_adapter` when omitted.

## Adapter Catalog

- Use `--adapter-catalog` to point to an adapter mapping TSV.
- Default catalog: `docs/unified_regression_adapter_catalog.tsv`.
- Catalog rows map:
  `adapter_id<TAB>engine<TAB>command_prefix`
- Adapter-generated command form:
  `<command_prefix> <suite_root> <adapter_args>`

## Key Options

- `--jobs N` for bounded parallel lane execution.
- `--lane-retries N` and `--lane-retry-delay-ms N` for bounded lane retries.
- `--resume` to reuse existing `summary.tsv`/`plan.tsv` artifacts and skip only
  lanes whose last status is `PASS` with a matching prior planned command.
- `--shard-count N --shard-index N` for deterministic suite sharding.

## Outputs

Default output directory: `./unified-regression-results`

- `summary.tsv`:
  `suite_id	engine	status	exit_code	elapsed_sec	log`
- `plan.tsv`:
  `suite_id	engine	command`
- `retry-summary.tsv`:
  `suite_id	engine	attempts	retries_used	status	exit_code`
- `engine_parity.tsv` (when `--engine both`):
  `suite_id	circt_status	circt_exit_code	xcelium_status	xcelium_exit_code	parity	reason`
- `logs/*.log` for executed lanes.

## Current Scope

This is a Phase 6.5 implementation focused on unified invocation/reporting with
retry/resume/sharding/parallelization + adapter-driven command normalization.
Additional features (cross-run artifact compaction, richer policy contracts)
are tracked in `docs/WHOLE_PROJECT_REFACTOR_TODO.md`.
