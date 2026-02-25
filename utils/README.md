# Utils

The supported entrypoint for running regressions in this repo is:

- `utils/run_regression_unified.sh`

It reads suite definitions from `docs/unified_regression_manifest.tsv` (and, for
adapter-driven lanes, `docs/unified_regression_adapter_catalog.tsv`) and writes
a summary to `--out-dir/summary.tsv`.

Everything else under `utils/` is implementation detail:

- suite runners invoked by the manifest (e.g. AVIP/sv-tests/OpenTitan/Ibex)
- internal checks and contract tests used by `test/Tools/*.test`
- one-off developer helpers
- Python helper scripts with extra dependencies (see `utils/python-requirements.txt`)

If you are adding a new regression suite, prefer:

1. Add a manifest entry in `docs/unified_regression_manifest.tsv`.
2. Ensure it can be executed via `utils/run_regression_unified.sh`.
