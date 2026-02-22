# Declarative Formal Suite Manifests

These manifests define formal suite/target metadata for Phase 3 refactor work.

## Schema

All manifest files in this folder use this TSV header:

`suite_id<TAB>default_root<TAB>bmc_runner<TAB>bmc_args<TAB>lec_runner<TAB>lec_args<TAB>profiles<TAB>notes`

Conventions:
- `default_root`: default checkout/root path for the suite.
- `bmc_runner` / `lec_runner`: repo-relative runner path or `-` when lane is intentionally absent.
- `bmc_args` / `lec_args`: additional arguments appended after root path; use `-` when none.
- `profiles`: comma-separated subset of `smoke,nightly,full`.
- `notes`: short human-readable intent/context.

These manifests are currently governance + planning artifacts and will be wired
into shared formal orchestration in subsequent Phase 3 slices.
