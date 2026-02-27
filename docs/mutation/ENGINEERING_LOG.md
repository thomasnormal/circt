# Mutation Engineering Log

## 2026-02-27

- realizations:
  - Adding arithmetic mutations (`ADD_TO_SUB`, `SUB_TO_ADD`) increases fault-class coverage for realistic RTL datapath bugs, but only if site detection stays context-aware.
  - `xrun` coverage API methods require functional coverage to be explicitly enabled (`-coverage functional`), otherwise `get_coverage()` is reported as `0.00`.
  - For stable xrun batch runs, each mutant needs isolated coverage output (`-covworkdir`/`-covtest`) or explicit overwrite; otherwise runs fail with existing DB errors.

- surprises:
  - Initial arithmetic site matching was over-inclusive and treated packed-range width math (for example `[W-1:0]`) as mutation targets.
  - Those range-level mutations dominated weighted schedules and reduced semantic diversity despite anti-dominance scoring.

- changes made:
  - Added arithmetic operators in CIRCT-only planner/mutator and mode mappings (`arith`, `inv`, `balanced/all`).
  - Added bracket-context exclusion for arithmetic site detection in both planner and mutator so site indexing remains aligned.
  - Added regression tests for arithmetic site indexing, unary-minus safety, and range-minus exclusion.
  - Re-ran seeded mutation sweeps with `xrun` and `circt`; no mismatches observed in coverage summaries for tested mutants.
