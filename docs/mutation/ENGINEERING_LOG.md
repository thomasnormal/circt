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

## 2026-02-27 (shift operators)

- realizations:
  - Shift-direction swaps (`<<`/`>>`) are a practical, realistic RTL bug class and complement compare/logic/arithmetic fault families.
  - Legacy-cycle assertions based on fixed line numbers are brittle as the native op set grows; tests should assert behavioral properties instead.

- surprises:
  - Local `yosys` wrappers in this environment recurse to themselves, so direct runtime decomp/help probes are unreliable here; static/native implementation work should not depend on invoking it.

- changes made:
  - Added `SHL_TO_SHR` and `SHR_TO_SHL` in planner/mutator with structural guards against `<<<`, `>>>`, `<<=`, and `>>=`.
  - Extended CIRCT-only mode mappings (`arith`, `inv`, `invert`, `balanced/all`) to include shift mutations.
  - Hardened cycle/index tests to avoid depending on an exact global operator count.
  - Ran a shift-focused seeded mutation campaign (24 mutants) through both `xrun` and `circt`; no mismatches observed.
