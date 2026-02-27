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

## 2026-02-27 (4-state compare operators)

- realizations:
  - X-propagation handling (`==` vs `===`, `!=` vs `!==`) is a distinct and realistic RTL fault class that should not be conflated with ordinary compare mutations.
  - Comparator token matching must be exact; naive `==` scanning can incorrectly match inside `===` and create semantically wrong mutants.

- changes made:
  - Added native operators `CASEEQ_TO_EQ` and `CASENEQ_TO_NEQ`.
  - Tightened comparator site detection in both planner and mutator so `EQ_TO_NEQ`/`NEQ_TO_EQ` no longer target `===`/`!==` sites.
  - Extended `arith`/`inv`/`invert`/`balanced` mode mappings to include 4-state compare mutations.

## 2026-02-27 (binary XOR-only tightening)

- realizations:
  - Mutating reduction XOR (`^a`) as if it were binary XOR is structurally valid textually but semantically a different fault class, and it skews campaigns toward less realistic operator substitutions.

- changes made:
  - Tightened `XOR_TO_OR` site detection in planner and mutator to binary XOR only.
  - Added guards for XNOR and assignment forms (`^~`, `~^`, `^=`), and ensured site-index parity between planner and mutator.

## 2026-02-27 (constant literal coverage broadening)

- realizations:
  - Many real SV designs use unsized tick constants and hex 1-bit literals (`'0`, `'1`, `1'h0`, `1'h1`) rather than only decimal/binary 1-bit constants.
  - Limiting stuck-at mutations to `1'b*`/`1'd*` leaves practical constant-fault space under-exercised.

- changes made:
  - Extended `CONST0_TO_1`/`CONST1_TO_0` detection and rewrite support to include `1'h0`/`1'h1` and `'0`/`'1`.
  - Refactored constant-flip rewrite logic into a single helper to avoid duplicated token-mapping code.

## 2026-02-27 (signedness cast mutations)

- realizations:
  - Missing or inverted signedness casts are a realistic non-trivial RTL bug class, especially around arithmetic comparisons and overflow behavior.

- changes made:
  - Added native cast operators `SIGNED_TO_UNSIGNED` and `UNSIGNED_TO_SIGNED`.
  - Added boundary-safe cast-call site detection with optional whitespace handling.
  - Integrated cast operators into CIRCT-only `arith`/`inv`/`invert`/`balanced` mode mappings.

## 2026-02-27 (seeded parity campaign hygiene)

- realizations:
  - A full xrun-vs-circt mutant mismatch can be a false alarm when the harness reads uninitialized state (`X`), even with fixed seeds.
  - `$random(seed)` sequence parity was confirmed separately; the mismatch source was testbench/design initialization, not RNG drift.

- changes made:
  - Switched seeded mini parity harnesses to deterministic reset initialization before signature comparison.
  - Re-ran a 24-mutant campaign after reset hygiene; parity result was `ok=24 mismatch=0 fail=0`.

## 2026-02-27 (bitwise operator fault class)

- realizations:
  - Logical (`&&`, `||`) and bitwise (`&`, `|`) operator faults are distinct and both appear in realistic RTL bug patterns.
  - Keeping them separate avoids operator-class blind spots and avoids duplicating existing logical mutation semantics.

- changes made:
  - Added native bitwise operators `BAND_TO_BOR` and `BOR_TO_BAND`.
  - Added binary-site filtering to skip reduction and assignment forms while preserving site-index determinism.
  - Added site-index mutation tests and mode-generation coverage tests for the new bitwise operators.

## 2026-02-27 (unary bitwise inversion fault class)

- realizations:
  - Dropped bitwise inversion (`~expr` accidentally removed) is a common RTL polarity bug and should be modeled separately from logical-negation drops (`!expr`).
  - Unary `~` must be token-aware to avoid mutating reduction and XNOR spellings (`~&`, `~|`, `~^`, `^~`).

- changes made:
  - Added native operator `UNARY_BNOT_DROP`.
  - Implemented site detection and mutation rewriting with guards for reduction/XNOR contexts.
  - Extended CIRCT-only mode mappings (`control`, `inv`, `invert`, `balanced/all`) to include `UNARY_BNOT_DROP`.
  - Added site-index rewrite and mode-generation regression tests.
  - Ran seeded `xrun` vs `circt` parity campaign (`30` mutants including `UNARY_BNOT_DROP`): `ok=30 mismatch=0 fail=0` (with retry guard for transient `Permission denied` relink races).
