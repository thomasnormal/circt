# Mutation Engineering Log

## 2026-02-28 (if-condition polarity mutation class)

- realizations:
  - Inverted `if` conditions are a common and realistic control-path bug class
    not captured by pure operator substitutions.
  - Mutation campaigns need explicit liveness caps (`timeout`, `--max-time`,
    `--max-deltas`) because some control mutations intentionally create
    non-terminating behavior (for example clock-toggle logic inversion).
  - Intermittent `Permission denied` on `circt-sim` invocation appears as an
    infra flake; re-running the same command in-place succeeds and should be
    treated as non-semantic noise in parity campaigns.

- changes made:
  - Added native operator `IF_COND_NEGATE`:
    - planner site detection via `if` word-boundary token matching and
      balanced-parenthesis parsing,
    - mutator rewrite from `if (cond)` to `if (!(cond))`,
    - fallback planner support in `native_mutation_plan.py`.
  - Extended CIRCT-only mode mappings (`control`, `connect`, `cnot0/cnot1`,
    `inv`/`invert`, `balanced/all`) to include `IF_COND_NEGATE`.
  - Added regression tests:
    - `native-create-mutated-if-cond-negate-site-index`
    - `native-create-mutated-if-cond-negate-nested-cond`
    - `circt-mut-generate-circt-only-control-mode-if-cond-negate-op`
  - Re-ran seeded xrun-vs-circt campaigns:
    - single-module signature campaign (`24` mutants): `ok=22`,
      no mismatches; remaining failures were bounded liveness/infra cases.
    - `cov_intro_seeded`-style coverage campaign (`24` mutants): no semantic
      mismatches on terminating mutants; `IF_COND_NEGATE` parity matched.

## 2026-02-28 (mux-arm swap mutation class)

- realizations:
  - Swapping ternary mux arms (`sel ? a : b` -> `sel ? b : a`) is a realistic
    control/data bug class and complements assignment-timing faults.
  - Ternary token matching must be assignment-context aware; wildcard case-item
    patterns (`2'b0?`) can otherwise be misclassified as ternary operators.

- changes made:
  - Added native operator `MUX_SWAP_ARMS` to CIRCT-only planner op set.
  - Implemented site detection with:
    - assignment-context gating (`=` or `<=` before the `?` at statement depth),
    - declaration/port/type disqualifier filtering,
    - nested-ternary-aware `?`/`:` pairing.
  - Added mutator rewrite support to swap ternary true/false arms while
    preserving surrounding whitespace slices.
  - Extended CIRCT-only mode mappings (`control`, `connect`, `cnot0/cnot1`,
    `inv`/`invert`, `balanced/all`) to include `MUX_SWAP_ARMS`.
  - Updated fallback native planner utility (`native_mutation_plan.py`) and
    native-op validation in `run_mutation_mcy_examples.sh`.
  - Added regression tests:
    - `native-create-mutated-mux-swap-arms-site-index`
    - `native-create-mutated-mux-swap-arms-nba`
    - `native-create-mutated-mux-skip-case-item`
    - `circt-mut-generate-circt-only-control-mode-mux-swap-op`
  - Added declaration-safety regression coverage for assignment timing:
    - `native-create-mutated-ba-skip-user-type-decl`
    - `circt-mut-generate-circt-only-ba-skip-user-type-decl`
  - Fixed a real mutation bug where `BA_TO_NBA` could rewrite typed
    declarations (`my_t v = ...`) into invalid syntax (`<=` in declarations).
    Added typed-declaration filters to both planner and mutator paths.
  - Re-ran seeded parity campaigns after the fix:
    - control-mode single-module signature campaign: `ok=24 mismatch=0 fail=0`
      (includes `MUX_SWAP_ARMS` mutants),
    - `cov_intro_seeded`-style coverage campaign: no xrun-vs-circt mismatches on
      terminating mutants; remaining failures are liveness/time-limit cases from
      clock-killing mutants (for example `UNARY_BNOT_DROP` on `clk = ~clk`).

## 2026-02-28 (assignment timing faults and seeded parity campaigns)

- realizations:
  - Assignment-style timing mutations (`=` vs `<=`) are a realistic RTL bug
    class and should be modeled explicitly instead of only via operator swaps.
  - Seeded parity campaigns need deterministic initialization boundaries; when a
    signature depends on uninitialized/X-heavy state, xrun-vs-circt mismatches
    can be false alarms for mutation parity.
  - Some constant-flip mutants on clock-generator literals produce liveness
    failures (`both_fail`) in both simulators; these should be classified as
    non-parity failures, not semantic mismatches.
  - Cross-instance scheduling in current `circt-sim` can shift observation
    timing for NBA-fed control signals in seeded harnesses; single-module
    parity harnesses are currently more reliable for mutation equivalence runs.

- changes made:
  - Added native assignment operators:
    - `BA_TO_NBA`
    - `NBA_TO_BA`
  - Added planner and mutator site detection with context guards to avoid:
    - declaration/typedef/port contexts,
    - comparator/assignment-token collisions,
    - continuous-assignment contexts.
  - Added CIRCT-only mode mapping coverage for assignment faults in
    `control`, `connect`, `cnot0/cnot1`, `inv`/`invert`, and `balanced/all`.
  - Added regression tests for assignment site indexing and safety filters:
    - `native-create-mutated-ba-to-nba-site-index`
    - `native-create-mutated-nba-to-ba-site-index`
    - `native-create-mutated-ba-skip-decl-and-comparator`
    - `native-create-mutated-ba-skip-continuous-assign`
    - `circt-mut-generate-circt-only-control-mode-assignment-ops`
  - Re-ran focused lit coverage for new assignment operators: all targeted
    tests pass.
  - Ran multiple seeded mutation parity campaigns:
    - assignment-focused campaign: `ok=24 mismatch=0 fail=0`
    - broader deterministic single-module campaign:
      `ok=37 mismatch=0 fail=11` (`both_fail` liveness mutants)

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

## 2026-02-27 (multiplication arithmetic fault class)

- realizations:
  - Operand/operator confusions between `+` and `*` are realistic datapath bugs and are semantically distinct from existing add/sub flips.
  - Binary `*` site detection must explicitly exclude wildcard and assignment contexts (`(*)`, `**`, `*=`) to avoid structurally invalid or semantically unrelated mutations.

- changes made:
  - Added native operators `MUL_TO_ADD` and `ADD_TO_MUL`.
  - Added bracket-depth filtering so width/index expressions in `[...]` are not targeted.
  - Integrated multiplication operators into CIRCT-only `arith`, `inv`/`invert`, and `balanced/all` mode mappings.
  - Added mutation rewrite/site-index regression tests plus arith-mode generation coverage tests.
  - Hardened const-site generation regression (`const-unsized-sites`) to avoid brittle dependence on exact global op count growth.
  - Ran seeded parity campaign on a multiplier-bearing mini design (`35` mutants): `ok=35 mismatch=0 fail=0`.

## 2026-02-27 (unary arithmetic sign fault class)

- realizations:
  - Missing unary minus is a realistic arithmetic polarity bug and is semantically distinct from binary operator substitutions.
  - Site detection must reject binary/compound minus contexts (`a-b`, `--`, `->`) to avoid invalid rewrites.

- changes made:
  - Added native operator `UNARY_MINUS_DROP`.
  - Implemented planner and mutator site matching for unary-minus tokens with binary/compound context guards.
  - Integrated `UNARY_MINUS_DROP` into CIRCT-only `arith`, `inv`/`invert`, and `balanced/all` mode mappings.
  - Added site-index and no-op fallback regression tests for unary-minus behavior, plus arith-mode generation test coverage.
  - Ran seeded parity campaign on unary-minus-bearing mini design (`35` mutants): `ok=35 mismatch=0 fail=0`.

## 2026-02-27 (division arithmetic fault class)

- realizations:
  - `/` vs `*` confusion is a realistic datapath bug class and complements the
    existing add/sub/mul operator substitutions.
  - Division site detection must be strict enough to avoid non-expression
    spellings (`/=`, `//`) and keep site-index parity between planner and
    mutator.

- changes made:
  - Added native operators `DIV_TO_MUL` and `MUL_TO_DIV`.
  - Refactored mutator-side mul/div token scanning into one helper to avoid
    duplicated site logic.
  - Integrated division operators into CIRCT-only `arith`, `inv`/`invert`, and
    `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-div-to-mul-site-index`
    - `native-create-mutated-mul-to-div-site-index`
    - `native-create-mutated-div-skip-contexts`
    - `circt-mut-generate-circt-only-arith-mode-div-ops`
  - Ran seeded xrun-vs-circt parity campaign including division/multiplication
    mutations (`40` mutants): `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (signed shift fault class)

- realizations:
  - Confusing logical and arithmetic right shifts (`>>` vs `>>>`) is a
    realistic RTL bug class, especially in signed datapaths.
  - Matching `>>>` requires distinct token handling from `>>` so we avoid
    consuming shift-assignment spellings (`>>>=`) and preserve site-index
    consistency.

- surprises:
  - The local `yosys` executable available in this environment is a
    self-recursing wrapper script, so direct binary decompilation was not
    possible here.

- changes made:
  - Added native operators `SHR_TO_ASHR` and `ASHR_TO_SHR`.
  - Implemented dedicated planner/mutator site detection for arithmetic
    right-shift tokens (`>>>`) with assignment-guard filtering.
  - Integrated the new operators into CIRCT-only `arith`, `inv`/`invert`, and
    `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-shr-to-ashr-site-index`
    - `native-create-mutated-ashr-to-shr-site-index`
    - `native-create-mutated-ashr-skip-assign`
    - `circt-mut-generate-circt-only-arith-mode-ashr-ops`
  - Hardened an existing xcompare generation regression that depended on a
    stale fixed op-count window (`circt-mut-generate-circt-only-xcompare-mode-ops`).
  - Ran seeded xrun-vs-circt parity campaign including shift-class mutations
    (`40` mutants, `SHR_TO_ASHR=6`, `ASHR_TO_SHR=6`): `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (logical vs bitwise confusion fault class)

- realizations:
  - Confusing short-circuit logical operators with bitwise operators
    (`&&`/`||` vs `&`/`|`) is a realistic RTL bug class and semantically
    distinct from pure invert/swap mutations.
  - Logical token matching should avoid assertion-style triple tokens so site
    indexing remains structurally valid.

- changes made:
  - Added native operators:
    - `LAND_TO_BAND`, `LOR_TO_BOR`
    - `BAND_TO_LAND`, `BOR_TO_LOR`
  - Implemented planner-side logical token detection with triple-token guards
    and operand-boundary checks.
  - Implemented mutator-side rewrites for all four operators with matching
    site-index behavior.
  - Integrated the new operators into CIRCT-only `control`, `connect`,
    `inv`/`invert`, and `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-land-to-band-site-index`
    - `native-create-mutated-band-to-land-site-index`
    - `circt-mut-generate-circt-only-control-mode-logbit-ops`
  - Updated brittle default-sequence regression to reflect deterministic op
    ordering after the new operator family was added.
  - Ran seeded xrun-vs-circt parity campaign including logical/bitwise
    confusion mutations (`40` mutants, each new op present 5 times):
    `ok=40 mismatch=0 fail=0`.

## 2026-02-27 (xor/xnor confusion fault class)

- realizations:
  - XOR vs XNOR confusion is a realistic logic fault class and is semantically
    distinct from the existing `XOR_TO_OR` mutation.
  - `XNOR_TO_XOR` must avoid reduction-XNOR forms so binary-site indexing stays
    aligned and structurally valid.

- changes made:
  - Added native operators `XOR_TO_XNOR` and `XNOR_TO_XOR`.
  - Implemented planner-side binary XNOR site detection for `^~` and `~^` with
    operand-boundary checks.
  - Implemented mutator-side rewrites for both directions with site-index
    parity against planner behavior.
  - Integrated XOR/XNOR operators into CIRCT-only `control`, `connect`,
    `inv`/`invert`, and `balanced/all` mode mappings.
  - Added regression tests:
    - `native-create-mutated-xor-to-xnor-site-index`
    - `native-create-mutated-xnor-to-xor-site-index`
    - `native-create-mutated-xnor-skip-reduction`
    - `circt-mut-generate-circt-only-control-mode-xnor-ops`
  - Updated weighted diversity regression to include the expanded logic-op set.
  - Ran seeded xrun-vs-circt parity campaign including xor/xnor mutations
    (`40` mutants, `XOR_TO_XNOR=8`, `XNOR_TO_XOR=8`):
    `ok=40 mismatch=0 fail=0`.

## 2026-02-28 (relational polarity swap fault class)

- realizations:
  - Opposite-direction relational swaps (`<`↔`>`, `<=`↔`>=`) are a realistic
    fault class and distinct from the existing widen/narrow comparator class
    (`LT_TO_LE`, `LE_TO_LT`, etc.).
  - Several generation regressions were brittle to exact operator-count
    windows; they needed to assert properties rather than fixed cycle lengths.

- changes made:
  - Added native operators:
    - `LT_TO_GT`, `GT_TO_LT`
    - `LE_TO_GE`, `GE_TO_LE`
  - Integrated operators in native planner op catalog and site detection.
  - Integrated operators in CIRCT-only mode mappings for `arith`,
    `inv`/`invert`, and `balanced/all`.
  - Extended native mutator rewrites and refactored standalone `<`/`>` token
    lookup into one helper to avoid duplicated matcher logic.
  - Added/updated regression coverage:
    - `native-create-mutated-lt-to-gt-site-index`
    - `native-create-mutated-gt-to-lt-site-index`
    - `native-create-mutated-le-to-ge-site-index`
    - `native-create-mutated-ge-to-le-site-index`
    - `circt-mut-generate-circt-only-arith-mode-relpol-ops`
  - Hardened brittle schedule-dependent tests:
    - `circt-mut-generate-circt-only-ignore-comments-strings`
    - `circt-mut-generate-circt-only-site-aware-cycle`
    - `circt-mut-generate-circt-only-weighted-fault-class-diversity`
  - Validation:
    - focused relpol lit slice: `5 passed`
    - broader CIRCT-only mutation suite: `83 passed`
    - seeded parity campaign on relational-signature mini design
      (`20` mutants): `ok=20 mismatch=0 fail=0`
    - seeded parity campaign on `cov_intro_seeded` (`12` mutants):
      `ok=12 mismatch=0 fail=0`

## 2026-02-28 (xcompare bidirectional fault class)

- realizations:
  - X-sensitivity bugs are often symmetric in practice: using `==`/`!=` where
    `===`/`!==` was intended (and vice versa) can silently change behavior
    around unknown values.
  - Weighted-applicable and context-priority regressions should assert the
    property they validate, not a single hard-coded operator when multiple
    same-site candidates are valid.

- changes made:
  - Added native operators:
    - `EQ_TO_CASEEQ`
    - `NEQ_TO_CASENEQ`
  - Integrated new operators into:
    - native planner op catalog + comparator site counting
    - CIRCT-only mode mappings (`arith`, `inv`/`invert`, `balanced/all`)
    - native mutator rewrite dispatch
  - Added regression tests:
    - `native-create-mutated-eq-to-caseeq-site-index`
    - `native-create-mutated-neq-to-caseneq-site-index`
    - `circt-mut-generate-circt-only-xcompare-bidir-ops`
  - Hardened schedule-sensitive tests to allow either applicable eq-site
    mutation when appropriate:
    - `circt-mut-generate-circt-only-weighted-applicable-only`
    - `circt-mut-generate-circt-only-weighted-context-priority`

- validation:
  - focused new-op lit slice: `3 passed`
  - broader CIRCT-only mutation suite: `86 passed`
  - seeded `cov_intro_seeded` parity campaign (`16` mutants):
    `ok=16 mismatch=0 fail=0`
  - seeded xcompare-signature parity campaign (`24` arith mutants):
    `ok=24 mismatch=0 fail=0`
  - direct-op parity checks:
    - `NATIVE_EQ_TO_CASEEQ@1`: `circt_sig==xrun_sig`
    - `NATIVE_NEQ_TO_CASENEQ@1`: `circt_sig==xrun_sig`

## 2026-02-28 (edge-polarity fault class + native-op validator alignment)

- realizations:
  - Event-control edge polarity (`posedge` vs `negedge`) is a realistic and
    semantically distinct control/timing fault class (clock/reset sensitivity
    bugs).
  - `run_mutation_mcy_examples.sh` had drifted to a legacy native-op allowlist;
    the validator rejected many supported native operators and needed to be
    synchronized with current CIRCT-only operator coverage.
  - Xcelium/xrun does not enable covergroup methods by default; parity harnesses
    should use explicit signatures/counters unless functional coverage is
    explicitly enabled.

- changes made:
  - Added native operators:
    - `POSEDGE_TO_NEGEDGE`
    - `NEGEDGE_TO_POSEDGE`
  - Implemented planner-side keyword site detection with identifier-boundary
    checks and integrated both operators into timing-family classification.
  - Implemented mutator-side rewrites and site-index behavior for both
    operators.
  - Integrated operators into CIRCT-only mode mappings:
    - `control`, `connect`, `inv`/`invert`, `balanced/all`, and primitive
      `cnot0/cnot1`.
  - Extended Python fallback planner (`native_mutation_plan.py`) with matching
    operator support and keyword site counting.
  - Added/updated regression tests:
    - `native-create-mutated-posedge-to-negedge-site-index`
    - `native-create-mutated-negedge-to-posedge-site-index`
    - `native-mutation-plan-edge-polarity`
    - `circt-mut-generate-circt-only-control-mode-edge-polarity-ops`
    - `run-mutation-mcy-examples-native-mutation-ops-edge-polarity-pass`
    - updated `circt-mut-generate-circt-only-weighted-fault-class-diversity`
      to include edge-polarity operators in control-family coverage checks.
  - Synchronized `run_mutation_mcy_examples.sh` native-op token validation with
    the full native CIRCT-only operator set.

- validation:
  - focused lit slice for new edge operators and script validation: `7 passed`
  - targeted xrun-vs-circt parity checks on fixed-seed `cov_intro_seeded`
    harness:
    - baseline: `SIG=2921849304`, `COV_TOTAL=84.38` (match)
    - `NATIVE_POSEDGE_TO_NEGEDGE@1`: match
    - `NATIVE_NEGEDGE_TO_POSEDGE@1`: match
    - `NATIVE_IF_COND_NEGATE@1`: match
  - seeded mini-campaign (`10` balanced mutants): `ok=10 mismatch=0 fail=0`.
