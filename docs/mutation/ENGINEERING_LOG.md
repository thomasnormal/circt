# Mutation Engineering Log

## 2026-02-28 (compound modulo/division assignment mutation class)

- realizations:
  - `%=`/`/=` confusion is a realistic arithmetic implementation bug class in
    sequential datapath/control updates and complements existing binary
    `%`/`/` mutation operators.
  - Lit tests that synthesize SV using shell `printf` require careful `%`
    escaping (`%%%%`) to avoid format-string interpretation side effects.

- changes made:
  - Added native operators:
    - `MOD_EQ_TO_DIV_EQ`
    - `DIV_EQ_TO_MOD_EQ`
  - Integrated operators in:
    - planner op catalog, compound-assignment site detection, family
      classification, and native apply rewrites
      (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`)
      in `tools/circt-mut/circt-mut.cpp`
    - native-op validator allowlist in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-mod-assign-ops.test`
    - `test/Tools/native-create-mutated-mod-eq-to-div-eq-site-index.test`
    - `test/Tools/native-create-mutated-div-eq-to-mod-eq-site-index.test`

- validation:
  - Red-first: all new tests fail before implementation and pass after.
  - Focused lit slice across compound-assign families: `13 passed`.
  - Seeded xrun-vs-circt parity campaign on `cov_intro_seeded_modassign`
    variation with safe native-op allowlist:
    - `count=24`, `seed=20260228`
    - result: `match=24`, `mismatch=0`
    - operators observed include both new ops:
      `NATIVE_MOD_EQ_TO_DIV_EQ`, `NATIVE_DIV_EQ_TO_MOD_EQ`
    - workspace: `/tmp/mut_parity_modassign_898860`

## 2026-02-28 (compound arithmetic-shift assignment mutation class)

- realizations:
  - Confusion between logical and arithmetic shift-assignment (`>>=` vs
    `>>>=`) is a realistic sequential fault class and semantically distinct
    from expression-level `>>`/`>>>` swaps.
  - Compound-assignment token matching must explicitly avoid overlap with
    neighboring longer/shorter tokens (`>>=` vs `>>>=`) to keep deterministic
    site-index contracts.
  - For parity campaigns, timeout-guarded safe-op subsets remain important to
    prevent liveness stalls from unrelated control-loop mutations.

- changes made:
  - Added native operators:
    - `SHR_EQ_TO_ASHR_EQ`
    - `ASHR_EQ_TO_SHR_EQ`
  - Integrated these operators in:
    - native op catalog, site collection, family classification, and apply
      rewrites (`tools/circt-mut/NativeMutationPlanner.cpp`)
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`)
      in `tools/circt-mut/circt-mut.cpp`
    - native-op token validation in
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-ashr-assign-ops.test`
    - `test/Tools/native-create-mutated-shr-eq-to-ashr-eq-site-index.test`
    - `test/Tools/native-create-mutated-ashr-eq-to-shr-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-ashr-eq-skip-ashr-assign.test`

- validation:
  - Red-first: new tests fail before implementation, pass after.
  - Focused lit slice over touched shift/compound/native-op tests:
    `18 passed`.
  - Seeded xrun-vs-circt parity campaign on deterministic `cov_intro_seeded`
    harness with safe-op allowlist including new operators:
    - `count=24`, `seed=20260228`
    - result: `match=24`, `mismatch=0`
    - operators observed include:
      `NATIVE_SHR_EQ_TO_ASHR_EQ` and `NATIVE_ASHR_EQ_TO_SHR_EQ`
    - workspace: `/tmp/mut_parity_shift_740018`

## 2026-02-28 (compound shift-assignment mutation class + seeded parity hardening)

- realizations:
  - `<<=`/`>>=` confusion is a realistic sequential datapath/control fault class
    and was missing from the native mutation operator set.
  - Seeded parity campaigns need per-mutant timeouts and a safe-op subset;
    unrestricted arithmetic sets can mutate loop updates and create liveness
    hangs (for example non-terminating `repeat/for` control).
  - A transient xrun-vs-circt mismatch in this campaign was harness-induced:
    `always @*` sanitizer initialization at time-zero is simulator-scheduling
    sensitive; switching to `always_comb` removed the artifact.

- changes made:
  - Added native operators:
    - `SHL_EQ_TO_SHR_EQ`
    - `SHR_EQ_TO_SHL_EQ`
  - Integrated operators into:
    - native op catalog, site collection, family classification, and apply:
      `tools/circt-mut/NativeMutationPlanner.cpp`
    - CIRCT-only mode mappings (`arith`, `invert`, `inv`, `balanced/all`):
      `tools/circt-mut/circt-mut.cpp`
    - native-op validator:
      `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-compound-shift-assign-ops.test`
    - `test/Tools/native-create-mutated-shl-eq-to-shr-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-shl-eq-site-index.test`
    - `test/Tools/native-create-mutated-shr-eq-to-shl-eq-skip-ashr-assign.test`

- validation:
  - Red-first: all four new tests fail before implementation, pass after.
  - Focused lit slices over compound-assign/native-plan/native-backend tests:
    all passing (`22 passed` in touched slice).
  - Seeded xrun-vs-circt parity campaign on deterministic `cov_intro_seeded`
    harness with safe native op subset (including both new ops):
    - `count=20`, `seed=20260228`
    - result: `match=20`, `mismatch=0`
    - campaign workspace: `/tmp/mut_parity_shift_740018`

## 2026-02-28 (compound-assign realism + array-index sentinel parity bug)

- realizations:
  - Compound-assignment confusion for `*=`/`/=` is a realistic arithmetic fault
    class and fits naturally with existing `+=`/`-=` mutations.
  - A real xrun-vs-circt mismatch surfaced under mutation:
    `NATIVE_LT_TO_LE@3` changed `i < 16` to `i <= 16`, and CIRCT interpreted an
    imported sentinel index as a valid in-range access.
  - Root cause was not randomness: deterministic seeded parity reproduced it
    consistently (`79/80` before fix, one stable mismatch).

- changes made:
  - Added native operators:
    - `MUL_EQ_TO_DIV_EQ`
    - `DIV_EQ_TO_MUL_EQ`
  - Integrated new operators in planner + mutator + mode mappings:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `tools/circt-mut/circt-mut.cpp`
    - `utils/mutation_mcy/templates/native_create_mutated.py`
    - `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions for new ops:
    - `test/Tools/native-create-mutated-mul-eq-to-div-eq-site-index.test`
    - `test/Tools/native-create-mutated-div-eq-to-mul-eq-site-index.test`
    - updated `circt-mut-generate-circt-only-arith-mode-compound-assign-ops.test`
  - Fixed circt-sim array OOB sentinel behavior:
    - added `isImportedArrayIndexOutOfBoundsSentinel(...)` in
      `LLHDProcessInterpreter`,
    - applied it to `hw.array_get`, `hw.array_inject`, and `llhd.sig.array_get`
      probe/drive paths.
  - Added circt-sim regression test:
    - `test/Tools/circt-sim/array-get-oob-sentinel-index.sv`

- validation:
  - Focused lit slices for compound-assign and sentinel regression pass.
  - Minimal reproducer before/after:
    - before: `COUNT=2` for out-of-bounds loop (`i <= 16`)
    - after: `COUNT=1` (matches xrun semantics)
  - Previously failing mutant now matches:
    - `NATIVE_LT_TO_LE@3` summary parity restored.
  - Full seeded parity rerun (`count=80`, `seed=31`) now:
    - `matches=80`, `fails=0`.

## 2026-02-28 (modulo/division confusion mutation class)

- realizations:
  - `%` vs `/` confusion is a realistic arithmetic bug class in datapath RTL
    and complements existing `MUL`/`DIV` confusion operators.
  - xrun-vs-circt parity must avoid harness artifacts:
    - `get_coverage()` in xrun requires `-coverage ...` enabled.
    - stimulus driven on the same `posedge` as DUT flops can introduce race
      order differences that look like simulator bugs.
    - uninitialized memories can create X-vs-value noise unrelated to mutation
      semantics.

- changes made:
  - Added native operators:
    - `MOD_TO_DIV`
    - `DIV_TO_MOD`
  - Integrated new operators into CIRCT-only planning:
    - native op catalog (`kNativeMutationOpsAll`)
    - site collection (`collectBinaryMulDivSites` extended to `%`)
    - op-site dispatch (`collectSitesForOp`)
    - arithmetic fault-family classification (`getOpFamily`)
  - Added CIRCT-only mode mapping coverage in `circt-mut generate`:
    - `arith`
    - `invert` / `inv`
    - `balanced` / `all`
  - Added native mutator rewrite support:
    - `utils/mutation_mcy/templates/native_create_mutated.py`
      - `% -> /` for `MOD_TO_DIV`
      - `/ -> %` for `DIV_TO_MOD`
  - Updated public native-op CLI validation list:
    - `utils/run_mutation_mcy_examples.sh`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-arith-mode-mod-ops.test`
    - `test/Tools/native-create-mutated-mod-to-div-site-index.test`
    - `test/Tools/native-create-mutated-div-to-mod-site-index.test`

- validation:
  - TDD step: new tests fail before implementation; pass after implementation.
  - Focused lit slices:
    - `circt-mut-generate-circt-only-arith-mode-(div|mod)-ops`
    - `native-create-mutated-(div-to-mul|mul-to-div|mod-to-div|div-to-mod)-site-index`
    - broader touched slice with related arith/native tests (`12 passed`)
  - Seeded parity campaign using generated mutants (`count=24`, `seed=20260228`)
    on a race-free deterministic `cov_intro_seeded`-style bench:
    - includes new ops `NATIVE_MOD_TO_DIV@1`, `NATIVE_DIV_TO_MOD@1`
    - `xrun` (with coverage enabled) vs `circt-sim` summary comparison:
      `ok=24 mismatch=0 fail=0`

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

## 2026-02-28 (xrun/circt mismatch root cause: uninitialized 4-state arrays)

- realizations:
  - The deterministic xrun/circt delta on seeded mutants was not RNG-related:
    it came from uninitialized `logic` memory semantics.
  - In `--ir-llhd` output, uninitialized `moore.variable : <uarray<... x lN>>`
    was lowered to a zeroed `llhd.sig` initializer instead of all-`X`.
  - This made `==/!=`-based control mutations diverge whenever reads touched
    uninitialized memory (xrun saw `X`, CIRCT saw `0`).
  - One batch-sweep mismatch set was a false positive caused by transient
    `Permission denied` while binaries were being updated concurrently; direct
    per-mutant rechecks matched.

- changes made:
  - Added regression test:
    - `test/Tools/circt-sim/uninitialized-logic-memory-reads-x.sv`
  - Implemented Moore-to-core fix for uninitialized fixed arrays of 4-state
    elements:
    - added `createAllXArrayOfFourStateValue(...)`
    - added compact integer-pattern replication helper
      `repeatIntegerPattern(...)`
    - wired `VariableOpConversion` fallback init path to use all-`X` for
      fixed-size arrays with 4-state leaf elements before generic zero init.

- validation:
  - TDD:
    - new lit test fails before fix (observed `RES r=0 ... hit_neq=1`),
      passes after fix (`RES r=x ... hit_neq=0`).
  - focused stability checks:
    - `class-null-compare.sv`, `readmemh-basic.sv`,
      `syscall-random-unseeded-process-stability.sv` all pass.
  - minimized repro:
    - xrun: `RES r=x hit_eq=0 hit_neq=0 hit_not_eq=0`
    - circt (post-fix): same.
  - seeded mutant parity rechecks:
    - previously divergent `NATIVE_EQ_TO_NEQ@1` and
      `NATIVE_IF_COND_NEGATE@3` now match xrun exactly.
    - previously flagged status mismatches (`51/52/55/56`) rechecked as
      xrun/circt matches (earlier failures were transient execution races).

## 2026-02-28 (inc/dec fault class + seeded parity triage)

- realizations:
  - Increment/decrement operator faults (`++`/`--`) are realistic counter/state
    bugs and were missing from the native operator set.
  - Some planner tests were brittle because they implicitly depended on full
    operator catalog length and default ordering instead of constraining
    operator scope in-test.
  - One xrun-vs-circt mutant mismatch on `cov_intro_seeded` was due to an
    intentional race introduced by `NBA_TO_BA` in clock-edge stimulus
    (`we <=` -> `we =`), not a circt scheduling bug.

- changes made:
  - Added native operators:
    - `INC_TO_DEC`
    - `DEC_TO_INC`
  - Integrated these operators into:
    - native planner catalog/site counting/family classification
    - CIRCT-only mode mappings (`arith`, `inv`/`invert`, `balanced/all`)
    - native mutator rewrite dispatch (`native_create_mutated.py`)
    - Python native-plan fallback operator set and site counting
  - Added regression tests:
    - `circt-mut-generate-circt-only-arith-mode-incdec-ops`
    - `native-create-mutated-inc-to-dec-site-index`
    - `native-create-mutated-dec-to-inc-site-index`
  - Stabilized planner tests by explicitly constraining ops where needed:
    - `native-mutation-plan-site-aware-cycle`
    - `native-mutation-plan-le-assignment-order`
    - `circt-mut-generate-circt-only-weighted-context-priority`
    - adjusted `circt-mut-generate-circt-only-inv-mode-unary-bnot-op` count
      for expanded operator catalog.

- validation:
  - focused lit slice for touched tests: pass
  - broader CIRCT-only mutation suite slice (`generate/native-plan/create-mutated`):
    `76 passed`
  - seeded `cov_intro_seeded` parity sweep (`40` mutants):
    - overall: `39 match / 1 diff`
    - diff: `NATIVE_NBA_TO_BA@3` with race-sensitive clock-edge blocking write
      in stimulus (`xrun rdata=252`, `circt rdata=56`)
    - excluding race-prone BA/NBA swaps: `38 match / 0 diff`
  - seeded `incdec_seeded` arith sweep (`20` mutants, includes inc/dec ops):
    `20 match / 0 diff`.

## 2026-02-28 (MooreToCore ref-arg store fallback deletion safety check)

- realizations:
  - The removed `AssignOpConversion` fallback in
    `lib/Conversion/MooreToCore/MooreToCore.cpp` was forcing `llvm.store` for
    *all* `llhd.ref` function block args, including nonblocking assignments.
  - That fallback contradicts current interpreter behavior, which now tracks
    ref-argument signal provenance across function/task boundaries via
    `refBlockArgSources` and related ref mapping paths.
  - The prior mismatch was semantic, not random: task NBA through ref params was
    being lowered with blocking-store behavior.

- changes made:
  - Kept the fallback removal in MooreToCore (no replacement path added there).
  - Added/kept regression to lock expected task NBA ordering:
    - `test/Tools/circt-sim/task-nonblocking-assign-in-task-order.sv`

- validation:
  - Targeted lit tests pass with current built tools:
    - `test/Conversion/MooreToCore/ref-param-store.mlir`
    - `test/Conversion/MooreToCore/interface-timing-after-inlining.sv`
    - `test/Tools/circt-sim/task-nonblocking-assign-in-task-order.sv`
    - `test/Tools/circt-sim/coverage-event-sampling-order-negedge.sv`
  - Deterministic seeded parity repro rerun:
    - xrun and circt both report:
      `SUMMARY addr_unique=12 we_unique=2 cov_total=87.50 sig=876`
  - Full rebuild/suite execution is currently blocked by concurrent dirty-file
    compile failures outside MooreToCore (`ImportVerilog` and
    `LLHDProcessInterpreter`), so safety claim is bounded to targeted semantics
    and parity checks above.

## 2026-02-28 (reset-polarity fault class in native mutation planner)

- realizations:
  - A reset polarity bug is a high-frequency, realistic control fault, but the
    native operator catalog only had generic `IF_COND_NEGATE`.
  - We need reset-focused site filtering (identifiers like `rst*` / `*reset*`)
    to avoid spending control mutations on unrelated conditions.
  - During parity sweeps, `xrun -R` with source files can silently ignore HDL
    inputs and reuse old snapshots; for reproducible mutant comparison we must
    compile-run without `-R`.

- changes made:
  - Added new native operator: `RESET_COND_NEGATE`.
  - Integrated operator into CIRCT native planner and mode catalogs:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `tools/circt-mut/circt-mut.cpp`
  - Added reset-aware condition scanning in planner:
    - detects reset-like identifiers in `if (...)` conditions, including escaped
      identifiers.
  - Added reset-aware mutation rewrite in native mutator template:
    - `utils/mutation_mcy/templates/native_create_mutated.py`
  - Updated Python native plan helper parity:
    - `utils/mutation_mcy/lib/native_mutation_plan.py`
  - Added TDD regressions:
    - `test/Tools/circt-mut-generate-circt-only-control-mode-reset-cond-negate-op.test`
    - `test/Tools/native-create-mutated-reset-cond-negate-site-index.test`

- validation:
  - TDD: new tests fail before implementation, pass after.
  - Rebuilt and tested:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
    - `llvm-lit -sv` slices for control-mode generation, native mutation plan,
      and native create-mutated tests (`124 passed` in broader touched slice).
  - Seeded campaign (`cov_intro_seeded.sv`, `count=24`, `seed=20260228`,
    mode=`balanced`) with per-mutant xrun/circt comparison:
    - semantic matches: all mutants after rerunning transient tool errors.
    - transient non-matches were infra only (`circt-sim` temporary
      `Permission denied` while binaries changed), not semantic divergence.

## 2026-02-28 (remove duplicated native planner implementation)

- realizations:
  - Native mutation planning logic existed in two places:
    C++ (`circt-mut` / `NativeMutationPlanner`) and Python
    (`utils/mutation_mcy/lib/native_mutation_plan.py`).
  - Keeping both planners in sync is brittle and caused duplicate maintenance
    work for each new mutation operator and site contract update.
  - The apply path remains Python (`native_create_mutated.py`), which is fine as
    long as planning has a single source of truth.

- changes made:
  - Added `circt-mut generate` option support for operator allowlisting:
    - `--native-op OP` (repeatable)
    - `--native-ops CSV`
  - Implemented allowlist filtering in CIRCT-only generation with deterministic
    ordering compatibility (applicable ops first, non-applicable requested ops
    preserved afterward).
  - Routed `--native-op(s)` runs to CIRCT-only planning even when
    `CIRCT_MUT_ALLOW_THIRD_PARTY=1`.
  - Wired allowlist into generation cache key material (`native_ops=`).
  - Removed Python planner implementation file:
    - deleted `utils/mutation_mcy/lib/native_mutation_plan.py`
  - Removed shell fallback planner duplication:
    - `utils/mutation_mcy/lib/native_mutation_plan.sh` now delegates directly to
      `circt-mut generate --native-ops ...` with
      `CIRCT_MUT_ALLOW_THIRD_PARTY=0`.
  - Updated MCY module docs:
    - `utils/mutation_mcy/README.md`
  - Updated planner regression tests to use `circt-mut generate` directly:
    - `test/Tools/native-mutation-plan-*.test`
  - Added invalid-allowlist regression:
    - `test/Tools/circt-mut-generate-circt-only-native-ops-invalid.test`
  - Updated native-backend runner stub test to accept `generate` in addition to
    `cover`:
    - `test/Tools/run-mutation-mcy-examples-native-mutation-plan-safe-ops-pass.test`

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - Regression slice:
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='native-mutation-plan-|circt-mut-generate-circt-only-native-ops-invalid|run-mutation-mcy-examples-native-mutation-plan-safe-ops-pass'` (`10 passed`)
  - Additional generate-option slice:
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='circt-mut-generate-circt-only-(native-ops-invalid|unsupported-options|modes-basic|mode-counts-basic|mode-weights-basic|profiles-basic)'` (`6 passed`)
  - Native MCY backend slice (fake `--circt-mut` harnesses):
    - `python3 llvm/llvm/utils/lit/lit.py -sv build_test/tools/circt/test/Tools --filter='run-mutation-mcy-examples-native-(mutation-ops|mutation-op-filter|mutation-plan-safe-ops|mutation-seed-order|noop-fallback|backend-no-yosys|real-wrapper)'` (`14 passed`)

## 2026-02-28 (parity campaign follow-up: BA->NBA root cause fixed)

- realizations:
  - Control-mode campaigns with seeded `cov_intro` surfaced deterministic
    xrun/circt mismatches for `NATIVE_BA_TO_NBA` sites.
  - The divergences were not RNG-related; they traced to lowering semantics.
  - A second mismatch class remained for `NATIVE_NEGEDGE_TO_POSEDGE`, but that
    class is race-prone (same-edge blocking drives), so cross-simulator output
    depends on scheduler ordering.

- changes made:
  - Added red parity reproducer test:
    - `test/Tools/circt-sim/mixed-ba-nba-shadow-global-visibility.sv`
  - Fixed mixed BA/NBA shadow rewrite bug in:
    - `lib/Dialect/Moore/Transforms/SimplifyProcedures.cpp`
    - Skip shadowing when a global has any nonblocking assignment users.
  - Added pass regression for mixed BA/NBA shadow behavior:
    - `test/Dialect/Moore/simplify-procedures.mlir` (`@MixedBANBA`).

- validation:
  - `utils/ninja-with-lock.sh -C build_test circt-opt circt-verilog`
  - `build_test/bin/llvm-lit -sv test/Dialect/Moore/simplify-procedures.mlir test/Tools/circt-sim/mixed-ba-nba-shadow-global-visibility.sv`
  - Reran all 80 mutants with updated circt and compared to stored xrun
    summaries:
    - `matches=78`, `fails=2`
    - both residual failures are duplicate `NATIVE_NEGEDGE_TO_POSEDGE@2`.

- next mutation-planner implication:
  - Treat edge-polarity swaps on synchronization waits as race-sensitive in
    parity mode (or classify as unstable) to avoid false bug attribution.

## 2026-02-28 (edge-polarity planning/apply alignment + race-aware filtering)

- realizations:
  - `POSEDGE_TO_NEGEDGE` / `NEGEDGE_TO_POSEDGE` previously matched raw keyword
    tokens, so planner could mutate procedural waits in `initial` blocks.
  - This produced parity noise from race-prone same-edge scheduling, not design
    logic defects.
  - Planner/apply drift was present: planner could classify edge sites as
    inapplicable while Python apply still rewrote a fallback site for unsuffixed
    ops.

- changes made:
  - Native planner edge-site collection is now context-aware:
    - only `always* @(...)` sensitivity edge keywords are eligible.
  - Added race-aware target-edge filter:
    - skip edge swaps that would move an `always*` sensitivity edge onto a
      non-`always` event-control edge of the same signal.
  - Synced Python mutation apply logic with planner:
    - edge-site selection uses the same `always*` sensitivity filter.
    - added same target-edge non-`always` conflict filter, so unsuffixed edge
      ops become no-op fallback when planner found no safe site.
  - Files:
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - `utils/mutation_mcy/templates/native_create_mutated.py`
  - Added TDD regressions:
    - `test/Tools/native-mutation-plan-edge-polarity-always-sensitivity-only.test`
    - `test/Tools/native-mutation-plan-edge-polarity-avoid-target-wait-race.test`
    - `test/Tools/native-create-mutated-negedge-to-posedge-skip-procedural-wait.test`

- validation:
  - Build:
    - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - Focused lit slices (edge planner/apply + reset-edge + control-mode edge
    generation):
    - `12 passed`
  - Seeded parity reruns on `cov_intro_seeded`-style design (`count=40`,
    `seed=41`, mode=`control`):
    - before alignment: `matches=39`, `fails=1` (`NATIVE_POSEDGE_TO_NEGEDGE@1`)
    - after planner/apply alignment: `matches=40`, `fails=0`
  - Verified unsuffixed race-pruned edge mutants now produce explicit
    no-op fallback marker in mutated source:
    - `// native_mutation_noop_fallback NATIVE_POSEDGE_TO_NEGEDGE`

## 2026-02-28 (native apply migration: remove Python mutator dependency)

- realizations:
  - Native mutation planning was C++, but apply remained in
    `native_create_mutated.py`, creating a split implementation surface.
  - Keeping planner/apply in one codepath is higher-value than maintaining a
    Python template copy in MCY worker glue.

- changes made:
  - Added native C++ apply API in planner implementation:
    - `tools/circt-mut/NativeMutationPlanner.h`
    - `tools/circt-mut/NativeMutationPlanner.cpp`
    - new exported entry: `applyNativeMutationLabel(...)`
  - Added `circt-mut apply` subcommand (`-i/-o/-d`) in:
    - `tools/circt-mut/circt-mut.cpp`
  - Migrated native create-mutated lit tests from Python invocation to
    `circt-mut apply`:
    - all `test/Tools/native-create-mutated-*.test`
    - added `test/Tools/circt-mut-apply-basic.test`
  - Updated MCY worker native backend to use a generated shell wrapper that
    calls `circt-mut apply` (no Python template copy):
    - `utils/mutation_mcy/lib/worker.sh`
  - Removed Python mutator template:
    - deleted `utils/mutation_mcy/templates/native_create_mutated.py`
  - Updated runner/module docs:
    - `utils/mutation_mcy/README.md`
    - `utils/run_mutation_mcy_examples.sh`
  - Tightened native real harness args validation to report shell-quoting
    errors before smoke-mode gating:
    - `utils/run_mutation_mcy_examples.sh`

- validation:
  - `utils/ninja-with-lock.sh -C build_test circt-mut`
  - `build_test/bin/llvm-lit -sv test/Tools/circt-mut-apply-basic.test`
  - `build_test/bin/llvm-lit -sv test/Tools --filter='native-create-mutated'`
    (`76 passed`)
  - `build_test/bin/llvm-lit -sv test/Tools --filter='(circt-mut-apply-basic|native-create-mutated|run-mutation-mcy-examples-native)'`
    (`111 passed`)
