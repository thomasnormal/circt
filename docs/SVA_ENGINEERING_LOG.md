# SVA Engineering Log

## 2026-02-22

- Iteration update (property `nexttime` / `s_nexttime`):
  - realization:
    - legal `nexttime`/`s_nexttime` forms on property operands were still
      importer errors even though the required lowering shape is the same
      delay-shifted property used by bounded `eventually`.
    - Slang enforces a single count for these operators (`[N]`), not a range.
  - implemented:
    - added property-operand lowering for:
      - `nexttime p`
      - `nexttime [N] p`
      - `s_nexttime p`
      - `s_nexttime [N] p`
    - lowering strategy:
      - `ltl.delay true, N`
      - `ltl.implication delayed_true, property`.
    - diagnostics retained for the still-open unary property wrappers:
      - `always`, `s_always`.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-nexttime-property.sv`
    - updated:
      - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
        (now checks `always p` diagnostic).
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-nexttime-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-nexttime-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-nexttime-property.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (bounded property `always` / `s_always`):
  - realization:
    - after unblocking bounded `eventually` and property `nexttime`, bounded
      `always` wrappers on property operands remained rejected even though they
      can be lowered compositionally as shifted-property conjunctions.
  - implemented:
    - added bounded lowering for property-typed:
      - `always [m:n] p`
      - `s_always [m:n] p`
    - lowering strategy:
      - shift property by each delay in `[m:n]` using delayed-true implication
      - combine shifted properties with `ltl.and`.
    - unbounded property forms still emit diagnostics:
      - `always p`
      - `s_always p`
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-bounded-always-property.sv`
    - retained negative guard:
      - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
        (`always p` unsupported diagnostic).
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-bounded-always-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-bounded-always-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-bounded-always-property.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (unbounded property `always` + range guardrail):
  - realization:
    - plain `always p` on property operands remained unsupported.
    - while adding unbounded support, we identified a semantic hazard: open
      upper-bound property ranges (`[m:$]`) in unary wrappers would otherwise
      be accidentally collapsed to a single delay if treated as finite loops.
  - implemented:
    - added unbounded property lowering for:
      - `always p`
    - lowering strategy:
      - `always p` -> `not(eventually(not p))` using strong `eventually`.
    - added explicit diagnostics for open upper-bound property ranges in
      unary wrappers to prevent unsound lowering:
      - unbounded `eventually` range on property expressions
      - unbounded `s_eventually` range on property expressions
      - unbounded `always` range on property expressions
      - unbounded `s_always` range on property expressions
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-unbounded-always-property.sv`
    - updated negative diagnostic regression:
      - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
        now checks unsupported `$past(..., enable)` without explicit clocking.
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-unbounded-always-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-unbounded-always-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-unbounded-always-property.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-nexttime-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-nexttime-property.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (open-range property `s_eventually` and `always`):
  - realization:
    - Slang accepts open-range property wrappers for:
      - `s_eventually [m:$] p`
      - `always [m:$] p`
    - importer still diagnosed these as unsupported, despite a direct lowering
      path being available from existing shifted-property and unbounded unary
      machinery.
  - implemented:
    - `s_eventually [m:$] p` now lowers as:
      - `eventually(shiftPropertyBy(p, m))`
    - `always [m:$] p` now lowers as:
      - `always(shiftPropertyBy(p, m))`
      - encoded via duality:
        `not(eventually(not(shiftPropertyBy(p, m))))`
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-open-range-property.sv`
    - retained nearby guard regressions:
      - `test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
      - `test/Conversion/ImportVerilog/sva-unbounded-always-property.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-open-range-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-open-range-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-open-range-property.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-unbounded-always-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-unbounded-always-property.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (packed sampled-values with explicit clocking):
  - realization:
    - sampled-value helpers used by explicit-clocking forms of
      `$rose/$fell/$stable/$changed` rejected packed operands with:
      `unsupported sampled value type ...`.
    - this blocked legal SVA such as:
      - `$changed(packed_struct, @(posedge clk))`
  - implemented:
    - sampled-value paths now normalize non-`IntType` packed operands through
      `convertToSimpleBitVector` before helper lowering and comparisons.
    - explicit-clocking helper type derivation now accepts packed types via
      simple-bit-vector extraction.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - revalidated:
      - `test/Conversion/ImportVerilog/sva-sampled-default-disable.sv`
      - `test/Conversion/ImportVerilog/sva-sampled-explicit-clock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-default-disable.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-default-disable.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-explicit-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (`$past` packed values with explicit clocking):
  - realization:
    - explicit-clocked `$past` helper lowering still required direct integer
      operands, rejecting packed values with:
      `unsupported $past value type with explicit clocking`.
  - implemented:
    - extended `$past` helper lowering to accept packed operands by:
      - normalizing sampled values to simple-bit-vector form for history state
      - converting sampled result back to the original packed type at use sites
        via materialized conversion.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv`
    - revalidated:
      - `test/Conversion/ImportVerilog/sva-past-explicit-clock-default-disable.sv`
      - `test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-explicit-clock-default-disable.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-explicit-clock-default-disable.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (packed sampled-values in regular assertion clocking):
  - realization:
    - the packed sampled-value enablement introduced for explicit clocking also
      broadens regular assertion-clocked forms (`$changed/$stable`), but this
      path lacked dedicated regression coverage.
  - implemented:
    - added focused importer regression for packed sampled-value usage under
      standard assertion clocking (no explicit sampled-value clock arg).
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-sampled-packed.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-packed.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (string sampled-values with explicit clocking):
  - realization:
    - explicit-clocked sampled-value helpers still rejected `string` operands
      even though bit-vector sampled context conversion (`string_to_int`) is
      already available in generic expression lowering.
  - implemented:
    - sampled helper type derivation now recognizes `string` operands and
      lowers them through 32-bit integer sampled-value state.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv`
    - revalidated:
      - `test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
      - `test/Conversion/ImportVerilog/sva-sampled-packed.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (string `$past` explicit-clocking regression hardening):
  - realization:
    - recent sampled/`$past` helper improvements also enabled explicit-clocked
      `$past` on string operands, but this behavior lacked dedicated coverage.
  - implemented:
    - added focused importer regression to lock string explicit-clocked `$past`
      lowering (`string_to_int` sampled state + `int_to_string` re-materialize
      at result use).
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (sampled explicit-clocking crash hardening):
  - realization:
    - sampled explicit-clocking lowering had a null-deref crash path when
      unsupported operands (e.g. unpacked arrays) hit `convertToSimpleBitVector`
      and returned failure; follow-up type checks dereferenced null values.
  - implemented:
    - added explicit null guards after sampled-value bit-vector conversion in:
      - sampled-value call lowering (`$changed/$stable/$rose/$fell`)
      - explicit-clocked `$past` helper lowering.
    - behavior now emits diagnostics instead of crashing.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (unpacked-array sampled values under assertion clocking):
  - realization:
    - regular assertion-clocked `$changed/$stable` on fixed-size unpacked
      arrays were still rejected by forced simple-bit-vector conversion.
  - implemented:
    - sampled-value conversion now preserves fixed-size unpacked arrays for
      `$changed/$stable` (instead of forcing bit-vector cast).
    - lowering compares sampled/current array values via `moore.uarray_cmp`
      and applies `moore.not` for `$changed`.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-sampled-unpacked-array.sv`
    - updated:
      - `test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
        (diagnostic text after crash-hardening path changes).
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-array.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-unpacked-array.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-unpacked-array.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-packed.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (bounded property `eventually` / `s_eventually`):
  - realization:
    - bounded unary temporal operators on property operands were being treated
      as sequence-only forms. We previously guarded this with diagnostics to
      avoid invalid IR, but that left legal bounded property forms unsupported.
  - implemented:
    - added bounded lowering for property-typed:
      - `eventually [m:n] p`
      - `s_eventually [m:n] p`
    - lowering strategy:
      - shift property by each delay in `[m:n]` using:
        - `ltl.delay true, k`
        - `ltl.implication delayed_true, property`
      - OR the shifted properties with `ltl.or`.
    - kept explicit diagnostics for still-missing property-typed unary forms:
      - `nexttime`, `s_nexttime`, `always`, `s_always`.
  - tests:
    - added:
      - `test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
    - updated:
      - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
        (now checks `nexttime p` diagnostic).
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-bounded-eventually-property.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

## 2026-02-21

- Goal for this iteration:
  - establish an explicit unsupported-feature inventory for SVA
  - close at least one importer gap with TDD

- Realizations:
  - event-typed assertion ports already have dedicated lowering support and
    coverage in `test/Conversion/ImportVerilog/sva-event-arg*.sv`.
  - several explicit "unsupported" diagnostics around timing-control assertion
    ports are now mostly defensive; legal event-typed usage is already routed
    through timing-control visitors.
  - concurrent assertion action blocks were effectively dropped for diagnostics
    in import output, even in simple `else $error("...")` cases.

- Implemented in this iteration:
  - preserved simple concurrent-assertion action-block diagnostics by extracting
    message text from simple system-task action blocks
    (`$error/$warning/$fatal/$info/$display/$write`) into
    `verif.*assert*` label attrs during import.
  - extended regression coverage for:
    - `$error("...")`
    - `$fatal(<code>, "...")`
    - `begin ... $warning("...") ... end`
    - `$display("...")`
    - multi-statement `begin/end` action blocks (first supported diagnostic call)
    - nested control-flow action blocks (`if (...) $display("...")`)
  - fixed a spurious importer diagnostic for nested event-typed assertion-port
    clocking in `$past(..., @(e))` paths by accepting builtin `i1` in
    `convertToBool`.
  - added regression:
    - `test/Conversion/ImportVerilog/sva-event-port-past-no-spurious-bool-error.sv`

- Surprises:
  - the action-block path did not emit a warning in the common
    `assert property (...) else $error("...")` shape; diagnostics were silently
    dropped.
  - module-level labeled concurrent assertions (`label: assert property ...`)
    could be lowered after module terminator setup, which split `moore.module`
    into multiple blocks and broke verification.

- Additional closure in this iteration:
  - fixed module-level concurrent assertion insertion to avoid post-terminator
    block splitting in `moore.module`.
  - added regression `test/Conversion/ImportVerilog/sva-labeled-module-assert.sv`.
  - revalidated yosys SVA smoke on `basic0[0-3]` after the importer fix
    (`8/8` mode cases passing).
  - added support for compound sequence match-item assignments on local
    assertion variables (`+=`, `-=`, `*=`, `/=`, `%=`, bitwise ops, shifts).
  - added regressions in `test/Conversion/ImportVerilog/sva-local-var.sv`
    for `z += 1` and `s <<= 1` match-item forms.
  - follow-up stabilization: compound assignment RHS in Slang can include
    synthesized lvalue references and normalized compound-expression trees.
    lowering now evaluates that RHS under a temporary lhs reference context,
    avoiding importer assertions and preserving single-application semantics.

- Next steps:
  - implement richer action-block lowering (beyond severity-message extraction),
    including side-effectful blocks and success/failure branch semantics.
  - continue inventory-driven closure on unsupported SVA items in
    `docs/SVA_BMC_LEC_PLAN.md`.

- Iteration update (unbounded `first_match` formal path):
  - realization:
    - ImportVerilog now accepts unbounded `first_match` forms, but the
      `LTLToCore` lowering still rejected some unbounded sequence forms with:
      `first_match lowering requires a bounded sequence`.
    - reproduction was stable with:
      `ltl.first_match(ltl.non_consecutive_repeat %a, 2)` under
      `verif.clocked_assert`.
  - implemented:
    - added `test/Conversion/LTLToCore/first-match-unbounded.mlir` as a
      dedicated regression.
    - updated `LTLToCore` first-match lowering to avoid hard failure on
      unbounded inputs and fall back to generic sequence lowering for now.
  - validation:
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-unbounded.mlir`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-first-match-unbounded.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-first-match-unbounded.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-first-match-unbounded.sv`
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core --lower-clocked-assert-like --externalize-registers --lower-to-bmc='top-module=unbounded_first_match bound=5'`

- Iteration update (`restrict property` support):
  - realization:
    - ImportVerilog rejected legal concurrent `restrict property` statements
      with `unsupported concurrent assertion kind: Restrict`.
  - implemented:
    - lowered `AssertionKind::Restrict` to assumption semantics in importer
      paths (plain, clocked, hoisted clocked, and immediate assertion path).
    - added import regression:
      - `test/Conversion/ImportVerilog/sva-restrict-property.sv`
    - added BMC pipeline regression:
      - `test/Tools/circt-bmc/sva-restrict-e2e.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-restrict-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-restrict-property.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-restrict-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-restrict-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_restrict bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-restrict-e2e.sv --check-prefix=CHECK-BMC`

- Iteration update (`cover sequence` support):
  - realization:
    - ImportVerilog rejected legal concurrent `cover sequence` statements with
      `unsupported concurrent assertion kind: CoverSequence`.
  - implemented:
    - lowered `AssertionKind::CoverSequence` through the same concurrent cover
      paths as `CoverProperty` (plain + clocked + hoisted clocked).
    - added import regression:
      - `test/Conversion/ImportVerilog/sva-cover-sequence.sv`
    - added BMC pipeline regression:
      - `test/Tools/circt-bmc/sva-cover-sequence-e2e.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-cover-sequence.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-cover-sequence.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-cover-sequence.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-cover-sequence-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_cover_sequence bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-cover-sequence-e2e.sv --check-prefix=CHECK-BMC`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)

- Iteration update (`accept_on` / `reject_on` support):
  - realization:
    - abort-style property operators (`accept_on`, `reject_on`,
      `sync_accept_on`, `sync_reject_on`) failed import with:
      `unsupported expression: Abort`.
  - implemented:
    - added lowering for `slang::ast::AbortAssertionExpr` in
      `AssertionExprVisitor`.
    - current lowering model:
      - accept variants: `ltl.or(condition, property)`
      - reject variants: `ltl.and(ltl.not(condition), property)`
    - added import regression:
      - `test/Conversion/ImportVerilog/sva-abort-on.sv`
    - added BMC pipeline regression:
      - `test/Tools/circt-bmc/sva-abort-on-e2e.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-abort-on.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-abort-on-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_abort_on_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-abort-on-e2e.sv --check-prefix=CHECK-BMC`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)

- Iteration update (`strong` / `weak` property wrappers):
  - realization:
    - `strong(...)` / `weak(...)` wrappers failed import with:
      `unsupported expression: StrongWeak`.
  - implemented:
    - added lowering for `slang::ast::StrongWeakAssertionExpr` in
      `AssertionExprVisitor`.
    - current behavior preserves the inner assertion expression in the lowering
      pipeline (end-of-trace semantic refinement remains follow-up work).
    - added import regression:
      - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
    - added BMC pipeline regression:
      - `test/Tools/circt-bmc/sva-strong-weak-e2e.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-strong-weak.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-strong-weak-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_strong_weak_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-strong-weak-e2e.sv --check-prefix=CHECK-BMC`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)

- Iteration update (`case` property expressions):
  - realization:
    - `case (...) ... endcase` in property expressions failed import with
      `unsupported expression: Case`.
  - implemented:
    - added lowering for `slang::ast::CaseAssertionExpr` in
      `AssertionExprVisitor`.
    - current lowering model:
      - selector/case item expressions are normalized to boolean `i1`.
      - item groups lower to prioritized nested conditional property logic.
      - no-default case lowers with false default branch.
    - added import regression:
      - `test/Conversion/ImportVerilog/sva-case-property.sv`
    - added BMC pipeline regression:
      - `test/Tools/circt-bmc/sva-case-property-e2e.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-case-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-case-property.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-case-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-case-property-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_case_property_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-case-property-e2e.sv --check-prefix=CHECK-BMC`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)

- Iteration update (`case` property bitvector semantics):
  - realization:
    - initial `case` lowering normalized selectors to boolean, which lost
      multi-bit `case` semantics and diverged from tool expectations.
  - implemented:
    - refined `CaseAssertionExpr` lowering to compare normalized simple
      bitvectors (with type materialization to selector type) rather than
      booleanized selector values.
    - kept prioritized item-group semantics and no-default fallback behavior.
    - upgraded regression coverage to multi-bit selector constants.
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-case-property.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-case-property.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-case-property.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-case-property-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_case_property_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-case-property-e2e.sv --check-prefix=CHECK-BMC`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)

- Iteration update (unbounded `first_match` semantic closure + perf):
  - realization:
    - the initial unbounded `first_match` enablement used generic sequence
      fallback semantics; this avoided hard errors but did not encode
      first-hit suppression.
    - transition masking in `first_match` lowering duplicated many equivalent
      `and` terms (same source state and condition), creating avoidable IR
      churn.
  - implemented:
    - added dedicated unbounded first-match lowering that computes `match` from
      accepting next states and masks all next-state updates with `!match`.
    - optimized both bounded and unbounded first-match paths with
      per-source-state/per-condition transition-mask caching to reduce
      duplicated combinational terms.
    - strengthened regression to assert the first-hit kill-switch structure:
      - `test/Conversion/LTLToCore/first-match-unbounded.mlir`
  - validation:
    - `ninja -C build-test circt-opt`
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-unbounded.mlir`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/first-match-unbounded.mlir`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core` (`~0.01s`)

- Iteration update (sequence warmup min-bound semantics + sequence-event perf):
  - realization:
    - sequence assertion warmup in `LTLToCore` was keyed to exact
      finite-length bounds only; unbounded sequences with known minimum length
      did not receive startup warmup gating.
    - sequence event-control lowering duplicated transition `and` terms per
      state in large NFAs, creating avoidable combinational churn.
  - implemented:
    - added `getSequenceMinLength` in `LTLToCore` and switched warmup gating
      to use minimum-length information (including unbounded-repeat forms).
    - optimized sequence event-control NFA lowering in
      `TimingControls.cpp` by caching per-source-state transition terms.
    - added regression:
      - `test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir`
  - validation:
    - `ninja -C build-test circt-opt circt-verilog`
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir`
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-unbounded.mlir`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sequence-event-control.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sequence-event-control.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sequence-event-control.sv` (`~0.01s`)

- Iteration update (both-edge clock support for clocked sequence/property lowering):
  - realization:
    - `LTLToCore::normalizeClock` still rejected `ltl::ClockEdge::Both` for
      `i1` clocks, which blocked direct `--lower-ltl-to-core` lowering of
      `verif.clocked_{assert,assume,cover}` on `!ltl.sequence` properties with
      `edge` clocks.
  - implemented:
    - removed the `both-edge clocks are not supported in LTL lowering` bailout
      in `normalizeClock`; both-edge now normalizes through `seq.to_clock`
      (no inversion), and edge discrimination continues in sequence lowering
      (`getClockTick`).
    - added regression:
      - `test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
  - validation:
    - `ninja -C build-test circt-opt`
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir build-test/test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir build-test/test/Conversion/LTLToCore/clocked-assert-edge-gating.mlir`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-opt test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir --lower-ltl-to-core` (`~0.01s`)

- Iteration update (sync abort-on clock sampling semantics):
  - realization:
    - importer lowered `accept_on`/`reject_on` and `sync_accept_on`/
      `sync_reject_on` identically, despite `AbortAssertionExpr::isSync`
      exposing synchronized semantics.
  - implemented:
    - `AssertionExprVisitor::visit(AbortAssertionExpr)` now applies assertion
      clock sampling to abort condition when `expr.isSync` is true, using
      current assertion clock/timing control (or default clocking) via
      `convertLTLTimingControl`.
    - strengthened regression expectations in:
      - `test/Conversion/ImportVerilog/sva-abort-on.sv`
      - sync variants now require inner `ltl.clock` on abort condition.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-abort-on.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-abort-on-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_abort_on_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-abort-on-e2e.sv --check-prefix=CHECK-BMC`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-abort-on.sv build-test/test/Tools/circt-bmc/sva-abort-on-e2e.sv`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-abort-on.sv` (`~0.03s`)

- Iteration update (strong/weak wrapper semantic split):
  - realization:
    - importer lowered `strong(...)` and `weak(...)` to equivalent behavior,
      which collapses expected progress semantics.
  - implemented:
    - `StrongWeakAssertionExpr` now lowers as:
      - `strong(expr)` -> `ltl.and(expr, ltl.eventually expr)`
      - `weak(expr)` -> `expr`
    - updated import regression:
      - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
      - split checks for `circt-translate` vs `circt-verilog --ir-moore`
        output forms.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-IMPORT`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-MOORE`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-strong-weak-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc="top-module=sva_strong_weak_e2e bound=2" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-strong-weak-e2e.sv --check-prefix=CHECK-BMC`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-strong-weak.sv build-test/test/Tools/circt-bmc/sva-strong-weak-e2e.sv`
    - formal smoke:
      - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv` (`~0.01s`)

- Iteration update (strong/weak wrapper semantic split):
  - realization:
    - `strong(...)` and `weak(...)` wrappers were lowered identically.
  - implemented:
    - `strong(expr)` now lowers as `ltl.and(expr, ltl.eventually expr)`.
    - `weak(expr)` remains direct lowering.
    - updated regression:
      - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-IMPORT`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-MOORE`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-strong-weak.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Iteration update (empty first_match support):
  - `LTLToCore` now lowers empty `first_match` sequences to immediate success.
  - regression:
    - `test/Conversion/LTLToCore/first-match-empty.mlir`
  - validation:
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-empty.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-empty.mlir`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/first-match-empty.mlir`

- Iteration update (`$future_gclk` forward temporal semantics):
  - realization:
    - `$future_gclk` was normalized to `$past` as an approximation, which
      inverted temporal direction for sampled-value semantics.
    - existing regression checks around global-clock sampled functions were too
      broad (`CHECK: verif.assert`) and could match later assertions.
  - implemented:
    - in `convertAssertionCallExpression`, `_gclk` normalization now maps
      `$future_gclk` to `$future`.
    - added direct `$future` lowering as `ltl.delay(<bool arg>, 1, 0)`.
    - tightened `test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
      checks to keep each function's pattern local, and to explicitly require
      `ltl.delay ..., 1, 0` for `$future_gclk`.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-sampled-functions.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-sampled-functions.sv` (`elapsed=0.032s`)

- Iteration update (unclocked `_gclk` global-clocking semantics):
  - realization:
    - unclocked properties using sampled `_gclk` calls lowered to unclocked
      `verif.assert` forms even when a scope-level `global clocking` existed.
    - root cause: `_gclk` normalization reused base sampled-value lowering but
      did not force clock timing when no local assertion/default clock applied.
  - implemented:
    - `_gclk` paths now consult `compilation.getGlobalClockingAndNoteUse`
      when no explicit/default assertion clock is present.
    - for unclocked `_gclk` assertion contexts, helper-lowered sampled values
      are boolean-normalized and wrapped with `convertLTLTimingControl` so
      assertions remain clocked on the global clocking event.
    - added regression:
      - `test/Conversion/ImportVerilog/gclk-global-clocking.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-global-clocking.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-global-clocking.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/gclk-global-clocking.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-global-clocking.sv --check-prefix=CHECK-MOORE`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-sampled-functions.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/gclk-global-clocking.sv build-test/test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-global-clocking.sv` (`elapsed=0.074s`)

- Iteration update (`$global_clock` timing controls + silent-drop hardening):
  - realization:
    - `assert property (@($global_clock) ...)` did not lower to a clocked
      assertion and could disappear from final IR.
    - assertion conversion failures in `Statements.cpp` were treated as dead
      generate code unconditionally (`if (!property) return success();`), which
      allowed diagnostics with success exit status and dropped assertions.
  - implemented:
    - `LTLClockControlVisitor` now recognizes `$global_clock` system-call event
      expressions and resolves them via
      `compilation.getGlobalClockingAndNoteUse(*currentScope)`, then lowers the
      resolved global clocking event recursively.
    - concurrent assertion lowering now skips silently only for
      `InvalidAssertionExpr` (dead generate); other failed assertion conversions
      now propagate `failure()`.
    - added regressions:
      - `test/Conversion/ImportVerilog/sva-global-clock-func.sv`
      - `test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-func.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-global-clock-func.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-global-clock-func.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-global-clock-func.sv --check-prefix=CHECK-MOORE`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv` (fails with `error: expected a 1-bit integer`)
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-global-clock-func.sv build-test/test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv build-test/test/Conversion/ImportVerilog/gclk-global-clocking.sv build-test/test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-func.sv` (`elapsed=0.077s`)

- Iteration update (`$global_clock` explicit sampled-value clocking args):
  - realization:
    - after adding `@($global_clock)` support in assertion LTL timing controls,
      sampled-value explicit clocking argument paths could still fail because
      they lower through generic event controls (`EventControlVisitor`) instead
      of `LTLClockControlVisitor`.
    - reproduction: `assert property ($rose(a, @($global_clock)));` failed
      import prior to this fix.
  - implemented:
    - added `$global_clock` handling in `EventControlVisitor` signal-event
      lowering, resolving through
      `compilation.getGlobalClockingAndNoteUse(*currentScope)` and recursively
      lowering the resolved global clocking event.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-func.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-global-clock-func.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv` (fails with `error: expected a 1-bit integer`)
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv build-test/test/Conversion/ImportVerilog/sva-global-clock-func.sv build-test/test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv` (`elapsed=0.031s`)

- Iteration update (assertion clock event-list lowering):
  - realization:
    - property clocking event lists (e.g. `@(posedge clk or negedge clk)`) were
      rejected with `unsupported LTL clock control: EventList`.
  - implemented:
    - added `EventListControl` handling in `LTLClockControlVisitor`.
    - each listed event is lowered with the same base sequence/property, then
      combined using `ltl.or`.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-clock-event-list.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-clock-event-list.sv`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-clock-event-list.sv build-test/test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv build-test/test/Conversion/ImportVerilog/sva-global-clock-func.sv build-test/test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-sampled-functions.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list.sv` (`elapsed=0.041s`)

- Iteration update (`$global_clock iff` guard preservation):
  - realization:
    - `$global_clock` support landed, but outer `iff` guards were dropped in
      both assertion LTL clocking and sampled-value explicit event-control
      lowering.
    - reproduction:
      - `assert property (@($global_clock iff en) (a |-> b));`
      - `assert property ($rose(a, @($global_clock iff en)));`
  - implemented:
    - in `LTLClockControlVisitor`, `$global_clock` now applies outer
      `iffCondition` by gating `seqOrPro` with `ltl.and` before clocking.
    - in `EventControlVisitor`, `$global_clock` now combines outer and inner
      `iff` guards and emits `moore.detect_event ... if ...` for sampled-value
      helper/event paths.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-global-clock-iff.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-iff.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-global-clock-iff.sv`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-global-clock-iff.sv build-test/test/Conversion/ImportVerilog/sva-global-clock-func.sv build-test/test/Conversion/ImportVerilog/sva-sampled-global-clock-arg.sv build-test/test/Conversion/ImportVerilog/sva-clock-event-list.sv build-test/test/Conversion/ImportVerilog/sva-invalid-clocking-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/gclk-sampled-functions.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gclk-sampled-functions.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh` (`2/2` mode cases pass)
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-iff.sv` (`elapsed=0.028s`)

- Iteration update (yosys SVA `counter` known-profile XPASS cleanup):
  - realization:
    - widened yosys SVA smoke (`TEST_FILTER='.'`) was clean functionally but
      still exited non-zero due stale expectation baseline:
      `XPASS(fail): counter [known]`.
    - this indicated the expected-failure baseline lagged behind current SVA
      behavior.
  - implemented:
    - removed stale `counter\tfail\tknown` expected-XFAIL entries from:
      - `utils/yosys-sva-bmc-expected.txt`
      - `utils/yosys-sva-bmc-xfail.txt`
  - validation:
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='^counter$' utils/run_yosys_sva_circt_bmc.sh`
      now reports `PASS(pass)` and `PASS(fail)` with zero xpass.
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
      now passes with no failures/xpass in the widened smoke set.
    - profiling sample:
      - `time BMC_SMOKE_ONLY=1 TEST_FILTER='^counter$' utils/run_yosys_sva_circt_bmc.sh` (`elapsed=1.777s`)

- Iteration update (assertion event-list duplicate clock dedup):
  - realization:
    - repeated assertion clock events (for example
      `@(posedge clk or posedge clk)`) lowered to duplicated `ltl.clock`
      operations plus a redundant `ltl.or`.
    - this is unnecessary IR churn and can hurt downstream compile/runtime on
      large generated assertion sets with accidental duplicate event entries.
  - implemented:
    - added structural equivalence helper for clocked LTL values
      (`edge + input + equivalent clock signal`).
    - `LTLClockControlVisitor::visit(EventListControl)` now filters duplicate
      entries before constructing the final OR.
    - duplicate temporary LTL ops are reclaimed with `eraseLTLDeadOps`.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
  - validation:
    - `ninja -C build-test circt-translate`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time BMC_SMOKE_ONLY=1 TEST_FILTER='^counter$' utils/run_yosys_sva_circt_bmc.sh` (`real=2.233s`)

- Iteration update (mixed sequence+signal event-list clock inference):
  - realization:
    - mixed event-list lowering required each sequence event to already be
      clocked (explicitly or via default clocking), so patterns like
      `always @(s or posedge clk)` with unclocked `s` failed with
      `sequence event control requires a clocking event`.
    - commercial tools typically infer sequence sampling from the uniform
      signal-event clock in this form.
  - implemented:
    - in `lowerSequenceEventListControl`, signal events are pre-parsed and
      tracked as concrete `(clock, edge)` tuples.
    - added inference path for unclocked sequence events: if signal events are
      uniform (same edge + equivalent clock signal), synthesize
      `ltl.clock(sequence, inferred_edge, inferred_clock)` before sequence
      event lowering.
    - retained failure for non-uniform signal clocks with updated targeted
      diagnostic.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv` (`real=0.039s`)

- Iteration update (sequence-valued assertion clocking events):
  - realization:
    - assertion timing controls accepted sequence clocking forms like `@s`, but
      lowering treated all clocking-event expressions as scalar signals and
      failed with `error: expected a 1-bit integer`.
    - reproduction:
      - `assert property (@s c);` with `s` a sequence and default clocking.
  - implemented:
    - added sequence-event path in `LTLClockControlVisitor` signal-event
      lowering.
    - sequence clocking event lowering now:
      - converts sequence expression,
      - applies default clocking when unclocked,
      - derives event predicate using `ltl.matched`,
      - clocks assertion input with `ltl.clock` on the match predicate.
    - retained explicit error for property-valued event expressions in this
      path.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-global-clock-iff.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-global-clock-iff.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv` (`real=0.050s`)

- Iteration update (default clocking interaction with explicit `@seq`):
  - realization:
    - after landing `@seq` support, explicit assertion clocking was still
      receiving default clocking at the outer conversion layer, yielding an
      extra `ltl.clock(ltl.clock(...))` wrapper.
    - this is semantically incorrect for explicit-clock-overrides-default and
      caused unnecessary IR nesting.
  - implemented:
    - in `convertAssertionExpression`, default clocking application now checks
      whether the result is already rooted at `ltl.clock`; if so, default
      clocking is skipped.
    - tightened regression expectations in
      `test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv` to
      assert no re-clocked `ltl.clock [[CLOCKED]]` before the assert.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-defaults-property.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-defaults-property.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-defaults.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-defaults.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv` (`real=0.053s`)

- Iteration update (non-uniform mixed event-list sequence inference):
  - realization:
    - unclocked sequence events in mixed lists were inferable only for uniform
      signal clocks. Non-uniform signal lists (for example
      `@(s or posedge clk or negedge rst)`) still failed despite enough timing
      context to synthesize a multi-clock sequence check.
  - implemented:
    - extended `lowerSequenceEventListControl` to infer per-signal clocked
      sequence variants when clocks are non-uniform.
    - generated variants are deduplicated by clocked-value structural
      equivalence before combining.
    - when this path is used, lowering routes through existing multi-clock
      sequence event-control machinery.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv` (`real=0.045s`)

- Iteration update (edge-specific wakeups in multiclock mixed waits):
  - realization:
    - multiclock mixed event-list lowering relied on generic
      `moore.detect_event any` wakeups, which is conservative but obscures
      explicit signal-event edge intent (`posedge` / `negedge`) in generated
      IR.
  - implemented:
    - added supplemental edge-specific detect emission for signal-event entries
      in `lowerMultiClockSequenceEventControl` wait block creation.
    - detects are deduplicated by equivalent clock + edge.
    - generic wakeups remain to preserve conservative sequence clock progress.
    - updated regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv` (`real=0.057s`)

- Iteration update (global-clocking fallback for unclocked sequence events):
  - realization:
    - unclocked sequence event controls only considered default clocking for
      clock inference; with only `global clocking` declared they still failed
      (`sequence event control requires a clocking event`).
  - implemented:
    - added shared helper to apply default-or-global clocking for sequence-ish
      event values.
    - integrated helper in:
      - `lowerSequenceEventControl` (`always @(s)` path),
      - `lowerSequenceEventListControl` (mixed/list path),
      - sequence-valued assertion clocking events in
        `LTLClockControlVisitor` (`@s` in assertion timing controls).
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv` (`real=0.048s`)

- Iteration update (mixed sequence event lists with named events):
  - realization:
    - mixed sequence event-list lowering assumed all non-sequence entries could
      be converted to 1-bit clock-like signals.
    - named event entries (`event e; always @(s or e) ...`) are event-typed and
      caused a hard failure (`expected a 1-bit integer`).
  - implemented:
    - added a direct-event fallback path in `lowerSequenceEventListControl` for
      mixed lists containing event-typed entries.
    - fallback emits:
      - `ltl.matched`-driven `moore.detect_event posedge` wakeups for sequence
        entries,
      - direct `moore.detect_event` wakeups for all explicit signal/named-event
        entries (including `iff` conditions).
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv` (`real=0.036s`)

- Iteration update (named events in assertion clock controls):
  - realization:
    - assertion clock-event lowering expected signal-like expressions and forced
      `convertToI1`; named events in assertion clocks failed with
      `expected a 1-bit integer`.
    - reproducer:
      - `assert property (@(e) c);`
      - `assert property (@(s or e) d);`
  - implemented:
    - in `LTLClockControlVisitor::visit(SignalEventControl)`, event-typed
      expressions are now lowered through `moore.event_triggered` before
      building `ltl.clock`.
    - this integrates with existing event-list clock composition and sequence
      event handling (`ltl.matched`) without changing established paths.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv` (`real=0.034s`)

- Iteration update (avoid default re-clock of composed explicit clocks):
  - realization:
    - explicit assertion clock lists that lower to composed roots (e.g.
      `ltl.or` of `ltl.clock`s) were treated as "unclocked" by defaulting logic
      because only direct `ltl.clock` roots were recognized.
    - this incorrectly reapplied default clocking to explicit mixed clocks,
      changing assertion timing semantics.
  - implemented:
    - explicit timing-control conversion now tags root ops with
      `sva.explicit_clocking`.
    - assertion default clock application now skips values tagged explicit,
      and still skips values that contain explicit clocks through graph scan.
    - strengthened regression:
      - `test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
        now checks the mixed explicit clock result is asserted directly and
        not rewrapped by an extra `ltl.clock`.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv` (`real=0.008s`)

- Iteration update (clocking-block entries in mixed sequence event lists):
  - realization:
    - mixed sequence event-list lowering handled sequence/signal expressions but
      did not resolve clocking-block symbols in that path.
    - reproducer:
      - `clocking cb @(posedge clk); ... always @(s or cb);`
      - failed as `unsupported arbitrary symbol reference 'cb'`.
  - implemented:
    - added clocking-block symbol expansion to canonical signal-event controls
      while parsing mixed sequence event lists.
    - for expanded entries, lowering is forced through multiclock machinery so
      mixed sequence/signal wakeup semantics are preserved.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv` (`real=0.007s`)

- Iteration update ($global_clock entries in mixed sequence event lists):
  - realization:
    - mixed sequence event-list lowering still failed (silently) for:
      - `always @(s or $global_clock);`
    - root cause was missing `$global_clock` resolution in the mixed-list
      parsing path; this path bypassed the dedicated event-control visitor logic.
  - implemented:
    - added explicit `$global_clock` handling while parsing mixed sequence event
      list signal entries.
    - `$global_clock` now resolves through scope global clocking and is lowered
      as the corresponding canonical signal event.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv` (`real=0.007s`)

- Iteration update (assertion mixed clock event-list with clocking blocks):
  - realization:
    - after adding symbol-resolution fallbacks for assertion timing controls,
      the new mixed list case `assert property (@(s or cb) c);` worked, but
      named-event regression appeared in `assert property (@(s or e) d);`.
    - root cause: sequence-clock inference in assertion `EventListControl`
      pre-scan unconditionally applied `convertToI1` to non-assertion entries,
      which rejects event-typed symbols.
  - implemented:
    - assertion event-list sequence-clock inference now mirrors the single-event
      lowering path for non-assertion entries:
      - if inferred expression is `event`-typed, lower via
        `moore.event_triggered` before boolean coercion.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
        (mixed assertion event-list with sequence + clocking block).
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv` (`real=0.007s`)

- Iteration update (sequence `.matched` method support):
  - realization:
    - sequence method `.matched` parsed successfully in assertion expressions but
      import failed with `unsupported system call 'matched'`.
    - `.triggered` was already supported; `.matched` should lower similarly for
      sequence-typed operands.
  - implemented:
    - added expression lowering support for method/system call `matched` on
      `!ltl.sequence` values via `ltl.matched`.
    - added regression:
      - `test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
  - surprise:
    - slang rejects procedural use `always @(posedge s.matched)` with
      `'matched' method can only be called from a sequence expression`; kept
      coverage in assertion context only.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-matched-method.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sequence-event-control.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sequence-event-control.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-matched-method.sv` (`real=0.007s`)

- Iteration update (`$assertcontrol` fail-message parity):
  - realization:
    - `$assertcontrol` lowering only mapped control types 3/4/5
      (off/on/kill for procedural assertion enable).
    - control types 8/9 (fail-message on/off) were ignored, even though
      `$assertfailon/$assertfailoff` already had dedicated lowering.
  - implemented:
    - extended `$assertcontrol` handling to also map:
      - `8` -> fail messages enabled
      - `9` -> fail messages disabled
    - wired through existing global state used by immediate-assert action-block
      fail-message gating (`__circt_assert_fail_msgs_enabled`).
    - added regression:
      - `test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/system-calls-complete.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/system-calls-complete.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv` (`real=0.008s`)

- Iteration update (bounded unary temporal operators on property operands):
  - realization:
    - legal SVA forms like `eventually [1:2] p` (with `p` a property)
      could generate invalid IR (`ltl.delay` on `!ltl.property`) and fail at
      MLIR verification time.
    - this produced an internal importer failure instead of a frontend
      diagnostic.
  - implemented:
    - added explicit frontend diagnostics in unary assertion lowering for
      property-typed operands where current LTL sequence ops are invalid:
      - bounded `eventually`
      - bounded `s_eventually`
      - `nexttime`
      - `s_nexttime`
      - `always`
      - `s_always`
    - new regression:
      - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-matched-method.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv` (`real=0.007s`)

- Iteration update (explicit-clocked unpacked-array sampled support):
  - realization:
    - helper-based explicit clocking for sampled-value functions still assumed
      scalar/bit-vector operands, so `$changed/$stable` on fixed-size unpacked
      arrays worked in regular assertion-clocking paths but failed when an
      explicit sampled clock forced helper lowering.
  - implemented:
    - extended `lowerSampledValueFunctionWithClocking` to support unpacked
      array operands for `$stable/$changed`:
      - store previous sampled value in typed unpacked-array state
      - compare with `moore.uarray_cmp eq`
      - derive `$changed` using `moore.not`
    - hardened frontend diagnostics for unpacked-array sampled operands used
      with `$rose/$fell` to emit consistent
      `unsupported sampled value type for $rose/$fell`.
    - regressions:
      - added `test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv`
      - updated `test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
  - surprise:
    - first test run used stale binaries (`build/bin` absent), so the new test
      still showed old behavior until rebuilding `build-test` tools.
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-array.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-string-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv` (`real=0.007s`)

- Iteration update (explicit-clocked `$past` on unpacked arrays):
  - realization:
    - explicit-clock sampled helpers were extended for unpacked arrays, but
      explicit-clock `$past` still hard-required bit-vector conversion and
      emitted `unsupported $past value type with explicit clocking` for legal
      fixed-size unpacked array operands.
  - implemented:
    - extended `lowerPastWithClocking` to support fixed-size unpacked arrays by
      using typed unpacked-array helper state for history/result storage.
    - retained existing behavior for scalar/packed/string via bit-vector path.
    - new regression:
      - `test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-string-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv` (`real=0.007s`)

- Iteration update (`$past` enable with implicit default clocking):
  - realization:
    - `$past(expr, ticks, enable)` without explicit clocking still failed with
      `unsupported $past enable expression without explicit clocking` in
      procedural context even when `default clocking` was available.
  - implemented:
    - resolved implicit clocking for `$past` regardless of assertion context,
      reusing existing current/default/global clock inference.
    - when implicit clock exists, routed to helper-based clocked `$past`
      lowering instead of erroring.
    - fixed a verifier bug discovered by the new test: helper init constants
      for module-level `$past` state are now built at module insertion point to
      avoid dominance violations.
    - new regression:
      - `test/Conversion/ImportVerilog/sva-past-enable-default-clocking.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-enable-default-clocking.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-enable-default-clocking.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-past-enable-default-clocking.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-packed-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-enable-default-clocking.sv` (`real=0.007s`)

- Iteration update (procedural sampled `_gclk` global-clock semantics):
  - realization:
    - procedural `_gclk` sampled calls (e.g. `$changed_gclk(d)` in
      `always_comb`) were lowered as plain unclocked `moore.past` expressions,
      ignoring global clocking declarations.
  - implemented:
    - for non-assertion sampled-value lowering, if `_gclk` variant is used and
      no explicit clocking argument is provided, importer now resolves global
      clocking in scope and routes through helper-based sampled clocked state.
    - new regression:
      - `test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-packed-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv` (`real=0.007s`)

- Iteration update (procedural sampled default-clocking inference):
  - realization:
    - procedural sampled-value calls without explicit clocking still ignored
      in-scope `default clocking` and lowered to unclocked `moore.past`.
  - implemented:
    - for non-assertion `$rose/$fell/$stable/$changed`, importer now infers
      `default clocking` when available and uses helper-based clocked sampled
      state lowering.
    - new regression:
      - `test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-gclk-procedural.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-default-disable.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv` (`real=0.007s`)

- Iteration update (interface concurrent assertions at instance sites):
  - realization:
    - interface instance elaboration only instantiated interface continuous
      assignments and ignored assertion-generated procedural members in
      interface bodies.
    - result: interface-contained concurrent assertions were silently dropped
      from module IR (no diagnostic and no `verif.assert`).
  - implemented:
    - extended interface per-instance elaboration to also visit/lower
      assertion-origin procedural blocks (`ProceduralBlockSymbol` with
      `isFromAssertion`) in the same instance-context path used for interface
      signal resolution.
    - new regression:
      - `test/Conversion/ImportVerilog/sva-interface-assert-instance.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-interface-assert-instance.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-interface-assert-instance.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-interface-assert-instance.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-interface-property.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-default-clocking-procedural.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling sample:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-interface-assert-instance.sv` (`real=0.006s`)

- Iteration update (unpacked-struct sampled values and explicit-clock `$past`):
  - realization:
    - sampled-value support had been extended to unpacked arrays but still
      rejected unpacked structs (`$changed/$stable` failed with
      `cannot be cast to a simple bit vector`).
    - explicit-clock `$past` helper storage similarly supported unpacked arrays
      but not unpacked structs.
  - implemented:
    - added recursive sampled stable-comparison helper for unpacked structs:
      - compare fields via `moore.struct_extract`
      - reuse sampled comparators recursively and reduce with logical and.
    - wired sampled call lowering (`$stable/$changed`) and explicit sampled
      helper lowering to treat unpacked structs as supported aggregate sampled
      values.
    - extended explicit-clock `$past` aggregate helper path to include unpacked
      structs (typed helper history/result storage).
    - new regressions:
      - `test/Conversion/ImportVerilog/sva-sampled-unpacked-struct.sv`
      - `test/Conversion/ImportVerilog/sva-past-unpacked-struct-explicit-clock.sv`
  - validation:
    - `ninja -C build-test circt-translate circt-verilog`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-struct.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sampled-unpacked-struct.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sampled-unpacked-struct.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-struct-explicit-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-past-unpacked-struct-explicit-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-past-unpacked-struct-explicit-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-explicit-clock.sv > /dev/null`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-explicit-clock.sv > /dev/null`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
    - profiling samples:
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sampled-unpacked-struct.sv` (`real=0.007s`)
      - `time build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-past-unpacked-struct-explicit-clock.sv` (`real=0.007s`)
