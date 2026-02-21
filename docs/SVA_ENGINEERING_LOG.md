# SVA Engineering Log

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
