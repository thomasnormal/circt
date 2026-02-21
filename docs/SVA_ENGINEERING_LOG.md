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
