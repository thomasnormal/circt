# ImportVerilog Engineering Log

## 2026-02-26

### Task
Start TDD-driven implementation of missing cross-bin select support in ImportVerilog, beginning with `!binsof(...)` negation.

### Realizations
- `moore.binsof` and runtime lowering already support a `negate` bit end-to-end; the missing piece was ImportVerilog not propagating unary bins-select negation into `BinsOfOp`.
- The `build_test` tree is the correct lit context for targeted ImportVerilog tests in this workspace; direct source-tree lit invocation can fail due config mismatch.

### Surprises
- `xrun` accepts the new regression syntax cleanly in elaboration mode (`xrun -sv ... -elaborate`) with no warnings, which is a good guardrail for notation correctness.
- Cross-bin `with (...)` syntax in slang does not use `item` in this context the same way coverpoint bin `with (item ...)` does; using undeclared `item` in cross select filters errors in xrun/slang.

### Changes Landed In This Slice
- Added regression: `test/Conversion/ImportVerilog/binsof-negate.sv`.
- Implemented negation propagation in `convertBinsSelectExpr` (`lib/Conversion/ImportVerilog/Structure.cpp`) and wired failure propagation at callsite.

### Validation
- `build_test/bin/circt-verilog test/Conversion/ImportVerilog/binsof-negate.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/binsof-negate.sv`
- `xrun -sv test/Conversion/ImportVerilog/binsof-negate.sv -elaborate -nolog`
- Baseline notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/binsof-intersect.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/binsof-avip-patterns.sv -elaborate -nolog`

### Task
Implement dist default-weight support end-to-end (`ImportVerilog -> Moore -> MooreToCore -> runtime`) and add regression coverage.

### Realizations
- Slang/circt-verilog accepts `default` in `dist` only in 1800-2023 mode (`--language-version 1800-2023`), and in practice parses `default :/ expr`.
- Carrying default weight as explicit attrs on `moore.constraint.dist` (`default_weight`, `default_per_range`) cleanly preserves information through later lowering.
- `DistConstraintInfo` crossed LLVM SmallVector default-inline-size limits after adding fields; `SmallVectorImpl<T>&` / explicit inline counts are required to keep compile-time checks happy.

### Surprises
- Cadence Xcelium 24.03 rejects `dist ... default :/ ...` syntax (`*E,ILLPRI`), so it cannot be used today as an oracle for that 2023 feature.
- Unittests consumed the old `__moore_randomize_with_dist` 4-argument signature and needed full callsite updates after extending runtime ABI.

### Changes Landed In This Slice
- Added ImportVerilog regression:
  - `test/Conversion/ImportVerilog/dist-default-2023.sv`
- Extended `moore.constraint.dist` op definition / verification:
  - `include/circt/Dialect/Moore/MooreOps.td`
  - `lib/Dialect/Moore/MooreOps.cpp`
- Imported default dist weight from slang AST:
  - `lib/Conversion/ImportVerilog/Expressions.cpp`
- Threaded default dist metadata through MooreToCore randomize lowering:
  - `lib/Conversion/MooreToCore/MooreToCore.cpp`
- Extended runtime ABI and implementation for default bucket sampling:
  - `include/circt/Runtime/MooreRuntime.h`
  - `lib/Runtime/MooreRuntime.cpp`
- Updated/extended runtime unit tests:
  - `unittests/Runtime/MooreRuntimeTest.cpp`

### Validation
- Build:
  - `ninja -C build_test circt-verilog circt-sim`
  - `ninja -C build_test MooreRuntimeTests`
- Unit tests:
  - `build_test/unittests/Runtime/MooreRuntimeTests --gtest_filter=MooreRuntimeDistTest.*`
- ImportVerilog regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/binsof-negate.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/binsof-negate.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/signed-instance-port-conversion.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/signed-instance-port-conversion.sv`
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/dist-default-2023.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/dist-default-2023.sv`
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/binsof-negate.sv build_test/test/Conversion/ImportVerilog/signed-instance-port-conversion.sv build_test/test/Conversion/ImportVerilog/dist-default-2023.sv`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/binsof-negate.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/signed-instance-port-conversion.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/dist-constraints.sv -elaborate -nolog`
  - FAIL (tool limitation, unsupported 2023 syntax): `xrun -sv test/Conversion/ImportVerilog/dist-default-2023.sv -elaborate -nolog`

## 2026-02-26

### Task
Audit ImportVerilog cross-bin select handling against IEEE 1800-2023 Syntax 19-4 from `/tmp/ieee1800-2023.txt` without changing ImportVerilog implementation.

### Realizations
- IEEE 1800-2023 uses `intersect`, not `intersecting`, in the grammar:
  - `select_condition ::= binsof ( bins_expression ) [ intersect { covergroup_range_list } ]` (`/tmp/ieee1800-2023.txt:36879`).
- `convertBinsSelectExpr` in ImportVerilog currently handles `Condition`, `Unary`, and `Binary` recursively, but:
  - does not distinguish `&&` from `||` (`BinaryBinsSelectExpr::Op` is ignored),
  - drops `with (...) [matches ...]` by recursing into the base expression only,
  - silently skips `SetExpr` and `CrossId` cases with no diagnostics.

### Surprises
- Multiple standard-valid 2023 forms are accepted by `circt-verilog` and lowered successfully, but with semantics erased (no diagnostics).
- `cross_identifier` (`bins c = X;`) lowers to an empty `moore.crossbin.decl` body. This can be semantically valid ("all tuples"), but `X with (...)` is also lowered the same way, which is incorrect.

### Probe Matrix (build_test/bin/circt-verilog --ir-moore)
- `select_condition`: `binsof(cp) intersect {0}` -> `exit=0`, 1 `moore.binsof`.
- `! select_condition`: `!binsof(cp) intersect {0}` -> `exit=0`, 1 `moore.binsof` with negate.
- `select_expression && select_expression` -> `exit=0`, 2 `moore.binsof`.
- `select_expression || select_expression` -> `exit=0`, 2 `moore.binsof` (same structural lowering as `&&` in tested case, no OR encoding).
- `select_expression with (...) matches 1` -> `exit=0`, lowers as if `with/matches` absent.
- `cross_identifier` (`bins c = X`) -> `exit=0`, 0 `moore.binsof`.
- `cross_identifier with (...) matches` -> `exit=0`, 0 `moore.binsof` (same as plain `X`; filter dropped).
- `cross_set_expression` (`bins c = '{ '{1,0}, '{0,1} };`) -> `exit=0`, 0 `moore.binsof` (expression dropped).
- Non-standard token `intersecting` -> parser error `expected ';'` (good rejection).

### Validation Commands
- Used one-shot case matrix over temporary files:
  - `case_condition`, `case_unary`, `case_and`, `case_or`, `case_with`, `case_crossid`, `case_crossid_with`, `case_setexpr`, `case_intersecting_kw`
  - command emitted summary tuples of `exit`, `err_lines`, `crossbins`, and `binsof`.

## 2026-02-26

### Task
Add detection (hard diagnostics) for unsupported IEEE 1800-2023 cross-bin select forms in ImportVerilog, without implementing full semantics.

### Realizations
- Silent acceptance was the highest-risk behavior; explicit rejection is safer than partial lowering.
- Plain `cross_identifier` (`bins x = X;`) can remain representable as an empty cross-bin filter body.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - reject `||` with `error: unsupported cross select expression operator '||'`
  - reject any `with (...)` select form with `error: unsupported cross select expression with 'with' clause`
  - reject `cross_set_expression` with `error: unsupported cross select expression: cross set expression`
  - keep plain `CrossId` (`X`) accepted
- Added regression tests:
  - `test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-crossid-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` with explicit 2023 cross-select support/unsupported matrix.

### Validation
- Pre-fix TDD baseline:
  - all three new `not circt-verilog ... | FileCheck` tests failed (tool accepted input and produced IR).
- Post-fix:
  - `llvm/build/bin/not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `llvm/build/bin/not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `llvm/build/bin/not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-crossid-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-crossid-supported.sv`

## 2026-02-26

### Task
Continue closing cross-bin select gaps by implementing:
- `select_expression with (with_covergroup_expression)` (finite-domain path)
- `cross_set_expression` tuple list lowering

### Realizations
- Slang already binds `with` filters with per-cross-item iterator symbols (`IteratorSymbol`), which lets us evaluate the filter expression against explicit value tuples using `EvalContext`.
- `cross_set_expression` binds to the cross `CrossQueueType`, and for practical regressions this arrives as constant queue / unpacked tuple data that can be lowered directly to grouped `moore.binsof` filters.
- Existing OR-group machinery (`group` attribute + runtime separator lowering) naturally represents tuple-union selections emitted from `with` / `setexpr`.

### Surprises
- Xcelium 24.03 accepts `X with (expr)` but rejects explicit `matches 1` in this context (`*E,COVMNS`), even though omitting `matches` is semantically equivalent (default policy 1).
- Xcelium 24.03 still rejects plain `bins all = X;` (`*E,COVMCP`), so that pre-existing regression cannot currently be xrun-validated despite CIRCT/slang accepting it.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add constant queue / tuple lowering for `SetExprBinsSelectExpr`
  - add finite-domain tuple enumeration + expression evaluation for `BinSelectWithFilterExpr`
  - preserve hard diagnostics for unsupported policies / spaces:
    - `matches $`
    - `matches` other than 1
    - non-constant `cross_set_expression`
    - `with` filters requiring large / non-finite domain enumeration
  - refactor cross-target symbol resolution for `moore.binsof` emission
- `test/Conversion/ImportVerilog/cross-select-with-supported.sv`:
  - flip to positive regression (supported lowering)
  - use `X with (a + b < 3)` (no explicit `matches 1` for xrun compatibility)
- `test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`:
  - flip to positive regression (supported tuple-list lowering)
- `test/Conversion/ImportVerilog/cross-select-or-supported.sv`:
  - RUN line fixed to non-negated command (feature now supported)
- renamed cross-select positive tests from `*-unsupported.sv` to
  `*-supported.sv` for clarity
- `docs/ImportVerilog-Limitations.md`:
  - update cross-select status/matrix to reflect supported OR/with/setexpr subset and remaining constraints

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regression checks (FileCheck):
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-crossid-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-crossid-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/binsof-negate.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/binsof-negate.sv`
- Runtime unit tests:
  - `build_test/unittests/Runtime/MooreRuntimeTests --gtest_filter=MooreRuntimeCrossCoverageTest.*`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-or-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/binsof-negate.sv -elaborate -nolog`
  - FAIL (tool limitation): `xrun -sv test/Conversion/ImportVerilog/cross-select-crossid-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue ImportVerilog cross-select gap closure by supporting nested `with`
composition in boolean select expressions.

### Realizations
- The previous `with` implementation only handled top-level `WithFilter`
  expressions; `buildCrossSelectDNF` intentionally rejected nested `with`,
  causing valid IEEE expressions like
  `binsof(a) with (...) || binsof(b) with (...)` to fail.
- For any select expression containing `with` or `setexpr` nodes, a finite
  tuple evaluator over the cross domain provides a robust fallback while
  preserving the symbolic DNF lowering for plain `binsof` trees.

### Surprises
- Xcelium 24.03 accepts nested `with` composition in cross select expressions
  (good oracle for syntax/intent), even though it still rejects some other
  2023 features (`matches` clause and plain `bins = X`).

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `containsWithOrSetExpr` tree scan
  - add recursive tuple evaluator (`evaluateCrossSelectExprOnTuple`) supporting
    `Condition`, `CrossId`, `Unary`, `Binary`, nested `WithFilter`, and
    `SetExpr` (`matches 1` policy)
  - add finite fallback emitter (`emitFiniteTupleCrossSelect`) for expression
    trees that contain nested `with` / `setexpr`
  - keep existing top-level specialized paths for `WithFilter` / `SetExpr`
    unchanged
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
    covering `binsof(a) with (...) || binsof(b) with (...)`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regression checks:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-or-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Re-audit and close stale task/event limitation tracking.

### Realizations
- The previously documented limitation for task-local timing controls
  referencing module variables is no longer present in the current tree.
- ImportVerilog now captures module refs into task function signatures and
  rewrites call sites accordingly (for this pattern).

### Changes Landed In This Slice
- Added regression:
  - `test/Conversion/ImportVerilog/task-event-module-var-capture.sv`
- Updated docs:
  - removed stale "Task Clocking Events with Module-Level Variables" limitation
    from `docs/ImportVerilog-Limitations.md`

### Validation
- CIRCT regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/task-event-module-var-capture.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/task-event-module-var-capture.sv`
- xrun notation check:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/task-event-module-var-capture.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing ImportVerilog cross-select gaps with TDD, while validating every new regression candidate against `xrun -elaborate` notation support.

### Realizations
- `xrun` 24.03 rejects cross-bin `matches` clause syntax (`*E,COVMNS` / parser ambiguity on set-expression forms), so those are not usable as oracle-backed regressions in this environment.
- A practical remaining gap that is xrun-compatible is non-constant `cross_set_expression` via helper functions that return `CrossQueueType` literals.
- Slang script-mode expression evaluation (`EvalFlags::IsScript`) plus an empty eval frame can evaluate function-call set expressions in the literal-return style:
  - supported: `function CrossQueueType mk(); mk = '{ '{1,0}, '{0,1} }; endfunction`
  - still unsupported: procedural queue building forms (for example `mk.push_back(...)` loops).

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `parseCrossMatchesPolicy` helper and route cross-set `matches` parsing through it.
  - add script-mode fallback for evaluating cross-set expressions:
    - `evaluateCrossSelectScriptExpr`
    - `evaluateCrossSetExpr`
  - thread script-evaluable cross-set expression handling through:
    - `emitSetExprBinsSelect`
    - `evaluateSetExprOnTuple`
  - keep explicit diagnostics for unevaluable cross-set expressions.
- New regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
- Docs update:
  - `docs/ImportVerilog-Limitations.md`
    - mark script-evaluable cross-set helper expressions as supported subset.
    - narrow unsupported set-expression statement to non-evaluable procedural forms.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-crossid-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-crossid-supported.sv`
- xrun notation check for the new regression in this slice:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`
- xrun probe notes (not kept as regressions due tool limitation):
  - FAIL: `matches` clause in cross-bin definitions (`*E,COVMNS` / `*E,SVNIMP` in 24.03).

## 2026-02-26

### Task
Extend the non-constant cross set-expression support to cover helper functions that construct `CrossQueueType` via `push_back`, and add xrun-backed regressions.

### Realizations
- Slang constant/script eval fails for these helpers because expression statements returning `void` are treated as eval failure in constant evaluation.
- A targeted ImportVerilog fallback can recover this subset by interpreting function bodies structurally:
  - walk `StatementList` / `BlockStatement`
  - collect `push_back` calls on the function return variable
  - evaluate pushed tuple arguments and synthesize an unpacked tuple list constant

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add structural fallback for cross-set helper calls:
    - `collectCrossSetPushBackTuples`
    - `evaluateCrossSetExprFromPushBackHelper`
  - wire fallback into `evaluateCrossSetExpr` after normal constant + script eval attempts.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support/validation list now includes push_back helper subset and test.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
- xrun notation checks for regressions added in this slice:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue ImportVerilog cross-set gap closure by supporting helper functions that build `CrossQueueType` using `for` loops plus `push_back`.

### Realizations
- TDD baseline: `cross-select-setexpr-function-forloop-pushback-supported.sv` was parsed by xrun but rejected by ImportVerilog as `unsupported non-constant cross set expression`.
- The prior helper fallback handled direct `push_back` expression statements but not loop control / loop-scoped declarations.
- A lightweight statement walker with `EvalContext` can evaluate loop bounds/steps and tuple expressions while collecting pushed tuples.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added `evaluateCrossSetHelperExpr` to evaluate expressions against a live `EvalContext`.
  - extended push-back helper collector with loop/control support:
    - `CrossSetPushBackCollectResult`
    - `collectCrossSetPushBackTuples` now handles:
      - `StatementList` / `BlockStatement`
      - `VariableDeclStatement`
      - `ForLoopStatement` (initializers, loop vars, stop condition, steps)
      - `Break` / `Continue`
      - `ExpressionStatement` `push_back` tuple extraction
  - updated `evaluateCrossSetExprFromPushBackHelper` to:
    - seed formal argument locals from call actuals,
    - run tuple collection under a script-mode eval context with bounded loop budget.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now marks common `for`-loop push_back helper construction as supported and narrows unsupported setexpr helpers to non-statically-extractable control flow / side effects.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
- xrun notation checks (new/updated helper regressions):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue cross-set helper support by adding `while`-driven tuple construction (`push_back`) and validating notation via xrun.

### Realizations
- TDD baseline failed in ImportVerilog (`unsupported non-constant cross set expression`) while xrun successfully elaborated the notation.
- Supporting `for` loops alone was insufficient because helper bodies often include non-`push_back` expression statements (e.g., `i = i + 1`) that must be evaluated, not rejected.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extended statement collector to handle:
    - `WhileLoopStatement` with bounded iteration budget
    - generic evaluable `ExpressionStatement` side effects (assignments, increments, etc.)
  - preserved tuple extraction only for `push_back` calls on the helper return queue.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now includes `while` loop helper construction in supported subset and validation list.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv --ir-moore | build-ot/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
- xrun notation checks (new+helper regressions):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing cross-set helper evaluation gaps by supporting `if`-driven tuple construction in `CrossQueueType` helper functions.

### Realizations
- TDD baseline for `if` helper failed in ImportVerilog while xrun accepted and elaborated the notation.
- The helper collector needed explicit support for `ConditionalStatement` branches.
- During verification, running many `circt-verilog` invocations in parallel intermittently triggered a tool crash (`std::system_error: Invalid argument`); sequential validation avoids this instability.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - include `ConditionalStatements.h`.
  - extend `collectCrossSetPushBackTuples` with `StatementKind::Conditional` handling.
  - evaluate conditional guards with `evaluateCrossSetHelperExpr` and recursively traverse chosen branch.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-if-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now includes `if` helper construction in supported subset and validation list.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions (sequential run):
  - `cross-select-setexpr-function-if-pushback-supported.sv`
  - `cross-select-setexpr-function-while-pushback-supported.sv`
  - `cross-select-setexpr-function-forloop-pushback-supported.sv`
  - `cross-select-setexpr-function-pushback-supported.sv`
  - `cross-select-setexpr-function-literal-supported.sv`
  - plus existing `cross-select-setexpr/with/or` coverage checks
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-if-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue cross-set helper gap closure for `do-while` control flow and ensure xrun-compatible regression coverage.

### Realizations
- `do-while` helper notation is accepted by xrun and is a realistic queue-construction style.
- The collector needed explicit `DoWhileLoopStatement` handling; reusing the existing loop budget keeps behavior bounded.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `StatementKind::DoWhileLoop` support in `collectCrossSetPushBackTuples`.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-dowhile-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support/validation sections now include `do-while` helper support.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `cross-select-setexpr-function-dowhile-pushback-supported.sv`
  - `cross-select-setexpr-function-if-pushback-supported.sv`
  - `cross-select-setexpr-function-while-pushback-supported.sv`
  - `cross-select-setexpr-function-forloop-pushback-supported.sv`
  - `cross-select-setexpr-function-pushback-supported.sv`
  - `cross-select-setexpr-function-literal-supported.sv`
  - `cross-select-setexpr-supported.sv`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-dowhile-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-if-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue cross-set helper control-flow support with `repeat` loops in `CrossQueueType` helper functions.

### Realizations
- `repeat`-based helper construction is accepted by xrun and common in compact helper code.
- The collector had no `RepeatLoopStatement` branch, so this form still failed despite other loop support.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `StatementKind::RepeatLoop` support in `collectCrossSetPushBackTuples`.
  - evaluate repeat count via helper expression evaluator, require non-negative constant integer, iterate body with loop budget.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-repeat-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support/validation sections now include `repeat` helper support.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `cross-select-setexpr-function-repeat-pushback-supported.sv`
  - `cross-select-setexpr-function-dowhile-pushback-supported.sv`
  - `cross-select-setexpr-function-if-pushback-supported.sv`
  - `cross-select-setexpr-function-while-pushback-supported.sv`
  - `cross-select-setexpr-function-forloop-pushback-supported.sv`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-repeat-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-dowhile-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-if-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing cross-set helper extraction gaps by supporting `case` and
`foreach` statement forms in `CrossQueueType` helper functions.

### Realizations
- `xrun` accepts both helper styles and elaborates them cleanly, making them
  good oracle-backed regressions for notation and intent.
- Existing helper extraction already had robust loop/branch plumbing; adding
  `CaseStatement::getKnownBranch(...)` and static-range `foreach` recursion was
  sufficient to unlock both forms.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `StatementKind::Case` support in `collectCrossSetPushBackTuples`
    using `CaseStatement::getKnownBranch(evalContext)`.
  - add `StatementKind::ForeachLoop` support in
    `collectCrossSetPushBackTuples` for static-range dimensions, including
    iterator local creation/update and bounded recursive iteration.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support and validation lists now include
    `case` and `foreach` helper construction.

### Validation
- TDD baseline (before patch):
  - PASS (`xrun`): both new regressions elaborate
  - FAIL (`circt-verilog`): both produced
    `error: unsupported non-constant cross set expression`
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv`
  - full helper subset sweep:
    - `cross-select-setexpr-function-literal-supported.sv`
    - `cross-select-setexpr-function-pushback-supported.sv`
    - `cross-select-setexpr-function-if-pushback-supported.sv`
    - `cross-select-setexpr-function-forloop-pushback-supported.sv`
    - `cross-select-setexpr-function-while-pushback-supported.sv`
    - `cross-select-setexpr-function-dowhile-pushback-supported.sv`
    - `cross-select-setexpr-function-repeat-pushback-supported.sv`
    - `cross-select-setexpr-supported.sv`
    - `cross-select-with-supported.sv`
    - `cross-select-with-nested-or-supported.sv`
    - `cross-select-or-supported.sv`
- xrun notation checks (new regressions in this slice):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue cross-set helper support by handling explicit `return` of helper-local
`CrossQueueType` values (not only direct pushes into the function return var).

### Realizations
- Slang queue method calls in this context carry the queue target in
  `CallExpression::arguments()[0]`; relying only on `thisClass()` is not robust.
- Queue `push_back` calls return `void` (bad `ConstantValue`), so helper
  extraction must preserve side effects without treating that as hard failure.
- Correct `return` behavior needs control-flow propagation (early function exit)
  in the statement walker.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extend `CrossSetPushBackCollectResult` with `Return`.
  - propagate `Return` through loop walkers and accept terminal `Return` in
    `evaluateCrossSetExprFromPushBackHelper`.
  - add tuple extraction from explicit return values:
    - `extractCrossSetTuplesFromValue(...)`
    - `StatementKind::Return` now supports `return <queue-like-expression>;`
      in helper extraction.
  - update `push_back` handling:
    - detect function-return-variable target via `arguments()[0]`.
    - preserve side-effect evaluation for non-return queue `push_back` calls.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now notes explicit local-queue return
    support and includes the new regression in validation list.

### Validation
- TDD baseline (before patch):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv -elaborate -nolog`
  - FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv --ir-moore`
    (`error: unsupported non-constant cross set expression`)
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv`
  - full helper + cross-select sweep (including new case/foreach regressions):
    - `cross-select-setexpr-function-case-pushback-supported.sv`
    - `cross-select-setexpr-function-foreach-pushback-supported.sv`
    - `cross-select-setexpr-function-return-local-queue-supported.sv`
    - `cross-select-setexpr-function-literal-supported.sv`
    - `cross-select-setexpr-function-pushback-supported.sv`
    - `cross-select-setexpr-function-if-pushback-supported.sv`
    - `cross-select-setexpr-function-forloop-pushback-supported.sv`
    - `cross-select-setexpr-function-while-pushback-supported.sv`
    - `cross-select-setexpr-function-dowhile-pushback-supported.sv`
    - `cross-select-setexpr-function-repeat-pushback-supported.sv`
    - `cross-select-setexpr-supported.sv`
    - `cross-select-with-supported.sv`
    - `cross-select-with-nested-or-supported.sv`
    - `cross-select-or-supported.sv`
- xrun notation checks (new regressions in this slice set):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing cross-set helper evaluation gaps for queue mutator and
return-assignment forms that are xrun-valid but still rejected by ImportVerilog.

### Realizations
- Remaining xrun-valid failures were concentrated in helper methods using:
  - `push_front`
  - `insert`
  - `mk = q` return-variable assignment path (without explicit `return q`)
- In slang method-call lowering for queues, helper target queue is represented
  in `CallExpression::arguments()[0]`, not reliably via `thisClass()`.
- To support assignment-based return flow, helper extraction must materialize
  and read the function return local variable after statement execution.

### TDD Baseline
Temporary probes (all `xrun` PASS, `circt-verilog` FAIL with
`unsupported non-constant cross set expression`):
- `/tmp/iv_gap_probes/push_front.sv`
- `/tmp/iv_gap_probes/insert.sv`
- `/tmp/iv_gap_probes/assign_retvar.sv`
- `/tmp/iv_gap_probes/local_queue_push_front_return.sv`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extend helper expression-statement handling to model queue mutators:
    - `push_back`
    - `push_front`
    - `insert`
  - keep side-effect pass-through support for queue void methods
    (`delete` / `sort` / `rsort` / `shuffle` / `reverse`) in helper mode.
  - initialize helper function return local (`returnValVar`) before statement
    walk and read it at function exit for tuple extraction.
  - support explicit return-value propagation by writing returned value into
    return local in `StatementKind::Return`.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-pushfront-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-insert-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-returnvar-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-pushfront-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support text and validation list now
    include queue mutator + return-assignment helper support.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regressions:
  - each new regression above via:
    - `build_test/bin/circt-verilog <test> --ir-moore | llvm/build/bin/FileCheck <test>`
  - full helper + cross-select sweep re-run:
    - case / foreach / return-local / pushfront / insert / assign-returnvar
      helper tests
    - existing literal/pushback/if/for/while/do-while/repeat helper tests
    - existing `cross-select-{setexpr,with,with-nested-or,or}` tests
- xrun notation checks (new regressions in this slice):
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-pushfront-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-insert-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-returnvar-supported.sv -elaborate -nolog`
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-pushfront-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue helper control-flow support with `forever` loops that terminate via
explicit `break`.

### Realizations
- `forever` + `break` helper construction is xrun-valid and appears in compact
  queue builder styles.
- Existing loop budget machinery made this straightforward: model `forever`
  similarly to other loops and require eventual `break` / `return` before
  budget exhaustion.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv --ir-moore`
  (`error: unsupported non-constant cross set expression`)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `StatementKind::ForeverLoop` support in
    `collectCrossSetPushBackTuples` with bounded iteration and `Break`/`Return`
    handling.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now includes `forever`-with-`break`
    helper support and test coverage list.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv`
- Full helper/cross-select sweep:
  - all existing and newly added helper regressions plus:
    - `cross-select-setexpr-supported.sv`
    - `cross-select-with-supported.sv`
    - `cross-select-with-nested-or-supported.sv`
    - `cross-select-or-supported.sv`
- xrun notation check:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue helper control-flow support with block-local `disable` in
`CrossQueueType` builder helpers.

### Realizations
- `disable <block_label>` inside helper functions is accepted by xrun and can
  materially change emitted tuple sets, so it should be import-evaluable.
- Generic statement eval in slang does not always return clean `Success` for
  these forms, so structural fallback still needs explicit `Disable` propagation
  and block-target handling.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv --ir-moore`
  (`error: unsupported non-constant cross set expression`)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - enhance `evaluateCrossSetExprFromPushBackHelper` with:
    - eval-context initialization helper reuse
    - generic `Statement::eval` attempt before structural walk
    - return-value extraction helper for queue/unpacked tuple return locals
  - extend structural walker result model:
    - add `CrossSetPushBackCollectResult::Disable`
    - handle `StatementKind::Disable` via `stmt.eval(evalContext)` and
      propagate disable state
    - teach `StatementKind::Block` handling to consume disable when target
      matches `block.blockSymbol`
    - propagate `Disable` through loop/foreach control paths.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now lists block-local `disable` helper
    support and validation coverage.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- CIRCT regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv`
- Full helper/cross-select sweep:
  - all `cross-select-setexpr-function-*.sv` regressions
  - plus `cross-select-setexpr-supported.sv`,
    `cross-select-with-supported.sv`,
    `cross-select-with-nested-or-supported.sv`,
    `cross-select-or-supported.sv`
- xrun notation checks:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv -elaborate -nolog`
  - PASS: full `cross-select-setexpr-function-*.sv` xrun sweep.

## 2026-02-26

### Task
Continue closing remaining xrun-valid cross-set helper gaps:
- dynamic `foreach` over local dynamic arrays
- helper-to-helper queue return calls (`return sub(lim);`)

### Realizations
- Nested helper calls need actual arguments evaluated in the *caller* eval
  context; evaluating actuals only in the callee context loses caller locals.
- Direct recursion guards are needed for nested helper extraction to avoid
  self-recursive helper loops.
- Dynamic `foreach` dimensions can be derived by evaluating the foreach array
  expression and indexing into container values by already-bound iterator dims.

### TDD Baseline
- Probe `foreach_dynamic.sv`:
  - PASS: `xrun -sv /tmp/iv_gap_probes6/foreach_dynamic.sv -elaborate -nolog`
  - FAIL: `build_test/bin/circt-verilog /tmp/iv_gap_probes6/foreach_dynamic.sv --ir-moore`
    (`unsupported non-constant cross set expression`)
- Probe `helper_calls_helper.sv`:
  - PASS: `xrun -sv /tmp/iv_gap_probes6/helper_calls_helper.sv -elaborate -nolog`
  - FAIL: unsupported / unstable ImportVerilog helper extraction (resolved in this slice).

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extend `evaluateCrossSetExprFromPushBackHelper`:
    - add optional caller eval-context for evaluating nested call actuals
    - add optional excluded-subroutine guard for recursion cut-off
  - extend `StatementKind::Return` handling in helper collector:
    - when return expression eval fails and is a call, attempt nested helper
      extraction with caller context (`return sub(...)` support).
  - extend `StatementKind::ForeachLoop` handling:
    - support dynamic dimensions (`loopDim.range == nullopt`) by computing
      dimension size from evaluated container values and iterating indices.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-dynamic-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-call-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now includes dynamic-foreach helper
    support and helper-to-helper return-call support.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-dynamic-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-dynamic-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-call-supported.sv`
- Full sweep:
  - all `test/Conversion/ImportVerilog/cross-select-setexpr-function-*.sv`
  - plus `cross-select-setexpr-supported.sv`, `cross-select-with-supported.sv`,
    `cross-select-with-nested-or-supported.sv`, `cross-select-or-supported.sv`
- xrun notation checks:
  - PASS: full `cross-select-setexpr-function-*.sv` xrun elaboration sweep.

## 2026-02-26

### Task
Close remaining xrun-valid nested-helper assignment gap:
`mk = sub(lim);` in `CrossQueueType` helper functions.

### Realizations
- Supporting only `return sub(lim);` is not enough; practical helper code also
  uses assignment-form returns (`function_name = helper_call(...)`).
- This requires nested-helper fallback from failed assignment expression eval,
  not just from `ReturnStatement` expressions.

### TDD Baseline
- Probe `/tmp/iv_gap_probes7/helper_assign_call.sv`:
  - PASS: `xrun -sv /tmp/iv_gap_probes7/helper_assign_call.sv -elaborate -nolog`
  - FAIL: `build_test/bin/circt-verilog /tmp/iv_gap_probes7/helper_assign_call.sv --ir-moore`
    (`unsupported non-constant cross set expression`)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - in helper `ExpressionStatement` handling, when assignment evaluation fails
    and RHS is a call, try nested helper extraction on RHS and write resulting
    value into the assignment LHS local symbol.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-assign-call-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support text + validation list now
    include assignment-form helper-to-helper calls.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-assign-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-assign-call-supported.sv`
- xrun notation:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-assign-call-supported.sv -elaborate -nolog`
- Full sweep:
  - all `cross-select-setexpr-function-*.sv`
  - plus `cross-select-setexpr-supported.sv`,
    `cross-select-with-supported.sv`,
    `cross-select-with-nested-or-supported.sv`,
    `cross-select-or-supported.sv`.

## 2026-02-26

### Task
Continue closing xrun-valid ImportVerilog cross-select gaps by fixing `with`
lowering for explicit coverpoint bins so selection is done on candidate
*bin tuples* (IEEE 1800-2023 §19.6.1.2), not only value tuples.

### Realizations
- Current lowering was value-tuple based; this is equivalent only when all
  coverpoint bins are singleton (auto-bin-like), but under-selects semantics
  for explicit multi-value bins.
- In this xrun 24.03 environment:
  - `with` filters on labeled coverpoints are accepted when using cross-item
    names (`cp_a`, `cp_b`) in the filter expression.
  - `cross_identifier`-only select (`bins all = X;`) and `with ... matches ...`
    are parser-limited in xrun and cannot be used for notation validation.

### TDD Baseline
- Added regression: `test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv -elaborate -nolog`
- FAIL (before fix): FileCheck expected bin-tuple lowering (bin-symbol refs),
  but ImportVerilog emitted value-level `intersect` tuples for `with`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added finite cross *bin-domain* construction that prefers explicit
    coverpoint bins and falls back to finite auto-bin domains.
  - added bin-tuple condition evaluation for cross select expressions.
  - changed `with` evaluation to:
    - evaluate subordinate select expression on candidate bin tuples
    - count satisfying value tuples per candidate bin tuple
    - apply parsed `matches` policy against that count.
  - extended finite select-expression fallback (`containsWithOrSetExpr`) to use
    bin-tuple evaluation and emit bin-symbol `moore.binsof` filters when
    explicit bins are selected.
  - cleaned now-unused value-tuple helper paths that were replaced.
- `lib/Conversion/MooreToCore/MooreToCore.cpp`:
  - when lowering `moore.binsof` filters, detect `@coverpoint::@bin` targets
    and populate runtime `bin_indices` / `num_bins` instead of dropping bin
    specificity.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` support/limitations text + validation
    list updated for explicit-bin `with` support.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv`
- Cross-select regression sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`
- xrun notation check:
  - PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing xrun-valid `cross_set_expression` helper extraction gaps by
supporting conditional helper-call expressions in return and assignment forms:
- `return cond ? h1() : h2();`
- `mk = cond ? h1() : h2();`

### Realizations
- Existing fallback only handled direct call expressions in return/assignment.
- xrun accepts ternary helper-call forms in cross helper functions; this is a
  practical coding style for compact queue construction.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv -elaborate -nolog`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv -elaborate -nolog`
- FAIL: both new regressions in ImportVerilog with
  `error: unsupported non-constant cross set expression`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `evaluateCrossSetExprWithFallback(...)` helper that extends
    queue-helper fallback from direct calls to conditional (`?:`) expressions by
    evaluating the condition in helper eval context and recursively extracting
    the selected branch.
  - use this fallback in helper `ReturnStatement` handling and in assignment
    expression handling (`mk = ...`) when direct evaluation fails.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` validation list includes both new
    regression tests.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue closing xrun-valid helper-expression gaps for `cross_set_expression`
extraction by supporting wrapper expressions around helper calls:
- casted helper return: `return CrossQueueType'(h1());`
- concatenated helper returns: `return {h1(), h2()};`

### Realizations
- Existing fallback handled direct call / conditional forms, but not
  `ConversionExpression` or `ConcatenationExpression` wrappers.
- xrun elaborates both forms, so this is a real notation/semantics gap in the
  importer.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv -elaborate -nolog`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv -elaborate -nolog`
- FAIL: both regressions in ImportVerilog with
  `error: unsupported non-constant cross set expression`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extend `evaluateCrossSetExprWithFallback(...)` to handle:
    - `ConversionExpression` by recursively extracting from the converted
      operand when direct eval fails.
    - `ConcatenationExpression` by evaluating each operand (direct or fallback)
      and concatenating queue/unpacked elements into a tuple list value.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` validation list includes both new
    regressions.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`

### Additional Probe Sweep
- Re-ran a focused xrun-first probe set after landing cast/concat fallback
  support (`paren_return`, nested conditional variants, local-assign variants,
  mixed/typed concatenation variants).
- No additional xrun-pass / circt-fail helper-expression forms were found in
  that probe matrix beyond the forms already converted to regressions above.

## 2026-02-26

### Task
Close remaining xrun-valid helper extraction gap for local declaration initializers:
- `CrossQueueType t = h1();`

### Realizations
- Helper-call fallback was wired for `return ...` and assignment statements, but
  not for `VariableDeclStatement` initializers.
- This caused valid helper code that declares a local queue from a helper call
  and then returns/mutates that local queue to fail static tuple extraction.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv --ir-moore`
  with `error: unsupported non-constant cross set expression`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - in `StatementKind::VariableDeclaration`, when direct initializer eval fails,
    apply `evaluateCrossSetExprWithFallback(...)` (with recursion guard) before
    rejecting.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv -elaborate -nolog`

## 2026-02-26

### Task
Close remaining xrun-valid helper extraction gap for `for` initializer / step
assignment expressions that call helper functions:
- `for (..., mk = h1(); ...; ..., mk = h2()) ...`

### Realizations
- `ForLoopStatement` initializers and steps were evaluated only through
  `evaluateCrossSetHelperExpr`, so assignment RHS helper calls that need
  structural fallback were rejected.
- The same assignment fallback logic used in statement expressions should be
  applied to loop initializer/step expressions.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv --ir-moore`
  with `error: unsupported non-constant cross set expression`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `evaluateForExpr` helper in `StatementKind::ForLoop` handling:
    - try direct eval
    - for assignments, fallback-evaluate RHS helper expression and assign LHS
      local symbol
    - fallback-evaluate other expression wrappers when needed
  - use `evaluateForExpr` for both `for` initializers and step expressions.
  - extend loop variable initializer handling to use the same fallback path when
    direct eval fails.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-step-assign-call-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-step-assign-call-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-step-assign-call-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-step-assign-call-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`

## 2026-02-26

### Task
Close semantic gap in finite `with` lowering for coverpoints that define explicit
`default` bins.

### Realizations
- `buildFiniteCrossBinDomains(...)` skipped `default` bins entirely and used only
  non-default explicit bins when present.
- This made `with` lowering semantically incomplete: expressions that should
  select default-bin tuples were under-selected (in the probe case, lowered to
  an always-false `moore.binsof ... negate`).

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck ...`
  with missing expected `moore.binsof @a_cp::@other` (tool emitted
  `moore.binsof @a_cp negate`).

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `buildFiniteIntegralCoverpointDomain(...)` helper for finite integral
    value-domain construction.
  - in `buildFiniteCrossBinDomains(...)`:
    - collect non-default explicit bin values as before,
    - collect `default` bins separately,
    - when default bins are present, compute finite complement of covered values
      over the coverpoint domain and add that value set for each default bin
      domain entry.
  - keep auto-bin fallback path unchanged when no explicit/default bins are
    available.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`

## 2026-02-26

### Task
Close remaining xrun-valid `with`-path gaps for cross select expressions that
reference `ignore_bins` / `illegal_bins` coverpoint bins.

### Realizations
- `buildFiniteCrossBinDomains(...)` still filtered domain entries to
  `CoverageBinSymbol::Bins` only.
- In finite `with` evaluation, this caused bin targets like
  `binsof(a_cp.ig)` / `binsof(a_cp.ill)` to be absent from the domain and thus
  under-selected to an always-false lowering (`moore.binsof @a_cp negate`).

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv -elaborate -nolog`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv -elaborate -nolog`
- FAIL: both regressions in ImportVerilog FileCheck with missing expected
  `@a_cp::@ig` / `@a_cp::@ill` filters and observed fallback
  `moore.binsof @a_cp negate`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - in `buildFiniteCrossBinDomains(...)`, remove bin-kind restriction to
    `CoverageBinSymbol::Bins` so finite domains include non-default
    `ignore_bins` / `illegal_bins` (subject to existing finite-value checks).
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`

## 2026-02-26

### Task
Close xrun-valid finite-`with` gap where unrelated unsupported coverpoint bin
shapes (transition bins or coverpoint-bin `with`/set forms) caused hard failure
for select expressions that only reference finite plain bins.

### Realizations
- `buildFiniteCrossBinDomains(...)` attempted to enumerate every bin in each
  cross target coverpoint and failed immediately on unsupported bin shapes.
- This was too strict for cases like:
  - coverpoint has `{ bins n = ...; bins tr = (2=>3); }`
  - cross select uses only `binsof(a_cp.n) with (...)`
  - xrun elaborates these forms, but ImportVerilog rejected due unrelated `tr`.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv -elaborate -nolog`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv -elaborate -nolog`
- FAIL (before fix): ImportVerilog rejected both with:
  - `unsupported transition bin target in cross select expression`
  - `unsupported 'with' coverpoint bin target in cross select expression`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add `CrossSelectBinRequirements` +
    `collectCrossSelectBinRequirements(...)` to track which coverpoint/bin
    targets are semantically required by the current select expression.
  - extend `buildFiniteCrossBinDomains(...)` to accept requirement info and:
    - skip unsupported bin shapes when they are not required,
    - keep hard diagnostics when unsupported bin shapes are required by the
      expression semantics.
  - wire requirement collection through:
    - `emitWithFilterBinsSelect(...)`
    - `emitFiniteTupleCrossSelect(...)`
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do build_test/bin/circt-verilog "$f" --ir-moore | llvm/build/bin/FileCheck "$f"; done`

### Additional Diagnostic Regressions (same slice)
- Added explicit diagnostic coverage to ensure unsupported *required* non-finite
  bin targets stay hard-failed (instead of silently approximated):
  - `test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-unsupported.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-unsupported.sv -elaborate -nolog`
- CIRCT diagnostic checks:
  - `not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv`
  - `not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-unsupported.sv`

## 2026-02-26

### Task
Close remaining xrun-valid finite-`with` gaps for:
- wider finite coverpoint domains used in `with` filtering (width 9)
- required coverpoint bins declared with `with (...)` clauses

### Realizations
- The finite domain width cap for auto/default coverpoint-domain expansion was
  still 8 bits, even though the tuple/value cap is 4096 (12-bit domain).
- Required coverpoint-bin `with` targets were still hard-rejected, but xrun
  elaborates both range-based `with` bins and id-with bins in cross-select
  `with` forms.
- For required bin-with targets, ImportVerilog only needs finite value sets for
  import-time tuple filtering; the emitted cross selection still references the
  bin symbol (`moore.binsof @cp::@bin`).

### TDD Baseline
- PASS: `xrun -sv /tmp/iv_probe_bin_with_id_target.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog /tmp/iv_probe_bin_with_id_target.sv --ir-moore`
  with `unsupported 'with' coverpoint bin target in cross select expression`
- PASS: xrun elaboration on width-9 probes (`/tmp/iv_probe_with_wide_auto_domain.sv`,
  `/tmp/iv_probe_with_wide_default_bin.sv`)
- FAIL (before width-cap bump): ImportVerilog rejected with
  `maximum supported width is 8`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - keep finite domain cap aligned with 4096-value limit (`kMaxDomainBits = 12`)
    for auto/default coverpoint-domain expansion.
  - add finite evaluation support for required coverpoint bins declared with
    `with (...)`:
    - collect explicit bin values via a shared helper,
    - for id-with bins (no explicit values), derive finite base domain from the
      coverpoint type,
    - evaluate bin `with` predicates over candidate values using `EvalContext`
      iterator bindings,
    - use filtered finite values for tuple selection.
  - keep hard diagnostics for required transition/set/default-sequence targets.
  - preserve requirement-aware skipping for unreferenced bin-with targets.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-wide-default-bin-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-supported.sv`
  - `test/Conversion/ImportVerilog/cross-select-with-coverpoint-id-with-target-supported.sv`
- Removed obsolete negative coverage:
  - replaced `cross-select-with-coverpoint-bin-with-target-unsupported.sv`
    with supported coverage (`...-supported.sv`).

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-wide-default-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-wide-default-bin-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-coverpoint-id-with-target-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-coverpoint-id-with-target-supported.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-wide-default-bin-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-coverpoint-id-with-target-supported.sv -elaborate -nolog`
- Boundary retention:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv -elaborate -nolog`
  - `llvm/build/bin/not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv`
- Cross-select supported sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do ...; done`
  - Observed one transient local crash during an initial sweep invocation;
    rerun completed and all supported cross-select tests passed FileCheck.

## 2026-02-26

### Task
Close xrun-valid cross-select `with` gap for required coverpoint bins declared
with set-expression initializers (for example `bins even = vals;`).

### Realizations
- xrun elaborates required set-expression bin targets in cross-select `with`
  forms, but ImportVerilog still hard-failed with
  `unsupported set coverpoint bin target in cross select expression`.
- `evaluateCrossSetExpr(...)` handled many expression forms, but direct symbol
  references to initialized variables (e.g. `vals`) required one more fallback:
  evaluate the referenced symbol initializer when direct evaluation is bad.

### TDD Baseline
- PASS: `xrun -sv /tmp/iv_probe_bin_set_target.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog /tmp/iv_probe_bin_set_target.sv --ir-moore`
  with `unsupported set coverpoint bin target in cross select expression`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add finite extraction for set-expression coverpoint bins via
    `collectCoverageSetExprValues(...)`.
  - allow required set-expression bins in finite domain construction; keep
    requirement-aware skipping for unreferenced set bins.
  - add symbol-reference initializer fallback for set-expression evaluation.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-coverpoint-set-target-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-coverpoint-set-target-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-coverpoint-set-target-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-coverpoint-set-target-supported.sv -elaborate -nolog`
- Re-validation of this slice's new regressions:
  - `cross-select-with-wide-auto-domain-supported.sv`
  - `cross-select-with-wide-default-bin-supported.sv`
  - `cross-select-with-coverpoint-bin-with-target-supported.sv`
  - `cross-select-with-coverpoint-id-with-target-supported.sv`
  - `cross-select-with-coverpoint-set-target-supported.sv`
  - each checked with both `circt-verilog ... | FileCheck` and xrun elaboration
- Boundary retention:
  - `llvm/build/bin/not build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-transition-target-unsupported.sv -elaborate -nolog`

## 2026-02-26

### Task
Close remaining xrun-valid cross-select `with` gaps for required transition-bin
and `default sequence` bin targets.

### Realizations
- Required transition-bin targets (`binsof(a_cp.tr) with (...)`) were still
  hard-failed even though xrun elaborates them.
- Required `default sequence` targets are xrun-valid too; they do not always
  require finite value-domain enumeration (for example when the `with` filter
  references only other coverpoints).
- Existing `with` tuple counting evaluated Cartesian products across all
  dimensions, even when the filter references only a subset of coverpoints.
  This was stricter than needed and blocked value-less bin targets.

### TDD Baseline
- PASS: `xrun -sv /tmp/iv_probe_default_sequence_target.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog /tmp/iv_probe_default_sequence_target.sv --ir-moore`
  with `unsupported default sequence coverpoint bin target in cross select expression`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv -elaborate -nolog`
- FAIL (before fix): same case previously failed with
  `unsupported transition bin target in cross select expression`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - add finite transition-bin value extraction from transition range items.
  - allow required transition-bin targets in finite cross-domain construction.
  - extend requirement tracking with value-domain needs:
    - `valueBins`
    - `valueCoverpoints`
    driven by intersects, set-expr, and `with` iterator usage.
  - allow required `default sequence` targets when no value-domain evaluation
    is required by the current expression; keep hard diagnostics when value
    domains are semantically required.
  - update `with` tuple counting to enumerate only coverpoint dimensions that
    are actually referenced by `with` iterators.
- Added regressions:
  - `test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv`
    (renamed from previous `...-unsupported.sv` and converted to positive)
  - `test/Conversion/ImportVerilog/cross-select-with-default-sequence-target-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-sequence-target-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-default-sequence-target-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-sequence-target-supported.sv -elaborate -nolog`
- Cross-select sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do ...; done`
  - PASS for supported tests (same existing local warning in
    `cross-select-setexpr-function-local-decl-init-call-supported.sv`).
- xrun sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-with-*.sv; do xrun -sv "$f" -elaborate -nolog; done`
  - PASS for all current `cross-select-with-*` regressions.

## 2026-02-26

### Task
Extend `default sequence` cross-select `with` support from value-independent
filters to value-dependent filters over finite integral coverpoint domains.

### Realizations
- xrun elaborates required `default sequence` target filters that reference the
  same coverpoint value (`binsof(a_cp.ds) with (a_cp > 0)`), while ImportVerilog
  still rejected these after the previous slice.
- For finite integral coverpoint widths, a practical finite-domain abstraction
  for `default sequence` bins enables value-dependent `with` evaluation and keeps
  lowering in the same finite tuple framework.

### TDD Baseline
- PASS: `xrun -sv /tmp/iv_probe_default_sequence_value_needed.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog /tmp/iv_probe_default_sequence_value_needed.sv --ir-moore`
  with `unsupported default sequence coverpoint bin target in cross select expression`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - in finite cross-bin domain construction, required `default sequence` bins
    now use finite integral coverpoint domains when value evaluation is required.
  - keep symbol-only default-sequence entries for value-independent cases.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-default-sequence-value-filter-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-sequence-value-filter-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-default-sequence-value-filter-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-sequence-value-filter-supported.sv -elaborate -nolog`
- Re-validation:
  - `cross-select-with-default-sequence-target-supported.sv`
  - `cross-select-with-transition-target-supported.sv`
  - both pass `circt-verilog | FileCheck` and xrun elaboration.
- Sweeps:
  - supported cross-select FileCheck sweep (`cross-select-*.sv`, excluding
    `*-unsupported.sv`) passes.
  - xrun elaboration sweep for all `cross-select-with-*.sv` passes.

## 2026-02-26

### Task
Close remaining silent-degradation gap for cross-select condition leaves:
`binsof(...) intersect { ... }` entries that are non-constant / unbounded under
IEEE 1800-2023 should be diagnosed instead of dropped.

### Realizations
- `evaluateIntersectList(...)` already emits hard errors for non-constant
  intersect values/ranges, but `emitBinsOfCondition(...)` previously rebuilt
  intersect arrays independently and silently ignored unsupported entries.
- This affected DNF-lowered condition leaves (for example plain
  `binsof(a) intersect {[0:$]}`), which could degrade to unfiltered
  `moore.binsof`.

### TDD Baseline
- Added failing regressions (both initially failed because no diagnostic was
  emitted and IR was produced):
  - `test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv`
  - `test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`:
  - make `emitBinsOfCondition(...)` require constant finite intersect
    ranges/values.
  - emit:
    - `unsupported non-constant intersect value range in cross select expression`
    - `unsupported non-constant intersect value in cross select expression`
  - cap finite expansion at 4096 values and emit existing
    `unsupported cross select expression due to large finite value set`.
- Added regressions:
  - `cross-select-intersect-open-range-unsupported.sv`
  - `cross-select-intersect-plusminus-unsupported.sv`
- Updated limitations doc:
  - `docs/ImportVerilog-Limitations.md` (`Cross Bin Select Expressions` section)
    to include the non-constant/unbounded intersect-range limitation and list
    the new tests.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regressions:
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv`
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv`
- No-regression spot checks:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-or-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-or-supported.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-supported.sv`
- Lit:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv build_test/test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv build_test/test/Conversion/ImportVerilog/cross-select-or-supported.sv build_test/test/Conversion/ImportVerilog/cross-select-with-supported.sv`

## 2026-02-26

### Task
Refactor cross-select lowering internals for maintainability:
1) centralize coverage bin kind mapping
2) split cross-select lowering into a dedicated translation unit
4) de-duplicate finite bin value collection logic

### Realizations
- `Structure.cpp` contained a very large cross-select lowering block that made
  covergroup conversion hard to navigate.
- duplicated `CoverageBinSymbol::binsKind` switching in two places produced
  warning-prone code (`binKind may be used uninitialized`).
- finite bin-domain extraction had repeated cap/sort/empty handling across
  explicit, transition, and set-expression paths.

### Changes Landed In This Slice
- Added dedicated cross-select lowering unit:
  - `lib/Conversion/ImportVerilog/CrossSelect.cpp`
  - moved `convertBinsSelectExpr(...)` and all related helper machinery there.
- Wired build + declaration:
  - add `CrossSelect.cpp` to `lib/Conversion/ImportVerilog/CMakeLists.txt`
  - add exported declaration of `convertBinsSelectExpr(...)` to
    `ImportVerilogInternals.h`
- Centralized bin-kind conversion:
  - add `convertCoverageBinKind(...)` helper in `Structure.cpp`
  - replaced duplicate switches for coverpoint bins and cross bins.
- De-duplicated finite value-collection internals in `CrossSelect.cpp`:
  - `appendFiniteCoverageBinValue(...)`
  - `appendFiniteCoverageBinRange(...)`
  - `finalizeFiniteCoverageBinValues(...)`
  - reused by explicit, transition, set-expression, and bin-with filtering
    value collection paths.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Cross-select regression sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*.sv; do ...; done`
  - PASS for supported tests (same existing warning in
    `cross-select-setexpr-function-local-decl-init-call-supported.sv`).
- xrun parity sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-with-*.sv; do xrun -sv "$f" -elaborate -nolog; done`
  - PASS for all `cross-select-with-*` regressions.
- Note:
  - observed one transient local `Permission denied` launching freshly linked
    `build_test/bin/circt-verilog`; immediate rerun succeeded.

## 2026-02-26

### Task
Continue cross-select intersect-range parity by supporting open-ended ranges in
condition leaves when they are resolvable to finite target domains.

### Realizations
- xrun elaborates `binsof(a_cp) intersect {[4:$]}` for finite integral
  coverpoints, but ImportVerilog still hard-failed these as
  `unsupported non-constant intersect value range`.
- For condition-leaf lowering (`emitBinsOfCondition`), open-ended ranges can be
  finite-expanded when target coverpoint domain is finite and small enough.
- The tuple-evaluation path (`evaluateIntersectList`) can support open-ended
  ranges without expansion by comparing against one-sided bounds.

### TDD Baseline
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv -elaborate -nolog`
- FAIL: `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv --ir-moore`
  with `unsupported non-constant intersect value range in cross select expression`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`:
  - add `isUnboundedConstantExpr(...)` helper.
  - extend `evaluateIntersectList(...)` to handle unbounded one-sided /
    two-sided ranges (`[x:$]`, `[$:y]`, `[$:$]`).
  - extend `emitBinsOfCondition(...)` to finite-expand unbounded range bounds
    when target coverpoint domain is finite integral (`<= 12` bits), preserving
    existing diagnostics when not resolvable.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv -elaborate -nolog`
- Boundary retention:
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv`
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv`
- Sweep:
  - supported cross-select FileCheck sweep passes (`cross-select-*.sv`,
    excluding `*-unsupported.sv`).

## 2026-02-26

### Task
Continue cross-select intersect parity by fixing tolerance-range (`+/-`, `+%-`) lowering semantics in condition leaves.

### Realizations
- Slang exposes tolerance ranges via `ValueRangeExpression::rangeKind`:
  - `AbsoluteTolerance` for `[A +/- B]`
  - `RelativeTolerance` for `[A +%- B]`
- ImportVerilog previously treated all value ranges as simple `[L:R]`, which mis-lowered tolerance syntax (for example `[8 +/- 3]` became `[3:8]`).
- Relative tolerance expression typing can be surprising for narrow integral contexts (the tolerance operand may context-convert), so the positive regression should use `int` coverpoints to avoid width-truncation artifacts.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv`
- Baseline failure (before fix):
  - `[8 +/- 3]` lowered to `[3, 4, 5, 6, 7, 8]`
  - `[8 +%- 25]` lowered incorrectly

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`:
  - add `evaluateIntersectToleranceRangeBounds(...)`
  - handle `ValueRangeKind::AbsoluteTolerance` and
    `ValueRangeKind::RelativeTolerance` in both:
    - `evaluateIntersectList(...)` (tuple-evaluation path)
    - `emitBinsOfCondition(...)` (DNF condition-leaf lowering path)
  - preserve existing diagnostics for non-constant ranges.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv`
- Updated docs:
  - `docs/ImportVerilog-Limitations.md` now lists constant tolerance range
    support and includes the new regression in validation coverage.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv`
- Boundary retention:
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv`
  - `llvm/build/bin/not build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv --ir-moore 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv`
- Existing supported case re-check:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv -elaborate -nolog`
  - FAIL (`*E,ILLPRI`) in Xcelium 24.03 parser for `+/-` / `+%-` covergroup range tokens (tool limitation).

## 2026-02-26

### Task
Continue ImportVerilog gap closure by converting stale TODO-only areas into explicit
regression coverage with xrun parity checks.

### Realizations
- Class-parameter access through an instance handle (`obj.P`) is already lowered as
  a constant in current ImportVerilog; the old TODO in `class-parameters.sv` was stale.
- `$realtime()` to `real` conversion is already implemented end-to-end via
  `time -> logic -> int -> uint_to_real -> fdiv(timescale)`, but did not have a
  focused regression.

### Changes Landed In This Slice
- Updated regression to assert supported class parameter instance constant access:
  - `test/Conversion/ImportVerilog/class-parameters.sv`
  - uncommented `$display("a = %d", test_obj.a);`
  - added `FileCheck` for constant-folded value `34` and display emission.
- Added focused regression for realtime-to-real conversion:
  - `test/Conversion/ImportVerilog/realtime-to-real-conversion.sv`
  - checks `moore.builtin.time` -> `moore.time_to_logic` -> `moore.logic_to_int`
    -> `moore.uint_to_real` -> `moore.fdiv` -> call to real sink.

### Validation
- CIRCT regressions:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/class-parameters.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/class-parameters.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/realtime-to-real-conversion.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/realtime-to-real-conversion.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/class-parameters.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/realtime-to-real-conversion.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue closing stale ImportVerilog gap tracking around explicit coverpoint bins by
turning commented "future" cases into active regression coverage.

### Realizations
- Explicit coverpoint bins (`bins low = {[0:3]};` etc.) are already lowered as
  `moore.coverbin.decl` with concrete value ranges.
- The older note in `coverage_explicit_bins.sv` claiming explicit bins were not
  fully supported was stale.

### Changes Landed In This Slice
- Updated `test/Conversion/ImportVerilog/coverage_explicit_bins.sv`:
  - converted commented-out explicit bin declarations into active syntax.
  - added FileCheck coverage for:
    - `@low` values `[[0, 3]]`
    - `@mid` values `[[4, 11]]`
    - `@high` values `[[12, 15]]`
  - removed redundant late check lines that conflicted with sequential check
    ordering after the new inline coverbin checks.

### Validation
- CIRCT regression:
  - `build_test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/coverage_explicit_bins.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/coverage_explicit_bins.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/coverage_explicit_bins.sv -elaborate -nolog` (PASS)
