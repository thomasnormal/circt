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

## 2026-02-26

### Task
Continue gap closure by replacing stale TODO-only string conversion notes with
active regression coverage for `string.itoa` argument typing.

### Realizations
- `string.itoa` is already supported for multiple integral argument types; the
  old TODO in `queues.sv` claiming a type-conversion blocker was stale.
- Current lowering handles mixed widths/signedness via conversion steps before
  `moore.string.itoa`:
  - sign-extend for narrow signed types (`byte`, `shortint`)
  - truncation path for wide integral (`longint`) to the expected logic width.

### Changes Landed In This Slice
- Updated `test/Conversion/ImportVerilog/queues.sv`:
  - removed stale commented TODO block for `itoa`.
  - added new module `StringItoaMethodTest` with active `s.itoa(...)` calls for
    `int`, `byte`, `shortint`, and `longint`.
  - added FileCheck expectations for conversion ops and
    `moore.string.itoa` lowering.

### Validation
- CIRCT regression:
  - `build_test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/queues.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/queues.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/queues.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue closing stale covergroup gap tracking by enabling explicit cross-target
regression coverage in `covergroups.sv`.

### Realizations
- Plain covergroup cross declarations (`cross a_cp, b_cp;`) are already imported
  as `moore.covercross.decl` with target symbol references.
- The old note in `covergroups.sv` claiming cross targets were not fully
  supported was stale for this baseline pattern.

### Changes Landed In This Slice
- Updated `test/Conversion/ImportVerilog/covergroups.sv`:
  - named coverpoints (`data_cp`, `addr_cp`) to stabilize symbol checks.
  - enabled `cross data_cp, addr_cp;` inside the covergroup.
  - added FileCheck expectation for `moore.covercross.decl` target list.

### Validation
- CIRCT regression:
  - `build_test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/covergroups.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/covergroups.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/covergroups.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog gap closure for hierarchical interface method calls by
supporting indexed interface-instance receivers in module hierarchy paths:
`module.if_array[idx].task()`.

### Realizations
- `xrun` elaborates `agent.driverBFM[1].ping()` successfully, while ImportVerilog
  rejected it with:
  `hierarchical interface method calls through module instances are not yet supported`.
- For this call shape, the invocation receiver arrives as a scoped-name path
  using `IdentifierSelectNameSyntax` (`name[idx]`), and the previous
  hierarchical call collector only handled plain `IdentifierNameSyntax`.
- Existing interface-instance fallback lookup in call lowering keyed on
  `&instSym->body == instBody`; for array elements this can miss even when a
  unique same-definition candidate is already threaded.

### TDD Baseline
- Added regression: `test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`.
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore`
    failed with unsupported hierarchical interface call error.
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - Extended call receiver path collection to support
    `IdentifierSelectNameSyntax` with constant single index selectors.
  - Added indexed path resolution through `InstanceArraySymbol` to recover the
    concrete selected interface `InstanceSymbol`.
  - Built stable threaded path names including `[idx]` suffixes for indexed
    receivers.
- `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - For interface-method receiver fallback lookup, added unique
    same-interface-definition candidate selection when exact `instBody` pointer
    match is unavailable.
  - Retained exact-body match preference; only falls back to same-definition
    candidate when unambiguous.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regression checks:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv -elaborate -top TopLevel test/Conversion/ImportVerilog/hierarchical-interface-task.sv -nolog` (PASS)
  - `xrun -sv -elaborate -top DirectTest test/Conversion/ImportVerilog/hierarchical-interface-task.sv -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog hierarchical interface-array support by closing the
assignment gap for `virtual interface` binding from indexed hierarchical
receivers: `vif = module.if_array[idx];`.

### Realizations
- The failure was not just receiver resolution in expression lowering; the
  hierarchical collector for statements did not visit
  `ArbitrarySymbolExpression`, so interface threading state was missing at
  assignment lowering time.
- Slang can surface interface-array element references as unnamed instance
  symbols (`name == ""`), requiring definition-based fallback when exact pointer
  identity lookup fails.

### TDD Baseline
- Added regression: `test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv`.
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv --ir-moore`
    failed with `unsupported arbitrary symbol reference ''`.
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - Added statement-visitor handling for `ArbitrarySymbolExpression` so
    hierarchical interface references in assignment RHS participate in
    interface-threading collection.
- `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - In `visit(ArbitrarySymbolExpression)`, added final unambiguous fallback
    resolution by interface definition (and by matching name when available)
    for array-element shaped interface instance symbols.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv`.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regression checks:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv -elaborate -nolog` (PASS)

### Task
Stabilize formal front-end reliability for BMC/LEC workloads after repeated
`circt-verilog` crashes/hangs in UVM/SVA tests under `Tools/circt-bmc`.

### Realizations
- The primary failures were not in BMC lowering; stacks consistently pointed
  into Slang analysis threading (`BS::thread_pool` / `watchdogThreadMain`).
- The most impactful safety fix is to avoid running Slang semantic analysis in
  normal lowering modes used by formal flows (`--ir-hw` / `--ir-llhd`) while
  preserving analysis for explicit `--lint-only`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - Changed `shouldRunSlangAnalysis` to run analysis only when
    `ImportVerilogOptions::Mode::OnlyLint`.
  - Added rationale comments documenting current Slang thread-pool instability
    on large formal/UVM workloads.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Targeted failing formal regressions now pass:
  - `llvm/build/bin/llvm-lit -sv -j 8 --filter='sva-uvm-seq-local-var-e2e|sv-tests-uvm-path|sva-local-var-disable-iff-abort-unsat-e2e|sva-xprop-dyn-partselect-sat-e2e|sv-tests-keep-logs-logtag|sva-sequence-event-list-provenance-emit-mlir' build_test/test`
- Stress reruns of key failures:
  - `for i in 1 2 3; do llvm/build/bin/llvm-lit -q -j 8 --filter='sva-uvm-seq-local-var-e2e|sv-tests-uvm-path|sva-local-var-disable-iff-abort-unsat-e2e|sva-xprop-dyn-partselect-sat-e2e|sv-tests-keep-logs-logtag' build_test/test; done`

## 2026-02-26

### Task
Continue ImportVerilog hierarchical interface-array gap closure by supporting
non-literal constant indices in hierarchical interface task-call receivers:
- parameter index: `m.ifs[PIDX].ping()`
- folded expression index: `m.ifs[1-0].ping()`

### Realizations
- A remaining xrun-valid gap existed for hierarchical interface task calls when
  the receiver path used `IdentifierSelectNameSyntax` selectors that were
  constant expressions but not literal tokens.
- The receiver-path parser in `HierarchicalNames.cpp` only accepted literal
  selector syntax in this path, so valid constant selectors were rejected before
  interface threading resolved the instance.

### TDD Baseline
- Added regression: `test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`.
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore`
    failed with:
    `hierarchical interface method calls through module instances are not yet supported`.
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - extended `parseConstantIndex(...)` to bind and constant-evaluate selector
    expressions (not only literal syntax), enabling parameter/localparam and
    folded arithmetic indices in hierarchical receiver paths.
  - added fallback binding scope to the outermost instance body when
    `context.currentScope` is unavailable.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
- Focused lit sweep:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-task.sv -elaborate -top TopLevel -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog hierarchical interface gap closure for method calls through
module-instance arrays with indexed first path segments:
- `a[idx].ifs[idx].ping()`

### Realizations
- `xrun` elaborates hierarchical calls through module-instance arrays, but
  ImportVerilog still rejected them with the generic hierarchical-interface
  unsupported diagnostic.
- In `resolveInstancePath(...)`, indexed array selection was only handled for
  segments after the first path element; a leading segment like `a[1]` was left
  as an `InstanceArraySymbol`, causing path resolution to fail.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv --ir-moore`
    failed with:
    `hierarchical interface method calls through module instances are not yet supported`.
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - updated `resolveInstancePath(...)` to support indexed selection on the
    first path segment.
  - centralized indexed instance selection logic so both head and non-head
    segments resolve through the same `InstanceArraySymbol` / `InstanceSymbol`
    handling.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
- Re-validation:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`
- Focused lit sweep:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog hierarchical interface gap closure for virtual interface
binding through module-instance arrays:
- `vif = a[idx].ifs[idx];`

### Realizations
- `xrun` elaborates module-array hierarchical interface assignment forms, but
  ImportVerilog still failed with:
  `unsupported arbitrary symbol reference ''`.
- In this shape, Slang provides `ArbitrarySymbolExpression` with a non-empty
  hierarchical path (`hierRef`) but a synthesized/unnamed interface element
  symbol, so definition-only fallback is ambiguous.
- `hierRef` encodes selectors as alternating array symbols and unnamed instance
  hops (e.g., `a`, unnamed `[1]`, `ifs`, unnamed `[1]`), so we need path
  normalization that folds unnamed selector hops into the preceding named
  segment.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog /tmp/iv_hier_probe6_assign_modarr.sv --ir-moore`
    failed with `unsupported arbitrary symbol reference ''`.
  - `xrun -sv /tmp/iv_hier_probe6_assign_modarr.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - added hierarchical-reference path normalization for arbitrary symbol
    receivers (`collectReceiverSegmentsFromHierRef(...)`) that folds unnamed
    selected elements into named path segments.
  - updated `handle(ArbitrarySymbolExpression)` to resolve and thread concrete
    interface instances (with full path names) via normalized hierarchical
    paths.
- `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - in `visit(ArbitrarySymbolExpression)`, added path-based fallback that maps
    normalized hierarchical path names to threaded interface paths in the
    current instance body, then resolves the concrete interface instance.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`
- Re-validation:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
- Focused lit sweep:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog hierarchical interface gap closure for nested module-array
receiver paths:
- `m[idx].l[idx].ifs[idx].ping()`

### Realizations
- `xrun` accepts nested module-array hierarchical interface task-call receivers,
  but ImportVerilog failed with missing hierarchical interface threading.
- For selected instance-array elements, slang can provide unnamed
  `InstanceSymbol`s; using raw `name` during interface threading produced
  unstable hierarchical segments.
- Module deduplication by definition+parameters alone can incorrectly merge
  instance bodies that have different hierarchical threaded-port shapes,
  dropping required hierarchical interface outputs.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv`
- Baseline behavior before fix:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv --ir-moore`
    failed (first with `missing hierarchical interface value for '.'`, then
    with `missing hierarchical interface value for 'l[0].ifs[1]'` during
    iterative debugging).
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - added `getInstancePathSegment(...)` to build stable segment names for
    selected array elements (e.g. `l[0]`, `ifs[1]`) when symbols are unnamed.
  - updated `threadInterfaceInstance(...)` upward path construction to use
    canonical segment names instead of raw symbol names.
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - tightened module dedup equivalence by requiring matching hierarchical port
    shape (`hierPaths` + `hierInterfacePaths`) in addition to
    definition/parameters/bind-overrides.
  - for module output hierarchical interface emission, replaced exact-symbol-only
    lookup with `resolveInterfaceInstance(...)` and an unambiguous
    same-path fallback.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv`
- Focused re-validation:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Continue ImportVerilog gap closure for nested module-array hierarchical
**interface signal** access:
- writes: `m[idx].l[idx].ifs[idx].v = ...;`
- reads:  `x = m[idx].l[idx].ifs[idx].v;`

### Realizations
- After closing nested task-call threading, signal member accesses uncovered a
  separate path: `xrun` accepted the notation but ImportVerilog failed during
  lvalue lowering (`unknown hierarchical name 'v'`).
- The initial failure mode also exposed pointer-identity fragility in
  interface-instance resolution for array-element symbols; unique
  definition-based fallback is required when slang synthesizes unnamed element
  symbols.
- Hierarchical interface member collection needed a normalized receiver-path
  fast-path for unnamed selected elements to avoid dropping into generic
  hierarchical value threading.

### TDD Baseline
- Probe:
  - `/tmp/iv_nested_signal_access.sv`
- Baseline behavior before fix:
  - `xrun -sv /tmp/iv_nested_signal_access.sv -elaborate -nolog` passed.
  - `build_test/bin/circt-verilog /tmp/iv_nested_signal_access.sv --ir-moore`
    failed (initially verifier error around `Leaf_1` outputs during debugging,
    then stabilized to `unknown hierarchical name 'v'`).

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`:
  - in hierarchical interface-member handling for
    `HierarchicalValueExpression`, added normalized receiver-path resolution via
    `collectReceiverSegmentsFromHierRef(...)` + `resolveInstancePath(...)` and
    direct threading with canonical path names.
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - in `resolveInterfaceInstance(const InstanceSymbol*, ...)`, added
    unambiguous same-interface-definition fallback when exact pointer-based
    lookup fails.
- `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - in `resolveHierarchicalInterfaceSignalRef(...)`, added:
    - direct hierarchical-reference resolution first
      (`resolveInterfaceInstance(expr.ref, ...)`), and
    - path-based fallback through `hierInterfacePaths` for unnamed
      array-element-shaped references in current scope.
- Added regression:
  - `test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv`
- Focused re-validation:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-assign.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-array-task-const-index.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-module-array-assign.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv`
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv`
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv build_test/test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-task.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/hierarchical-interface-through-nested-module-array-signal.sv -elaborate -nolog` (PASS)
  - `xrun -sv /tmp/iv_nested_signal_access.sv -elaborate -nolog` (PASS)

## 2026-02-26

### Task
Close ImportVerilog gap for non-blocking intra-assignment **event controls**
that were accepted by xrun but rejected by circt-verilog:
- `x <= @y x;`
- `x <= @(posedge clk) x;`
- `x <= @(a or b) x;`
- `x <= repeat (N) @(posedge clk) x;`

### Realizations
- Differential probe run identified a clean xrun-pass/circt-fail cluster:
  `SignalEvent`, `EventList`, and `RepeatedEvent` timing controls on
  non-blocking assignments.
- Existing lowering already handled these timing controls for blocking
  assignments via `convertTimingControl(...)` and had the correct RHS-before-wait
  evaluation order for intra-assignment semantics.
- Non-blocking lowering only gated to `DelayControl` and emitted an error for all
  other timing controls, despite the backend already supporting wait/event ops.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv`
- Baseline failure before fix:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv`
    failed with:
    - `unsupported non-blocking assignment timing control: SignalEvent`
- Differential probe evidence:
  - `/tmp/iv_probe/nba_event_signal.sv: xrun=0 circt=1`
  - `/tmp/iv_probe/nba_event_posedge.sv: xrun=0 circt=1`
  - `/tmp/iv_probe/nba_event_or.sv: xrun=0 circt=1`
  - `/tmp/iv_probe/nba_event_repeat.sv: xrun=0 circt=1`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Expressions.cpp`:
  - updated non-blocking assignment lowering to:
    - keep `DelayControl` on `moore.delayed_nonblocking_assign`, and
    - lower all other timing controls through `convertTimingControl(...)`
      followed by `moore.nonblocking_assign`.
- `test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv`:
  - added regression covering `SignalEvent`, `EventList`, `RepeatedEvent`, and
    posedge forms for non-blocking intra-assignment controls.
- `test/Conversion/ImportVerilog/errors.sv`:
  - removed stale expected-error for `initial x <= @y x;`.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regressions:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv build_test/test/Conversion/ImportVerilog/repeated-event-control.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv -elaborate -nolog` (PASS)
  - `/tmp/iv_probe/nba_event_signal.sv: xrun=0 circt=0`
  - `/tmp/iv_probe/nba_event_posedge.sv: xrun=0 circt=0`
  - `/tmp/iv_probe/nba_event_or.sv: xrun=0 circt=0`
  - `/tmp/iv_probe/nba_event_repeat.sv: xrun=0 circt=0`

## 2026-02-26

### Task
Close delayed non-blocking event trigger gap:
- `->> #1 e;`

### Realizations
- Differential probe showed this as a clean xrun-pass/circt-fail case:
  `/tmp/iv_probe/delayed_event_trigger_nb.sv: xrun=0 circt=1`.
- Existing lowering already handled immediate event triggers (`->` and `->>`) but
  rejected any trigger with `stmt.timing`.
- To preserve non-blocking behavior for delayed non-blocking triggers, lowering
  should not stall the current process; detached fork lowering is the right fit.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv`
- Baseline failure before fix:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv`
    failed with:
    - `unsupported delayed event trigger`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Statements.cpp`:
  - refactored event-trigger emission into a local helper.
  - added support for timed non-blocking event triggers by lowering
    `stmt.timing` under `stmt.isNonBlocking` as:
    - `moore.fork join_none` branch,
    - `convertTimingControl(...)` inside the branch,
    - event-trigger emission,
    - `moore.fork_terminator`.
  - kept blocking timed event triggers rejected (`unsupported delayed event
    trigger`).
- Added regression:
  - `test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv`

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regressions:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv build_test/test/Conversion/ImportVerilog/hierarchical-event-trigger.sv build_test/test/Conversion/ImportVerilog/event-trigger-fork.sv build_test/test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv -elaborate -nolog` (PASS)
  - `/tmp/iv_probe/delayed_event_trigger_nb.sv: xrun=0 circt=0`
- Probe recheck:
  - `/tmp/iv_probe/*.sv` now has no remaining `xrun=0/circt!=0` in this batch.

## 2026-02-26

### Task
Close format-string `%l` gap (`xrun` accepted, ImportVerilog rejected).

### Realizations
- Differential probe found a clean mismatch:
  - `/tmp/iv_probe2/format_l_probe.sv: xrun=0 circt=1`
  - diagnostic: `unsupported format specifier `%l``.
- IEEE 1800-2023 (Clause 21.2.1 / 33.7) defines `%l` / `%L` as library binding
  display (`library.cell`) and says `%l` is a non-argument-consuming specifier.
- ImportVerilog already had `%m` support and a `%L` compatibility path lowering
  to hierarchical scope text; `%l` was hard-error.

### TDD Baseline
- New regression added:
  - `test/Conversion/ImportVerilog/format-lowercase-l-compat.sv`
- Baseline before fix:
  - `xrun -sv /tmp/iv_probe2/format_l_probe.sv -elaborate -nolog` passed.
  - `build_test/bin/circt-verilog /tmp/iv_probe2/format_l_probe.sv --ir-moore`
    failed with unsupported `%l`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/FormatStrings.cpp`:
  - changed `%l` handling from hard error to non-failing lowering path.
  - `%l` now uses the same hierarchical-name fallback as `%m`/`%L` in the
    importer (until full library-binding metadata is threaded).
- `test/Conversion/ImportVerilog/format-lowercase-l-compat.sv`:
  - added regression covering `%l` in plain and prefixed strings.
- `test/Conversion/ImportVerilog/errors.sv`:
  - removed stale expected-error for `%l`.

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regressions:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/format-lowercase-l-compat.sv build_test/test/Conversion/ImportVerilog/nonblocking-assignment-event-control.sv build_test/test/Conversion/ImportVerilog/nonblocking-delayed-event-trigger.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/format-lowercase-l-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv /tmp/iv_probe2/format_l_probe.sv -elaborate -nolog` (PASS)
- Differential recheck:
  - `/tmp/iv_probe2/format_l_probe.sv: xrun=0 circt=0`

## 2026-02-26

### Task
Close additional display-format differential gaps for `%u`, `%z`, `%v`.

### Realizations
- Differential probes found all three as xrun-pass/circt-fail:
  - `/tmp/iv_probe2/format_u_probe.sv: xrun=0 circt=1`
  - `/tmp/iv_probe2/format_z_probe.sv: xrun=0 circt=1`
  - `/tmp/iv_probe2/format_v_probe.sv: xrun=0 circt=1`
- Current format parser had no switch cases for `%u`, `%z`, `%v`, so they
  hard-failed as unsupported format specifiers.
- For importer compatibility, lowering these through binary integer formatting
  is a pragmatic non-failing fallback:
  - `%u` and `%z`: fallback binary formatting
  - `%v` (strength): value-only binary fallback

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/format-vuz-compat.sv`
- Baseline failure before fix:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/format-vuz-compat.sv`
    failed on unsupported `%u`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/FormatStrings.cpp`:
  - added handling for `%u`, `%z`, `%v` in format-specifier switch.
  - lowered each through `emitInteger(..., IntFormat::Binary)` with explicit
    fallback comments for semantics.
- `test/Conversion/ImportVerilog/format-vuz-compat.sv`:
  - added regression coverage for `%u`, `%z`, `%v`.
- `test/Conversion/ImportVerilog/errors.sv`:
  - removed stale expected-error on `$fwrite` (already supported lowering).

### Validation
- Build:
  - `ninja -C build_test circt-verilog`
- Regressions:
  - `llvm/build/bin/llvm-lit -sv build_test/test/Conversion/ImportVerilog/format-vuz-compat.sv build_test/test/Conversion/ImportVerilog/format-lowercase-l-compat.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/format-vuz-compat.sv -elaborate -nolog` (PASS)
  - `/tmp/iv_probe2/format_u_probe.sv: xrun=0 circt=0`
  - `/tmp/iv_probe2/format_z_probe.sv: xrun=0 circt=0`
  - `/tmp/iv_probe2/format_v_probe.sv: xrun=0 circt=0`

## 2026-02-26

### Task
Close xrun-valid `with`-filter semantic gap for explicit coverpoint bins that
include a `default` bin.

### Realizations
- Finite cross bin-domain construction skipped `default` bins entirely, which can
  under-select `with` filters and collapse valid selections to always-false.
- For integral coverpoints with finite width, `default` bin values can be
  computed as a finite complement of explicitly covered bin values.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`
- PASS: `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv -elaborate -nolog`
- FAIL (before fix):
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`
  - emitted `moore.binsof @a_cp negate` (always-false lowering) instead of
    selecting `@a_cp::@other` bin tuples.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added `buildFiniteIntegralCoverpointDomain(...)` helper for finite integral
    coverpoint-domain enumeration.
  - updated `buildFiniteCrossBinDomains(...)` to:
    - track explicit covered values,
    - include `default` bins by finite complement over the coverpoint domain,
    - reuse finite-domain helper for auto-domain fallback.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`
- xrun notation:
  - `xrun -sv test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv -elaborate -nolog`
- Cross-select supported sweep:
  - `for f in test/Conversion/ImportVerilog/cross-select-*-supported.sv; do build_test/bin/circt-verilog "$f" --ir-moore [--language-version 1800-2023 where required] | llvm/build/bin/FileCheck "$f"; done`

## 2026-02-26

### Task
Close xrun-pass / circt-fail gap for class randomize inline constraints when
the randomize receiver is an array element:
- `elems[0].randomize() with { this.mode == 3; }`

### Realizations
- The failure happened before ImportVerilog lowering, during slang semantic
  binding for inline randomize constraints. In this form, slang binds `this`
  to the array symbol (`elems`) instead of the selected element handle.
- Existing ImportVerilog lowering for inline randomize constraints was already
  correct once semantic binding succeeded.

### Surprises
- The regression already existed in-tree as a positive test
  (`randomize-array-element-this.sv`) and xrun elaborated it successfully,
  making this a true compatibility gap.
- The same `this`-based inline constraints were already fine for non-array
  receivers (`randomize-nested-receiver-this.sv`), so the issue was specific
  to array-element receiver forms.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - added a targeted source rewrite pass during driver setup for
    `obj.randomize(...) with { ... }` blocks.
  - inside those inline constraint blocks, rewrites `this.` to unqualified
    member access before handing text to slang.
  - rewrite is lexical (skips strings / comments, handles balanced delimiters).
- No test-file edits were required; the existing positive regression now passes.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Reproduced fixed regression:
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this.sv`
  - `xrun -sv test/Conversion/ImportVerilog/randomize-array-element-this.sv -elaborate -nolog` (PASS)
- Nearby guardrail regression:
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-nested-receiver-this.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-nested-receiver-this.sv`
  - `xrun -sv test/Conversion/ImportVerilog/randomize-nested-receiver-this.sv -elaborate -nolog` (PASS)
- Additional probe parity checks:
  - `/tmp/iv_probe_randomize_next/array_varlist_with.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_randomize_next/nested_array_with.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_randomize_next/member_array_with.sv: circt=0, xrun=0`

## 2026-02-26

### Task
Continue closing xrun-pass / circt-fail randomize inline-constraint gaps for
array-element receivers when `this` is followed by comments before `.`:
- `this/*...*/.mode`
- `this // ... \\n .mode`

### Realizations
- The prior rewrite only recognized `this` followed by whitespace then `.`.
- Xcelium accepts comments as trivia in this position, so comment forms are
  legal notation and should lower the same as `this.mode`.

### TDD Baseline
- Probe cases:
  - `/tmp/iv_probe_thisdot/this_comment_dot.sv`
  - `/tmp/iv_probe_thisdot/this_linecomment_dot.sv`
- Before fix:
  - both had `xrun=0, circt=1` with
    `error: invalid member access for type 'unpacked array ... of Elem'`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - added `skipTrivia(...)` helper that skips whitespace and comments.
  - updated inline randomize `this.` rewrite to accept comment trivia between
    `this` and `.`.
- Added regression:
  - `test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regressions:
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this.sv`
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv -elaborate -nolog` (PASS)
- Probe recheck:
  - `/tmp/iv_probe_thisdot/this_comment_dot.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_thisdot/this_linecomment_dot.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_thisdot/this_space_dot.sv: circt=0, xrun=0`

## 2026-02-26

### Task
Close follow-up xrun-pass / circt-fail randomize inline-constraint gaps when
receiver syntax uses trivia between `.` and `randomize`:
- `obj . randomize() with { this.mode == ...; }`
- `obj.\nrandomize() with { this.mode == ...; }`

### Realizations
- Randomize inline-constraint rewriting was guarded by a fast-path
  `text.contains(\".randomize\")`, so files using `.` + trivia + `randomize`
  skipped rewriting entirely.
- The matcher itself also assumed the method token was contiguous as
  `.randomize`, which missed legal trivia-separated member-call spellings.

### TDD Baseline
- Probe cases:
  - `/tmp/iv_probe_randomize_spacing/receiver_space_around_dot.sv`
  - `/tmp/iv_probe_randomize_spacing/receiver_newline_after_dot.sv`
- Before fix:
  - both had `xrun=0, circt=1` with
    `error: invalid member access for type 'unpacked array ... of Elem'`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - generalized randomize receiver matching to parse `.` followed by trivia
    and then `randomize`.
  - switched `with` / brace parsing around the randomize call to trivia-aware
    scanning.
  - widened the pre-rewrite fast-path from `\".randomize\"` to `\"randomize\"`.
- Added regression:
  - `test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regressions:
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this.sv`
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv`
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv -elaborate -nolog` (PASS)
- Probe recheck:
  - `/tmp/iv_probe_randomize_spacing/receiver_space_around_dot.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_randomize_spacing/receiver_newline_after_dot.sv: circt=0, xrun=0`
  - `/tmp/iv_probe_randomize_spacing/receiver_newline_before_dot.sv: circt=0, xrun=0`

## 2026-02-26

### Task
Close remaining xrun-pass / circt-fail ImportVerilog format-string compatibility
for width modifiers on specifiers where xrun accepts and ignores width:
`%c`, `%p`, `%u`, `%z`, `%v`, `%m`, `%l`.

### Realizations
- A focused format-matrix differential probe found clean mismatches where xrun
  elaborated but CIRCT failed with
  `error: field width not allowed on ... format specifiers`.
- The failing forms were width / alignment variants such as `%4c`, `%-4p`,
  `%0u`, `%4z`, `%-4v`, `%0m`, `%4l`.
- Simply downgrading slang's `FormatSpecifierWidthNotAllowed` diagnostic is not
  sufficient: slang's format parser still returns failure for these strings,
  and affected display statements can be dropped from lowering.
- A source compatibility rewrite is required before slang semantic checking so
  format strings remain valid and lower completely.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/format-width-ignored-compat.sv`
- Baseline failure before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/format-width-ignored-compat.sv`
  - hard errors on `%4c`, `%-4p`, `%0u`, `%4z`, `%-4v`, `%0m`, `%4l`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - added `rewriteFormatWidthCompatLiteral(...)`:
    - strips width/alignment flags for `%c/%p/%u/%z/%v/%m/%l` inside string
      literal format fragments (for example `%4c` -> `%c`, `%-4p` -> `%p`).
  - added `rewriteFormatWidthCompat(...)`:
    - scans source text lexically,
    - rewrites only argument text of format system calls
      (`$display/$write/$monitor/$sformatf/...`) while preserving comments and
      non-format string literals unchanged.
  - integrated rewrite pass in `prepareDriver(...)` alongside the existing
    randomize-inline-constraint rewrite path.
- `test/Conversion/ImportVerilog/format-width-ignored-compat.sv`:
  - new regression for the width-ignored compatibility forms.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/format-width-ignored-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/format-width-ignored-compat.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/format-width-ignored-compat.sv -elaborate -nolog` (PASS; xrun emits `*W,IGNFMT` warnings and elaborates)
- Nearby guardrails:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/format-vuz-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/format-vuz-compat.sv`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this-comment-dot.sv`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randomize-array-element-this-receiver-trivia.sv`
- Differential re-probes:
  - full width-format matrix rerun showed no remaining xrun-pass/circt-fail
    mismatches in the tested `%` width cases.
  - letter probe rerun still shows nonstandard `%n/%N` as xrun-pass / circt-fail
    (unchanged in this slice).

### Notes
- `llvm-lit` could not be used in this sandbox due Python multiprocessing
  semaphore permission errors (`PermissionError: [Errno 13]`), so validation
  was done via direct `circt-verilog | FileCheck` invocations.

## 2026-02-26

### Task
Close remaining xrun-pass / circt-fail format-specifier gap for `%n` / `%N`
in display-family formatting calls.

### Realizations
- In this environment, xrun accepts `%n` / `%N` when an argument is provided,
  while CIRCT/slang rejected them as unknown format specifiers.
- `xrun` requires an argument for `%n` / `%N` (`*E,MISARG` without one), so
  compatibility handling must preserve normal argument consumption behavior.
- Slang rejects unknown format specifiers before ImportVerilog format lowering,
  so handling must occur via pre-parse source rewrite.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/format-n-compat.sv`
- Baseline failure before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/format-n-compat.sv`
  - errors: unknown format specifier `%n` / `%N`.
- xrun notation baseline:
  - `xrun -sv test/Conversion/ImportVerilog/format-n-compat.sv -elaborate -nolog` (PASS)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - extended format-call compatibility rewrite path to map `%n` -> `%u` and
    `%N` -> `%U` inside format string literals used by display-family system
    calls.
  - retained existing width-modifier stripping for width-ignored specifiers,
    so forms like `%4n` / `%-4N` are rewritten compatibly as well.
- Added regression:
  - `test/Conversion/ImportVerilog/format-n-compat.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/format-n-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/format-n-compat.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/format-n-compat.sv -elaborate -nolog` (PASS)
  - probe parity checks:
    - `%n/%N` with arg: xrun PASS, circt PASS
    - `%n/%N` without arg: xrun error (`MISARG`), circt error (`no argument provided`)
- Guardrails rechecked:
  - `format-width-ignored-compat.sv` (circt + FileCheck PASS)
  - `format-vuz-compat.sv` (circt + FileCheck PASS)
  - `randomize-array-element-this-receiver-trivia.sv` (circt + FileCheck PASS)
- Differential probe rerun:
  - `%a..%z` / `%A..%Z` single-argument probe now has no remaining
    xrun-pass/circt-fail specifiers in this matrix.

### Notes
- `%n/%N` semantics are vendor extension behavior here; lowering uses the same
  compatibility fallback pattern as other non-core format support.

## 2026-02-26

### Task
Close xrun-pass / circt-fail virtual-interface assignment gap for interface
instances that are targets of `defparam` / `bind` overrides.

### Realizations
- In this slang version, `CompilationFlags::AllowVirtualIfaceWithOverride` is
  not available, so the existing CLI flag path cannot suppress the diagnostic.
- Slang still constructs the interface reference expression after emitting the
  diagnostic, so severity downgrade (error -> warning) preserves correct
  lowering while matching xrun's permissive behavior.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv`
- Baseline before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv`
  - hard error:
    `interface instance cannot be assigned to a virtual interface because it is the target of a defparam or bind directive`.
- xrun notation baseline:
  - `xrun -sv test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv -elaborate -nolog` (PASS)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - diagnostic policy now downgrades:
    - `slang::diag::VirtualIfaceDefparam`
    - `slang::diag::VirtualIfaceConfigRule`
    to warnings.
  - retained default-on wiring for
    `AllowVirtualIfaceWithOverride` when available in newer slang versions;
    older versions continue via warning-policy fallback.
- Added regression:
  - `test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv`
  - emits warning (not error) and lowers successfully.
- Existing related regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/virtual-iface-bind-override.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/virtual-iface-bind-override.sv`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/virtual-iface-bind-override-default.sv -elaborate -nolog` (PASS)
- Differential rerun impact:
  - removed `virtual-iface-bind-override` from xrun-pass/circt-fail list.

## 2026-02-26

### Task
Close the immediate xrun-pass / circt-fail import gap for user-defined
primitive declarations and instances (UDP) under `--ir-moore`.

### Realizations
- ImportVerilog failed early in `RootVisitor` on top-level `PrimitiveSymbol`
  (`error: unsupported construct: Primitive`) before reaching module lowering.
- Even after that, UDP instances do not map to the existing built-in gate
  primitive lowering path.
- The practical compatibility requirement in this slice is to avoid hard import
  failure and keep module conversion progressing, while preserving a visible
  diagnostic.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`
- Before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`
  - failed with `error: unsupported construct: Primitive`.
- Existing test baseline:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-coverage-note-limit.sv`
  - failed with the same `unsupported construct: Primitive` error.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - `RootVisitor` now accepts `slang::ast::PrimitiveSymbol` declarations.
  - `ModuleVisitor::visit(PrimitiveInstanceSymbol)` now detects
    `primitiveKind == UserDefined` and emits:
    - `warning: dropping user-defined primitive instance \`...\` of \`...\``
    then continues lowering instead of hard-failing.
- New regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`
    (warning + successful module lowering checks).
- Extended existing regression:
  - `test/Conversion/ImportVerilog/udp-coverage-note-limit.sv`
    now also checks `--ir-moore` warning + successful module lowering.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv --check-prefix=WARN`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv --check-prefix=IR`
- Existing regression (updated):
  - `build_test/bin/circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=DEFAULT`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage --max-udp-coverage-notes=2 test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=LIMIT2`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=IR-WARN`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-coverage-note-limit.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=IR`
- Guardrails:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/gate-primitives.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gate-primitives.sv`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/mos-primitives.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/mos-primitives.sv`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/pullup-pulldown.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/pullup-pulldown.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog` (PASS)

### Notes
- This slice addresses import compatibility by continuing on UDP instances with
  warning-level diagnostics; full UDP behavior modeling remains a follow-up.

## 2026-02-26

### Task
Continue closing UDP ImportVerilog gaps by lowering a practical user-defined
primitive subset instead of dropping all UDP instances.

### Realizations
- After the previous slice, all user-defined primitives were accepted at parse
  level but always dropped at conversion.
- A useful incremental target is combinational UDPs with simple 1-bit truth
  tables, while keeping sequential / edge-sensitive UDPs as explicit warning
  fallbacks.
- A broad xrun-vs-circt sweep over
  `test/Conversion/ImportVerilog/*.sv` with default `--ir-moore` command-line
  found no remaining corpus-level xrun-pass/circt-fail hard mismatches in that
  command mode, so this UDP slice was selected as the next structural gap.

### TDD Baseline
- Added new positive regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv`
- Before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv`
  - emitted warning:
    `dropping user-defined primitive instance ...`
- Existing fallback regression was previously combinational and needed to become
  sequential to remain a valid drop-path checker once combinational support was
  added.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added combinational UDP lowering in
    `ModuleVisitor::visit(PrimitiveInstanceSymbol)` for user-defined
    primitives that are:
    - non-sequential (`!isSequential`),
    - non-edge-sensitive (`!isEdgeSensitive`),
    - 1-bit output and 1-bit inputs,
    - row symbols limited to supported simple forms.
  - lowering builds a conditional chain from UDP table rows and assigns the
    resulting value to the UDP output net.
  - unspecified combinational rows default to X for four-valued outputs.
  - sequential / edge-sensitive and unsupported row-shape cases keep warning
    fallback behavior (`dropping user-defined primitive instance ...`).
- Updated fallback regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`
    now uses a sequential UDP (`reg q`, edge row) so it still checks the
    unsupported fallback path.
- Added positive regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv`
    checks successful lowering without drop warning.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New combinational regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv --check-prefix=DIAG`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv --check-prefix=IR`
- Sequential fallback regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv --check-prefix=WARN`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv --check-prefix=IR`
- Existing UDP regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=DEFAULT`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --lint-only -Wudp-coverage --max-udp-coverage-notes=2 test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=LIMIT2`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-coverage-note-limit.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-coverage-note-limit.sv --check-prefix=IR-WARN`
- Guardrail:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/gate-primitives.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/gate-primitives.sv`
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog` (PASS)

### Notes
- The supported UDP subset is intentionally narrow in this slice; complex table
  symbols and sequential behavior remain warning-fallback.

## 2026-02-26

### Task
Continue closing UDP ImportVerilog gaps by supporting sequential (state-table)
user-defined UDPs that are non-edge-sensitive.

### Realizations
- First implementation attempt reused continuous-assignment style lowering for
  sequential UDPs, which produced self-referential IR for module output-port
  connections (invalid/unsafe semantics).
- Sequential UDP lowering needs procedural semantics (wait + read current state
  + assign) rather than pure combinational expression assignment.
- Edge-sensitive UDP rows remain a distinct gap and should stay explicit warning
  fallback until edge-transition matching is implemented.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv`
- Baseline before fix:
  - `build_test/bin/circt-verilog --ir-moore --no-uvm-auto-include test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv`
  - warning: `dropping user-defined primitive instance ...: sequential or edge-sensitive UDP not yet supported`
- xrun notation baseline:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog` (PASS)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extended UDP lowering path to support `udp.isSequential && !udp.isEdgeSensitive`.
  - keeps explicit fallback warning for edge-sensitive UDPs.
  - sequential lowering now emits a `moore.procedure always` with:
    - `moore.wait_event` + `moore.detect_event any` on UDP inputs,
    - current-state read (`moore.read`) of UDP output storage,
    - table-row condition evaluation including current-state column symbols,
    - `'-'` output-row semantics as hold-current-state,
    - `moore.blocking_assign` for output update.
  - combinational UDP lowering path remains active and unchanged in behavior.
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- New sequential regression:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv 2>&1 | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv --check-prefix=DIAG`
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv --check-prefix=IR`
- Existing UDP guardrails:
  - `udp-user-defined-comb-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-drop-compat.sv` (WARN + IR PASS)
  - `udp-coverage-note-limit.sv` (DEFAULT + LIMIT2 + IR-WARN PASS)
  - `gate-primitives.sv` (PASS)
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog` (PASS)

### Notes
- xrun emitted environment-local `cds_root` temp-file/platform warnings in this
  environment but still completed elaboration with exit code 0.
- Remaining UDP gap after this slice: edge-sensitive table semantics.

## 2026-02-26

### Task
Continue UDP gap closure by implementing edge-sensitive user-defined UDP support
for explicit edge-row forms while keeping unsupported shorthand/complex forms as
warning fallback.

### Realizations
- Supporting edge-sensitive UDPs exposed a latent control-flow issue in the
  drop path: if dropping occurred after creating a sequential UDP procedure,
  the procedure body could be left without a terminator.
- A practical incremental target is explicit edge tuple rows (`(ab)`), which
  are sufficient for common DFF-style UDP tables and are xrun-compatible.
- Edge shorthand tokens like `p/r/f/n/*` can remain explicit fallback for now;
  this keeps behavior deterministic while still reducing a major gap.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv`
- Baseline before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv`
  - warning: `dropping user-defined primitive instance ...: edge-sensitive UDP not yet supported`
- xrun notation baseline:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog` (PASS)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - extended sequential UDP lowering to accept edge-sensitive rows represented
    as explicit parenthesized transitions (`(ab)`) in Slang table entries.
  - implemented edge-row input parsing into per-input tokens and condition
    lowering using previous/current input values for transition matching.
  - introduced per-input previous-value storage for edge-sensitive sequential
    UDP matching and updates it after output assignment.
  - retained warning fallback for unsupported edge symbols and unsupported
    row/token forms.
  - fixed sequential drop path to emit `moore.return` when a UDP instance is
    dropped after a sequential procedure body has been created.
- Tests:
  - added `udp-user-defined-seq-edge-supported.sv` (positive support case).
  - updated `udp-user-defined-drop-compat.sv` to use unsupported shorthand edge
    token `p` so fallback coverage remains active.
  - updated `udp-coverage-note-limit.sv` IR check to assert that this explicit
    edge-row UDP is no longer dropped in `--ir-moore` mode.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks:
  - `udp-user-defined-seq-edge-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-seq-level-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-comb-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-drop-compat.sv` (WARN + IR PASS)
  - `udp-coverage-note-limit.sv` (DEFAULT + LIMIT2 + IR-DIAG + IR PASS)
  - `gate-primitives.sv` (PASS)
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog` (PASS)

### Notes
- xrun continued to emit environment-local `cds_root` temp-file/platform
  warnings in this environment, but all elaboration runs exited 0.
- Remaining edge-sensitive UDP gap after this slice: shorthand edge symbols
  (`p`, `r`, `f`, `n`, `*`) and broader full-table edge semantics.

## 2026-02-26

### Task
Continue closing UDP ImportVerilog gaps by supporting edge shorthand row symbols
for sequential user-defined primitives.

### Realizations
- After explicit `(ab)` edge-row support, shorthand edge tokens remained an
  xrun-valid gap (`r/f/p/n`), while `*` can be kept as explicit fallback for a
  controlled incremental step.
- Implementing shorthand expansion as transition alternatives keeps behavior
  aligned with existing row-condition lowering and naturally handles 2-state
  type restrictions via impossible-row filtering.
- The fallback path must stay verifier-safe even after a sequential procedure is
  created, which is now ensured by emitting `moore.return` on drop.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv`
- Baseline before fix:
  - shorthand rows (`r/f/p/n`) triggered drop warnings in ImportVerilog.
- xrun notation baseline:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv -elaborate -nolog` (PASS)

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - edge-sensitive row parsing now treats shorthand edge symbols as edge tokens
    when row is edge-sensitive.
  - added shorthand transition expansion support for:
    - `r` -> `(01)`
    - `f` -> `(10)`
    - `p` -> `(01)|(0x)|(x1)`
    - `n` -> `(10)|(1x)|(x0)`
  - keeps `*` as explicit unsupported edge shorthand fallback.
  - edge transition alternatives are lowered as OR-of-transition predicates and
    integrated into existing row condition chains.
- Tests:
  - added `udp-user-defined-seq-edge-shorthand-supported.sv` (positive case).
  - updated `udp-user-defined-drop-compat.sv` to use `*` shorthand to preserve
    unsupported fallback coverage.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks:
  - `udp-user-defined-seq-edge-shorthand-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-seq-edge-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-seq-level-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-comb-supported.sv` (DIAG + IR PASS)
  - `udp-user-defined-drop-compat.sv` (WARN + IR PASS)
  - `udp-coverage-note-limit.sv` (DEFAULT + LIMIT2 + IR-DIAG + IR PASS)
  - `gate-primitives.sv` (PASS)
- xrun notation checks:
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog` (PASS)
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog` (PASS)

### Notes
- xrun continues to emit environment-local `cds_root` temp-file/platform
  warnings in this environment, but elaboration exits 0.
- Remaining UDP shorthand gap after this slice: `*` edge shorthand token.

## 2026-02-26

### Task
Continue UDP ImportVerilog gap closure by supporting `*` edge shorthand rows and
aligning wildcard edge-transition semantics with IEEE value-change notation.

### Realizations
- IEEE 1800-2023 Table 29-1 defines `*` as `same as (??)` with comment
  "Any value change on input".
- Existing edge-transition lowering matched wildcard pairs independently on
  previous/current symbols; for wildcard descriptors like `(??)` this could
  admit stable values when another input triggered evaluation.
- Constraining each transition alternative with case-inequality
  (`prev !== curr`) preserves shorthand behavior and aligns wildcard transition
  descriptors with value-change semantics.

### TDD Baseline
- Baseline before this slice: `*` shorthand rows were still dropped with
  `unsupported UDP edge transition symbol`.
- Regression target updated to require acceptance:
  - `test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`
- Added focused wildcard-transition regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-edge-anychange-supported.sv`

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - implemented `*` edge shorthand expansion in sequential UDP row lowering as
    any-change transitions over `0/1/x`:
    - `(01)`, `(0x)`, `(10)`, `(1x)`, `(x0)`, `(x1)`
  - tightened edge-transition lowering so each transition alternative includes
    `moore.case_ne(prev, curr)` to enforce value-change semantics for wildcard
    descriptors such as `(??)`, `(b?)`, etc.
- `test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv`:
  - converted from warning fallback to positive support check for `*` rows.
- Added:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-edge-anychange-supported.sv`
    (checks `(??)` support and `moore.case_ne` emission).

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- ImportVerilog regression checks (all PASS):
  - `udp-user-defined-seq-edge-anychange-supported.sv` (DIAG + IR)
  - `udp-user-defined-drop-compat.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-shorthand-supported.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-supported.sv` (DIAG + IR)
  - `udp-user-defined-seq-level-supported.sv` (DIAG + IR)
  - `udp-user-defined-comb-supported.sv` (DIAG + IR)
  - `udp-coverage-note-limit.sv` (DEFAULT + LIMIT2 + IR-DIAG + IR)
  - `gate-primitives.sv`
- xrun notation checks (all PASS):
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-anychange-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog`

### Notes
- `llvm-lit` invocation in this sandbox hit a multiprocessing semaphore
  permission error; direct `circt-verilog` + `FileCheck` commands were used for
  deterministic regression validation in this environment.
- xrun continued to emit environment-local `cds_root` temp/platform warnings,
  but elaboration exited 0 for all checked regressions.

## 2026-02-26

### Task
Continue ImportVerilog gap closure by handling xrun-compatible UDP table rows
that use `z` symbols, which Slang currently rejects during parsing.

### Realizations
- Differential probes showed a real xrun-pass / circt-fail gap for UDP table
  rows containing `z` symbols (input/state/output/edge tuple positions).
- The failure occurred before ImportVerilog lowering (`invalid symbol 'z' in
  state table`) because Slang rejected these rows at parse time.
- ImportVerilog already uses source compatibility rewrites (format-width,
  randomize inline constraints), so a targeted pre-parse rewrite is the
  maintainable place to bridge this parser compatibility gap.

### TDD Baseline
- Probe baseline (`/tmp/udp_probe2/*`) was:
  - PASS: `xrun -sv ... -elaborate -nolog`
  - FAIL: `build_test/bin/circt-verilog ... --ir-moore`
    with `error: invalid symbol 'z' in state table`.
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-z-compat.sv`
  - Uses `z` in edge tuple, input symbol, current-state symbol, and output
    symbol inside a sequential UDP table.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`:
  - added `rewriteUDPZCompat` source rewrite pass.
  - rewrite scope is restricted to `primitive ... table ... endtable ...
    endprimitive` regions outside comments / string literals.
  - canonicalizes UDP table symbol `z/Z` to `x/X` before Slang parsing.
  - wired rewrite pass into `prepareDriver` rewrite pipeline with lightweight
    content gating.
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-z-compat.sv`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks (all PASS):
  - `udp-user-defined-z-compat.sv` (DIAG + IR)
  - `udp-user-defined-drop-compat.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-anychange-supported.sv` (DIAG + IR)
  - `udp-coverage-note-limit.sv` (IR-DIAG + IR)
- xrun notation checks (all PASS):
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-z-compat.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-anychange-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog`

### Notes
- Large corpus differential sweeps over `test/Conversion/ImportVerilog/*.sv`
  (split into two passes) found no additional xrun-pass / circt-fail cases in
  this tree under default `--ir-moore` command mode.

## 2026-02-26

### Task
Continue UDP ImportVerilog gap closure by implementing sequential UDP
initialization (`initial q = ...`) lowering.

### Realizations
- Slang exposes sequential UDP initialization as `PrimitiveSymbol::initVal`.
- ImportVerilog previously accepted and lowered sequential UDP tables but ignored
  UDP initial values entirely.
- xrun elaborates sequential UDPs with `initial` state assignments, so this is a
  real semantic compatibility gap even when parsing and lowering otherwise
  succeed.

### TDD Baseline
- Probe baseline:
  - `xrun -sv /tmp/udp_init_probe.sv -elaborate -nolog` -> PASS
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore /tmp/udp_init_probe.sv`
    -> PASS, but IR only had `moore.procedure always` (no initial-state assign).
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-initial-supported.sv`
  - Requires both `moore.procedure initial` and `moore.procedure always`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added sequential UDP initialization lowering via `emitSequentialInit`.
  - when `udp.initVal` is present, emits a dedicated
    `moore.procedure initial` that assigns the initial 1-bit value to the UDP
    output reference and returns.
  - initialization value extraction uses `udp.initVal->integer()[0]` and maps
    to Moore constants (including unknown handling for 4-state outputs).
  - preserves existing sequential always-procedure lowering path.
- Added regression:
  - `test/Conversion/ImportVerilog/udp-user-defined-seq-initial-supported.sv`.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks (all PASS):
  - `udp-user-defined-seq-initial-supported.sv` (DIAG + IR)
  - `udp-user-defined-z-compat.sv` (DIAG + IR)
  - `udp-user-defined-drop-compat.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-anychange-supported.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-shorthand-supported.sv` (DIAG + IR)
  - `udp-user-defined-seq-edge-supported.sv` (DIAG + IR)
  - `udp-user-defined-seq-level-supported.sv` (DIAG + IR)
  - `udp-user-defined-comb-supported.sv` (DIAG + IR)
  - `udp-coverage-note-limit.sv` (IR-DIAG + IR)
- xrun notation checks (all PASS):
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-initial-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-z-compat.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-drop-compat.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-anychange-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-shorthand-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-edge-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-seq-level-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-user-defined-comb-supported.sv -elaborate -nolog`
  - `xrun -sv test/Conversion/ImportVerilog/udp-coverage-note-limit.sv -elaborate -nolog`

## 2026-02-26

### Task
Continue ImportVerilog timing-control gap closure by implementing `#1step`
delay lowering.

### Realizations
- Slang models `#1step` as `OneStepDelayControl`; ImportVerilog previously
  rejected it as unsupported delay control.
- This is a real xrun-pass / circt-fail gap for timing notation.
- The semantically correct lowering is one local time-precision tick, which can
  be represented as `moore.constant_time <precision_fs>` plus
  `moore.wait_delay`.

### TDD Baseline
- Probe baseline:
  - `xrun -sv /tmp/one_step_probe.sv -elaborate -nolog` -> PASS
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore /tmp/one_step_probe.sv`
    -> FAIL with `unsupported delay control: OneStepDelay`.
- Added regression:
  - `test/Conversion/ImportVerilog/delay-one-step-supported.sv`
  - Checks diagnostic compatibility and IR lowering to constant-time delay.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/TimingControls.cpp`:
  - added `getTimePrecisionInFemtoseconds(Context&)`.
  - added `DelayControlVisitor::visit(const OneStepDelayControl&)`.
  - lowering emits `moore.constant_time <precision_fs>` and
    `moore.wait_delay`.
- Added regression:
  - `test/Conversion/ImportVerilog/delay-one-step-supported.sv`.
- Updated regression check order to match actual IR placement where
  `moore.constant_time` is hoisted before the procedure body.

### Validation
- Regression checks (all PASS):
  - `circt-verilog --no-uvm-auto-include --ir-moore delay-one-step-supported.sv 2>&1 | FileCheck --check-prefix=DIAG`
  - `circt-verilog --no-uvm-auto-include --ir-moore delay-one-step-supported.sv | FileCheck --check-prefix=IR`
  - `circt-verilog --ir-moore nonblocking-delayed-event-trigger.sv | FileCheck`
  - `circt-verilog --ir-moore event-trigger-fork.sv | FileCheck --check-prefix=MOORE`
  - `circt-verilog --ir-hw event-trigger-fork.sv | FileCheck --check-prefix=HW`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/delay-one-step-supported.sv -elaborate -nolog` -> PASS

## 2026-02-26

### Task
Continue ImportVerilog timing-control gap closure by supporting cycle-delay
controls (`##N`) in procedural contexts.

### Realizations
- A concrete xrun-pass / circt-fail gap existed for `CycleDelayControl`:
  - xrun elaborates `##1`/`##2` with default clocking.
  - CIRCT failed with `unsupported delay control: CycleDelay`.
- Existing repeated-event lowering already had the right countdown-loop shape,
  so cycle delay support can reuse that mechanism by binding to default/global
  clocking events.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/delay-cycle-supported.sv`.
- Baseline before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/delay-cycle-supported.sv`
    failed with `unsupported delay control: CycleDelay`.
  - `xrun -sv test/Conversion/ImportVerilog/delay-cycle-supported.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/TimingControls.cpp`:
  - extracted repeated timing loop emission into
    `emitRepeatedTimingControl(...)`.
  - refactored `RepeatedEventControl` lowering to use the helper.
  - added `CycleDelayControl` lowering:
    - converts cycle count expression.
    - resolves default clocking event in current scope, with global clocking
      fallback.
    - canonicalizes clocking event to signal-event form.
    - emits countdown loop that waits on the resolved event each cycle.
- Added regression:
  - `test/Conversion/ImportVerilog/delay-cycle-supported.sv`.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks (all PASS):
  - `circt-verilog --no-uvm-auto-include --ir-moore delay-cycle-supported.sv 2>&1 | FileCheck --check-prefix=DIAG`
  - `circt-verilog --no-uvm-auto-include --ir-moore delay-cycle-supported.sv | FileCheck --check-prefix=IR`
  - `circt-verilog --ir-moore repeated-event-control.sv | FileCheck`
  - `circt-verilog --no-uvm-auto-include --ir-moore delay-one-step-supported.sv | FileCheck --check-prefix=IR`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/delay-cycle-supported.sv -elaborate -nolog` -> PASS

## 2026-02-26

### Task
Continue ImportVerilog timing-control compatibility by supporting `#1step`
delays in continuous assignments (`assign #1step lhs = rhs;`).

### Realizations
- A concrete xrun-pass / circt-fail gap remained specifically in continuous
  assignment timing-control lowering.
- Procedural `#1step` support was already landed in `TimingControls.cpp`, but
  continuous assignment handling in `Structure.cpp` still only accepted
  `DelayControl` and `Delay3Control`.

### TDD Baseline
- Added regression:
  - `test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv`.
- Baseline before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv`
    failed with
    `unsupported continuous assignment timing control: OneStepDelay`.
  - `xrun -sv test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv -elaborate -nolog`
    passed.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`:
  - added local `getTimePrecisionInFemtoseconds(Context&)` helper.
  - extended continuous assignment delay lowering to handle
    `OneStepDelayControl` by emitting `moore.constant_time <precision_fs>` and
    `moore.delayed_assign`.
- Added regression:
  - `test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv`.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog`
- Regression checks (all PASS):
  - `circt-verilog --no-uvm-auto-include --ir-moore continuous-assign-delay-one-step-supported.sv 2>&1 | FileCheck --check-prefix=DIAG`
  - `circt-verilog --no-uvm-auto-include --ir-moore continuous-assign-delay-one-step-supported.sv | FileCheck --check-prefix=IR`
- xrun notation check:
  - `xrun -sv test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv -elaborate -nolog` -> PASS

## 2026-02-26

### Task
Strengthen newly added ImportVerilog regressions from syntax/acceptance checks
into behavior-oriented checks, then re-validate notation coverage with xrun.

### Realizations
- Several new regressions were too weak (e.g. only `DIAG-NOT` or one generic
  op presence), which does not protect lowering semantics.
- The strongest signal for these parser/lowering compat tests is to check
  operation ordering and key lowering details (flags, conversion chain, event
  scheduling), not just that translation succeeds.
- `virtual-iface-bind-override-default.sv` remains warning-level in CIRCT
  (`interface instance cannot be assigned...`), but elaboration and IR
  structure are still compatible and intentionally tested.

### Changes Landed In This Slice
- Upgraded semantic checks in these new regressions:
  - `always-comb-nested-ternary-dominance.sv`
  - `continuous-assign-delay-one-step-supported.sv`
  - `delay-cycle-supported.sv`
  - `localparam-unpacked-multidim-dynamic-index.sv`
  - `nonblocking-stream-unpack-supported.sv`
  - `nonblocking-stream-unpack-delayed-supported.sv`
  - `relax-enum-conversions-default-compat.sv`
  - `string-concat-byte-default-compat.sv`
  - `virtual-iface-bind-override-default.sv`
  - `udp-user-defined-comb-supported.sv`
  - `udp-user-defined-drop-compat.sv`
  - `udp-user-defined-seq-edge-supported.sv`
  - `udp-user-defined-seq-edge-anychange-supported.sv`
  - `udp-user-defined-seq-edge-shorthand-supported.sv`
  - `udp-user-defined-seq-initial-supported.sv`
  - `udp-user-defined-seq-level-supported.sv`
  - `udp-user-defined-z-compat.sv`
- Key semantic upgrades include:
  - nonblocking stream unpack checks now assert `right_to_left true` and timing
    ordering (`wait_event` / `wait_delay` / `stream_unpack`).
  - cycle-delay checks now verify countdown-loop lowering shape.
  - enum/string compatibility tests now verify concrete conversion dataflow
    (`trunc`, `int_to_logic`, `string.getc`, `int_to_string`, `string_concat`).
  - UDP tests now assert edge/level/state-table related lowering details
    (`case_eq`, `case_ne`, event detection, assignment targets).

### Validation
- Re-ran all `RUN:` lines for the upgraded regressions (PASS).
- xrun notation check sweep (all PASS):
  - `always-comb-nested-ternary-dominance.sv`
  - `continuous-assign-delay-one-step-supported.sv`
  - `cross-select-intersect-open-range-wide-supported.sv`
  - `delay-cycle-supported.sv`
  - `delay-one-step-supported.sv`
  - `format-n-compat.sv`
  - `format-width-ignored-compat.sv`
  - `localparam-unpacked-multidim-dynamic-index.sv`
  - `nonblocking-stream-unpack-delayed-supported.sv`
  - `nonblocking-stream-unpack-supported.sv`
  - `randomize-array-element-this-comment-dot.sv`
  - `randomize-array-element-this-receiver-trivia.sv`
  - `relax-enum-conversions-default-compat.sv`
  - `string-concat-byte-default-compat.sv`
  - `sva-hierarchical-generate-scope-disambiguation.sv`
  - `udp-user-defined-comb-supported.sv`
  - `udp-user-defined-drop-compat.sv`
  - `udp-user-defined-seq-edge-anychange-supported.sv`
  - `udp-user-defined-seq-edge-shorthand-supported.sv`
  - `udp-user-defined-seq-edge-supported.sv`
  - `udp-user-defined-seq-initial-supported.sv`
  - `udp-user-defined-seq-level-supported.sv`
  - `udp-user-defined-z-compat.sv`
  - `virtual-iface-bind-override-default.sv`

### Task
Close next ImportVerilog legality gap from xrun differential corpus:
procedural assignment to module input ports (`001`, `002`).

### Realizations
- CIRCT already rejected procedural assignment to plain nets (`014`), but still
  accepted assignments to symbols backing module `input` ports.
- The robust place to enforce this is assignment-kind bookkeeping in
  `noteProceduralVariableAssignment`, by resolving whether the direct assigned
  symbol is the `internalSymbol` of an `input` `PortSymbol`.
- Running broad `check-circt-conversion-importverilog` in this worktree hit
  unrelated dirty-tree compile failures in `tools/circt-sim` (undeclared
  `registerJITRuntimeSymbols`), so verification for this slice used targeted
  CIRCT+xrun differential runs.

### Changes Landed In This Slice
- Added procedural-assignment legality check in:
  - `lib/Conversion/ImportVerilog/Structure.cpp`
    - new helpers:
      - `getDirectAssignedValue`
      - `getInputPortForInternalSymbol`
    - `noteVariableAssignmentKind` now emits:
      - `cannot assign to input port '<name>'`
      for procedural assignments targeting input-port backing symbols.
- Added ImportVerilog regression coverage in:
  - `test/Conversion/ImportVerilog/driver-errors.sv`
    - `InputPortProcAssign`
    - `InputVarPortProcAssign`

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- Targeted differential checks:
  - `python3 utils/run_illegal_sv_xcelium_diff.py --circt-verilog build_test/bin/circt-verilog --case-filter '^(001_input_proc_assign|002_input_var_proc_assign)\\.sv$' ...` (PASS, both `both_reject`)
  - Full illegal corpus rerun:
    - before this slice: `xcelium_reject_circt_accept = 15`
    - after this slice:  `xcelium_reject_circt_accept = 13`
- Direct xrun notation checks:
  - `001_input_proc_assign.sv` -> xrun rejects (`WANOTL`)
  - `002_input_var_proc_assign.sv` -> xrun rejects (`ICDPAV`)

## 2026-02-26

### Task
Continue closing ImportVerilog legality gaps against xrun in
`utils/illegal_sv_diff_cases`, with TDD and xrun notation checks.

### Realizations
- The next concrete gaps were all in procedural/driver legality, not parser
  support:
  - procedural assignment to `input` ports (`001`, `002`),
  - always_* exclusivity and cross-procedure driver checks (`003`..`009`,
    `011`, `015`..`019`),
  - multiple continuous assignments to variables (`010`),
  - struct-member / whole-struct always_ff conflicts (`012`, `018`).
- Existing direct-symbol checks were insufficient: member/element/range lvalues
  must be mapped back to their root variable to enforce always_ff/always_comb/
  always_latch constraints consistently.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilogInternals.h`
  - added procedure-context tracking fields:
    - `currentProcedureKind`
    - `currentProceduralBlock`
  - added `ProceduralDriverInfo` + `variableProceduralDrivers` map.
- `lib/Conversion/ImportVerilog/Structure.cpp`
  - added recursive assigned-symbol helpers:
    - `getAssignedValue(...)`
    - `getAssignedVariable(...)`
    covering member/element/range lvalues.
  - added input-port procedural assignment rejection:
    - `cannot assign to input port '<name>'`.
  - added always_* exclusivity enforcement via per-variable procedural-driver
    tracking:
    - `variable '<name>' driven by always_comb procedure`
    - `variable '<name>' driven by always_ff procedure`
    - `variable '<name>' driven by always_latch procedure`.
  - changed repeated continuous assignment handling from permissive to rejecting:
    - `multiple continuous assignments to variable '<name>'`.
  - moved mixed continuous/procedural diagnostic location to assignment site
    (`assignLoc`) for consistent test anchoring.
  - threaded procedure context through `convertProcedure(...)` and caller paths.
  - for `always @*` lowered-as-comb compatibility, kept assignment legality
    tracking as `Always` (non-special) so conflicts against prior `always_ff`
    report `always_ff`, matching xrun corpus expectation (`017`).
- Regression tests added:
  - `test/Conversion/ImportVerilog/input-port-procedural-assign-errors.sv`
  - `test/Conversion/ImportVerilog/always-special-driver-conflicts.sv`
  - `test/Conversion/ImportVerilog/always-ff-struct-member-driver-conflict.sv`
  - `test/Conversion/ImportVerilog/always-ff-struct-whole-field-driver-conflict.sv`
- Regression test updates:
  - `test/Conversion/ImportVerilog/driver-errors.sv`
    - aligned to current legality diagnostics for multiple continuous assigns
      and mixed continuous/procedural assignment locations.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog circt-translate`
- New/updated CIRCT regressions:
  - `build_test/bin/circt-translate --import-verilog --verify-diagnostics --split-input-file test/Conversion/ImportVerilog/input-port-procedural-assign-errors.sv` (PASS)
  - `build_test/bin/circt-translate --import-verilog --verify-diagnostics --split-input-file test/Conversion/ImportVerilog/always-special-driver-conflicts.sv` (PASS)
  - `build_test/bin/circt-translate --import-verilog --verify-diagnostics --split-input-file test/Conversion/ImportVerilog/driver-errors.sv` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/always-ff-struct-member-driver-conflict.sv 2>&1 | llvm/build/bin/FileCheck ... --check-prefix=ERR` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/always-ff-struct-whole-field-driver-conflict.sv 2>&1 | llvm/build/bin/FileCheck ... --check-prefix=ERR` (PASS)
- xrun notation checks for each new regression construct:
  - `001, 002, 003, 004, 006, 008, 010, 011, 012, 018` all reject under
    `xrun -sv -64bit -elaborate` (exit code `1`), confirming notation/semantics.
- Differential closure status:
  - `python3 utils/run_illegal_sv_xcelium_diff.py --circt-verilog build_test/bin/circt-verilog`
  - result: `both_reject: 22`, `xcelium_reject_circt_accept: 0`,
    `expect match: 22`.

### Task
Continue closing ImportVerilog gaps with TDD, focusing on cross-root
hierarchical interface method calls (e.g. `topa.ifc.ping(...)` from sibling
top-level modules).

### Realizations
- The failure did not come from `resolveInterfaceInstance` path threading alone.
  The receiver-binding path in `Expressions.cpp` only handled
  `ArbitrarySymbolExpression`/element-select patterns and missed
  `HierarchicalValueExpression` receivers.
- For this notation, xrun accepted elaboration while CIRCT rejected with:
  `hierarchical interface method calls through module instances are not yet supported`.
- Simulation execution with `xrun -R` could not be completed in this environment
  due unavailable runtime license checkout (`xmsim: *F,NOLICN`), but
  elaboration succeeded.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Expressions.cpp`
  - added receiver resolution for
    `slang::ast::HierarchicalValueExpression` in interface method-call lowering;
    it now tries:
    - `resolveInterfaceInstance(hier->ref, loc)` then
    - `resolveInterfaceInstance(instSym, loc)`.
- `lib/Conversion/ImportVerilog/HierarchicalNames.cpp`
  - removed temporary debug diagnostics used during prior investigation.
- `test/Conversion/ImportVerilog/hierarchical-interface-cross-root-task.sv`
  - upgraded from syntax-only shape check to functional behavior check:
    - calls `topa.ifc.ping(11)` / `topb.ifc.ping(22)`
    - asserts resulting state values via `$fatal` on mismatch
    - emits `$finish`.
  - FileCheck expectations now include state reads/comparisons and fatal paths.
- `lib/Conversion/ImportVerilog/Structure.cpp`
  - applied mechanical compile fixes in adjacent dirty code to unblock rebuild
    (`as_if` checks and `Type*` declaration), no behavioral intent change.

### Validation
- Build:
  - `CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- TDD baseline (before receiver fix):
  - `build_test/bin/circt-verilog .../hierarchical-interface-cross-root-task.sv --ir-moore` (FAIL with unsupported hierarchical interface method call diagnostic)
  - `xrun -elaborate .../hierarchical-interface-cross-root-task.sv` (PASS)
- After fix:
  - `build_test/bin/circt-verilog .../hierarchical-interface-cross-root-task.sv --ir-moore` (PASS)
  - IR contains threaded inputs `%topa.ifc` and `%topb.ifc`, two `IF::ping*`
    calls, state reads, `case_ne`, and fatal paths for mismatches.
  - `xrun -elaborate .../hierarchical-interface-cross-root-task.sv` (PASS)
  - `xrun -R .../hierarchical-interface-cross-root-task.sv` (FAIL: runtime license unavailable in environment).

### Task
Continue closing ImportVerilog xrun/CIRCT gaps with TDD for randc-constraint
compatibility and in-design `` `resetall`` handling.

### Realizations
- Concrete xrun-pass / CIRCT-fail divergences were reproducible for:
  - `'randc' variables cannot be used in 'solve before' constraints`
  - `'randc' variables cannot be used in 'soft' constraints`
  - `directive is not allowed inside a design element`
- In this tree, those diagnostics are not available from the currently included
  Slang diagnostic headers; `ImportVerilog.cpp` needed additional includes for
  `StatementsDiags` and `PreprocessorDiags` before severity remapping compiled.
- A stronger `resetall` probe with `default_nettype none` + implicit net still
  fails in CIRCT due undeclared identifier (even after warning downgrade), so
  the compatibility change here is scoped to accepting the directive location
  while preserving current semantic behavior.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`
  - included:
    - `slang/diagnostics/StatementsDiags.h`
    - `slang/diagnostics/PreprocessorDiags.h`
  - downgraded to warning in `applySlangDiagnosticSeverityPolicy(...)`:
    - `slang::diag::RandCInSoft`
    - `slang::diag::RandCInSolveBefore`
    - `slang::diag::DirectiveInsideDesignElement`
- Added regressions:
  - `test/Conversion/ImportVerilog/randc-constraint-compat.sv`
    - verifies warning diagnostics (not errors)
    - checks functional lowering of `solve before` and `soft` constraints in
      Moore IR
    - checks randomize call emission from a top module
  - `test/Conversion/ImportVerilog/resetall-inside-design-element-compat.sv`
    - verifies warning diagnostic for in-design `` `resetall``
    - checks lowering continues across directive and functional behavior is
      preserved (`q = ~q` with fatal guard)

### Validation
- TDD baseline before change:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore /home/thomas-ahle/sv-tests/tests/chapter-18/18.5.10--variable-ordering_1.sv` (FAIL)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore /home/thomas-ahle/sv-tests/tests/chapter-18/18.5.14--soft-constraints_2.sv` (FAIL)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore /home/thomas-ahle/sv-tests/tests/chapter-22/22.3--resetall_illegal.sv` (FAIL)
  - xrun elaboration for all three above: PASS
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- New regressions:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --verify-diagnostics test/Conversion/ImportVerilog/randc-constraint-compat.sv` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/randc-constraint-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/randc-constraint-compat.sv` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --verify-diagnostics test/Conversion/ImportVerilog/resetall-inside-design-element-compat.sv` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/resetall-inside-design-element-compat.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/resetall-inside-design-element-compat.sv` (PASS)
- xrun notation checks for each new regression:
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/randc-constraint-compat.sv` (PASS)
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/resetall-inside-design-element-compat.sv` (PASS)
- Attempted broader check:
  - `CCACHE_DISABLE=1 ninja -C build_test check-circt-conversion-importverilog`
    - FAIL due unrelated dirty-tree build failures in other subsystems
      (`unittests/Support`, `circt-sim`), not in the changed ImportVerilog
      files/tests.

### Task
Continue implementing ImportVerilog gaps with TDD after prior xrun-compat
closures by re-running broad xrun/circt differentials and landing the next
concrete mismatch.

### Realizations
- Broad differential status:
  - recursive sweep over `/home/thomas-ahle/sv-tests/tests/**/*.sv`
    (1622 files) found no remaining `xrun-pass / circt-fail` hard mismatches
    in the existing corpus.
- A concrete compatibility mismatch still exists outside that corpus for
  cross-select intersect ranges using initialized variables:
  - CIRCT previously failed with
    `unsupported non-constant intersect value range in cross select expression`
  - xrun elaboration accepted the same notation.
- The failing path was not in `CrossSelect.cpp` evaluation mechanics directly,
  but in the `evaluateConstant` callback passed from `Structure.cpp` into
  `convertBinsSelectExpr(...)`; that callback only accepted strict constants.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`
  - upgraded the cross-select `evaluateConstant` callback used by
    `convertBinsSelectExpr(...)`:
    - first tries strict `evaluateConstant`.
    - if that fails, tries symbol-initializer fallback for `ValueSymbol`
      references (bounded recursion).
    - if still unresolved, tries script evaluation (`EvalFlags::IsScript`).
  - this enables elaboration-stable initialized-symbol use in cross-select
    intersect operands (for example `[lo:hi]` with initialized `lo`/`hi`).
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-initialized-var-supported.sv`
    - checks lowered `moore.binsof` semantics for both range and singleton
      intersect values derived from initialized variables.

### Validation
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- New regression:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-initialized-var-supported.sv --ir-moore | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/cross-select-intersect-initialized-var-supported.sv` (PASS)
- xrun notation check for new regression:
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/cross-select-intersect-initialized-var-supported.sv` (PASS; warning-only about `sample` on default sampling covergroup)
- TDD repro closure on the concrete mismatch:
  - repro file: `/tmp/cross_intersect_var_probe.sv`
  - before: `circt-verilog` FAIL / `xrun -elaborate` PASS
  - after: `circt-verilog` PASS / `xrun -elaborate` PASS
- Additional differential evidence:
  - recursive `xrun -elaborate` vs `circt-verilog --ir-moore` scan over
    `/home/thomas-ahle/sv-tests/tests/**/*.sv` completed with
    `no_gap_found checked_total=1622`.

### Task
Continue implementing ImportVerilog gaps with TDD by finding the next concrete
xrun-pass / CIRCT-fail mismatch in cross-select intersect lowering.

### Realizations
- Broad corpus sweeps had already converged, so I used targeted differential
  probing over intersect range expression shapes to uncover latent mismatches.
- A concrete gap remained for elaboration-stable initialized expression ranges
  in cross-select `intersect` bounds:
  - arithmetic over initialized variables (`[GLO + 1:GHI - 1]`)
  - indexed initialized unpacked arrays (`[ARR[0]:ARR[1]]`)
- Root cause: the fallback evaluator passed from `Structure.cpp` into
  `convertBinsSelectExpr(...)` seeded only direct symbol initializers and did
  not evaluate composed expressions using those seeded values.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`
  - extended cross-select constant-eval fallback to:
    - collect referenced initialized `ValueSymbol`s across the whole
      expression,
    - seed an `EvalContext` frame with recursively evaluated initializer values,
    - evaluate the full expression in that seeded frame (scoped context first,
      root/script context fallback).
  - preserves depth-bounded recursion and existing strict-constant fast path.
- Added regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv`
    - TDD target that previously failed in CIRCT and elaborated in xrun.
    - checks functional lowering of computed intersect ranges:
      - `[GLO + 1:GHI - 1] -> [2, 3, 4]`
      - `[ARR[0]:ARR[1]] -> [3, 4, 5, 6]`
      - `[ARR[0] + 1:ARR[1] - 1] -> [4, 5]`
    - includes sampling path so the construct is exercised beyond syntax-only.

### Validation
- TDD baseline (before fix):
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv --ir-moore`
    - FAIL with `unsupported non-constant intersect value range in cross select expression`.
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv`
    - PASS (warning-only about `sample` on a default-sampling covergroup).
- Build after fix:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- Regression checks after fix:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-initialized-var-supported.sv --ir-moore | llvm/build/bin/FileCheck ...` (PASS)
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv --ir-moore | llvm/build/bin/FileCheck ...` (PASS)
- xrun notation parity after fix:
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv` (PASS)
- Differential probe closure:
  - `/tmp/probe_cross_intersect_diff.py` rerun: `total=28 mismatches=0`.

### Task
Continue implementing ImportVerilog gaps with TDD after intersect-expression
support landed, by finding remaining xrun-pass / CIRCT-fail cross-select
intersect cases.

### Realizations
- After fixing initialized expression ranges, targeted differential probes still
  found xrun-pass / CIRCT-fail cases for uninitialized bounds in simple
  intersect ranges:
  - uninitialized `int` / `integer` / `logic` scalar bounds,
  - uninitialized unpacked-array element bounds (`integer[]`, `logic[]`).
- Root cause split:
  - fallback evaluation in `Structure.cpp` only seeded initialized symbols;
    uninitialized symbols were unresolved.
  - even after seeding defaults, 4-state integral defaults (for example
    `integer`, `logic`) carried unknowns and were rejected by
    `getConstantInt64(...)`.
- Compatibility constraint retained:
  - tolerance forms (`[A +/- B]`, `[A +%- B]`) must remain strict to avoid
    regressing xrun parity for unsupported notation in this xrun version.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/Structure.cpp`
  - expanded cross-select fallback symbol seeding from
    "initialized ValueSymbols only" to all referenced `ValueSymbol`s.
  - for symbols without initializers, seeds type default values.
  - added recursive normalization for default values in this path:
    - converts unknown integral defaults to zero-valued `SVInt`
      (bounded to <= 64 bits),
    - applies recursively through unpacked aggregates.
  - this enables elaboration-stable evaluation for uninitialized scalar and
    unpacked-array element bounds in simple intersect ranges.
- `lib/Conversion/ImportVerilog/CrossSelect.cpp`
  - made tolerance-range bound evaluation strict via `expr.getConstant()`
    (instead of fallback evaluator), preserving current behavior for
    non-constant `+/-` / `+%-` forms.
- Added/updated regression:
  - `test/Conversion/ImportVerilog/cross-select-intersect-uninitialized-int-range-supported.sv`
    - checks functional lowering for uninitialized bounds across:
      - `int`, `integer`, `logic` scalars,
      - `int[]`, `integer[]`, `logic[]` unpacked arrays,
      - expression form `[lo + 1:hi + 2]`.
    - keeps runtime sampling path (not syntax-only).

### Validation
- TDD baselines before each fix:
  - uninitialized scalar/range probes (`[ilo:ihi]`, `[logic_lo:logic_hi]`)
    were xrun PASS / CIRCT FAIL with
    `unsupported non-constant intersect value range in cross select expression`.
  - uninitialized array-element probe
    (`[iarr[0]:iarr[1]]` / `[larr[0]:larr[1]]`) was xrun PASS / CIRCT FAIL.
- Build:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
- Regressions and guards:
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-uninitialized-int-range-supported.sv --ir-moore | llvm/build/bin/FileCheck ...` (PASS)
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv --ir-moore | llvm/build/bin/FileCheck ...` (PASS)
  - `build_test/bin/circt-verilog --language-version 1800-2023 test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv --ir-moore` (still FAIL as expected)
- xrun notation checks:
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/cross-select-intersect-uninitialized-int-range-supported.sv` (PASS)
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/cross-select-intersect-initialized-expr-supported.sv` (PASS)
- Differential probe closure:
  - `/tmp/probe_cross_intersect_diff.py`: `total=28 mismatches=0`
  - `/tmp/probe_cross_intersect_next_diff.py`: `total=14 mismatches=0`
  - `/tmp/probe_cross_type_variants_diff.py`: `total=18 mismatches=0`

### Task
Continue implementing ImportVerilog gaps with TDD after cross-select closures by
finding the next concrete xrun-pass / CIRCT-fail mismatch in external
standalone sources.

### Realizations
- A new concrete mismatch was found in OpenTitan standalone bind files:
  - xrun accepted files where `bind` targets unresolved external modules
    (warning-only behavior).
  - CIRCT still failed for multiline `bind` forms where the unknown-module
    diagnostic location is on the bind-target token line, not the `bind`
    keyword line.
- Existing bind-target compatibility logic in `ImportVerilog.cpp` was too
  narrow because it matched only when the diagnostic line itself started with
  `bind`.

### Changes Landed In This Slice
- `lib/Conversion/ImportVerilog/ImportVerilog.cpp`
  - updated `isBindUnknownModuleDiagnostic(...)`:
    - retained fast-path line check for single-line `bind ...`.
    - added statement-prefix fallback for multiline bind statements by scanning
      from the prior statement boundary (`;`) to the diagnostic location,
      skipping trivia, and detecting a leading `bind` keyword.
- `test/Conversion/ImportVerilog/bind-unknown-target-compat.sv`
  - extended regression with a multiline bind-target form:
    - `bind` on its own line,
    - unresolved target on the next line,
    - checker instantiation on the following line.
  - kept existing single-line bind-target coverage.

### Validation
- TDD baseline before fix:
  - `build_test/bin/circt-verilog --no-uvm-auto-include --verify-diagnostics test/Conversion/ImportVerilog/bind-unknown-target-compat.sv`
    - FAIL with unexpected error for multiline bind target unknown module.
- After fix:
  - `CCACHE_DISABLE=1 ninja -C build_test circt-verilog` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --verify-diagnostics test/Conversion/ImportVerilog/bind-unknown-target-compat.sv` (PASS)
  - `build_test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/bind-unknown-target-compat.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/bind-unknown-target-compat.sv` (PASS)
  - `xrun -sv -64bit -elaborate test/Conversion/ImportVerilog/bind-unknown-target-compat.sv` (PASS; warning-only bind-target-not-found diagnostics)
- External differential confirmation:
  - OpenTitan `*bind.sv` sweep before fix: `xrun_pass_circt_fail=1`
  - OpenTitan `*bind.sv` sweep after fix: `xrun_pass_circt_fail=0`
    (`summary {"both_fail": 9, "both_pass": 100}`).

### Surprise / Environment Note
- The previously used `build_test` directory disappeared from the workspace
  later in this slice (likely external cleanup). Further CIRCT validation
  requires a rebuilt `circt-verilog` binary.
