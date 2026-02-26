# ImportVerilog Known Limitations

This document tracks known limitations in the ImportVerilog conversion.

## Cross Bin Select Expressions (IEEE 1800-2023 Syntax 19-4)

**Status**: Partially Supported
**Severity**: Medium

### Description

ImportVerilog supports most commonly used cross-bin select forms used in
covergroup cross bins.

Supported:
- `binsof(...) [intersect {...}]`, including constant tolerance ranges
  (`[A +/- B]`, `[A +%- B]`) and open-ended ranges when they can be resolved to
  finite integral target domains
- `!binsof(...)`
- `&&` and `||` composition
- plain cross identifier selection, e.g. `bins all = X;`
- finite-domain `with (...)` filtering over candidate bin tuples
  (default / parsed `matches` policies), including explicit `default`
  coverpoint bins when their finite complement can be enumerated, and
  `ignore_bins` / `illegal_bins` targets with finite value domains, required
  coverpoint bins declared with `with (...)` clauses and finite set-expression
  bins when their domains are import-time evaluable, required transition-bin
  targets, and required `default sequence` bin targets over finite integral
  coverpoint domains, while ignoring unrelated non-finite coverpoint bin
  shapes in the same coverpoint
- `cross_set_expression` tuple lists from constants or script-evaluable helper
  expressions (e.g., `CrossQueueType` helper function returning a queue literal
  or building one via queue mutator methods (`push_back`, `push_front`,
  `insert`) including common `if` / `case` / `for` / `foreach` / `while` /
  `do-while` / `repeat` / `forever`-with-`break` control-flow construction
  patterns (including static and dynamic `foreach`) and block-local `disable`
  control flow, while returning helper-local queue variables via explicit
  `return` statements or via assignment to the function return variable, and
  helper-to-helper queue calls via both `return sub(...);`, `mk = sub(...);`,
  and local declaration initializers like `CrossQueueType t = sub(...);` when
  arguments are import-time evaluable, including `for` initializer / step
  assignment forms (e.g., `for (..., mk = sub(...); ...; ..., mk = sub(...))`)
  and common
  wrapper-expression forms such as conditional (`?:`) call selection, casted
  helper returns, and concatenation of helper-returned queue fragments

Currently unsupported (diagnosed with hard errors):
- `with` filters that require non-finite or too-large tuple enumeration
- cross-select `with` forms that semantically require non-finite coverpoint
  bin value domains (for example unevaluable/non-finite coverpoint-bin `with`
  / set-expression forms)
- `cross_set_expression` forms that cannot be evaluated at import time
  (for example, helper functions with data-dependent control flow or other
  side effects that prevent static tuple extraction)
- non-constant `intersect` value ranges in condition leaves, and unbounded
  ranges that cannot be resolved to a finite integral target domain at import
  time (for example `[0:$]` over wide/infinite domains, or non-constant
  tolerance forms like `[center +%- span]`)

Examples that are supported:

```systemverilog
bins c1 = binsof(a) intersect {0} || binsof(b) intersect {0};
bins c2 = X with (a + b < 3);
bins c3 = '{ '{1,0}, '{0,1} };
```

### Root Cause

Cross-bin lowering is represented as `moore.binsof` filters in OR-of-AND form.
`with` and `cross_set_expression` support is implemented by constructing explicit
finite tuple selections and lowering them to grouped `moore.binsof` filters.
This requires finite enumeration and still rejects non-evaluable/dynamic forms
that cannot be resolved during import.

### Detection Behavior

Unsupported forms are rejected during ImportVerilog lowering with explicit
diagnostics instead of being silently ignored.

### Files Affected

- `lib/Conversion/ImportVerilog/CrossSelect.cpp`
- `lib/Runtime/MooreRuntime.cpp` (cross-bin grouped filter matching)

### Validation

Regression tests:
- `test/Conversion/ImportVerilog/cross-select-or-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-nested-or-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-explicit-bins-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-default-coverpoint-bin-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-ignore-coverpoint-bin-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-illegal-coverpoint-bin-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-unreferenced-transition-bin-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-unreferenced-coverpoint-bin-with-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-coverpoint-bin-with-target-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-coverpoint-id-with-target-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-coverpoint-set-target-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-transition-target-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-default-sequence-target-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-default-sequence-value-filter-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-with-wide-default-bin-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-literal-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-pushfront-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-insert-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-returnvar-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-local-queue-pushfront-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-forever-break-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-disable-block-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-dynamic-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-helper-assign-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-call-then-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-local-decl-init-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-init-assign-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-step-assign-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-conditional-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-assign-conditional-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-cast-call-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-return-concat-calls-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-if-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-case-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-forloop-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-foreach-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-while-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-dowhile-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-setexpr-function-repeat-pushback-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-crossid-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-intersect-open-range-unsupported.sv`
- `test/Conversion/ImportVerilog/cross-select-intersect-open-range-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-intersect-tolerance-range-supported.sv`
- `test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv`
