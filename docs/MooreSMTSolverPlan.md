# Moore Constraint SMT Solver Plan

This document proposes a full SMT-based constraint solver for Moore
randomization, building on the existing SMT dialect and Z3 lowering.

## Goals
- Close the remaining ~6% of constraint coverage by supporting complex
  constraints (implication, if/else, unique, foreach, solve-before, dist).
- Reuse in-tree SMT dialect and Z3 lowering where possible.
- Provide a permissive, in-process solver backend with an SMT-LIB fallback.

## Non-goals (initial phase)
- Exhaustive or optimal distribution sampling (dist): weighted sampling plus
  SMT validation is sufficient for now.
- Full-blown optimization beyond soft constraints (no cost functions yet).
- Cross-module compilation or whole-program formal checking.

## Existing Infrastructure
- Constraint parsing into `moore.constraint.*` ops:
  `lib/Conversion/ImportVerilog/Structure.cpp`
- Range/soft constraints lowered directly to runtime stubs:
  `lib/Conversion/MooreToCore/MooreToCore.cpp`
- Runtime constraint stubs (explicitly marked placeholder for SMT/Z3):
  `lib/Runtime/MooreRuntime.cpp`
  `include/circt/Runtime/MooreRuntime.h`
- SMT dialect + Z3 lowering already in-tree:
  `include/circt/InitAllDialects.h`
  `lib/Conversion/SMTToZ3LLVM/LowerSMTToZ3LLVM.cpp`
  `docs/Dialects/SMT.md`

## Backend Choice
- Primary: Z3 (MIT), in-process via existing SMT-to-Z3 lowering.
- Secondary: SMT-LIB export to external solvers (e.g., cvc5 BSD-3).

## Proposed Architecture
1) Lower `moore.constraint.*` into the SMT dialect.
2) Emit `smt.solver` region with `smt.assert` for hard constraints.
3) Solve and extract model values for randomizable fields.
4) Write model values back to the class instance.
5) Keep the current range-only path as a fast fallback.

### Data Flow
`moore.randomize` -> new pass `MooreConstraintsToSMT` -> `smt.solver` ->
Z3 lowering (or SMT-LIB export) -> runtime model extraction ->
store to class fields.

## Runtime API Sketch
Add a generic solver entry point for randomization:

```c++
// Proposed runtime interface (sketch).
struct MooreRandVarDesc {
  const char *name;
  uint32_t bitWidth;
  uint32_t fieldOffsetBytes;
  uint8_t isSigned;
};

// SMT-LIB based randomization.
int __moore_randomize_with_smtlib(void *obj, size_t objSize,
                                  const MooreRandVarDesc *vars,
                                  size_t numVars,
                                  const char *smtlib,
                                  uint64_t seed);
```

Z3 in-process path may bypass SMT-LIB and directly query model values.

## Constraint Encoding (Moore -> SMT)
- `constraint.expr`: convert to SMT boolean term, `smt.assert`.
- `constraint.inside`: disjunction of range checks.
- `constraint.unique`: `smt.distinct` on elements.
- `constraint.implication`: `smt.implies`.
- `constraint.if_else`: encode both branches with `smt.ite` or gated asserts.
- `constraint.foreach`: unroll for static arrays; for dynamic arrays, emit
  a warning and fall back to runtime iteration (phase 2).
- `constraint.solve_before`: staged solving (see below).
- `constraint.disable`: filter out named soft constraints prior to encoding.
- `constraint.dist`: weighted sampling (see below).

## Soft Constraints
Use a two-phase solver:
1) Assert all hard constraints.
2) Attempt each soft constraint (in priority order); if UNSAT, drop it.

If Z3 Optimize is available, use MaxSAT-style soft constraints later.

## Solve-before / Staged Randomization
Partition variables into stages based on solve-before constraints:
- Solve stage 0, extract model values, substitute as constants.
- Push new solver context, solve stage 1, repeat.
- Guarantees ordering without requiring full dependency analysis.

## Distribution (dist)
Implement as a front-end sampler:
- Choose a bucket based on weights (seeded RNG).
- Assert the corresponding range/set.
- Solve; if UNSAT, resample or fall back to hard constraints only.

## Model Extraction and Storage
- For each randomized property, query the solver for a model value.
- Truncate/extend to the declared bit width.
- Store to the class field using existing property path info.

## Integration Points
### New Pass
`MooreConstraintsToSMT` in `lib/Conversion/MooreToSMT/`:
- Consumes `moore.constraint.*` ops.
- Produces `smt` dialect ops with a `smt.solver` region.

### Randomize Lowering
Update `RandomizeOpConversion` in
`lib/Conversion/MooreToCore/MooreToCore.cpp`:
- If complex constraints present, call the SMT backend path.
- If only simple ranges, keep the existing fast path.

### Tooling and CLI
Add options to `circt-verilog` or `circt-opt` pipelines:
- `--moore-use-smt` (default on if Z3 is available)
- `--moore-smtlib-export` (debug option)
- `--moore-rand-seed=<n>`

## Testing Strategy
- Unit tests for each constraint kind in `test/Conversion/MooreToSMT/`.
- Runtime tests in `unittests/Runtime/MooreRuntimeTest.cpp`:
  - Soft constraint conflict handling.
  - Dist sampling respects weights (statistical).
  - Solve-before ordering.
- End-to-end AVIP regressions using existing pipeline tests.

## Milestones
1) SMT lowering for `expr/inside/unique/implication/if_else`.
2) Z3 runtime integration for model extraction + randomize.
3) Soft constraints + staged solving.
4) Foreach + dist support.
5) Performance tuning and deterministic seeding.

## Risks and Mitigations
- Solver performance: keep fast path for simple constraints.
- Non-determinism: explicit seed parameter and randomized model blocking.
- Dynamic arrays in foreach: fall back to runtime iteration or warn.
