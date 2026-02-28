# Project Gaps Manual Writeup

This file is a manual pass over `out/project-gap-todo.all.txt`, in source order, with one judgment paragraph per entry.

## Entries 1-40

### [ ] 1. `utils/check_avip_circt_sim_mode_parity.py:94`
This is mostly a strict input-validation path, not a product feature gap by itself: the script rejects allowlist rows whose mode is not `exact`, `prefix`, or `regex`. What is missing is either broader syntax support for mode aliases (for example `contains` or `glob`) or clearer user-facing documentation that only those three modes are valid. The practical fix is to choose one direction explicitly: either extend the parser and matching engine with additional modes plus tests, or keep the current grammar and document it prominently in the allowlist format spec and error help text.

### [ ] 2. `utils/run_mutation_cover.sh:406`
This guard currently limits mutation input format to `il|v|sv`, which blocks experiments that might naturally emit other formats (for example direct MLIR/FIRRTL artifacts) before conversion. The missing piece is either native support for extra formats or a normalized pre-conversion step so the script can accept broader inputs safely. A good fix is to add a format adapter layer that maps new formats into one of the existing internal representations, then add end-to-end CLI tests proving each accepted format reaches execution and is not silently misparsed.

### [ ] 3. `utils/summarize_circt_sim_jit_reports.py:113`
Like entry 1, this is a strict parser decision point: unknown allowlist modes are rejected rather than interpreted. The gap is not correctness in current behavior, but missing flexibility and possibly inconsistent expectations across tools that use similar allowlist files. The fix should be shared with the other allowlist consumers via a common parser module so mode semantics are centralized, and tests should assert identical behavior for accepted and rejected modes across all scripts.

### [ ] 4. `unittests/Runtime/MooreRuntimeTest.cpp:10074`
`unimplemented_test` here is intentionally used to verify warning behavior for an unregistered UVM test, so this line is not itself a missing implementation. The underlying gap signal is that grep-based scans treat this expected test fixture string as a real product TODO, which creates noise in project-level audits. The fix is to keep the behavior test, but tag the scan pipeline to suppress known test-fixture literals (or support a local suppression annotation) so audit output better tracks true implementation debt.

### [ ] 5. `unittests/Runtime/MooreRuntimeTest.cpp:10081`
This assertion confirms that the runtime warning output contains the string `unimplemented_test`, which again is deliberate negative-case testing and not an unfinished runtime feature. The missing capability is better audit precision: today, mechanical scanning cannot distinguish “testing an unimplemented path” from “this codebase still needs implementation.” The fix is to add scanner exclusions for known test assertions, or move such sentinel names to values that don’t trigger TODO-gap regexes while preserving test intent.

### [ ] 6. `unittests/Analysis/CMakeLists.txt:1`
This is a real TODO: linting tests are disabled pending “slang header fixes,” so analysis-lint coverage is currently absent from regular test execution. What is missing is the dependency hygiene or include structure needed to compile and run that test subtree reliably. The concrete fix is to resolve the header incompatibilities, re-enable `add_subdirectory(Linting)`, and add CI coverage so the linting target stays live instead of regressing back to a commented-out state.

### [ ] 7. `utils/run_opentitan_fpv_circt_lec.py:414`
The option text explicitly says cover evidence emission is off by default because case-level LEC checks cannot yet model native cover reachability semantics, so this is a genuine capability gap. What is missing is a trustworthy semantic bridge between FPV cover objectives and the case-level LEC health model. The fix is to define and implement that mapping first (including edge cases like vacuous/unreachable cover points), then enable cover evidence by default only after parity tests with the reference flow confirm interpretation is correct.

### [ ] 8. `utils/formal/lib/runner_common.py:117`
This is another allowlist-mode parser with strict rejection for unknown modes, mirroring entries 1 and 3. The gap is ecosystem consistency and maintainability: three separate parsers can drift over time and create confusingly different user behavior. The best fix is to consolidate allowlist parsing into one shared library API used by all formal/analysis scripts, with a single test suite that validates supported modes and diagnostics.

### [ ] 9. `utils/run_sv_tests_circt_sim.sh:516`
The script infers `UNSUPPORTED` by grepping free-form log text for phrases like `unsupported` or `not yet implemented`, which is brittle and can misclassify failures. What is missing is a structured machine-readable failure taxonomy from `circt-verilog`/`circt-sim` (for example diagnostic IDs or explicit status codes). The right fix is to key classification on structured diagnostics rather than substring matching, then keep a compatibility fallback only for legacy logs.

### [ ] 10. `utils/run_sv_tests_circt_sim.sh:517`
Setting `result="UNSUPPORTED"` from the grep heuristic encodes policy in a way that can hide real regressions if unrelated errors contain matching words. The missing piece is confidence that this bucket means “known feature gap” instead of “random failure with unfortunate wording.” The fix is to gate this assignment behind explicit diagnostic classes (or allowlisted IDs) and add regression tests showing true unsupported cases and true failures are separated deterministically.

### [ ] 11. `tools/circt-sim-compile/circt-sim-compile.cpp:630`
This rejection reason for `builtin.unrealized_conversion_cast:unsupported_arity` indicates the compilability check only handles a restricted cast arity shape and bails on others. What is missing is legalization support for multi-input/multi-result conversion-cast patterns that can appear after upstream transformations. The fix is to implement arity-general handling (or normalize such casts earlier), and add tests with representative multi-arity casts to ensure they are either lowered correctly or rejected with precise actionable diagnostics.

### [ ] 12. `tools/circt-sim-compile/circt-sim-compile.cpp:1417`
The `unsupported` tracking flag marks a conservative synthesis strategy for native module-init generation: unsupported constructs cause full module-init fallback. The gap is functional coverage, because currently many modules are skipped even when most of their init body is otherwise compatible. The fix is to incrementally widen the accepted op subset and dependency forms while preserving safety guarantees, then measure emitted-module ratio improvements with regression metrics.

### [ ] 13. `tools/circt-sim-compile/circt-sim-compile.cpp:1427`
`unsupported_call:<callee>` shows direct calls outside the allowlisted native-init set are treated as blockers, which prevents native init emission for modules with otherwise simple call usage. What is missing is a call-lowering path for a vetted subset of callees (for example pure helper functions with legal argument/result types). The fix is to add call eligibility analysis plus lowering support for that subset, and keep explicit skip reasons for remaining unsafe callees.

### [ ] 14. `tools/circt-sim-compile/circt-sim-compile.cpp:1429`
`unsupported_call:indirect` is a hard stop for indirect calls in module-init synthesis, which is understandable for safety but leaves capability on the table. What is missing is at least a restricted strategy for cases where the target set is statically known or single-target after analysis. The fix is to add conservative devirtualization/constant-target detection and support that narrow case, while retaining skip behavior for truly dynamic indirect calls.

### [ ] 15. `tools/circt-sim-compile/circt-sim-compile.cpp:1431`
The generic `unsupported_op:<name>` reason is broad and currently catches any non-native-init op, which can make diagnosis coarse and make feature expansion less targeted. The missing piece is finer-grained capability accounting per op family so developers can prioritize highest-impact additions. The fix is to split this into explicit op-category checks and maintain per-category counters/tests, so newly supported ops can be added deliberately without broad behavior changes.

### [ ] 16. `tools/circt-sim-compile/circt-sim-compile.cpp:1434`
Setting `unsupported = true` here causes an immediate break and module skip once an unsupported op is found, even if remaining ops are harmless. What is missing is partial extraction capability that could still compile a safe prefix/slice of init behavior. The fix is to decide whether partial semantics are acceptable for module-init; if yes, implement safe slicing plus explicit residual handling, and if not, keep full-skip semantics but improve reporting so users can quickly see the first blocking op.

### [ ] 17. `tools/circt-sim-compile/circt-sim-compile.cpp:1439`
`unsupported_op:hw.struct_extract` indicates struct extraction support is limited to a constrained form (`isSupportedNativeModuleInitStructExtract`). The missing behavior is broader handling of struct extraction shapes that are semantically safe for init-time evaluation. The fix is to expand the support predicate and corresponding lowering paths stepwise, with focused tests for each newly admitted struct-extract pattern.

### [ ] 18. `tools/circt-sim-compile/circt-sim-compile.cpp:1440`
This flag assignment is the control point that enforces the struct-extract limitation from entry 17. The capability gap is not this line itself but the narrow admissible struct-extract set behind it. The fix is to broaden that admissible set and keep this guard as the final safety net, while recording distinct skip reasons for every still-unsupported variant.

### [ ] 19. `tools/circt-sim-compile/circt-sim-compile.cpp:1446`
`unsupported_op:hw.struct_create` shows struct construction is also only partially supported during native module-init synthesis. The missing feature is handling for more struct-create forms (especially those composed from already-supported operands and deterministic control flow). The fix is to define legal struct-create criteria, implement lowering/mapping for those cases, and add regression tests that prove both acceptance and correct emitted values.

### [ ] 20. `tools/circt-sim-compile/circt-sim-compile.cpp:1447`
As with entry 18, this is the switch that enforces the struct-create support boundary. What is missing remains wider verified struct-create support rather than changes to the boolean itself. The fix is to evolve `isSupportedNativeModuleInitStructCreate` with test-first additions, then keep this fast-fail behavior for residual unsupported shapes.

### [ ] 21. `tools/circt-sim-compile/circt-sim-compile.cpp:1453`
Region-bearing ops (other than `scf.if`) are currently rejected for native init, which excludes loops/switch-like constructs even when they may be statically evaluable. The missing capability is a safe structural subset of region ops for init-time compilation. The fix is to admit additional region ops one by one with strict constraints (boundedness, side-effect profile, supported operand types) and introduce semantic tests to verify equivalence with interpreter behavior.

### [ ] 22. `tools/circt-sim-compile/circt-sim-compile.cpp:1467`
This blocker rejects module body operands that are block arguments unless they match a narrow probe exception. The missing support is richer handling of block-argument-fed values during native init synthesis, especially when provenance is still deterministic and read-only. The fix is to extend block-argument admissibility analysis (including explicit source categories), then add tests for accepted and rejected cases so accidental over-acceptance does not creep in.

### [ ] 23. `tools/circt-sim-compile/circt-sim-compile.cpp:1480`
Dependencies on “skipped ops” currently force unsupported status for dependent ops, which can cascade and eliminate otherwise valid module-init candidates. The missing behavior is dependency rewriting for specific skipped-op patterns that can be normalized into supported forms. The fix is to add targeted rewrites (as already done for some probe patterns) and verify that transformed dependencies preserve runtime semantics before being admitted to native init emission.

### [ ] 24. `tools/circt-sim-compile/circt-sim-compile.cpp:1484`
`operand_dep_unsupported:<op>` captures unsupported producers in the same block, but the diagnostic granularity remains low for triage because it reports only the producer op name. The missing piece is richer diagnostic context such as consumer op, operand index, and maybe a compact value trail. The fix is to enrich skip reason metadata and surface it in summary reports, enabling developers to prioritize the most common dependency blockers.

### [ ] 25. `tools/circt-sim-compile/circt-sim-compile.cpp:1486`
This is the enforcement point for unsupported dependencies from entry 24. What is missing is not the guard itself, but support for more producer ops or conversion paths that would avoid tripping it. The fix is to progressively legalize high-frequency producer patterns and keep this fallback for genuinely unsupported dependencies.

### [ ] 26. `tools/circt-sim-compile/circt-sim-compile.cpp:1492`
Breaking out as soon as `unsupported` is observed gives deterministic control flow, but it also means only the first blocker is discovered per module. The missing capability is multi-reason collection for better developer feedback and faster gap reduction. The fix is to add an optional “collect all blockers” analysis mode for diagnostics while preserving first-failure short-circuit in the fast path.

### [ ] 27. `tools/circt-sim-compile/circt-sim-compile.cpp:1497`
`if (unsupported || opsToClone.empty())` skips module-init emission even for empty-op modules, conflating “unsupported” and “nothing to emit.” The missing clarity is separate accounting for each outcome so optimization and diagnostics can distinguish a true feature gap from a no-op case. The fix is to split counters/reasons for `unsupported` versus `empty`, and reflect that separation in emitted stats and regression expectations.

### [ ] 28. `tools/circt-sim-compile/circt-sim-compile.cpp:1498`
The nested `if (unsupported)` branch only increments skip reasons when unsupported, but still shares outer control flow with empty-op skips. The missing piece is cleaner semantics for reporting and downstream tooling that consumes these stats. The fix is to isolate unsupported handling into its own branch and ensure every skip path has an explicit, semantically meaningful reason code.

### [ ] 29. `tools/circt-sim-compile/circt-sim-compile.cpp:1500`
`unsupported:unknown` is a useful fallback but also a quality-of-diagnostics smell: it means a blocker occurred without a specific reason string. What is missing is complete reason coverage for every early-exit path. The fix is to audit all branches that set `unsupported`/`cloneUnsupported`, guarantee they populate structured reason codes, and add a regression assertion that unknown skip reasons remain at zero in normal test suites.

### [ ] 30. `tools/circt-sim-compile/circt-sim-compile.cpp:1514`
`cloneUnsupported` starts a second unsupported-tracking phase during op cloning, indicating that pre-scan acceptance does not fully guarantee clone-time success. The missing capability is a single unified admissibility model that makes clone failure rare and predictable. The fix is to push clone-time constraints back into the pre-scan predicates where possible, reducing double-phase failure and making skip reasons easier to interpret.

### [ ] 31. `tools/circt-sim-compile/circt-sim-compile.cpp:1521`
This clone-time unsupported branch fires when probing a block argument that fails supported probe constraints, even after earlier filtering. What is missing is either stronger up-front filtering or broader clone-time handling of probe forms. The fix is to align `isSupportedNativeModuleInitBlockArgProbe` checks across both scan and clone stages and add dedicated tests for every rejected probe subtype.

### [ ] 32. `tools/circt-sim-compile/circt-sim-compile.cpp:1550`
Here clone-time support is limited to integer and pointer probe result types, rejecting other result kinds outright. The missing feature is typed conversion support for additional legal probe result types that could be faithfully represented in the generated init function. The fix is to enumerate and implement safe conversions for additional types (with explicit width/sign semantics), and retain rejection for types lacking a clear ABI-safe mapping.

### [ ] 33. `tools/circt-sim-compile/circt-sim-compile.cpp:1575`
This branch erases a partially built init function whenever cloning encountered unsupported content, preserving correctness but losing any work that might have been salvageable. The missing capability is either transactional partial emission with verified semantics or richer artifact retention for debugging. The fix is to keep correctness-first erase behavior, but optionally emit debug metadata or temporary IR traces explaining exactly what blocked finalization.

### [ ] 34. `tools/circt-sim-compile/circt-sim-compile.cpp:1577`
`cloneSkipReason = "unsupported:unknown"` mirrors the earlier unknown-reason fallback and has the same triage weakness. The missing part is complete reason initialization for clone-time failure branches. The fix is to force every `cloneUnsupported = true` site to assign a specific code and to add tests or runtime asserts that unknown clone reasons are not produced in covered paths.

### [ ] 35. `tools/circt-sim-compile/circt-sim-compile.cpp:1592`
The comment admits that support for post-SCF block-argument `hw.struct_extract` forms is still incomplete, and the code works around that with a conservative rewrite. What is missing is first-class handling of those forms without requiring this special-case rewrite path. The fix is to generalize struct-extract lowering across SCF boundaries, then simplify or retire this compensating pass once direct support is robust.

### [ ] 36. `tools/circt-sim-compile/circt-sim-compile.cpp:1859`
This comment describes a deliberate degradation strategy in format-string lowering: unsupported fragments become placeholders instead of causing hard failure. The missing capability is complete `sim.fmt.*` coverage for printable fragments so placeholder text rarely appears in production logs. The fix is to expand fragment support by op/type combination and keep placeholders only as a last-resort escape hatch with explicit diagnostics when triggered.

### [ ] 37. `tools/circt-sim-compile/circt-sim-compile.cpp:1872`
If a format value has no defining op, lowering injects `<unsupported>`, which avoids crashes but loses user-visible fidelity. The missing behavior is robust handling of block arguments or externally produced format values in this path. The fix is to add explicit support for those value origins (or an upstream normalization guaranteeing local definers), then validate with tests where format values flow through control/data boundaries.

### [ ] 38. `tools/circt-sim-compile/circt-sim-compile.cpp:1899`
For decimal formatting, failure to convert the integer into a `printf`-compatible width currently emits `<unsupported>`. The missing capability is conversion support for more integer-like value kinds that can still be represented safely in native printing. The fix is to widen `convertIntegerForPrintf` coverage (or insert legalizing casts before formatting), and add tests proving decimal output stays correct across widths and signedness.

### [ ] 39. `tools/circt-sim-compile/circt-sim-compile.cpp:1911`
The same placeholder fallback appears for hexadecimal formatting when conversion fails, so hex rendering coverage is still incomplete for some value types. What is missing is a reliable path from supported IR integer forms to hex-printable operands without dropping content. The fix is to close remaining conversion gaps and add targeted tests for hex formatting on boundary widths/types that currently fall through to `<unsupported>`.

### [ ] 40. `tools/circt-sim-compile/circt-sim-compile.cpp:1923`
Octal formatting has the same fallback behavior as decimal and hex, which signals shared conversion limitations rather than an octal-specific bug. The missing piece is comprehensive conversion normalization before selecting radix-specific format verbs. The fix is to centralize integer normalization once, reuse it across radix handlers, and add cross-radix parity tests so one conversion bug cannot silently affect only one format mode.

### [ ] 41. `tools/circt-sim-compile/circt-sim-compile.cpp:1936`
Binary formatting currently falls back to `<unsupported>` when argument conversion fails, even though the operation has already selected a hex-based fallback verb for binary output. What is missing is robust conversion support that guarantees the value reaches a print-compatible integer representation in this path. The fix is to harden conversion of integer-like operands before formatting and add tests where `sim.fmt.bin` exercises edge-width and cast-heavy values.

### [ ] 42. `tools/circt-sim-compile/circt-sim-compile.cpp:1948`
Character formatting inserts `<unsupported>` when the value cannot be converted to a 32-bit integer for `%c`, which silently degrades output fidelity. The missing capability is reliable narrowing/normalization for char-printable sources that are semantically valid but not already in the expected type. The fix is to extend conversion coverage for char operands and verify non-ASCII and boundary-value behavior with explicit runtime checks.

### [ ] 43. `tools/circt-sim-compile/circt-sim-compile.cpp:1961`
Scientific formatting currently supports `f64` and widening from `f32`, but all other float-like or numeric forms degrade to `<unsupported>`. What is missing is a broader numeric-to-float bridge for cases that are semantically safe to print in scientific notation. The fix is to decide which additional source types are legal (for example integer promotion rules), implement those casts explicitly, and add semantic comparison tests against reference simulator output.

### [ ] 44. `tools/circt-sim-compile/circt-sim-compile.cpp:1975`
The `%f` path has the same conversion boundary as scientific mode, meaning unsupported source types are dropped to placeholders instead of being printed. The missing piece is consistent floating-point print coverage across all float format ops. The fix is to unify float conversion policy across `%f/%e/%g`, implement it once, and lock behavior with a shared test matrix across these three formatting operations.

### [ ] 45. `tools/circt-sim-compile/circt-sim-compile.cpp:1989`
General floating formatting (`%g`) still emits `<unsupported>` for types outside the narrow accepted set, so behavior depends on frontend typing details instead of user intent. What is missing is tolerant but principled coercion for values that can be represented as `double` without ambiguity. The fix is to add a controlled coercion ladder and add parity tests proving `%g` output aligns with `%f/%e` conversions where mathematically equivalent.

### [ ] 46. `tools/circt-sim-compile/circt-sim-compile.cpp:2009`
Dynamic-string formatting requires a specific `{ptr,len}` struct shape and still falls back to `<unsupported>` if `len` cannot be converted for `%.*s`. The missing capability is broader acceptance of length field representations that are common in lowered IR. The fix is to broaden integer-length normalization (and fail loudly when unsafe), then add tests for multiple length widths/signs so string truncation behavior is deterministic.

### [ ] 47. `tools/circt-sim-compile/circt-sim-compile.cpp:2026`
This final catch-all `<unsupported>` means any unrecognized `sim.fmt.*` producer silently appears as a placeholder instead of hard-failing lowering. The missing part is visibility: placeholder rendering is useful for liveness, but without explicit diagnostics developers may not realize formatting semantics were dropped. The fix is to keep the placeholder fallback but emit structured diagnostics/counters whenever this path is hit, and add a regression that asserts those counters remain zero on supported suites.

### [ ] 48. `tools/circt-sim-compile/circt-sim-compile.cpp:3487`
This `unsupported` flag is used in call-indirect lowering to LLVM calls, indicating the lowering path is intentionally limited to ABI-compatible argument/return types. The gap is that any unsupported type causes the rewrite to be skipped entirely, leaving potential performance or functionality on the table for partially legal calls. The fix is to incrementally extend the supported ABI surface with test-first additions and retain explicit skips for truly non-lowerable cases.

### [ ] 49. `tools/circt-sim-compile/circt-sim-compile.cpp:3493`
Setting `unsupported = true` when argument type lowering fails means there is no fallback conversion for that argument beyond the current compatibility helper. The missing capability is richer ABI adaptation for argument types that are representable but not currently recognized as LLVM-compatible. The fix is to expand type conversion support (or pre-normalize signatures) and add tests where previously skipped indirect calls are now lowered correctly.

### [ ] 50. `tools/circt-sim-compile/circt-sim-compile.cpp:3499`
The return-type path only proceeds when arguments were all supported, so return handling is gated behind earlier success and cannot provide independent diagnostics for mixed failures. What is missing is better reason reporting when both arguments and return types are problematic. The fix is to collect independent arg/ret compatibility outcomes for diagnostics while preserving a single skip decision for lowering.

### [ ] 51. `tools/circt-sim-compile/circt-sim-compile.cpp:3504`
Return types that cannot be lowered to LLVM-compatible form trigger unsupported status, preventing indirect-call conversion. The missing behavior is return-type bridging for additional legal forms that can be packed/unpacked safely. The fix is to add result-type adaptation rules (or trampoline-based fallback for this rewrite site) and include regression tests for representative formerly-unsupported return signatures.

### [ ] 52. `tools/circt-sim-compile/circt-sim-compile.cpp:3508`
`if (unsupported) continue;` silently drops the rewrite opportunity without recording why a specific call_indirect stayed unlowered. The missing part is observability for skipped transformations in optimization/debug workflows. The fix is to add optional skip diagnostics or counters keyed by failure reason so users can understand why indirect calls remain in the IR.

### [ ] 53. `tools/circt-sim-compile/circt-sim-compile.cpp:3526`
Argument materialization failures during operand adaptation set `unsupported` and abandon the conversion, indicating gaps in value-level casting even when signature-level type checks passed. What is missing is robust per-operand bridging for hard cases (for example mixed dialect/value forms). The fix is to strengthen `materializeLLVMValue` coverage and add tests that exercise operand conversions under call-indirect rewriting.

### [ ] 54. `tools/circt-sim-compile/circt-sim-compile.cpp:3532`
This second `if (unsupported) continue;` repeats the silent-skip pattern after operand rewriting, which can make transformed coverage hard to reason about. The missing capability is structured reporting of rewrite success/failure rates at pass scope. The fix is to track conversion attempts and skip reasons in pass statistics and expose them in logs so unsupported cases are actionable.

### [ ] 55. `tools/circt-sim-compile/circt-sim-compile.cpp:3825`
Returning zero slots for a nested unsupported field type in trampoline ABI flattening blocks trampoline generation for composite signatures that include that field. The missing capability is flattening support for more nested type shapes that can still map to the `uint64_t` slot convention. The fix is to extend `countTrampolineSlots` and corresponding pack/unpack logic together, with round-trip tests on nested arrays/structs.

### [ ] 56. `tools/circt-sim-compile/circt-sim-compile.cpp:3830`
The base-case `return 0; // unsupported` marks any unrecognized type as ABI-inexpressible for trampoline dispatch. The gap is breadth: many types may be semantically supportable but are currently excluded by this conservative default. The fix is to explicitly enumerate additional supported kinds and keep this zero-slot fallback only for genuinely unsafe/unmapped types.

### [ ] 57. `tools/circt-sim-compile/circt-sim-compile.cpp:4171`
This comment documents a real architectural gap: trampolines are needed specifically because some functions still contain unsupported ops and cannot be compiled directly. What is missing is broader direct compilation coverage that reduces reliance on interpreted fallback boundaries. The fix is to continue expanding unsupported-op coverage in the compiled path while keeping trampoline fallback as a compatibility safety net.

### [ ] 58. `tools/circt-sim-compile/circt-sim-compile.cpp:4215`
`sawUnsupportedReferencedExternal` tracks whether referenced externals could not be trampolined, which currently forces failure for that generation pass. The missing capability is partial progress handling where unsupported externals are isolated without aborting all trampoline generation. The fix is to decide policy (strict vs best-effort) explicitly and, if best-effort is allowed, emit precise diagnostics while still generating valid trampolines for supported externals.

### [ ] 59. `tools/circt-sim-compile/circt-sim-compile.cpp:4253`
Vararg externals are currently rejected for trampoline generation and set the global unsupported flag. The missing behavior is a defined vararg ABI strategy, which is difficult but sometimes necessary for real-world C library interop. The fix is either to implement a constrained vararg trampoline ABI or to preserve strict rejection but improve tooling guidance so users can route such calls through direct extern linkage paths.

### [ ] 60. `tools/circt-sim-compile/circt-sim-compile.cpp:4259`
`hasUnsupported` begins ABI compatibility checks for function parameters/returns, and any unsupported part currently rejects trampoline generation for that function. The missing piece is finer-grained ABI support and clearer distinction between “unsupported forever” and “not implemented yet.” The fix is to turn this check into a capability table with explicit per-type rationale and to grow it in a controlled, tested manner.

### [ ] 61. `tools/circt-sim-compile/circt-sim-compile.cpp:4260`
This dedicated `unsupportedReason` string is useful, but only captures a single first failure reason and can hide secondary incompatibilities. What is missing is multi-reason reporting that helps prioritize the dominant blockers across a codebase. The fix is to collect all incompatible ABI features for a symbol in diagnostic mode, while still selecting one canonical reason for concise normal-mode output.

### [ ] 62. `tools/circt-sim-compile/circt-sim-compile.cpp:4265`
Setting `hasUnsupported = true` when a parameter flattens to zero slots is the exact point where non-flattenable parameter types are excluded. The missing capability is type flattening support for those parameter types or a fallback passing convention that can represent them. The fix is to extend slot-packing semantics for additional legal types and validate with interpreter trampoline round-trip tests.

### [ ] 63. `tools/circt-sim-compile/circt-sim-compile.cpp:4266`
The parameter-type reason string is informative but still plain text and not machine-typed, making automated triage across many failures harder. The missing part is structured diagnostic metadata for unsupported ABI reasons. The fix is to emit stable reason codes plus human text so dashboards and scripts can aggregate top blockers reliably.

### [ ] 64. `tools/circt-sim-compile/circt-sim-compile.cpp:4271`
Return-type checking is deferred until parameter checks pass, which is efficient but can obscure full incompatibility profiles for a function signature. The missing behavior is optional comprehensive validation for diagnostic workflows. The fix is to add a non-fast-path mode that inspects both params and return unconditionally and reports combined ABI incompatibilities.

### [ ] 65. `tools/circt-sim-compile/circt-sim-compile.cpp:4275`
Unsupported return type detection currently depends on zero-slot flattening and collapses all failure causes into one branch. The missing detail is differentiation between specific return-type incompatibility classes (for example nested aggregate vs unsupported primitive kind). The fix is to expose typed reason codes at this decision point and prioritize implementation based on observed frequency.

### [ ] 66. `tools/circt-sim-compile/circt-sim-compile.cpp:4276`
Guarding reason assignment behind `if (hasUnsupported)` is correct, but it currently overwrites context with a broad “return type …” message when return checking fails. The missing capability is retaining richer causal context for debugging (e.g., exact nested field causing flatten failure). The fix is to thread inner flatten diagnostics up to this point and include them in the final reason text/code.

### [ ] 67. `tools/circt-sim-compile/circt-sim-compile.cpp:4277`
The return-type `unsupportedReason` payload is human-readable but not normalized, which limits automated bucketing across runs. What is missing is stable ABI error coding that still preserves rich type text for humans. The fix is to emit both a compact reason code and full rendered type string, then use the code in summaries and the text in detailed logs.

### [ ] 68. `tools/circt-sim-compile/circt-sim-compile.cpp:4279`
This branch rejects trampoline generation whenever unsupported ABI traits are found, prioritizing correctness over degraded execution. The missing capability is any alternative execution path for those specific extern signatures besides failing the overall trampoline-generation step. The fix is to either add fallback dispatch mechanisms for more signatures or fail earlier with clearer user guidance on how to avoid unsupported extern shapes.

### [ ] 69. `tools/circt-sim-compile/circt-sim-compile.cpp:4282`
The emitted diagnostic text `unsupported trampoline ABI (...)` is accurate but still relies on free-form messaging. The missing piece is integration with structured diagnostics so tooling can classify this failure without string matching. The fix is to attach a stable diagnostic ID/category to this error and add tests that assert both message clarity and ID stability.

### [ ] 70. `tools/circt-sim-compile/circt-sim-compile.cpp:4284`
Marking `sawUnsupportedReferencedExternal = true` ensures failures are propagated, but it also means one unsupported external can fail a larger compile scope. The missing capability is policy control: some workflows may prefer warnings and partial output over hard failure. The fix is to introduce a strictness option (strict by default) that can downgrade selected unsupported-external cases to non-fatal diagnostics when explicitly requested.

### [ ] 71. `tools/circt-sim-compile/circt-sim-compile.cpp:4283`
The line that prints `unsupportedReason` into the trampoline ABI error is useful but still too coarse when debugging nested aggregate ABI failures. What is missing is deeper reason decomposition so users can see exactly which nested field or shape made flattening impossible. The fix is to preserve and print nested flatten context (for example field index paths) and add regression tests that assert actionable diagnostics for complex unsupported signatures.

### [ ] 72. `tools/circt-sim-compile/circt-sim-compile.cpp:4289`
`if (sawUnsupportedReferencedExternal) return failure();` enforces all-or-nothing behavior for trampoline generation. The missing capability is best-effort output when only a subset of externals are problematic. The fix is to make this policy configurable and, in non-strict mode, emit supported trampolines while reporting unsupported externals as structured warnings.

### [ ] 73. `tools/circt-sim-compile/circt-sim-compile.cpp:7498`
This unsupported flag appears in a separate declaration-cloning path where `func::FuncOp` types are converted into `LLVMFuncOp` declarations for referenced symbols. The gap is that only a narrow type subset (integers, pointers, floats, index) is accepted, so valid-but-richer function signatures are silently skipped. The fix is to broaden signature lowering support in lockstep with trampoline ABI support, and surface skip statistics so dropped declarations are visible.

### [ ] 74. `tools/circt-sim-compile/circt-sim-compile.cpp:7506`
Unsupported is set when any input type falls outside the currently handled scalar/index set, which makes declaration availability dependent on type shape rather than call reachability needs. What is missing is a fallback declaration strategy for additional argument kinds. The fix is to add explicit lowering rules for additional legal MLIR function input types and keep strict rejection only for truly unrepresentable signatures.

### [ ] 75. `tools/circt-sim-compile/circt-sim-compile.cpp:7510`
The `if (!unsupported)` gate means one unsupported input type suppresses declaration creation entirely, even if downstream code could still handle partial cases via interpreter fallback. The missing piece is clearer coordination between declaration emission and trampoline generation. The fix is to document and enforce a unified policy: either declarations are required only for trampoline-able signatures, or a broader stub strategy is introduced with explicit runtime fallback.

### [ ] 76. `tools/circt-sim-compile/circt-sim-compile.cpp:7522`
Unsupported is set for return types outside the small accepted set, so multi-value or aggregate returns are excluded from this declaration path. The missing capability is return-type lowering parity with the rest of the call boundary machinery. The fix is to support additional return encodings where safe, or require trampoline wrapping for those signatures and emit explicit reasoned diagnostics when declaration emission is skipped.

### [ ] 77. `tools/circt-sim-compile/circt-sim-compile.cpp:7525`
This branch rejects functions with more than one return value for declaration conversion, reflecting a current ABI simplification. The missing behavior is support for multi-result function signatures (or canonical tuple lowering) at this interface boundary. The fix is to introduce a canonical multi-result flattening approach and validate it with call-site and trampoline integration tests.

### [ ] 78. `tools/circt-sim-compile/circt-sim-compile.cpp:7527`
The final `if (!unsupported)` check controls whether an external declaration gets emitted at all, so unsupported signatures become invisible entries in the table. What is missing is explicit accounting of skipped declarations for developer feedback. The fix is to record skipped symbol names/reasons and expose them in compile summaries so missing declarations are discoverable without debugging IR manually.

### [ ] 79. `tools/circt-sim-compile/circt-sim-compile.cpp:7536`
The comment admits skipped entries are tolerated with null table slots when symbols are unsupported or absent, which avoids crashes but can delay failure to runtime dispatch. The missing capability is earlier user feedback for null-entry use paths. The fix is to report null-table symbol coverage at compile time and add runtime assertions that identify unresolved entries with clear symbol names when invoked.

### [ ] 80. `utils/run_mutation_mcy_examples.sh:1006`
This is a scan false positive: `mktemp ...XXXXXX` was matched because the audit regex includes `XXX`, but the line is normal temporary-file creation rather than an implementation gap. What is missing is scanner precision, not product functionality. The fix is to tighten TODO token matching to word boundaries (or require `TODO|FIXME|...` as standalone markers) so mktemp templates do not pollute the gap list.

### [ ] 81. `utils/run_mutation_mcy_examples.sh:2379`
This line is the same false-positive pattern as entry 80 (`mktemp` with `XXXXXX`), not a true unsupported/TODO gap. The missing part is the quality of the gap-discovery query. The fix is to refine scanning rules and optionally add a suppression list for known lexical false positives in shell scripts.

### [ ] 82. `utils/run_mutation_mcy_examples.sh:2380`
Again, this `mktemp ...XXXXXX` match is not technical debt in behavior; it is an artifact of using `XXX` as a broad scan token. What is missing is better discrimination between deliberate placeholders and canonical random-suffix templates. The fix is to change the audit pattern to avoid matching inside all-uppercase temp suffix literals.

### [ ] 83. `utils/run_mutation_mcy_examples.sh:2381`
This is another `XXXXXX` false positive and should not be treated as an implementation gap. The missing capability is a robust audit filter that can distinguish housekeeping shell idioms from actionable TODO markers. The fix is to post-process scan output through a false-positive filter and keep only semantically meaningful matches.

### [ ] 84. `utils/run_mutation_mcy_examples.sh:2382`
This is the fourth consecutive `mktemp` false positive in the same block and confirms the scan regex is currently over-inclusive. The missing piece is audit hygiene: without filtering, these matches dilute attention from real unsupported paths. The fix is to update the scanner and regenerate the audit so these noise entries disappear from future manual passes.

### [ ] 85. `frontends/PyRTG/src/pyrtg/support.py:60`
`assert False, "Unsupported value"` indicates the conversion helper has no fallback for unknown CIRCT value wrapper types and will hard-stop in debug-oriented style. What is missing is either comprehensive value-type coverage or an exception path that reports unsupported inputs with actionable context instead of an assertion. The fix is to replace assertion-based termination with structured exceptions and extend `_FromCirctValue` mappings for newly introduced RTG value classes.

### [ ] 86. `frontends/PyRTG/src/pyrtg/support.py:117`
`raise ValueError("unsupported type")` in `_FromCirctType` is a hard boundary for type conversion coverage in PyRTG. The missing behavior is support for additional CIRCT/RTG type nodes that users can encounter from evolving dialect features. The fix is to add new type cases as dialect support grows and include round-trip tests that confirm Python wrappers are generated for each supported type form.

### [x] 87. `test/Runtime/uvm/uvm_phase_wait_for_state_test.sv:2`
Status update (2026-02-28): this gap is closed. The test now has execution-backed `circt-sim` RUN lines and `FileCheck` assertions in addition to parse-only coverage, so runtime behavior is exercised in the standard lit workflow.

### [x] 88. `test/Runtime/uvm/uvm_phase_aliases_test.sv:2`
Status update (2026-02-28): this gap is closed. The test now includes LLHD lowering + `circt-sim` execution checks with `FileCheck`, so phase-handle alias behavior is validated beyond parse-only.

### [ ] 89. `utils/run_avip_circt_sim.sh:249`
Rejecting `AVIP_SET` values outside `core8|all9` is likely an intentional guard, but it also hard-codes set names and blocks extension without script edits. What is missing is a configurable registry for AVIP sets so users can add curated subsets without patching the script. The fix is to load sets from a config file (or env-provided manifest) and keep these strict checks as validation against that dynamic registry.

### [ ] 90. `utils/run_avip_circt_sim.sh:269`
Likewise, limiting `CIRCT_SIM_MODE` to `interpret|compile` enforces current known modes but prevents smooth introduction of new execution modes. The missing behavior is mode extensibility with clear capability discovery. The fix is to centralize mode definitions (with help text and validation) and source this script from that single definition so new modes can be added once without scattering guard updates.

### [ ] 91. `frontends/PyRTG/src/pyrtg/control_flow.py:92`
This FIXME is a real maintainability gap in PyRTG control-flow lowering: it rebuilds `scf.if` ops in a workaround because Python MLIR bindings do not support deleting region blocks cleanly in this flow. What is missing is a robust structural edit path that preserves block semantics without cloning-and-append gymnastics. The fix is to move to an API pattern that avoids block deletion requirements (or upgrade bindings and refactor), then add regression tests for nested `If/Else` rewrites to ensure SSA/value mapping stays correct.

### [ ] 92. `frontends/PyRTG/src/pyrtg/control_flow.py:174`
Setting deleted locals to `None` instead of removing them is a pragmatic workaround, but it leaks temporary names and can subtly alter user frame behavior. The missing behavior is proper local cleanup semantics after control-flow capture. The fix is to safely delete locals where possible (or isolate execution scope to avoid touching caller locals directly) and add tests asserting no stray loop/if locals remain visible after block exit.

### [ ] 93. `frontends/PyRTG/src/pyrtg/contexts.py:73`
This TODO identifies a concrete scalability issue: context capture currently hoovers up all locals, not just values actually used inside the sequence body. What is missing is dependency-minimal capture, which impacts readability and can bloat generated sequence signatures. The fix is to compute used-value sets (AST analysis or tracing) and only pass needed values into context sequences, then verify generated argument lists shrink without semantic regression.

### [ ] 94. `frontends/PyRTG/src/pyrtg/contexts.py:87`
Assuming `_context_seq` is a reserved prefix is fragile and can collide with user symbols, so this is a legitimate naming-hygiene gap. What is missing is proper symbol uniquing integrated with module symbol tables. The fix is to switch to a deterministic uniquing utility (or MLIR symbol utilities) and add tests covering collisions with user-provided names to prove generated sequence symbols remain conflict-free.

### [ ] 95. `utils/run_avip_arcilator_sim.sh:103`
Restricting `AVIP_SET` to `core8|all9` is likely intentional, but it hard-codes policy and makes extension cumbersome. The missing piece is a configurable AVIP-set registry so new curated sets can be added without editing script logic. The fix is to externalize set definitions to data (TSV/JSON/env list), validate them at startup, and keep strict erroring for unknown set names.

### [ ] 96. `utils/create_mutated_yosys.sh:95`
Rejecting design extensions outside `.il/.v/.sv` is a clear boundary that may be intentional, but it limits pipeline interoperability when upstream tools emit other forms. What is missing is either richer format support or a documented canonical pre-conversion step. The fix is to add adapters for additional common inputs where feasible, or keep strict validation but provide a companion conversion helper so users can normalize inputs automatically.

### [ ] 97. `utils/create_mutated_yosys.sh:106`
The output-extension restriction mirrors entry 96 and can block downstream consumers expecting other file types. The missing behavior is pluggable output backend selection rather than fixed extension branching. The fix is to parameterize output mode independently from filename extension and add tests that verify each backend emits the expected syntactically valid artifact.

### [ ] 98. `utils/check_opentitan_connectivity_cover_parity.py:101`
This is the same recurring allowlist grammar boundary: unsupported mode names are rejected. The missing piece is shared parsing infrastructure so all parity scripts behave identically and evolve together. The fix is to move allowlist parsing to one common helper module and cover it with shared tests used by every script that consumes allowlists.

### [ ] 99. `utils/check_opentitan_connectivity_status_parity.py:101`
This entry repeats the allowlist-mode consistency issue, and the real gap is duplication, not correctness at this call site. What is missing is de-duplication of parser logic across parity checkers. The fix is to import a shared parser API and keep script-level behavior focused on domain checks, not format parsing.

### [ ] 100. `test/CMakeLists.txt:53`
This TODO marks a real build/test coverage gap: `circt-verilog-lsp-server` is excluded from test dependencies due to slang API compatibility issues. What is missing is compatibility glue (or versioned API handling) that allows the LSP server target to build and test with the current slang frontend. The fix is to resolve API drift and re-enable the target in `CIRCT_TEST_DEPENDS`, then add CI coverage to prevent silent re-breakage.

### [ ] 101. `utils/run_opentitan_connectivity_circt_bmc.py:120`
Again this is a duplicated allowlist parser rejecting unknown modes. The missing behavior is centralized parser governance, especially because this script participates in high-volume OpenTitan workflows where consistency matters. The fix is the same shared helper approach with a single compliance test suite for `exact/prefix/regex` semantics and diagnostics.

### [ ] 102. `utils/run_opentitan_connectivity_circt_bmc.py:412`
Failing on unknown connectivity rule types in the manifest is correct for data integrity, but it also means new manifest schema variants cannot roll out incrementally. The missing capability is versioned rule-type handling and forward-compatibility strategy. The fix is to add schema-version negotiation with explicit support tables (and optional strict mode), so known-new rule types can be introduced without hard-breaking older runners.

### [ ] 103. `frontends/PyCDE/test/test_esi.py:170`
`# TODO: fixme` is a low-information TODO and therefore a process-quality gap: it flags debt but gives no actionable scope or acceptance criteria. What is missing is a precise statement of why `PureTest` is disabled and what condition would allow re-enablement. The fix is to replace this placeholder with a concrete issue description and expected behavior, then add/enable a regression once that behavior is implemented.

### [ ] 104. `include/circt/Runtime/MooreRuntime.h:4208`
This is a scanner false positive (`super.XXX_phase()` in docs), not a TODO/unsupported implementation gap. What is missing is scan precision for tokens like `XXX` when used as literal examples in comments. The fix is to tighten audit regexes (for example word-boundary TODO markers only) or maintain an allowlist for known documentation phrases.

### [ ] 105. `frontends/PyCDE/test/test_polynomial.py:120`
This TODO is real: the test notes that IR verification fails before all modules are generated because `hw.instance` references are unresolved. What is missing is staged generation/verification semantics that tolerate forward references or enforce topological emission order. The fix is to either support deferred symbol resolution during generation or reorder module emission so the intermediate IR is always verifiable.

### [ ] 106. `utils/refactor_continue.sh:6`
This is another scan false positive: `TODO_PATH` is just a variable name and not unfinished implementation. The missing issue is audit specificity, not script behavior. The fix is to exclude identifiers containing `TODO` unless they appear in recognized marker syntax (`# TODO`, `TODO:`) or known diagnostic strings.

### [ ] 107. `utils/refactor_continue.sh:14`
`plan/todo` in usage text is documentation wording, not implementation debt. The gap is scanner overreach into ordinary prose. The fix is to constrain scans to marker prefixes or code comments, not arbitrary help text.

### [ ] 108. `utils/refactor_continue.sh:18`
This `--todo` option description is similarly non-actionable and should not be treated as a gap. What is missing is classification hygiene in the audit pipeline. The fix is to add filtering rules for command-line option names that happen to contain `todo`.

### [ ] 109. `utils/refactor_continue.sh:41`
The `--todo)` case label is functional CLI parsing and not technical debt. The gap is again false-positive identification in the scan results. The fix is to filter shell `case` labels that merely include the token text.

### [ ] 110. `utils/refactor_continue.sh:42`
Assigning `TODO_PATH="$2"` is normal option processing, not a TODO marker. What is missing is semantic filtering during audit generation. The fix is to require TODO matches in comments/messages rather than variable assignments unless explicitly configured.

### [ ] 111. `utils/refactor_continue.sh:61`
Checking for file existence of `TODO_PATH` is expected runtime validation and not a feature gap. The missing piece is separating “string contains TODO” from actual TODO markers. The fix is scanner refinement rather than code changes here.

### [ ] 112. `utils/refactor_continue.sh:62`
Printing `todo file not found` is a normal error path, not unsupported functionality. The gap is audit noise introduced by matching lowercase `todo` text in user diagnostics. The fix is to restrict scan rules to upper-case marker conventions or comment annotations.

### [ ] 113. `utils/refactor_continue.sh:66`
The canonical prompt string includes `TODO_PATH` as content and is not evidence of unfinished code. What is missing is a more context-aware scanner that understands operational text versus debt markers. The fix is to run token scans on comments/diagnostics only, or add per-file suppression for known helper scripts.

### [ ] 114. `utils/refactor_continue.sh:81`
This reference to `"$TODO_PATH"` in an awk invocation is operational plumbing, not a TODO item. The gap remains scanner precision. The fix is to post-filter entries where the matched substring is part of a variable name rather than a marker.

### [ ] 115. `utils/refactor_continue.sh:92`
Same as entry 114: use of `TODO_PATH` as a parameter is functional code, not missing implementation. The fix is no code change here; it is refining how audit matches are interpreted and filtered.

### [ ] 116. `utils/refactor_continue.sh:122`
`printf 'TODO: %s\n'` is user-facing status output, not a debt marker about this script. The missing piece is distinguishing display labels from unresolved tasks during scanning. The fix is to mark this script as a known false-positive hotspot or tighten marker detection.

### [ ] 117. `utils/run_yosys_sva_circt_bmc.sh:142`
`UNSUPPORTED_SVA_POLICY` is a configuration knob name, and its appearance in the scan does not itself indicate a gap. The real underlying gap is elsewhere: policy exists because unsupported SVA constructs are still present in the flow. The fix is to keep the policy control, but track and reduce unsupported-SVA incidence so `lenient` mode becomes less necessary over time.

### [ ] 118. `utils/run_yosys_sva_circt_bmc.sh:298`
Validating policy values (`strict|lenient`) is good hygiene, but it also reflects that only two policies are currently expressible. What is missing is finer policy granularity (for example warn-only, per-diagnostic overrides) if workflows need it. The fix is to define a richer policy model only if demanded by users; otherwise keep strict validation and document semantics clearly.

### [ ] 119. `utils/run_yosys_sva_circt_bmc.sh:299`
The invalid-policy diagnostic is not a feature gap on its own; it is a boundary message for entry 118’s guard. The missing capability, if any, is policy extensibility rather than message text. The fix is to keep this explicit diagnostic but drive accepted values from a central enum/table to avoid drift with future policy additions.

### [ ] 120. `utils/run_yosys_sva_circt_bmc.sh:2747`
Rejecting unsupported JSONL line formats in history files is intentional data validation, but it can be brittle for slightly malformed legacy artifacts. What is missing is resilient migration support for a broader set of legacy line shapes. The fix is to extend migration parsing with explicit compatibility cases and keep hard-fail behavior only when recovery would be ambiguous.

### [ ] 121. `utils/run_yosys_sva_circt_bmc.sh:2867`
Unsupported drop-event hash modes are currently rejected, which is safe but rigid. The missing capability is pluggable hash-provider support when teams need stronger/stabler IDs than built-ins. The fix is to define a vetted extension point for hash algorithms and retain strict rejection for unknown modes by default.

### [ ] 122. `utils/run_yosys_sva_circt_bmc.sh:3026`
Rejecting unknown lock backends is correct for synchronization safety, but it means backend selection logic is closed to extension. The missing piece is a clean backend abstraction layer with explicit capabilities/timeouts. The fix is to formalize backend interfaces (`flock`, `mkdir`, future backends), then validate each with lock-contention tests before enabling.

### [ ] 123. `utils/run_yosys_sva_circt_bmc.sh:3186`
Failing on unsupported history-TSV headers is a strict schema guard that protects data quality, but it blocks gradual schema evolution. What is missing is explicit versioned header compatibility with migration paths. The fix is to track schema versions in-band and add deterministic migrators so older headers can be upgraded instead of hard-failed when possible.

### [ ] 124. `utils/run_yosys_sva_circt_bmc.sh:8522`
The comment states that smoke mode treats sim-only tests as unsupported and skips them, which is a real coverage trade-off. What is missing is lightweight bounded checking for sim-only cases in smoke that preserves determinism without complete omission. The fix is to introduce a cheap smoke-safe fallback (or a separate tiny sim lane) so these tests still get minimal signal.

### [ ] 125. `utils/run_yosys_sva_circt_bmc.sh:8616`
`lenient` policy injects `--sva-continue-on-unsupported`, which is useful operationally but confirms unsupported constructs are still common enough to require bypass. The missing capability is broader SVA lowering support so lenient mode is exceptional rather than routine. The fix is to track which diagnostics trigger this path and prioritize implementation against the highest-frequency unsupported constructs.

### [ ] 126. `utils/run_yosys_sva_circt_bmc.sh:8617`
This flag insertion is the mechanism for entry 125 and carries the same semantic gap: it trades strict correctness for throughput under unsupported SVA. What is missing is confidence that skipped assertions are transparent to users. The fix is to ensure every continue-on-unsupported event is surfaced in structured reports and tied to explicit pass/fail policy decisions.

### [ ] 127. `utils/run_yosys_sva_circt_bmc.sh:8676`
This second insertion point for lenient policy in another execution branch suggests duplicated policy plumbing. The missing maintainability piece is centralized argument synthesis so policy behavior cannot drift between branches. The fix is to factor policy-to-arg mapping into a helper function and test both code paths for parity.

### [ ] 128. `utils/run_yosys_sva_circt_bmc.sh:8677`
Same as entry 127: this is duplicated flag wiring for lenient mode. The missing capability is single-source policy mapping with consistent diagnostics across all run modes. The fix is refactoring plus regression tests that diff generated command lines across branches.

### [ ] 129. `utils/run_yosys_sva_circt_bmc.sh:8761`
Lenient policy also affects `circt-bmc` arguments here, which is correct but again duplicated branch logic. What is missing is policy cohesion across frontend and bmc tool invocation assembly. The fix is to generate both frontend and bmc policy args from one shared function/object so future policy changes stay synchronized.

### [ ] 130. `utils/run_yosys_sva_circt_bmc.sh:8762`
Adding `--drop-unsupported-sva` in lenient mode is a real semantic compromise that can hide assertion coverage holes if not tracked. The missing behavior is robust accountability for dropped properties during result interpretation. The fix is to require per-case drop accounting in outputs and optionally fail when dropped-SVA count exceeds configured thresholds.

### [x] 131. `lib/Runtime/MooreRuntime.cpp:2481`
Status update (2026-02-28): this gap is closed. `__moore_wait_condition` now has scheduler-assisted poll-callback support in runtime, `circt-sim` wires callback install/clear in simulation run lifecycle, and regression coverage includes wait-condition poll callback lifecycle tracing plus wait-condition/UVM execution checks.

### [ ] 132. `lib/Runtime/MooreRuntime.cpp:12227`
Array-element signal lookup currently strips indices and returns the base signal handle with a TODO for index calculation, so element-precise access is not implemented. What is missing is mapping from parsed index vectors to actual element offsets/handles for packed/unpacked arrays. The fix is to implement index resolution against registered signal metadata and add tests for multidimensional and out-of-range cases.

### [ ] 133. `lib/Runtime/MooreRuntime.cpp:14402`
The comment says test creation via UVM factory is TODO, but the function now contains concrete factory-create logic; this is stale documentation debt rather than missing executable code at this exact site. What is missing is comment-to-implementation consistency, which matters for maintainability. The fix is to update the comment to describe current behavior accurately and reserve TODO tags for truly unfinished work.

### [ ] 134. `lib/Runtime/MooreRuntime.cpp:17781`
Backdoor register read currently does not integrate with HDL path access and simply returns mirror state, which is a real functional gap for UVM backdoor semantics. What is missing is a bridge from register metadata/HDL path to live design state readback. The fix is to implement HDL path callbacks/resolution for backdoor reads and validate parity with frontdoor/mirror behavior under synchronized and desynchronized scenarios.

### [ ] 135. `lib/Runtime/MooreRuntime.cpp:17823`
Backdoor register write has the same missing HDL path integration as entry 134, so writes update mirror state but do not propagate through configured HDL access mechanisms. What is missing is actual design-state writeback when backdoor paths are present. The fix is to implement HDL path write hooks, define error behavior for unresolved paths, and add tests that verify mirrored and physical states converge correctly.

### [ ] 136. `frontends/PyCDE/test/test_instances.py:124`
This TODO marks an explicit test-coverage gap: physical region support tests were removed/disabled and not restored. What is missing is either feature readiness in the implementation or stable test scaffolding around region APIs. The fix is to re-enable region creation/bounds tests once the API is stable and ensure failures are clearly attributable to implementation regressions, not test harness fragility.

### [ ] 137. `frontends/PyCDE/test/test_instances.py:156`
Anonymous reservation tests are similarly commented out, so reservation behavior currently lacks active regression coverage in this file. What is missing is verification that placedb reservation semantics still work end-to-end. The fix is to reinstate these tests (or replace with updated equivalents) and assert reservation conflict/lookup behavior explicitly.

### [ ] 138. `utils/check_opentitan_fpv_bmc_evidence_parity.py:111`
This is another duplicated allowlist parser with strict `exact|prefix|regex` mode handling; the implementation is fine but duplicated across many scripts. The missing piece is a shared parser to eliminate drift and repeated bug fixes. The fix is to centralize allowlist parsing and migrate these parity scripts to the shared helper with a common test suite.

### [ ] 139. `utils/check_opentitan_target_manifest_drift.py:87`
Same pattern as entry 138: unsupported mode rejection is acceptable behavior, but parser duplication is the real maintainability gap. What is missing is single-source allowlist semantics across target-manifest and parity tooling. The fix is shared library extraction plus migration tests to guarantee unchanged user-visible behavior.

### [ ] 140. `utils/run_avip_circt_verilog.sh:157`
This `mktemp ...XXXXXX` hit is a scan false positive caused by matching `XXX` inside the random-suffix template. There is no implementation gap at this line. The fix is to tighten scanner patterns so `XXXXXX` templates are excluded from TODO-style matches.

### [ ] 141. `utils/run_avip_circt_verilog.sh:184`
This is the same `mktemp ...XXXXXX` false positive as entry 140. The missing issue is audit precision, not script functionality. The fix is identical: refine marker regexes or apply false-positive suppression for mktemp suffix literals.

### [ ] 142. `include/circt/Transforms/Passes.td:86`
The word “unsupported” here appears in pass documentation describing why index conversion exists; it is not itself a TODO in this file. The true gap, if any, is upstream pass dependence on this conversion to avoid unsupported index arithmetic in downstream mapping. The fix is scanner filtering for descriptive prose and, separately, continued reduction of downstream unsupported index operations so this pass becomes less critical.

### [ ] 143. `unittests/Support/TestReportingTest.cpp:54`
`tc.skip("Not implemented")` is intentional test data for skip-state behavior, not an implementation gap in reporting. What is missing is scanner discrimination between fixture messages and real code debt. The fix is to suppress known test-fixture phrases like “Not implemented” in audit output.

### [ ] 144. `unittests/Support/TestReportingTest.cpp:58`
This assertion checks the same fixture string and is not actionable product debt. The missing capability is better audit signal-to-noise around unit-test literals. The fix is to filter assertions that compare expected fixture messages from TODO/unsupported scans.

### [ ] 145. `unittests/Support/TestReportingTest.cpp:195`
Again this is a deliberate skipped-test fixture setup, not a runtime/reporting feature gap. The scanner is over-attributing gap semantics to test literals. The fix is to classify unit-test string fixtures separately and remove them from actionable gap lists.

### [ ] 146. `unittests/Support/TestReportingTest.cpp:203`
This output substring check intentionally looks for “Not implemented” and should not be treated as missing implementation. What is missing is context-aware scanning. The fix is to refine audit tooling to ignore expected-output assertions in unit tests by default.

### [ ] 147. `utils/check_opentitan_connectivity_contract_fingerprint_parity.py:90`
This is the same repeated allowlist mode parser pattern, with duplication being the real debt. What is missing is a canonical parsing utility to keep behavior aligned across all connectivity/fingerprint checkers. The fix is to migrate this script to shared parser code and delete local copies.

### [ ] 148. `utils/mutation_mcy/lib/native_mutation_plan.py:57`
Rejecting unknown native mutation ops is correct for safety, but it also means operator extensibility depends on code edits rather than declarative registration. What is missing is a pluggable op registry or capability discovery path. The fix is to keep strict validation but source allowed ops from a centralized registry definition so extending mutation ops is controlled and testable.

### [ ] 149. `utils/mutation_mcy/lib/drift.sh:646`
This is another `mktemp ...XXXXXX` false positive and not a TODO/unsupported code gap. The missing issue is scan pattern quality. The fix is to exclude mktemp suffix templates from TODO token matching.

### [ ] 150. `utils/mutation_mcy/lib/drift.sh:647`
Same false-positive category as entry 149: routine temp-file creation matched by `XXX` token scan. The fix is scanner filtering, not code change.

### [ ] 151. `utils/mutation_mcy/lib/drift.sh:648`
Again a false positive from mktemp suffix text. What is missing is a robust noise filter in the audit toolchain. The fix is to add lexical exclusions for uppercase temp patterns.

### [ ] 152. `utils/mutation_mcy/lib/drift.sh:649`
Fourth false-positive in the same block, confirming this category should be filtered globally. The fix is to implement and keep a false-positive suppression rule for `mktemp` templates before future manual review passes.

### [ ] 153. `utils/run_regression_unified.sh:120`
Rejecting unsupported profile tokens is expected schema validation, but supported values are hard-coded and may drift across tooling layers. What is missing is centralized profile schema definition shared by producers and consumers. The fix is to define profile enums in one source of truth and validate manifests against that shared definition.

### [ ] 154. `utils/check_opentitan_fpv_objective_parity.py:170`
This is another duplicated allowlist parser boundary. The missing behavior is not at this branch itself, but in ecosystem-level parser reuse and consistency. The fix is to consolidate parser logic and migrate objective-parity scripts to the shared implementation.

### [ ] 155. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm_time.svh:105`
`// ToDo: Check resolution` in vendored UVM core indicates an unresolved semantic detail in time scaling logic. What is missing is explicit validation that conversion respects expected timescale resolution behavior under all legal inputs. The fix is to add targeted UVM time-resolution tests and either resolve this TODO upstream or patch locally with clear divergence notes.

### [ ] 156. `utils/run_opentitan_connectivity_circt_lec.py:130`
This is the same allowlist mode parser duplication seen in BMC/connectivity scripts. The missing piece is centralization to prevent behavior drift between BMC and LEC wrappers. The fix is shared helper adoption with cross-tool parity tests.

### [ ] 157. `utils/run_opentitan_connectivity_circt_lec.py:366`
Failing on unsupported connectivity rule types is robust for strict manifests, but it prevents staged rollout of new rule kinds in LEC flows. What is missing is versioned manifest schema handling with explicit compatibility policy. The fix is to implement schema-version gates and supported-kind negotiation so new rule types can be introduced safely.

### [ ] 158. `utils/internal/checks/wasm_cxx20_contract_check.sh:45`
This “accepted unsupported C++ standard override” string is a negative test assertion, not a product TODO. The script is intentionally verifying that unsupported standards are rejected. The missing issue is scan-context awareness for test/check scripts. The fix is to suppress these contract-check expectation messages from actionable gap inventories.

### [ ] 159. `utils/run_opentitan_fpv_bmc_policy_workflow.sh:323`
`unsupported mode: $MODE` is a strict CLI guard with a closed mode set; functionally valid, but extensibility requires touching this script. What is missing is declarative workflow mode registration and shared validation with callers. The fix is to define allowed modes in one table and reuse it across argument parsing/help so adding a mode is a data change, not branch surgery.

### [ ] 160. `utils/select_opentitan_connectivity_cfg.py:276`
Rejecting unknown connectivity CSV row kinds (`CONNECTION|CONDITION`) enforces manifest hygiene, but it hard-fails future schema extension. What is missing is explicit schema versioning and controlled forward compatibility in CSV ingestion. The fix is to pair row-kind validation with schema-version checks and migration logic, keeping hard failures only for unknown kinds under the active schema.

### [ ] 161. `utils/run_avip_xcelium_reference.sh:94`
This is the same constrained AVIP-set guard seen in other AVIP runner scripts: only `core8` and `all9` are accepted. What is missing is a shared, data-driven AVIP set definition so support doesn’t diverge between scripts. The fix is to centralize AVIP set metadata and have all runner scripts validate against that shared source.

### [ ] 162. `utils/check_opentitan_fpv_objective_parity_drift.py:83`
Again this is duplicated allowlist mode parsing (`exact|prefix|regex`) rather than a unique local gap. The missing capability is parser reuse across parity and drift tools to avoid subtle behavior drift. The fix is to consolidate parser logic and move this script to the shared helper.

### [ ] 163. `utils/generate_mutations_yosys.sh:696`
Rejecting unsupported design extensions is safe but rigid; it requires users to pre-normalize files into `.il/.v/.sv`. What is missing is either broader front-end format support or an integrated conversion step. The fix is to add optional conversion/adaptation for common additional formats while preserving strict validation for unknown types.

### [ ] 164. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:76`
This macro defines a “not implemented” error for TLM-2 interface tasks in base interface scaffolding, which reflects incomplete method bodies by design. The missing behavior is concrete task implementations in derived interfaces/components where these APIs are expected to function. The fix is to ensure runtime integrations override these stubs and to add tests that fail if base “not implemented” paths are reached in supported flows.

### [ ] 165. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:81`
Same as entry 164, but for function paths: the macro exists as a default error for unimplemented TLM-2 functions. The missing capability is validated functional overrides for all required API points used by CIRCT-supported UVM workloads. The fix is coverage tests that exercise each required TLM-2 function and assert no default “not implemented” macro fires.

### [ ] 166. `test/firtool/firtool.fir:77`
This TODO is real and useful: the test currently depends on brittle behavior around aggressive port removal, so it can fail for non-semantic reasons. What is missing is a more robust assertion strategy that checks intended semantics rather than incidental port shape. The fix is to rewrite the test with narrower, behavior-focused checks and minimize sensitivity to unrelated canonicalization changes.

### [ ] 167. `unittests/Support/PrettyPrinterTest.cpp:75`
This `xxxxxxxx...` identifier is stress-test fixture data for long-token wrapping, not a TODO/unsupported gap. The missing issue is scan noise from `XXX`-like substrings inside test literals. The fix is to exclude long literal fixtures from marker scans.

### [ ] 168. `unittests/Support/PrettyPrinterTest.cpp:204`
Same fixture pattern as entry 167: this line intentionally validates pretty-print wrapping with long names. There is no implementation debt signaled here. The fix is audit filtering for known fixture strings.

### [ ] 169. `unittests/Support/PrettyPrinterTest.cpp:237`
Another intentional expected-output line using long `x...` tokens for formatter tests. What is missing is not product functionality but scan precision. The fix is to suppress these fixture regions from TODO/unsupported audits.

### [ ] 170. `unittests/Support/PrettyPrinterTest.cpp:247`
This is still fixture text in expected pretty-printed output, not a gap marker. The fix remains scanner refinement to avoid matching generic `xxx` sequences in test literals.

### [ ] 171. `unittests/Support/PrettyPrinterTest.cpp:279`
Same category: expected-output fixture, not unresolved work. The missing part is audit quality. The fix is to filter test expectation string blocks in this file.

### [ ] 172. `unittests/Support/PrettyPrinterTest.cpp:285`
Again this is intentionally long token text used for line-breaking tests and should not be treated as TODO debt. The fix is no code change here, only scan-rule tightening.

### [ ] 173. `unittests/Support/PrettyPrinterTest.cpp:304`
This match comes from expected formatting output and is non-actionable from a feature-gap perspective. The missing issue is false positives in static scans. The fix is to update scan heuristics.

### [ ] 174. `unittests/Support/PrettyPrinterTest.cpp:317`
This line is test fixture content for nested-call wrapping behavior, not missing functionality. The fix is scanner suppression for this literal family.

### [ ] 175. `unittests/Support/PrettyPrinterTest.cpp:321`
Again fixture text, not gap. The missing work is in tooling around audits, not in pretty printer implementation at this location.

### [ ] 176. `unittests/Support/PrettyPrinterTest.cpp:335`
Expected-output literal with long `x` token; no TODO or unsupported path implied. The fix is to avoid matching these literals in debt scans.

### [ ] 177. `unittests/Support/PrettyPrinterTest.cpp:339`
Same as entries 167–176: intentional fixture string. This should be filtered from actionable gap tracking.

### [ ] 178. `unittests/Support/PrettyPrinterTest.cpp:350`
Long prototype string in a margin-2048 test is deliberate stress data, not unresolved functionality. The fix is better scan discrimination, not production code changes.

### [ ] 179. `unittests/Support/PrettyPrinterTest.cpp:356`
Another long fixture prototype for nested call formatting; non-actionable as a gap. The fix is scanner filtering.

### [ ] 180. `unittests/Support/PrettyPrinterTest.cpp:362`
Same fixture pattern in test expectations; no missing feature indicated. The fix is to treat this as scan noise.

### [ ] 181. `unittests/Support/PrettyPrinterTest.cpp:671`
`StringToken("xxxxxxxxxxxxxxx")` is test input to force line wrapping behavior. It is not a TODO/unsupported marker. The fix is exclusion of synthetic token literals from scans.

### [ ] 182. `unittests/Support/PrettyPrinterTest.cpp:690`
This expected output line with repeated x/y strings is formatter test data, not debt. The missing item is audit precision only.

### [ ] 183. `unittests/Support/PrettyPrinterTest.cpp:697`
Again expected formatter output containing long x/y words; not an implementation gap. The fix is scanner suppression for this test section.

### [ ] 184. `unittests/Support/PrettyPrinterTest.cpp:702`
Same non-actionable fixture output as previous entries. No code change needed in this location for gap closure.

### [ ] 185. `unittests/Support/PrettyPrinterTest.cpp:709`
This is indentation fixture output (`>>>>>>xxxxxxxx...`) for pretty-printer behavior validation. It should not be interpreted as TODO debt.

### [ ] 186. `unittests/Support/PrettyPrinterTest.cpp:716`
Same fixture-category false positive as entry 185. The fix is audit filtering.

### [ ] 187. `unittests/Support/PrettyPrinterTest.cpp:723`
Expected output text in tests; not unresolved implementation. The missing issue is scan noise reduction.

### [ ] 188. `unittests/Support/PrettyPrinterTest.cpp:730`
Again a test fixture string used for printer layout checks, not a real gap marker. The fix is no code change, only scan heuristic refinement.

### [ ] 189. `frontends/PyCDE/src/pycde/types.py:576`
This TODO is real: the >63-bit `Bits` constant workaround for Python binding limits has not been generalized to `UInt` and `SInt`. What is missing is equivalent large-constant construction paths for signed/unsigned typed wrappers. The fix is to factor the chunked-constant creation logic into shared code and apply type-correct adaptation for `UInt`/`SInt` with dedicated tests.

### [ ] 190. `unittests/Support/FVIntTest.cpp:26`
`"XXXX1"` here is a four-valued logic expectation in a unit test, not a TODO marker. The missing problem is scanner confusion between X-state test vectors and “XXX” debt tags. The fix is to exclude FVInt test literals from `XXX`-based scans.

### [ ] 191. `unittests/Support/FVIntTest.cpp:104`
This `XXXXZZZZ` value is deliberate test data for unknown/high-impedance logic handling, not an implementation gap. The fix is scanner filtering for four-valued literal patterns.

### [ ] 192. `unittests/Support/FVIntTest.cpp:108`
`000001XX0XXX0XXX` is expected result data in logical-ops tests and should not be treated as TODO debt. The missing item is audit precision around literal matching.

### [ ] 193. `unittests/Support/FVIntTest.cpp:109`
Same as entry 192: this is expected-value text in tests, not unresolved work. The fix is to narrow marker scans to actual comments/messages.

### [ ] 194. `unittests/Support/FVIntTest.cpp:110`
`01XX10XXXXXXXXXX` is again intentional expected output for FVInt behavior tests. This is non-actionable from a gap perspective and should be filtered.

### [ ] 195. `utils/check_opentitan_connectivity_objective_parity.py:151`
This is the repeated allowlist mode parser pattern seen throughout parity tooling. The missing piece is still parser deduplication and shared governance. The fix is to replace local parsing code with a shared helper and keep one conformance test suite.

### [ ] 196. `test/firtool/spec/refs/read_subelement_add.fir:9`
`XXX: ADDED` is annotation text in spec-reference test input, not a TODO debt marker. The missing issue is scanner inability to distinguish editorial comments in test fixtures. The fix is to exclude `; XXX: ADDED` style spec-note comments from actionable scans.

### [ ] 197. `test/firtool/spec/refs/read_subelement_add.fir:10`
Same as entry 196: this is a fixture comment to indicate spec-example augmentation, not unresolved implementation. The fix is scanner filtering for this annotation pattern.

### [ ] 198. `test/firtool/spec/refs/read_subelement_add.fir:11`
Again `XXX: ADDED` in test fixture text, not a project gap. The fix is no code change here; improve scan heuristics.

### [ ] 199. `test/firtool/dedup-modules-with-output-dirs.fir:7`
`"dirname": "XXX"` is a literal directory token in a test and not a TODO/unsupported marker. The missing issue is false positives from generic `XXX` matching. The fix is to ignore quoted literal values in tests for debt scans.

### [ ] 200. `test/firtool/dedup-modules-with-output-dirs.fir:18`
`CHECK: FILE "XXX..."` is expected-output pattern text in a test, not an implementation debt signal. The fix is to suppress FileCheck pattern lines from TODO/unsupported scanning, since they routinely use placeholder strings.

### [ ] 201. `test/firtool/dedup-modules-with-output-dirs.fir:72`
`"dirname": "ZZZ/XXX"` is a test fixture value used to verify output-directory dedup behavior, not a TODO/unsupported marker. The missing issue is scanner overmatching of placeholder directory names. The fix is to ignore quoted fixture values in FIRRTL test refs during debt scans.

### [ ] 202. `test/firtool/dedup-modules-with-output-dirs.fir:95`
`CHECK: FILE "ZZZ...XXX...A.sv"` is again expected pattern text and non-actionable as implementation debt. The fix is to treat FileCheck directives as test assertions, not TODO indicators.

### [ ] 203. `tools/circt-mut/circt-mut.cpp:2368`
Rejecting unsupported mutant formats for native prequalification is reasonable, but it limits native coverage to `.il/.v/.sv` and forces fallback workflows otherwise. What is missing is broader native format ingestion or integrated conversion. The fix is to expand native prequalification input support (or automate conversion before native path) while keeping strict erroring for truly unknown formats.

### [ ] 204. `tools/circt-mut/circt-mut.cpp:6369`
The TOML parser supports only a small escape subset and reports other escapes as unsupported, which can reject valid-ish user config expectations. What is missing is fuller TOML string escape support consistent with documented config grammar. The fix is to either implement the missing escapes or document the restricted subset clearly and validate configs with precise diagnostics.

### [ ] 205. `tools/circt-mut/circt-mut.cpp:14642`
This comment is a real migration-state gap: unknown generate options trigger fallback to the script backend because native option parity is incomplete. What is missing is full native CLI compatibility with legacy script options. The fix is to close option-by-option parity, add compatibility tests, and eventually remove fallback for unsupported options once native coverage is complete.

### [ ] 206. `tools/circt-mut/circt-mut.cpp:15299`
Native generate mode still rejects non-`.il/.v/.sv` design extensions, which mirrors tooling gaps elsewhere. What is missing is format flexibility or builtin conversion for additional source forms. The fix is to extend native loader capabilities or provide a first-class conversion stage before read command synthesis.

### [ ] 207. `test/firtool/spec/refs/read_subelement.fir:10`
`XXX: ADDED` is editorial annotation in spec reference material, not unresolved implementation. The fix is to filter this annotation style from actionable scans.

### [ ] 208. `test/firtool/spec/refs/read_subelement.fir:11`
Same annotation false positive as entry 207. No product gap is implied at this line.

### [ ] 209. `test/firtool/spec/refs/read_subelement.fir:12`
Again `XXX: ADDED` fixture text, not technical debt. Scanner filtering is the correct remedy.

### [ ] 210. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:34`
This macro marks default “not implemented” behavior for base TLM1 interface tasks, indicating stub semantics remain in foundational interfaces. What is missing is guaranteed override coverage in supported runtime flows. The fix is to verify all required TLM1 API paths are implemented by concrete classes used in CIRCT-supported environments and to fail tests if base stubs are reached.

### [ ] 211. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:35`
Same as entry 210 for function APIs: base macro indicates unimplemented default behavior by design. The missing part is enforcement that supported workflows do not depend on these defaults. The fix is regression coverage for concrete function-path overrides and clear diagnostics when stubs are accidentally invoked.

### [ ] 212. `frontends/PyCDE/src/pycde/system.py:122`
This TODO is real and architectural: cf-to-handshake lowering is disabled in PyCDE because required dialects are not registered through current Python hooks. What is missing is a reliable dialect-loading/registration path from Python side. The fix is to expose stronger bindings (likely pybind11 module support as noted), register required dialects, and re-enable the handshake pass pipeline with tests.

### [ ] 213. `frontends/PyCDE/src/pycde/system.py:249`
Symbolref handling across potentially renamed symbols is still TODO, so imported MLIR objects can lose robust cross-reference fidelity after renaming. What is missing is symbol remapping through the import pipeline. The fix is to build a symbol-translation map during import and resolve symbolrefs through it, with regression tests around renamed modules/ops.

### [ ] 214. `frontends/PyCDE/src/pycde/system.py:574`
This TODO documents a broken MLIR live-operation cleanup hook and leaves PyCDE without an equivalent replacement, which can obscure leaked op references during debugging. What is missing is a supported lifecycle/introspection mechanism for op liveness in Python. The fix is to adopt a replacement API (or add one) and restore warnings/metrics for leaked operation references.

### [ ] 215. `test/circt-verilog/roundtrip-register-enable.sv:4`
`// UNSUPPORTED: valgrind` is test metadata due external tool/runtime constraints, not a product feature gap in register-enable lowering itself. What is missing is valgrind-lane compatibility or a reliable suppression rationale audit. The fix is to periodically reassess these UNSUPPORTED tags and retire them when upstream tool issues are resolved.

### [ ] 216. `test/firtool/spec/refs/read.fir:10`
`XXX: ADDED` is spec-fixture annotation, not unresolved implementation. The fix is scanner suppression for this editorial marker.

### [ ] 217. `test/firtool/spec/refs/read.fir:11`
Same fixture annotation false positive as entry 216. No actionable product gap at this line.

### [ ] 218. `test/firtool/spec/refs/read.fir:12`
Again fixture/comment marker, not technical debt. Scanner filtering should remove this.

### [ ] 219. `utils/run_opentitan_fpv_circt_bmc.py:231`
Unsupported stopat selector formats are rejected, which enforces strict selector grammar but can surprise users with legacy/informal selector shapes. What is missing is either richer selector normalization or clearer migration tooling. The fix is to document accepted grammar precisely and optionally add compatibility normalization for common legacy selector variants.

### [ ] 220. `utils/run_opentitan_fpv_circt_bmc.py:373`
This is the repeated allowlist parser mode gate found across many OpenTitan scripts. The missing capability remains shared parser infrastructure and single-point maintenance. The fix is parser deduplication into a common module.

### [ ] 221. `utils/run_opentitan_fpv_circt_bmc.py:1607`
`unsupported_stopat_selector` is an internal error classification when normalization fails; the gap is that selector-language support is narrower than some contract inputs. What is missing is support for additional selector forms or richer user guidance for correction. The fix is to either extend selector parsing safely or provide precise remediation hints tied to the failing row/value.

### [ ] 222. `test/circt-verilog/registers.sv:4`
Another `UNSUPPORTED: valgrind` test metadata line, not direct implementation debt. The missing piece is environmental test portability under valgrind, not register semantics support per se. The fix is to track and periodically revalidate valgrind exclusions.

### [ ] 223. `test/firtool/spec/refs/probe_export_simple.fir:7`
`XXX: Added width.` is explanatory fixture commentary in spec refs and not a TODO gap. The fix is scanner suppression for `XXX:` annotations in reference tests.

### [ ] 224. `test/firtool/spec/refs/nosubaccess.fir:8`
`XXX: Modified ...` is similarly a fixture note documenting adaptation of a spec example, not unresolved implementation. The fix is to ignore this annotation class in actionable debt reports.

### [ ] 225. `utils/check_opentitan_compile_contract_drift.py:95`
This is the same allowlist mode parser duplication pattern. The missing behavior is shared parser reuse across contract drift/parity tools. The fix is to migrate to one common parser and drop per-script copies.

### [ ] 226. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:35`
Default “task not implemented” macro in TLM FIFO base indicates stub behavior for unimplemented FIFO task APIs. What is missing is assurance that supported runtime paths bind to concrete implementations instead of stub defaults. The fix is targeted runtime tests and explicit override verification.

### [ ] 227. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:36`
Same as entry 226 for function APIs: this macro is a default error path, not full implementation. The missing capability is validated coverage of concrete FIFO function behavior in CIRCT-supported UVM flows. The fix is to add/strengthen tests that ensure these defaults are never hit in expected scenarios.

### [ ] 228. `test/circt-verilog/memories.sv:4`
`UNSUPPORTED: valgrind` here is another environmental test exclusion marker, not a direct memory-lowering TODO. The missing piece is tool/environment compatibility to run this test under valgrind reliably. The fix is to revisit exclusions as toolchain issues improve.

### [ ] 229. `tools/circt-sim/AOTProcessCompiler.h:211`
The comment notes that functions with unsupported ops are skipped during AOT function-body compilation, which is a real capability boundary. What is missing is wider AOT compilability coverage and/or better fallback accounting for skipped functions. The fix is to expand `isFuncBodyCompilable()` support set and produce structured skip diagnostics so unsupported-op hotspots are visible.

### [ ] 230. `test/circt-verilog/redundant-files.sv:4`
This `UNSUPPORTED: valgrind` marker is the same class as entries 215/222/228: test-runner environment exclusion, not core feature debt at this location. The fix is ongoing periodic validation of whether valgrind exclusions are still necessary.

### [ ] 231. `lib/Dialect/LLHD/Transforms/Deseq.cpp:824`
This TODO is a real correctness gap in deseq simplification: values that depend on trigger signals are currently accepted and folded through `getKnownValue`, which can hide trigger-coupled behavior. What is missing is explicit dependence rejection for trigger-derived values. The fix is to run a backward dependence walk against trigger SSA roots and refuse simplification when dependence is detected.

### [x] 232. `lib/Runtime/uvm-core/src/base/uvm_root.svh:1280`
Status update (2026-02-28): this gap is closed in this workspace. Run-phase time-zero guarding was centralized into `m_check_run_phase_start_time()` and wired into `run_test` and phase lifecycle callbacks, removing reliance on the previous inline TODO site. Regression coverage was added in `test/Runtime/uvm/uvm_run_phase_time_zero_guard_test.sv`.

### [x] 233. `lib/Runtime/uvm-core/src/base/uvm_root.svh:1281`
Status update (2026-02-28): same closure as entry 232.

### [ ] 234. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:86`
Expanded macros are explicitly not handled in this range-processing path, which means symbol indexing/navigation can silently miss or misplace macro-origin entities. What is missing is macro-expansion-aware source mapping. The fix is to resolve expansion vs spelling locations through Slang source manager APIs and add macro-heavy LSP index tests.

### [ ] 235. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:449`
This parser only supports one serialized location style (`@[...]`), creating a compatibility gap when location formatting settings differ. What is missing is multi-format location parsing or normalization. The fix is to support alternate location-info styles (or a shared parser abstraction) with roundtrip tests per style.

### [ ] 236. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:472`
This path also skips expanded macros, so reference mapping can be incomplete for macro-generated code. What is missing is robust handling of expanded macro source ranges in reference indexing. The fix is macro-aware range translation and targeted tests for macro definition/usage references.

### [ ] 237. `lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:108`
This TODO is a performance/scalability gap: branch-decision aggregation is eager across control-flow, potentially doing unnecessary work on large regions. What is missing is a bounded region-aware decision analysis. The fix is to restrict analysis to blocks between dominator and target (as the comment suggests) and benchmark compile-time improvements.

### [ ] 238. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:45`
Hardcoding the Verilated module class name is a real configurability gap and risks mismatches with CLI/user-specified Verilator naming. What is missing is attribute/config-driven class resolution. The fix is to derive class names from interop config attributes (or pass options) and validate with integration tests over custom module names.

### [ ] 239. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:125`
Indirect evaluation still uses a temporary `func::CallIndirectOp` fallback, leaving a dialect-boundary cleanup debt. What is missing is native `systemc::CallIndirectOp` use and removal of extra func-dialect coupling. The fix is to migrate once the dependency lands, then prune the fallback dependency chain and add interop call-lowering tests.

### [ ] 240. `include/circt/Dialect/HWArith/HWArithOps.td:170`
`AnyInteger` here is a type constraint declaration in ODS, not a TODO/unsupported marker. This is a scan false positive. The fix is to refine scanners to ignore ordinary type-constraint tokens in TableGen op definitions.

### [ ] 241. `include/circt/Dialect/HWArith/HWArithOps.td:171`
Same as entry 240: `AnyInteger` result typing is expected ODS content and not unresolved work. The fix is scanner filtering for non-comment, non-diagnostic TableGen tokens.

### [ ] 242. `include/circt/Dialect/SV/SVStatements.td:68`
This TODO marks an ergonomic/tooling gap in ODS: custom builders are required only to ensure implicit region terminator installation. What is missing is a declarative ODS mechanism to request this behavior directly. The fix is upstream ODS support (or local helper abstraction) to eliminate repetitive custom-builder boilerplate.

### [ ] 243. `include/circt/Dialect/SV/SVStatements.td:124`
Same pattern as entry 242 for another op: boilerplate builders exist due ODS terminator limitations, increasing maintenance surface. The missing piece is shared declarative terminator support. The fix is centralizing builder generation once ODS capability exists and removing duplicated custom code.

### [ ] 244. `include/circt/Dialect/SV/SVStatements.td:169`
Again, this is the same ODS limitation on implicit terminator insertion. What is missing is declarative region-terminator control at ODS level. The fix is an ODS enhancement and follow-up cleanup to standard builders.

### [ ] 245. `include/circt/Dialect/SV/SVStatements.td:335`
This is the same terminator-builder workaround repeated for loop-body region ops. The missing capability remains ODS-native region terminator provisioning. The fix is to adopt such support and remove custom builders across SV statement ops.

### [ ] 246. `lib/Dialect/SystemC/SystemCOps.cpp:746`
Copy-pasting from the func dialect is a real maintainability gap: behavior parity depends on manual sync with upstream changes. What is missing is reusable shared implementation hooks for call-like op verification/parsing/printing. The fix is to factor common function-interface utilities upstream or into shared CIRCT helpers.

### [ ] 247. `lib/Dialect/SystemC/SystemCOps.cpp:751`
This explicit `FIXME` confirms an exact upstream copy for symbol-use verification. What is missing is deduplicated function-interface logic. The fix is to replace copied verification code with shared utility calls and add parity tests to avoid drift.

### [ ] 248. `lib/Dialect/SystemC/SystemCOps.cpp:816`
The FuncOp implementation is largely copied from upstream, creating drift risk and larger review surface for future updates. What is missing is a composable function-op implementation layer reusable by SystemC ops. The fix is to upstream/factor common pieces and minimize local forks.

### [ ] 249. `lib/Dialect/SystemC/SystemCOps.cpp:878`
Parser code is copied because SSA argument-name access requires custom handling. What is missing is an upstream parser extension point that exposes argument-name handling without full copy-paste. The fix is to add that hook upstream (or local utility wrapper) and delete duplicated parser logic.

### [ ] 250. `lib/Dialect/SystemC/SystemCOps.cpp:981`
Printing logic is inlined from upstream to customize attribute elision, which again creates divergence risk. What is missing is configurable upstream print helpers. The fix is to expose attribute-elision policy hooks and switch SystemC ops back to shared printers.

### [ ] 251. `lib/Dialect/SystemC/SystemCOps.cpp:1012`
Clone-operation code is copied from upstream; this is maintainability debt and a parity hazard when upstream clone semantics evolve. What is missing is reusable cloning helpers for function-interface ops. The fix is to consolidate on shared clone utility code and add behavior lock tests.

### [ ] 252. `lib/Dialect/SystemC/SystemCOps.cpp:1123`
ReturnOp implementation is also copy-pasted, repeating the same drift issue. What is missing is shared return-op validation/building infrastructure for function-like dialects. The fix is helper extraction and replacement of local copies.

### [x] 253. `lib/Runtime/uvm-core/src/base/uvm_resource_base.svh:554`
Status update (2026-02-28): this entry is closed as stale scanner noise. `record_xxx_access` at this line is an API-family placeholder in explanatory text, not unresolved implementation work.

### [ ] 254. `lib/Bindings/Tcl/circt_tcl.cpp:67`
This TODO is a real user-facing feature gap: Tcl `load FIR` is explicitly unimplemented. What is missing is FIR file ingestion through the Tcl binding path. The fix is to implement FIR loader plumbing (or remove/guard the command with clear capability docs) and add Tcl integration tests.

### [ ] 255. `lib/Dialect/LLHD/Transforms/InlineCalls.cpp:246`
Hardcoded C++ class-field offsets (`m_parent`, `m_name`) are a real correctness/portability gap and can break with ABI/layout changes. What is missing is metadata-driven offset resolution. The fix is to compute offsets from class metadata (or pass them explicitly from frontend lowering) and remove magic constants.

### [ ] 256. `include/circt/Dialect/SV/SVTypes.td:18`
This TODO is a low-priority structural note, not an immediate capability blocker: it suggests moving definitions to `SVTypesImpl.td` once SV-specific types exist. What is missing is file-organization cleanup only when/if types are added. The fix is to keep it as deferred refactor and execute during first real SV type introduction.

### [ ] 257. `lib/Dialect/LLHD/Transforms/HoistSignals.cpp:652`
Using ad-hoc constants where a semantic `llhd.dontcare` would be more precise is a real IR expressiveness gap. What is missing is first-class don't-care materialization in this transform path. The fix is to introduce/route through an LLHD don't-care representation and update downstream consumers/tests accordingly.

### [ ] 258. `lib/Bindings/Python/support.py:281`
This TODO marks exception taxonomy debt: unconnected-backedge diagnostics still use generic text instead of `UnconnectedSignalError`. What is missing is consistent typed error signaling for Python users. The fix is to emit `UnconnectedSignalError` and update tests/documentation to match the richer exception type.

### [ ] 259. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:62`
Supported logic-op simulation is intentionally narrow (`aig::AndInverterOp` only), limiting cut rewriting opportunities. What is missing is broader combinational op support (`comb.and/xor/or` noted in TODO). The fix is to extend `isSupportedLogicOp` and simulation semantics for these ops with regression coverage.

### [ ] 260. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:174`
Assuming `numOutputs == 1` is a functional limitation in truth-table extraction and blocks multi-output cuts. What is missing is generalized multi-output table construction/return. The fix is to refactor `simulateCut` to produce all outputs and thread multi-output support through callers/tests.

### [ ] 261. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:343`
This TODO is a real performance gap in cut merging: the code re-derives topological order instead of merging already-sorted operation lists. What is missing is linear-time merge of pre-sorted cut operation vectors. The fix is to implement merge-sort-by-index for `operations`/`other.operations` and keep duplicate elimination cheap.

### [ ] 262. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:402`
Merged cut inputs are not explicitly sorted by defining operation, which can make ordering unstable and affect downstream reproducibility/caching. What is missing is deterministic canonical input ordering. The fix is to sort merged inputs by stable op index and lock with deterministic-output tests.

### [ ] 263. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:403`
Area and delay are not recomputed on merged cuts, so cost modeling is incomplete after merge. What is missing is cost recomputation with merged-structure awareness. The fix is to update area/delay bookkeeping during merge and validate with cut-ranking regression tests.

### [ ] 264. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:552`
Using full sorting where a priority queue would suffice is a known efficiency gap in cut selection. What is missing is incremental best-cut retrieval. The fix is to replace repeated sort-heavy selection with a priority queue and benchmark compile-time wins on large netlists.

### [ ] 265. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:684`
Cut enumeration currently rejects variadic ops and non-single-bit results, constraining applicability. What is missing is generalized operation/result-shape support. The fix is to add variadic handling and bit-slicing/multi-bit result support in enumeration and simulation.

### [ ] 266. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:887`
Patterns with multiple outputs are still hard-rejected, which blocks richer pattern libraries. What is missing is multi-output matching/rewrite support. The fix is to remove this guard once multi-output truth-table/extraction support is implemented end-to-end.

### [ ] 267. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:957`
Primary-input delays ignore global arrival-time context, so delay estimation is incomplete. What is missing is explicit arrival-time modeling on IR values. The fix is to propagate/capture arrival-time metadata and include it in input delay computation.

### [ ] 268. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:122`
This FIXME indicates conservative SOP cut limits are used because CutRewriter performance is currently weak. What is missing is efficient enough cut rewriting to use stronger defaults comparable to ABC/mockturtle. The fix is to optimize CutRewriter and then retune `maxCutInputSize` defaults with benchmark-backed thresholds.

### [ ] 269. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:139`
Pipeline TODO notes missing major synthesis stages (balancing/rewriting/FRAIG, etc.). What is missing is fuller logic-optimization flow parity. The fix is to add these passes with configurable ordering and regression/quality-of-result benchmarks.

### [ ] 270. `lib/Dialect/Synth/Transforms/TechMapper.cpp:135`
Mapped instance naming is placeholder-quality, which hurts readability/debugging and can complicate downstream tooling. What is missing is stable, meaningful naming policy. The fix is to derive names from source op/module/cut identifiers with collision-safe suffixing.

### [ ] 271. `lib/Dialect/Synth/Transforms/TechMapper.cpp:174`
Technology-library metadata is currently represented as an ad-hoc attribute dictionary. What is missing is structured IR representation for techlib semantics. The fix is to introduce dedicated techlib ops/types/attrs and migrate mapper logic to them.

### [ ] 272. `lib/Dialect/Synth/Transforms/TechMapper.cpp:183`
Mapping currently runs broadly rather than being constrained to target hierarchy scopes. What is missing is hierarchy-scoped mapping control. The fix is to add scope selection (attribute/option driven) and gate mapping to explicit module subtrees.

### [ ] 273. `lib/Dialect/Synth/Transforms/TechMapper.cpp:201`
This line is ordinary numeric conversion (`convertToDouble`) and not a TODO/fixme/unsupported marker. This is a scanner false positive. The fix is to tighten marker matching to explicit debt tokens instead of incidental symbols/identifiers.

### [ ] 274. `lib/Dialect/Synth/Transforms/TechMapper.cpp:207`
Delay parsing assumes integer attributes, which is a temporary limitation against richer timing models/units. What is missing is typed timing attributes with unit semantics. The fix is to migrate to structured cell timing attributes and update parsing/validation accordingly.

### [ ] 275. `lib/Dialect/Synth/Transforms/LowerWordToBits.cpp:189`
Known-bits fallback uses a depth-limited, uncached path, which is both potentially imprecise and slower. What is missing is cache-aware, depth-robust known-bits computation integration. The fix is to thread cached known-bits analysis and avoid repeated recomputation.

### [ ] 276. `lib/Dialect/Synth/Transforms/LowerVariadic.cpp:134`
Only top-level ops are lowered due missing topological handling across nested regions, leaving nested-region variadics untouched. What is missing is region-aware traversal/scheduling. The fix is to implement nested topological rewrite order and extend coverage with nested-region tests.

### [ ] 277. `lib/Tools/circt-lec/ConstructLEC.cpp:54`
Fetched LLVM globals are not fully sanity-checked before reuse, which risks attribute/type mismatches going unnoticed. What is missing is strict global-shape validation. The fix is to verify linkage/type/attrs against expected schema and emit actionable diagnostics on mismatch.

### [ ] 278. `lib/Tools/circt-lec/ConstructLEC.cpp:219`
The TODO calls out avoidable LLVM-specific construction in result plumbing. What is missing is a cleaner dialect-level reporting path. The fix is to replace LLVM-constant/return scaffolding with higher-level ops or a dedicated reporting abstraction.

### [ ] 279. `lib/Tools/circt-lec/ConstructLEC.cpp:231`
Result reporting is currently implemented by injecting LLVM-print style plumbing, which is acknowledged as inelegant. What is missing is first-class LEC result-reporting mechanism. The fix is a dedicated report op/API and streamlined lowering to the final presentation layer.

### [ ] 280. `lib/Analysis/FIRRTLInstanceInfo.cpp:229`
`anyInstanceUnderDut` is a normal API function name, not a debt marker. This is a scanner false positive likely triggered by broad token matching. The fix is to match only explicit TODO/FIXME/TBD/unsupported markers in comments/diagnostics.

### [ ] 281. `lib/Analysis/FIRRTLInstanceInfo.cpp:239`
Same as entry 280: this is routine API code (`anyInstanceUnderEffectiveDut`), not unresolved work. The fix is scanner narrowing to actionable marker patterns.

### [ ] 282. `lib/Analysis/FIRRTLInstanceInfo.cpp:240`
Again this is ordinary implementation code returning computed status, not a TODO/unsupported gap. The fix is improved audit heuristics to avoid generic identifier false positives.

### [ ] 283. `lib/Analysis/FIRRTLInstanceInfo.cpp:247`
`anyInstanceUnderLayer` is part of analysis API surface, not a debt comment. This is non-actionable and should be filtered from gap scans.

### [ ] 284. `lib/Analysis/FIRRTLInstanceInfo.cpp:257`
`anyInstanceInDesign` is also normal function code with no TODO semantics. The missing item is scan precision, not implementation work.

### [ ] 285. `lib/Analysis/FIRRTLInstanceInfo.cpp:267`
Same false-positive class as entries 280-284: ordinary API naming captured as debt. The fix is to require explicit marker comments/messages for audit inclusion.

### [ ] 286. `lib/Bindings/Python/dialects/synth.py:57`
This TODO marks a real Python binding gap: wrapper objects are not yet linked back to MLIR value/op identities. What is missing is stable association to underlying IR objects for diagnostics/introspection. The fix is to store MLIR handles in the wrapper and add lifecycle-safe accessors/tests.

### [ ] 287. `lib/Dialect/Synth/Transforms/AIGERRunner.cpp:207`
Repeated `comb.extract` creation in bit expansion is uncached, leaving avoidable overhead in large multi-bit mappings. What is missing is extract-op reuse. The fix is local caching keyed by `(value, bitPosition)` to reuse existing extracts and reduce IR bloat.

### [ ] 288. `include/circt/Dialect/HW/HWStructure.td:247`
This TODO acknowledges a modeling hack around `verilogName` due missing proper parameterized-type representation in HW dialect. What is missing is first-class parameterization/type abstraction. The fix is to introduce parameterized type modeling and retire this naming workaround.

### [ ] 289. `lib/Analysis/CMakeLists.txt:67`
Linting subdirectory is disabled behind a TODO due Slang header-path issues, leaving analysis linting unavailable in normal builds. What is missing is robust include-path integration for CIRCTLinting. The fix is to resolve header path configuration and re-enable linting target with CI coverage.

### [ ] 290. `lib/Dialect/Synth/SynthOps.cpp:68`
Majority folding is incomplete for constant patterns (`maj(x,1,1)=1`, `maj(x,0,0)=0`), missing easy canonicalizations. What is missing is these constant-folding rules in `MajorityInverterOp::fold`. The fix is to implement these cases and add fold regression tests.

### [ ] 291. `lib/Dialect/Synth/SynthOps.cpp:298`
The current variadic-and/inverter lowering uses a balanced binary tree regardless of signal timing or fanout, which is correct but leaves QoR on the table. What is missing is a cost-aware tree-construction strategy that can optimize for critical-path delay and/or area. The fix is to add a heuristic or analysis-guided decomposition policy and validate it with timing/size regressions on representative netlists.

### [ ] 292. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:97`
This TODO captures a real dialect-conversion limitation: direct RAUW is used as a workaround because `replaceAllUsesWith` is not safely supported in this conversion context. What is missing is a first-class replacement path compatible with `ConversionPatternRewriter` semantics. The fix is to migrate to the canonical conversion API once available (or encapsulate the workaround centrally) and add regression coverage to prevent mutate-after-erase assertions.

### [ ] 293. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:319`
Same underlying gap as entry 292: this path still depends on direct RAUW during dialect conversion due framework limitations. What is missing is a robust conversion-safe value replacement mechanism for this forwarding case. The fix is shared migration to supported conversion rewrite utilities and removal of ad hoc RAUW once upstream support lands.

### [ ] 294. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:359`
Again this is the same conversion-framework workaround pattern, indicating repeated technical debt in the portref lowering pass. What is missing is a unified, safe replacement strategy instead of duplicated direct RAUW sites. The fix is to refactor these replacement points behind one helper and switch to native `DialectConversion` support when available.

### [x] 295. `lib/Runtime/uvm-core/src/base/uvm_port_base.svh:528`
Status update (2026-02-28): this gap is closed in this workspace. Late-connection phase-state gating now routes through a shared helper (`m_is_end_of_elaboration_complete`) instead of open-coded duplicated checks.  
Additional semantic closure (2026-02-28): port bookkeeping now uses stable fallback keys when `get_full_name()` degrades to non-stable dynamic placeholders at runtime, which unblocked `resolve_bindings()`/TLM connection semantics under `circt-sim`.

### [x] 296. `lib/Runtime/uvm-core/src/base/uvm_port_base.svh:638`
Status update (2026-02-28): same closure as entry 295. `debug_connected_to` now reuses the same shared phase-state helper.  
Additional semantic closure (2026-02-28): validated with semantic runtime coverage (`uvm_port_connect_semantic_test.sv`, `uvm_tlm_port_test.sv`) so connect/resolve behavior is now exercised beyond parse/lowering checks.

### [ ] 297. `include/circt/Dialect/HW/HWOps.h:42`
This TODO is an architectural gap: module helper functions are free functions instead of being surfaced through a `hw::ModuleLike` interface. What is missing is a uniform interface abstraction that makes module-like operations interchangeable across passes. The fix is to move these helpers into interface methods, migrate call sites, and keep compatibility shims only during transition.

### [ ] 298. `lib/Tools/circt-bmc/LowerToBMC.cpp:3312`
The pass currently injects LLVM dialect ops just to return `0` from generated `main`, which mixes abstraction levels and complicates lowering boundaries. What is missing is a cleaner dialect-level return construction in this stage. The fix is to construct `main` purely with non-LLVM core ops (or dedicated helper API) and leave LLVM materialization to later lowering passes.

### [ ] 299. `lib/Dialect/Kanagawa/Transforms/KanagawaPassPipelines.cpp:62`
This TODO points to a real verification gap: there is no pass ensuring unexpected `memref.alloca` values are gone after `mem2reg`. What is missing is an explicit pipeline invariant check before downstream SSA assumptions. The fix is to add a verifier/cleanup pass that rejects illegal residual allocas (except approved member-variable cases) with actionable diagnostics.

### [ ] 300. `lib/Analysis/TestPasses.cpp:238`
`anyInstanceUnderDut` here is expected diagnostic-print text in a test analysis pass, not unresolved implementation work. This is a scanner false positive from broad token matching. The fix is to ignore ordinary API identifier text in pass debug output when building debt lists.

### [ ] 301. `lib/Analysis/TestPasses.cpp:242`
This is continuation of expected analysis output formatting (`anyInstanceUnderEffectiveDut`) and is non-actionable from a gap perspective. The missing issue is scanner precision, not code functionality. The fix is marker filtering to comments/diagnostics that explicitly encode TODO/FIXME/TBD/unsupported debt.

### [ ] 302. `lib/Analysis/TestPasses.cpp:243`
Same false-positive class as entry 301: this line simply prints a computed analysis value. No implementation gap is implied. The fix is to exclude ordinary stream-output lines from debt scans.

### [ ] 303. `lib/Analysis/TestPasses.cpp:246`
`anyInstanceUnderLayer` label output in a test pass is intentional and not a TODO or unsupported marker. This is non-actionable audit noise. The fix is tighter pattern heuristics that avoid API-name collisions.

### [ ] 304. `lib/Analysis/TestPasses.cpp:247`
Same as entry 303: expected test-pass printout, not deferred work. The missing capability is scanner disambiguation between identifiers and debt markers. The fix is filtering for comment/diagnostic contexts only.

### [ ] 305. `lib/Analysis/TestPasses.cpp:250`
`anyInstanceInDesign` appears in normal debug output for analysis tests, which should not be classified as a project gap. This is another false positive from token overmatching. The fix is to treat this class of output-string lines as non-actionable by default.

### [ ] 306. `lib/Analysis/TestPasses.cpp:254`
This line is also expected label output (`anyInstanceInEffectiveDesign`) in a test pass and does not indicate missing implementation. The true gap is scan quality. The fix is debt-marker parsing that requires explicit TODO-like syntax.

### [ ] 307. `lib/Analysis/TestPasses.cpp:255`
Same false-positive pattern as entries 300-306: a printed value expression, not a gap marker. No code change is needed in this file for feature completeness. The fix is scanner refinement.

### [ ] 308. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:211`
Name extraction for block arguments only handles `hw::HWModuleOp` and falls back to `<unknown-argument>` otherwise, reducing analysis/debug fidelity on other operation kinds. What is missing is generalized argument naming across additional module-like ops. The fix is to extend `getNameImpl` dispatch to other relevant parent ops and add tests that verify stable naming in those contexts.

### [ ] 309. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:729`
Debug-point collection is always instantiated, which adds memory/runtime overhead even when path-debug data is not needed. What is missing is a mode to disable this bookkeeping for performance-focused runs. The fix is to gate debug-point factory creation and propagation behind an option, keeping output parity only when debug mode is enabled.

### [ ] 310. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1307`
Memory write endpoints are modeled without incorporating the write address, which collapses distinct memory locations and can blur path precision. What is missing is address-sensitive endpoint tracking for `seq::FirMemWriteOp`. The fix is to include address information in endpoint keys/path state and add regressions demonstrating distinct per-address behavior.

### [ ] 311. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1312`
Same precision gap as entry 310 for `seq::FirMemReadWriteOp`: endpoint modeling ignores address. What is missing is consistent address-aware handling across both write-capable memory op forms. The fix is to apply the same address-aware representation and tests to read-write ports.

### [ ] 312. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:293`
Argument/result name collection is open-coded with a TODO noting it belongs in `ModulePortInfo`, so port metadata ownership is currently fragmented. What is missing is a single structured source of port naming data. The fix is to extend `ModulePortInfo` to carry canonical arg/result names and remove local reconstruction logic.

### [ ] 313. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:314`
This is the same dialect-conversion RAUW workaround pattern as entries 292-294, repeated in container-to-HW lowering. What is missing is conversion-safe replacement support that avoids direct RAUW in these patterns. The fix is to centralize and eventually eliminate this workaround once the conversion rewriter supports the needed API.

### [ ] 314. `lib/Analysis/DependenceAnalysis.cpp:169`
`replaceOp` currently performs a full scan of all dependence entries to rewrite source references, which is simple but can become expensive at scale. What is missing is reverse indexing from source op to dependent edges for efficient updates. The fix is to maintain an inverted index alongside `results` and update both maps on insertion/replacement.

### [ ] 315. `lib/Analysis/DebugInfo.cpp:165`
Fallback debug-info construction for instances does not track port assignments, leaving incomplete connectivity context in the resulting debug model. What is missing is representation of instance port-to-value bindings in fallback DI paths. The fix is to add a port-assignment structure to debug-info instances and populate it when visiting `hw.instance`.

### [ ] 316. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:537`
`TBD add more useful debug` marks observability debt in phase execution tracing, not a direct semantic bug. What is missing is richer debug output for phase control transitions/termination behavior. The fix is to define a concise debug schema and emit consistent trace points guarded by the existing phase-trace switch.

### [x] 317. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:553`
Status update (2026-02-28): this gap is closed in this workspace. `uvm_phase::m_aa2string` was refactored to use a straightforward separator-first pattern (removing the `TBD tidy` marker) for clearer and deterministic predecessor/successor edge-string formatting. Focused UVM phase/runtime regressions pass.

### [x] 318. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:762`
Status update (2026-02-28): this gap is closed in this workspace. `uvm_phase::add` now validates that all relationship parameters (`with_phase`, `before_phase`, `after_phase`, `start_with_phase`, `end_with_phase`) resolve within the current schedule graph before mutation. Cross-schedule misuse now emits immediate `PH_BAD_ADD` diagnostics, with regression coverage in `test/Runtime/uvm/uvm_phase_add_scope_validation_test.sv`.

### [ ] 319. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1218`
State-change callback payload fields are manually poked with a comment noting no official setter path, which indicates API design debt around phase-state transitions. What is missing is a sanctioned constructor/setter interface for `m_state_chg` updates. The fix is to introduce a formal update helper/API and route all state transitions through it.

### [ ] 320. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1383`
`find` is explicitly marked as not doing a full search, so phase lookup coverage is intentionally incomplete in some graph/scope patterns. What is missing is comprehensive traversal semantics (with proper scope controls) for phase search. The fix is to implement full search behavior with cycle-safe traversal and regression tests for in-scope and cross-scope queries.

### [ ] 321. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1403`
`find_by_name` is still marked `TBD full search`, so lookup behavior remains intentionally partial for some graph/scope cases. What is missing is full traversal parity with expected UVM phase graph semantics. The fix is to implement complete predecessor/successor search with scope controls and add regressions for ambiguous/cross-scope names.

### [ ] 322. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1432`
`is_before` currently hardcodes out-of-scope traversal (`m_find_successor(..., 0, ...)`) and explicitly lacks `stay_in_scope=1` support. What is missing is scope-aware ordering queries. The fix is to add a scoped variant (or parameter) and ensure phase-order checks honor domain/schedule boundaries when requested.

### [ ] 323. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1442`
Same gap as entry 322 for `is_after`: it does not yet support scoped ordering semantics. What is missing is symmetric `stay_in_scope` behavior for both before/after APIs. The fix is parallel implementation and tests that prove `is_before`/`is_after` stay consistent under scoped and unscoped modes.

### [ ] 324. `lib/Tools/arcilator/pipelines.cpp:98`
This TODO points to pass-pipeline duplication/ordering debt: `RemoveUnusedArcArguments` and `SinkInputs` likely overlap and could be merged. What is missing is a consolidated optimization stage with clear invariants. The fix is to evaluate overlap, merge where semantics align, and keep one canonical pass to reduce redundant IR churn.

### [ ] 325. `lib/Tools/arcilator/pipelines.cpp:111`
`LowerClocksToFuncsPass` is flagged as possibly incorrect for `scf.if`/nested regions, which is a real structural correctness risk in non-trivial control-flow modules. What is missing is region-aware lowering logic for clock conversion. The fix is to make the pass traverse/transform nested regions correctly and add regressions containing muxes and nested `scf.if` in both `hw.module` and `arc.model` contexts.

### [ ] 326. `lib/Tools/arcilator/pipelines.cpp:114`
`InlineArcs` has the same nested-region limitation and currently blocks enabling `MuxToControlFlowPass` in this pipeline. What is missing is robust inlining across `scf.if` boundaries. The fix is to harden `InlineArcs` region handling, then re-enable the downstream pass and cover the previously disabled scenario with a regression test.

### [ ] 327. `lib/Dialect/Kanagawa/Transforms/KanagawaAddOperatorLibrary.cpp:46`
This is implementation-style debt rather than feature debt: helper boilerplate exists because templated lambdas are deferred pending C++20. What is missing is modernized helper abstraction that removes explicit builder threading. The fix is a C++20 cleanup refactor once toolchain baseline permits, with no semantic change expected.

### [ ] 328. `include/circt/Dialect/RTG/Transforms/RTGPasses.td:208`
The pass description explicitly leaves test-grouping strategy unfinished (`TODO`), so output partitioning behavior is under-specified. What is missing is completed grouping policy (single file, per-test, property buckets like mode). The fix is to define supported grouping modes as options, implement them, and document deterministic naming/layout.

### [x] 329. `lib/Runtime/uvm-core/src/base/uvm_objection.svh:1359`
Status update (2026-02-28): this entry was stale. `+UVM_TIMEOUT` runtime override is already implemented in `uvm_root::m_do_timeout_settings()` and covered by `test/Runtime/uvm/uvm_timeout_plusarg_test.sv`; the `uvm_objection.svh` location now documents that path instead of carrying an unresolved plusarg TODO.

### [ ] 330. `lib/Dialect/ESI/ESIServices.cpp:98`
Cosim service lowering currently iterates bundle channels manually instead of accepting bundle types directly, indicating an abstraction gap between ESI service ops and SV primitive lowering. What is missing is first-class bundle-typed cosim op support. The fix is to allow bundle inputs at the op level and lower them later into channel primitives with matching runtime semantics.

### [ ] 331. `lib/Target/ExportSystemC/Patterns/SystemCEmissionPatterns.cpp:140`
Destructor emission hardcodes `override` based on current assumptions (`sc_module` inheritance), which can break future extensibility to custom class hierarchies. What is missing is explicit ownership of override semantics in IR metadata. The fix is to represent `override` as an operation attribute/property and emit it conditionally from modeled class relationships.

### [ ] 332. `lib/Runtime/uvm-core/src/base/uvm_misc.svh:331`
This TODO indicates current unknown-bit counting logic could be improved using `$countbits(...,'z)`, so utility behavior is functional but suboptimal/less direct. What is missing is a cleaner builtin-based implementation for Z-bit accounting. The fix is to switch when tool compatibility allows and validate equivalent behavior across simulators.

### [ ] 333. `lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:72`
Inlinability checks for `CallOpaqueOp` currently reject template arguments, leaving templated call expressions unsupported in this EmitC/SystemC path. What is missing is template-argument-aware call emission. The fix is to extend matching/printing to carry template arguments and add coverage for single-result templated calls.

### [ ] 334. `lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:84`
Same feature gap as entry 333 in statement matching: template-argument calls are excluded. What is missing is consistent template support for both expression and statement contexts. The fix is shared support in the call matcher and code emitter, with tests for void and non-void templated calls.

### [ ] 335. `lib/Dialect/Handshake/Transforms/Materialization.cpp:83`
The pass uses op-kind checks as a proxy for “already replaced/erased” state and questions this indicator, so transformation bookkeeping is brittle. What is missing is a robust explicit marker/state model for replaced operations. The fix is to track replacement status directly (attribute/set/map) instead of inferring from opcode classes.

### [x] 336. `lib/Dialect/Sim/SimOps.cpp:97`
Status update (2026-02-28): this entry is closed as stale scanner noise. `convertToDouble()` in this context is normal formatting code and not unresolved Sim implementation debt.

### [x] 337. `lib/Dialect/Sim/SimOps.cpp:100`
Status update (2026-02-28): same closure as entry 336.

### [ ] 338. `lib/Dialect/Handshake/Transforms/LockFunctions.cpp:61`
The `TODO is this UB?` comment flags uncertainty about `replaceAllUsesExcept` on function args during sync insertion, which is a correctness-risk marker. What is missing is a proven legality argument (or safer rewrite sequence) for this mutation pattern. The fix is to formalize dominance/use constraints, assert them, and add a regression that would fail if UB-like behavior occurs.

### [ ] 339. `lib/Target/ExportSystemC/ExportSystemC.cpp:39`
Header guard sanitization still allows leading digits, which can produce invalid macro identifiers in some cases. What is missing is full C/C++ identifier normalization for guard names. The fix is to enforce non-digit first character (for example prefixing `_` when needed) and keep character filtering for the rest.

### [ ] 340. `include/circt/Dialect/RTG/IR/RTGInterfaces.td:42`
Interface verification is acknowledged as incomplete because this TableGen interface path lacks a dedicated `verify` hook. What is missing is reliable invariant checking for `SetType` compatibility in interface users. The fix is to implement equivalent verification in concrete op verifiers or add supporting infrastructure so interface-level constraints are enforced centrally.

### [x] 341. `lib/Runtime/uvm-core/src/base/uvm_globals.svh:238`
Status update (2026-02-28): this gap is closed in this workspace. `uvm_string_to_severity` now delegates to `uvm_enum_wrapper#(uvm_severity)::from_name`, removing the duplicate severity-name conversion table and centralizing enum string parsing. Regression coverage was added in `test/Runtime/uvm/uvm_string_to_severity_test.sv`.

### [ ] 342. `lib/Dialect/Handshake/HandshakeUtils.cpp:276`
`NoneType` mapping carries a transitional note pending handshake switch to `i0`, so type-bridge semantics are still in migration state. What is missing is finalized zero-width type convention alignment between handshake and ESI wrapping. The fix is to complete the handshake `i0` migration and remove transitional conversion branches.

### [ ] 343. `lib/Dialect/ESI/runtime/python/CMakeLists.txt:76`
Windows stub generation for Python runtime bindings is explicitly disabled with a TODO, leaving platform parity incomplete. What is missing is a working Windows stubgen workflow for wheel builds. The fix is to resolve environment/path/tooling issues and re-enable stub generation (or ship a deterministic fallback mechanism).

### [ ] 344. `lib/Runtime/uvm-core/src/base/uvm_event.svh:466`
This `\todo` is an external-tracker note tied to Mantis 6450 and documents a temporary caveat, not an immediate CIRCT implementation gap by itself. The missing piece is lifecycle hygiene once the upstream issue is resolved. The fix is to periodically revalidate against the referenced Mantis item and remove/update the note when no longer applicable.

### [ ] 345. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:527`
Service-port object creation still relies on ad hoc type checks without a formal registration mechanism. What is missing is extensible, pluggable registration for service port wrappers. The fix is to introduce a registry keyed by backend port type/metadata and route construction through it.

### [ ] 346. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:565`
`Future.result()` ignores the timeout argument and blocks unconditionally, so API behavior diverges from caller expectations. What is missing is timeout-aware wait/get behavior with appropriate exception semantics. The fix is to pass timeout to the C++ future layer (or poll with deadline) and raise on expiration.

### [x] 347. `lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:80`
Status update (2026-02-28): this gap is closed for scalar floating-point support in this workspace. `LowerDPIFunc` now accepts both integer and float DPI port types, and regression coverage was added in `test/Dialect/Sim/lower-dpi-float.mlir`. Non-scalar/aggregate DPI ABI expansion remains future work.

### [x] 348. `lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:100`
Status update (2026-02-28): this gap is closed in this workspace. `--sim-lower-dpi-func` now validates that a reused external `func.func` (looked up via `verilogName`) exactly matches the expected lowered DPI signature and emits a targeted mismatch diagnostic before building invalid calls. Regression coverage was added in `test/Dialect/Sim/lower-dpi-errors.mlir`.

### [ ] 349. `lib/Support/TruthTable.cpp:217`
Canonicalization currently uses factorial/exponential search (`O(n! * 2^(n+m))`), which does not scale beyond tiny inputs. What is missing is a more efficient semi-canonical or heuristic minimization strategy. The fix is to replace brute-force enumeration with a scalable canonical-form algorithm and benchmark correctness/performance tradeoffs.

### [ ] 350. `lib/Dialect/ESI/runtime/python/esiaccel/codegen.py:373`
Bitfield layout handling acknowledges implementation-defined behavior, so generated packing can be non-portable across toolchains/ABIs. What is missing is explicit, deterministic bitfield layout and pack/unpack policy. The fix is to move to byte-aligned explicit serialization helpers (or equivalent deterministic schema) and test round-trip compatibility.

### [ ] 351. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:113`
The binding module explicitly suppresses leak warnings and documents known callback-related leaks, so memory ownership is not fully resolved in the Python bridge. What is missing is deterministic lifetime management for registered Python callbacks and associated C++ wrapper objects. The fix is to audit ownership edges, introduce explicit unregister/destructor paths, and re-enable leak warnings once clean.

### [ ] 352. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:209`
`ModuleInfo` bindings omit the `extra` field, leaving Python-side metadata incomplete relative to the underlying model. What is missing is full metadata exposure parity in the nanobind API. The fix is to bind the field (with a stable Python representation) and add round-trip tests for manifest/module info completeness.

### [ ] 353. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:373`
Callback connection code is marked as potentially crashing Python under certain conditions, indicating unresolved safety/ownership/threading issues in cross-language callback dispatch. What is missing is a reproducible failure characterization and hardened callback invocation model. The fix is to reproduce deterministically, then enforce GIL/lifetime/thread-safety invariants around callback execution.

### [ ] 354. `include/circt/Dialect/FIRRTL/FIRRTLVisitors.h:442`
This is documentation of default visitor chaining behavior, not unresolved implementation debt. The line is non-actionable from a project-gap perspective. The fix is scanner refinement to avoid classifying ordinary explanatory comments as TODO debt.

### [ ] 355. `lib/Support/JSON.cpp:64`
`convertToDouble()` here is routine float serialization logic and not a TODO/FIXME/unsupported marker. This is a scanner false positive. The fix is to restrict debt scans to explicit marker patterns.

### [ ] 356. `test/Tools/run-formal-all-strict-gate-bmc-lec-contract-fingerprint-parity-defaults.test:8`
This is a test harness `RUN:` script fixture line and does not represent unresolved product work. The missing issue is scanner overmatching within embedded shell text in tests. The fix is to suppress or separately classify fixture-script content in debt reports.

### [ ] 357. `lib/Dialect/Seq/SeqOps.cpp:610`
A canonicalization is currently pass-local because aggregate HW constants are not yet supported in folders, leaving fold logic split across layers. What is missing is aggregate-constant folder support that would centralize this transform. The fix is to add aggregate constant folding support and migrate this canonicalization into fold hooks.

### [ ] 358. `lib/Dialect/Seq/SeqOps.cpp:693`
Register simplification currently handles only 1D array cases and omits nested arrays/bundles. What is missing is generalized aggregate traversal for deeper container types. The fix is recursive support for nested arrays/bundles plus regressions over mixed aggregate shapes.

### [ ] 359. `lib/Dialect/Seq/SeqOps.cpp:743`
`FirRegOp::fold` does not yet handle preset values, so optimization opportunities are left unused when presets are present. What is missing is preset-aware fold semantics that preserve initialization behavior. The fix is to model preset interactions explicitly and add tests covering preset/no-preset equivalence cases.

### [ ] 360. `include/circt/Dialect/OM/OMAttributes.td:72`
Assembly currently requires explicit element type instead of inferring it from elements, making syntax noisier than necessary. What is missing is custom assembly parsing/printing that infers type where unambiguous. The fix is to implement custom format inference and retain explicit form for ambiguous cases.

### [ ] 361. `lib/Runtime/uvm-core/src/base/uvm_component.svh:3337`
Config precedence sorting still uses older API and notes migration to `sort_by_precedence_q` (Mantis 7354), leaving behavior tied to legacy ordering machinery. What is missing is alignment with the newer precedence sort utility. The fix is to switch once dependency availability is confirmed and verify ordering parity.

### [ ] 362. `lib/Dialect/Seq/Transforms/LowerSeqHLMem.cpp:120`
For higher latencies, the pass does not yet optimize whether buffering address or data is cheaper, so area/perf tradeoffs are not exploited. What is missing is latency-window-aware buffering strategy selection. The fix is cost-based buffering decisions (width/latency driven) with QoR regressions.

### [ ] 363. `lib/Dialect/HW/Transforms/HWAggregateToComb.cpp:340`
`ArraySliceOp` is not yet included in aggregate-to-comb legalization targets, leaving a partial aggregate op surface unsupported by this pass. What is missing is slice lowering support consistent with other array ops. The fix is to add `ArraySliceOp` conversion patterns and tests for slice composition cases.

### [ ] 364. `include/circt/Dialect/FIRRTL/FIRRTLTypes.td:248`
`PrintfOp` still accepts broader operand typing via a compatibility type instead of fully using `FStringType`, indicating unfinished type-system cleanup. What is missing is explicit conversion operations from `FIRRTLBaseType` to `FStringType` and migration of operand constraints. The fix is to introduce those conversions and tighten `PrintfOp` typing.

### [ ] 365. `lib/Dialect/Seq/Transforms/HWMemSimImpl.cpp:652`
Randomization logic is duplicated across `HWMemSimImpl` and `LowerToHW`, increasing maintenance drift risk. What is missing is a shared randomization utility layer. The fix is to extract common helpers and consolidate call sites so behavior changes happen in one place.

### [ ] 366. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:55`
Bitcast support excludes unions, packed arrays, and enums, so conversion coverage is incomplete for common HW type forms. What is missing is conversion rules for these type categories. The fix is to add legality + lowering for each type class with targeted negative/positive tests.

### [ ] 367. `lib/Support/PrettyPrinter.cpp:43`
This `(TODO)` sits in high-level algorithm notes and does not point to a specific missing implementation task. It is documentation debt rather than concrete feature debt. The fix is either to replace with actionable design notes or exclude such generic prose TODO markers from gap tracking.

### [ ] 368. `lib/Dialect/HW/ModuleImplementation.cpp:157`
Printing locations currently depends on an upstream alias-emission quirk, requiring local flag-based workaround logic. What is missing is upstream fix for `printOptionalLocationSpecifier` alias behavior. The fix is to remove workaround once upstream is corrected and keep temporary guard logic until then.

### [ ] 369. `lib/Dialect/HW/ModuleImplementation.cpp:189`
Same upstream-printing workaround as entry 368 in another print path. What is missing is consistent location alias control without ad hoc branching. The fix is shared with entry 368 and should be deduplicated when upstream is fixed.

### [ ] 370. `lib/Dialect/HW/ModuleImplementation.cpp:441`
Third occurrence of the same alias-emission workaround, confirming repeated technical debt across module printing paths. What is missing is centralized handling or removal post-upstream fix. The fix is to unify these sites behind one helper and prune once upstream behavior is corrected.

### [ ] 371. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:57`
Service lookup helper is reused for engines and explicitly labeled a hack, indicating abstraction mismatch in manifest parsing. What is missing is separate typed construction paths for services versus engines. The fix is to split the parser responsibilities and avoid mutating shared service state for engine creation.

### [ ] 372. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:178`
Constant parsing does not validate/coerce JSON values against declared types, risking mismatches reaching runtime consumers. What is missing is type-guided conversion/validation for manifest constants. The fix is strict conversion with clear diagnostics on incompatible values.

### [ ] 373. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:290`
Engine creation currently only supports top-level manifest location, so nested/lower-level engine declarations are ignored. What is missing is hierarchical engine discovery/instantiation. The fix is recursive traversal support with deterministic app-id path handling.

### [ ] 374. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:324`
`createEngine` is explicitly described as a giant hack to reuse service code paths, reinforcing architectural debt in runtime manifest handling. What is missing is a proper engine construction API separated from service registration. The fix is dedicated engine parsing + factory plumbing.

### [ ] 375. `lib/Dialect/SV/Transforms/SVExtractTestCode.cpp:82`
Backward recursion logic intentionally bails on some parent-op block structures, which can miss relevant test-code extraction opportunities. What is missing is principled control-flow-aware backward traversal across multi-block parents. The fix is to define safe recursion criteria and add coverage for multi-block region cases.

### [ ] 376. `lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:422`
Service creation dispatch is hardcoded by explicit `typeid` checks instead of using a registration table, limiting extensibility. What is missing is plugin-style service factory registration. The fix is a registry mapping service type identifiers to constructors.

### [ ] 377. `lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:435`
Name-to-type lookup is similarly hardcoded with string comparisons, duplicating the manual dispatch issue in entry 376. What is missing is one authoritative service registry for both lookup and construction. The fix is to unify discovery and instantiation through shared registration metadata.

### [ ] 378. `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:998`
`hw.inout` on outputs is still unsupported and guarded by an assert path, so this transformation is incomplete for that direction. What is missing is lowering semantics for output-side inouts. The fix is to implement explicit conversion strategy for output inouts and remove the assert-only fallback.

### [ ] 379. `include/circt/Dialect/FIRRTL/FIRParser.h:77`
The location-caching API is marked as overly awkward and leaks implementation details to callers. What is missing is a cleaner parser interface that encapsulates cache behavior. The fix is API redesign around simpler contract boundaries and migration of call sites.

### [ ] 380. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcClient.cpp:180`
Manifest fetch currently spins in a retry loop to dodge a race condition, pending a DPI API change. What is missing is explicit synchronization/readiness signaling in the backend protocol. The fix is to implement the DPI/API change and replace polling loops with deterministic handshake semantics.

### [ ] 381. `lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:184`
Prettification currently handles only a narrow concatenation pattern and is explicitly not generalized to ranges/arbitrary concatenations. What is missing is broader normalization logic for sliced and mixed concatenation forms. The fix is to extend field extraction to cover range-based patterns and arbitrary concat trees with correctness tests on emitted Verilog readability and equivalence.

### [ ] 382. `lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:351`
Event-control expression filtering uses local conditions and notes divergence from `ExportVerilog`’s `allowExprInEventControl` policy, risking inconsistent legality between passes. What is missing is shared event-control admissibility logic. The fix is to centralize the predicate (or reuse ExportVerilog helper) so Prettify and export paths make identical decisions.

### [ ] 383. `lib/Scheduling/SimplexSchedulers.cpp:924`
Operation prioritization is acknowledged as simplistic, which can degrade schedule quality under resource pressure. What is missing is a richer priority heuristic incorporating dependency depth, slack, and resource contention. The fix is to implement a tunable priority function and benchmark schedule QoR.

### [ ] 384. `lib/Scheduling/SimplexSchedulers.cpp:1149`
A tie-break condition still relies on an ad hoc final comparator instead of graph-aware analysis, which can produce unstable/non-optimal movement choices. What is missing is principled graph-based tie resolution. The fix is to replace the fallback with explicit dependence/criticality analysis.

### [ ] 385. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:427`
The comment explicitly says callback-side types are wrong and should be channel-wrapped, so runtime type semantics are currently incomplete. What is missing is correct channel-typed wrapping for cosim service endpoints. The fix is to update type mapping and endpoint plumbing so read/write callbacks use proper channel abstractions.

### [ ] 386. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:489`
Host-memory read callback trusts requested locations without checking mapping validity, which is a safety/correctness gap. What is missing is mapping verification before memory access. The fix is to validate requested address ranges against mapped regions and reject invalid access with diagnostics.

### [ ] 387. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:517`
Same issue as entry 386 on write path: missing mapped-memory validation before writes. What is missing is symmetric bounds/mapping checks for write requests. The fix is to perform mapping validation and return structured errors on unmapped writes.

### [ ] 388. `lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:211`
Trace backend read logic only supports bitwidth-resolvable types and throws for other runtime types. What is missing is support for additional type classes (or graceful conversion) in trace reads. The fix is to add handlers for currently unsupported types and preserve clear errors for truly unknown ones.

### [ ] 389. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:153`
RPC server still uses insecure credentials by default, even if localhost-limited, which leaves security hardening unfinished. What is missing is configurable secure transport/authentication for deployments beyond trusted local use. The fix is TLS/credential support with secure-by-default options where practical.

### [ ] 390. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:310`
Write reactor loop polls with sleeps pending a future notification mechanism, which is inefficient and latency-sensitive. What is missing is event-driven notification from write-queue activity. The fix is to switch to condition-variable or callback-driven wakeups and remove busy waiting.

### [ ] 391. `lib/Dialect/ESI/Passes/ESIBuildManifest.cpp:32`
This is maintainability debt: code is called “ugly but works,” implying structure/readability issues rather than immediate feature loss. What is missing is pass-internal cleanup and clearer decomposition. The fix is a refactor into smaller helpers with preserved behavior and regression checks.

### [ ] 392. `lib/Dialect/ESI/ESIOps.cpp:478`
Container conversion/type checking falls through with a TODO for “other container types,” so support is incomplete beyond currently handled cases. What is missing is coverage for additional container categories. The fix is to add type-switch cases and validation for the remaining supported container families.

### [ ] 393. `include/circt/Dialect/MSFT/MSFTConstructs.td:28`
This TODO is documentation debt for op description completeness, not an implementation blocker. What is missing is mature, user-facing semantics documentation once the op stabilizes. The fix is to expand description text with invariants/examples after behavior is finalized.

### [ ] 394. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:396`
The loop currently yields while polling and explicitly seeks a better mechanism, indicating suboptimal wait strategy. What is missing is explicit notification/wakeup for completion or queue activity. The fix is to replace poll+yield with synchronization primitives to cut idle overhead.

### [ ] 395. `test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:7`
This line is test harness fixture script text, not unresolved product work. It is non-actionable from a gap perspective. The fix is scanner filtering for embedded `RUN:` script payloads in lit tests.

### [ ] 396. `test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:24`
This is expected output (`PARITY`) assertion content in a regression test, not technical debt. The missing issue is scan precision, not implementation. The fix is to exclude FileCheck expectation lines from gap-marker extraction.

### [ ] 397. `lib/Dialect/Datapath/DatapathFolds.cpp:154`
A fold that should conceptually live as `CompressOp` canonicalization remains implemented elsewhere due current operand-introduction constraints. What is missing is canonicalization placement parity and cleaner pattern ownership. The fix is to relax/adjust constraints or pattern infrastructure so this rewrite can move to `CompressOp` canonicalization.

### [ ] 398. `lib/Dialect/Datapath/DatapathFolds.cpp:316`
The transformation does not yet exploit known-bits analysis to capture all constant-one opportunities. What is missing is known-bits-driven constant extraction for broader fold coverage. The fix is to integrate known-bits data in this pattern and add cases that currently miss simplification.

### [ ] 399. `lib/Dialect/Datapath/DatapathFolds.cpp:426`
Partial-product folding lacks constant multiplication support, leaving obvious constant-driven reductions unavailable. What is missing is constant-multiplier handling in `PartialProductOp` logic. The fix is to implement constant multiplication folding and verify width/sign behavior.

### [ ] 400. `lib/Dialect/Datapath/DatapathFolds.cpp:484`
Current rewrite path requires equal input widths and bails on mixed-width cases. What is missing is heterogeneous-width support. The fix is to add sign/extension-aware handling for differing widths and extend regression coverage accordingly.

### [ ] 401. `include/circt/Dialect/MSFT/DeviceDB.h:50`
`getLeaf` exposes mutable access only, and TODO calls for a read-only variant. What is missing is const-correct API surface for safe lookup callers. The fix is to add const overload/read-only accessor and migrate non-mutating users.

### [ ] 402. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:322`
Driver-walk logic intentionally stops before looking through certain unary ops, limiting traceability in some single-driver cases. What is missing is optional look-through for unambiguous unary wrappers. The fix is to add guarded unary-op traversal when semantics are provably single-driver.

### [ ] 403. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:333`
Behavior for flipped types is unresolved and code asserts passive types, so mixed-flow traversal semantics are not defined. What is missing is explicit policy for reverse-flow fields during walking. The fix is to define flip handling rules and implement them without passive-type-only assumptions.

### [ ] 404. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:452`
Same unary-op look-through gap as entry 402 in a related walker path. What is missing is consistent optional traversal through safe unary wrappers. The fix is shared extension across both walker implementations.

### [ ] 405. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:726`
Unexpected type cases currently fatal-error instead of returning structured failures to callers. What is missing is plumbed error propagation and recoverable diagnostics. The fix is to convert fatal path to error result handling through caller chain.

### [ ] 406. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:768`
Enum handling in aggregate walking is semantically uncertain (“are enums aggregates or not”), leaving type-walk behavior potentially inconsistent. What is missing is a clear enum classification policy in walkGroundTypes. The fix is to decide and codify enum traversal semantics, then align callers/tests.

### [ ] 407. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:827`
Property insertion sorts after append and notes desire for always-sorted structure, implying avoidable overhead. What is missing is data-structure/API support for ordered insertion. The fix is to maintain sorted order incrementally (binary insert) and avoid full resort each time.

### [ ] 408. `include/circt/Dialect/ESI/ESIStdServices.td:56`
Out-of-order return support is acknowledged but port modeling is unfinished. What is missing is explicit service ports/protocol for OOO returns. The fix is to define the return-channel interface and implement end-to-end lowering/runtime handling.

### [x] 409. `test/Tools/circt-sim/syscall-strobe.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-strobe.sv`), so the previously documented immediate-vs-postponed `$strobe` mismatch is no longer present in current behavior.

### [ ] 410. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:4`
This is a lit fixture line constructing expected LEC TSV input, not an unresolved implementation marker. It is audit noise from scanning test setup text. The fix is to exclude test fixture data lines from actionable debt extraction.

### [ ] 411. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:5`
This is test setup fixture content (`allow.txt` generation) and not unresolved implementation work. It is non-actionable from a product-gap perspective. The fix is to suppress lit `RUN:` fixture payload lines from debt scans.

### [ ] 412. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:12`
This is expected parity output assertion text in a regression test, not a TODO/unsupported implementation marker. The missing issue is scanner overmatching on test expectations. The fix is to exclude FileCheck `PARITY` lines from actionable gap extraction.

### [ ] 413. `lib/Dialect/Datapath/DatapathOps.cpp:129`
Compressor tree construction does not yet fold constant-one bits, leaving simplification potential unused during bit-level reduction planning. What is missing is constant-one aware folding alongside existing known-zero handling. The fix is to integrate known-bits one detection and reduce compressor workload accordingly.

### [ ] 414. `lib/Dialect/Datapath/DatapathOps.cpp:216`
Current algorithm still leans on Dadda assumptions of uniform arrival, which is mismatched with timing-driven objectives in this context. What is missing is true arrival-time-aware compression strategy. The fix is to replace/augment Dadda scheduling with timing-driven selection using per-signal arrival metadata.

### [ ] 415. `lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:98`
Direction semantics in `BundleType` are under-documented, which can cause ambiguity for API users and backend implementers. What is missing is clear directional contract documentation (`To`/`From`) with examples. The fix is to flesh out API docs with host/device perspective and channel-direction mapping.

### [ ] 416. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:4`
This line is test data fixture construction for LEC contracts, not unresolved implementation debt. It should be treated as non-actionable audit text. The fix is scanner filtering for lit fixture payload lines.

### [ ] 417. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:12`
This is FileCheck expected output for a failure-mode parity test and does not indicate missing code. The true issue is scan precision in test expectation contexts. The fix is to suppress expectation lines from debt reports.

### [x] 418. `test/Tools/circt-sim/syscall-shortrealtobits.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-shortrealtobits.sv`), so the previously documented negative-value bit-pattern mismatch is no longer reproducible.

### [ ] 419. `lib/Dialect/FIRRTL/FIRRTLReductions.cpp:1821`
Port-name namespace workarounds remain necessary because downstream expectations still rely on uniqueness assumptions. What is missing is consistent IR/pass invariants that make local namespace generation unnecessary. The fix is to align passes on naming contracts or centralize uniqueness enforcement.

### [x] 420. `test/Tools/circt-sim/syscall-randomize-with.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-randomize-with.sv`), so the previously documented inline-constraint gap is no longer observable in this scenario.

### [ ] 421. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/Trace.h:44`
Trace backend explicitly lacks full trace mode support, so behavior is partial by design. What is missing is complete mode coverage (read/write/validation/generation combinations) as documented. The fix is to implement full trace mode semantics and add backend conformance tests.

### [x] 422. `test/Tools/circt-sim/syscall-random.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-random.sv`), so the seed update behavior mismatch is not reproducible on current code.

### [ ] 423. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:25`
`RpcServer` is noted as not yet a “proper backend,” reflecting architectural incompleteness in backend abstraction conformance. What is missing is full backend-interface parity (lifecycle/config/error handling). The fix is to align `RpcServer` with standard backend contracts and remove special-case behavior.

### [ ] 424. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:36`
Manifest publication has a startup race where clients can connect before manifest is set. What is missing is a protocol/API guarantee that manifest availability precedes client interactions. The fix is DPI/API ordering enforcement or connection gating until manifest initialization completes.

### [ ] 425. `lib/Dialect/RTG/Transforms/ElaborationPass.cpp:1978`
The pass currently relies on assumptions about allocate-effect reorderability that are acknowledged as misaligned with MLIR `MemoryEffects` intent. What is missing is a dedicated effect model/trait for this transformation. The fix is to define a custom ordering interface or preserve explicit ordering when materializing operations.

### [ ] 426. `lib/Dialect/RTG/Transforms/ElaborationPass.cpp:2094`
Sequences are cloned even when a unique remaining reference could permit reuse, adding unnecessary duplication. What is missing is ownership/reference-aware reuse optimization. The fix is to detect sole-remaining references and avoid clone when safe.

### [ ] 427. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:53`
Active-interval management uses a simple vector+sort approach with known scalability limits. What is missing is a more suitable data structure for interval expiration queries. The fix is to use an ordered set/heap structure keyed by end index.

### [ ] 428. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:82`
The allocator assumes fully elaborated IR but does not verify it, risking silent misbehavior on unexpected input states. What is missing is explicit precondition checking. The fix is to assert/diagnose non-elaborated constructs before allocation.

### [ ] 429. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:119`
Current live-range computation lacks support for labels/jumps/loops, so control-flow-heavy code is not modeled correctly. What is missing is control-flow-aware liveness across non-linear execution. The fix is CFG-based interval construction rather than linear-block approximation.

### [ ] 430. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:131`
Fixed-register overlap handling intentionally overapproximates, which can cause unnecessary register pressure and suboptimal allocation. What is missing is tighter conflict modeling. The fix is to refine overlap constraints to exact interference intervals.

### [ ] 431. `lib/Dialect/ESI/runtime/cpp/include/esi/Utils.h:78`
Queue pop path copies data defensively to avoid invalidated references, trading safety for extra copying overhead. What is missing is zero-copy-safe ownership model. The fix is to redesign buffer/lifetime handling (move semantics/stable storage) to avoid mandatory copies.

### [x] 432. `test/Tools/circt-sim/syscall-monitor.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-monitor.sv`), indicating monitor retrigger behavior in this test case is working.

### [ ] 433. `include/circt/Dialect/Calyx/CalyxPrimitives.td:543`
`AnyInteger` here is a type constraint name in op definition, not TODO or unsupported debt. This is a scanner false positive caused by token matching. The fix is to constrain scans to explicit debt markers rather than generic identifiers.

### [x] 434. `test/Tools/circt-sim/syscall-isunbounded.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-isunbounded.sv`), so the documented class/type-parameter failure is not currently reproducible.

### [ ] 435. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:96`
Assembly emission assumes `//` as line-comment syntax, which is not portable across all target assembly dialects. What is missing is target-aware comment syntax abstraction. The fix is configurable/target-specific comment emit rules.

### [ ] 436. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:120`
Binary emission hardcodes `.word`, limiting portability to ISAs/assemblers using different directives. What is missing is target-specific data directive selection. The fix is backend-configurable directive mapping rather than literal `.word`.

### [ ] 437. `lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:335`
Callback API currently returns no handle/future for completion/error observation, reducing composability and diagnosability. What is missing is callback completion signaling primitive. The fix is to have callback registration return an object for wait/status/notification.

### [ ] 438. `lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:458`
`getAs()` constness is flagged as likely wrong relative to intended user access model, indicating API const-correctness mismatch. What is missing is coherent mutability policy for bundle-port accessors. The fix is to revisit const surface and update caller interfaces accordingly.

### [ ] 439. `include/circt/Dialect/Calyx/CalyxPasses.td:27`
Pass description still lists unresolved TODO for multi-write signal read replacement by disjunction, implying transformation semantics are incomplete in that corner case. What is missing is defined handling for conflicting writes. The fix is to implement/read-resolve strategy for multi-write scenarios and validate with pass tests.

### [x] 440. `test/Tools/circt-sim/syscall-generate.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-generate.sv`), so the documented width/padding mismatch is not observed in current behavior.

### [x] 441. `test/Tools/circt-sim/syscall-fread.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-fread.sv`), indicating `$fread` support is present for this tested flow.

### [ ] 442. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:304`
`getDeclarationKind` duplicates traversal logic already present in `foldFlow`, indicating unnecessary duplication and drift risk. What is missing is a unified analysis path returning both flow and declaration kind information. The fix is to combine walkers into one shared routine with structured return type.

### [ ] 443. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:1331`
This is another local workaround for upstream location alias printing behavior. What is missing is upstream `printOptionalLocationSpecifier` fix so local flag-based guards can be removed. The fix is to keep temporary guard and clean up once upstream behavior is corrected.

### [ ] 444. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:1676`
Parsing still stores visibility in attributes while noting migration to operation properties. What is missing is full property-based representation for visibility. The fix is to move parse/print/verify paths to properties and deprecate legacy attribute plumbing.

### [ ] 445. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:2100`
Same migration gap as entry 444 in another parser helper: visibility handling is not property-native yet. What is missing is consistent property-based handling across FIRRTL ops. The fix is shared property migration and compatibility tests.

### [ ] 446. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:3461`
Missing bundle flip on read ports is marked FIXME but not enforced, leaving verifier weakness around memory port directionality. What is missing is explicit erroring/diagnostics when expected flip constraints are violated. The fix is to implement the check and produce actionable verifier errors.

### [ ] 447. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4076`
Verifier currently rejects some read sources conservatively and calls out future relaxation for output ports and instance/memory input ports. What is missing is broader legal-source modeling in connect-like checks. The fix is to extend flow rules for those source categories with safety tests.

### [ ] 448. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4321`
`ref.sub` destination flow policy is unresolved, leaving conservative prohibition without finalized semantics. What is missing is a clear flow contract for `ref.sub` in destination position. The fix is to decide source-only vs broader flow semantics and encode verifier rules accordingly.

### [ ] 449. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4949`
Bundle create verification checks type but skips flow validation. What is missing is flow-consistency verification for each element. The fix is to add flow checks and reject mismatched directional composition.

### [ ] 450. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4962`
Vector create has the same omission as entry 449: type checks exist but flow checks are TODO. What is missing is flow validation for vector element assembly. The fix is to implement flow checks parallel to bundle case.

### [ ] 451. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6791`
`ref.sub` with `rwprobe` behavior is explicitly undecided and currently allowed pending clarification, which risks semantic ambiguity. What is missing is specified semantics and tests for this interaction. The fix is to choose behavior (likely demotion path as noted), implement it, and add conformance tests.

### [ ] 452. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6923`
Verifier does not check that target type matches op type for one target-using op. What is missing is type-equality validation between referenced target and operation result. The fix is to add explicit type match verification with clear mismatch diagnostics.

### [ ] 453. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6935`
Same target-type verification gap as entry 452 on the sibling op. What is missing is consistent type-check enforcement across both operations. The fix is shared verifier utility to prevent drift.

### [ ] 454. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:7023`
Verifier intentionally avoids full non-passive `ConnectLike` checking due complexity, leaving partial validation. What is missing is comprehensive direction/drive verification for non-passive cases. The fix is to encode full connect direction analysis and expand verifier coverage.

### [x] 455. `test/Tools/circt-sim/syscall-feof.sv:2`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-feof.sv`), so the previously described integration-level file I/O failure is not reproducible.

### [ ] 456. `include/circt/Dialect/Calyx/CalyxLoweringUtils.h:204`
Lowering utility lacks a post-insertion invariant check for use-def ordering among scheduled groups. What is missing is enforcement of ordering invariants after conversion. The fix is to add post-insertion validation and fail fast when invariant is violated.

### [ ] 457. `lib/Dialect/Comb/CombFolds.cpp:1091`
Constant combination only handles trailing constants and misses opportunities when constants appear elsewhere in operand list. What is missing is position-independent constant aggregation. The fix is to gather/fold all constants regardless of placement.

### [ ] 458. `lib/Dialect/Comb/CombFolds.cpp:1109`
Current replicate-and-mask rewrite is limited to single-bit replicated operands. What is missing is generalized handling for wider replicated operands. The fix is to extend pattern arithmetic for multi-bit operand replication.

### [ ] 459. `lib/Dialect/Comb/CombFolds.cpp:1205`
`and(..., x, not(x))` complement fold is unimplemented, leaving obvious simplification unavailable. What is missing is complement pair detection in variadic and-folds. The fix is to add complement canonicalization to zero.

### [ ] 460. `lib/Dialect/Comb/CombFolds.cpp:1421`
Equivalent complement fold for OR (`or(..., x, not(x)) -> all ones`) is also missing. What is missing is complement detection for variadic or-folds. The fix is complementary OR canonicalization.

### [ ] 461. `lib/Dialect/Comb/CombFolds.cpp:2194`
Case analysis from comparisons handles only exact equality/inequality and not range predicates like `x < 2`. What is missing is richer predicate decomposition into multiple case entries. The fix is to split range predicates into equivalent discrete/interval cases where tractable.

### [ ] 462. `lib/Dialect/Comb/CombFolds.cpp:2507`
A rewrite is limited to concat though similar opportunities exist for and/or/xor/icmp-not forms. What is missing is pattern generalization across common boolean/combinational ops. The fix is to abstract the rewrite framework and reuse it for those op families.

### [ ] 463. `lib/Dialect/Comb/CombFolds.cpp:3254`
Zero-bit concat behavior remains behind FIXME workaround due upstream/merge issue, preventing clean constant accumulation through zero-width spans. What is missing is robust zero-width concat handling. The fix is to resolve the zero-bit concat limitation and remove workaround branching.

### [ ] 464. `include/circt/Dialect/Calyx/CalyxHelpers.h:49`
Control-leaf helper still omits `Invoke`, tracked in issue 1679, so control-graph utilities are incomplete. What is missing is `Invoke` integration in leaf-node classification. The fix is to extend helper logic and add tests covering invoke-containing control trees.

### [ ] 465. `lib/Dialect/ESI/runtime/cosim_dpi_server/CMakeLists.txt:14`
Build/install currently packages a dummy simulator library to avoid runtime link errors, marked as improper. What is missing is correct linkage strategy against simulator-provided symbols without shipping placeholders. The fix is to rework link/install rules and loader behavior to remove dummy packaging.

### [ ] 466. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:541`
A hand-written fold pattern is marked for DRR migration, indicating maintainability/consistency debt in fold implementation style. What is missing is declarative rewrite representation for this fold. The fix is to port to DRR and keep behavior parity tests.

### [ ] 467. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:854`
`eq` fold shortcuts do not yet support `SInt<1>` left-hand side variants. What is missing is signed 1-bit equivalence handling in this fold family. The fix is to extend the predicate conditions for `SInt<1>` and related compatible cases.

### [ ] 468. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:904`
Same limitation as entry 467 for `neq` fold path. What is missing is `SInt<1>` support in inequality canonicalization. The fix is parallel extension and tests for both eq/neq.

### [ ] 469. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:950`
`IntegerAddOp::fold` still lacks constant folding and related simplifications (tracked upstream). What is missing is baseline arithmetic folding support for integer add. The fix is to implement constant fold rules and algebraic identities.

### [ ] 470. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:956`
`IntegerMulOp::fold` similarly has no constant folding implementation yet. What is missing is multiplicative constant/algebraic fold support. The fix is to implement fold rules (zero/one/constants) and verify overflow/width semantics.

### [ ] 471. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1657`
Mux folding is missing the common `x ? ~0 : 0 -> sext(x)` optimization. What is missing is this canonical boolean-to-mask transform in FIRRTL folds. The fix is to detect all-ones/zero constant arms and rewrite to sign-extension pattern.

### [ ] 472. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1658`
General constant-arm mux tricks are acknowledged but not implemented, leaving multiple canonicalization opportunities unused. What is missing is broader `x ? c1 : c2` algebraic simplification coverage. The fix is to implement a table of constant-pattern rewrites and guard with width/sign correctness.

### [ ] 473. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1660`
Pattern `x ? a : 0` is not yet lowered to a more compact mask form. What is missing is selective-zero arm simplification. The fix is to canonicalize into `sext(x) & a` (or equivalent typed form) where legal.

### [ ] 474. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1663`
Canonical arm ordering for `x ? c1 : y` is unimplemented, reducing normalization consistency. What is missing is rewrite to preferred predicate/arm form (e.g. inverted select with swapped arms). The fix is to add deterministic canonicalization for constant-vs-variable arm placement.

### [ ] 475. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2042`
Optimization currently requires `index` to be `uint<1>`, skipping equivalent cases with wider indices. What is missing is index-width-general handling. The fix is to support wider-but-constrained index forms via known-bits/range checks.

### [ ] 476. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2155`
`MatchingConnectOp` canonicalization lacks normalization toward explicit extensions and flips. What is missing is a canonical form policy for extension/flip representation. The fix is to rewrite implicit forms into explicit extension/flip ops for consistent downstream behavior.

### [ ] 477. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2207`
Fold/delete behavior may ignore `dontTouch`/annotation sensitivity in some paths, risking unsafe canonicalization. What is missing is full annotation-aware safety gating. The fix is to thread annotation checks through these rewrites before erasing/rewiring ops.

### [ ] 478. `lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:227`
Reference-type targeting check uses a local workaround and notes missing `containsReference()` utility. What is missing is robust reusable reference-containment query in annotation helpers. The fix is to implement `containsReference()` and replace ad hoc checks.

### [ ] 479. `lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:268`
Same helper gap as entry 478 on another annotation path. What is missing is centralized reference-detection utility to avoid duplicated logic. The fix is shared utility adoption across both call sites.

### [ ] 480. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:133`
Endpoint registration API currently multiplexes read/write registration in one function and lacks handle-returning structure. What is missing is cleaner split API with explicit handles. The fix is to separate read/write registration entrypoints and return stable endpoint handles.

### [ ] 481. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:357`
Low-level DPI entry points are documented as degraded after gRPC conversion and effectively unmaintained. What is missing is functional restoration of these entry points or their formal deprecation. The fix is to revive and test them (or remove dead paths with migration guidance).

### [ ] 482. `lib/Dialect/Comb/Transforms/IntRangeAnnotations.cpp:110`
Int-range annotation pass omits subtraction support, limiting arithmetic coverage. What is missing is subtraction transfer-function handling in range inference/annotation. The fix is to add subtraction rules and associated correctness tests.

### [ ] 483. `lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:86`
Driver lacks configurable max simulation speed/cycle pacing for interactive debug use. What is missing is runtime throttling control. The fix is a configurable cycles-per-second option (CLI or API) with deterministic pacing behavior.

### [ ] 484. `lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:94`
Reset sequencing does not yet implement ESI reset handshake protocol. What is missing is protocol-level reset handshake support in the cosim driver. The fix is to model handshake states/signals and gate startup traffic accordingly.

### [ ] 485. `lib/Dialect/RTG/IR/RTGAttributes.cpp:26`
Dense-set hashing uses simple XOR and notes weak collision resistance. What is missing is stronger order-insensitive hash composition. The fix is to use a better combination strategy (e.g. commutative mix with robust avalanche properties).

### [ ] 486. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1282`
Emitter still has manual memory-port emission where `emitAssignLike` abstraction should likely be used. What is missing is shared assignment-like emission path. The fix is to refactor this output through common emitter helper for consistency.

### [ ] 487. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1386`
Another emission site bypasses common assign-like machinery, increasing duplication. What is missing is unification on assignment emission helper semantics. The fix is shared helper migration and output regression checks.

### [ ] 488. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1465`
Literal emission currently lacks configurable radix formatting. What is missing is user/control option for base-2/8/10/16 output. The fix is an emitter option and deterministic formatting policy per base.

### [ ] 489. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1477`
Type-alias declaration emission is missing in one code path, so aliases may be inlined/erased in emitted FIR. What is missing is explicit alias decl emission support. The fix is to emit alias declarations and references consistently.

### [ ] 490. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1743`
Same alias-declaration emission gap as entry 489 in another type-printing path. What is missing is complete alias support across all emitter branches. The fix is shared alias-emission utility usage.

### [ ] 491. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1843`
Location emission does not properly handle `FusedLoc` or deduplicate repeated file names. What is missing is robust location formatting/uniquification policy. The fix is fused-location flattening and per-file dedup logic in emitter output.

### [ ] 492. `lib/Dialect/Comb/Transforms/BalanceMux.cpp:216`
Mux balancing splits ranges evenly rather than by arrival timing, so critical-path optimization is limited. What is missing is timing-aware separator selection. The fix is to choose split points using arrival-time/cost heuristics.

### [ ] 493. `lib/Dialect/FIRRTL/Import/FIRLexer.cpp:99`
Lexer escape handling omits octal/unicode escapes, so import coverage is incomplete for valid string literal forms. What is missing is full FIRRTL escape-sequence support. The fix is to implement remaining escapes with parser tests.

### [ ] 494. `include/circt/Dialect/Arc/ArcOps.td:333`
`AnyInteger` here is a normal ODS type constraint, not a TODO or unsupported marker. This is a scanner false positive. The fix is debt scanning that ignores ordinary type-constraint identifiers.

### [ ] 495. `include/circt/Dialect/Arc/ArcOps.td:335`
Same as entry 494: regular ODS result type declaration, non-actionable for gap tracking. The fix is scanner filtering.

### [ ] 496. `include/circt/Dialect/Arc/ArcOps.td:411`
Again an `AnyInteger` type constraint in op arguments, not unresolved work. This is audit noise. The fix is explicit marker-only scanning.

### [ ] 497. `include/circt/Dialect/Arc/ArcOps.td:413`
Same false-positive class as entries 494–496: routine ODS type use, not debt. The fix is scanner refinement.

### [ ] 498. `include/circt/Dialect/Arc/ArcOps.td:427`
`AnyInteger` occurrence in argument list is not actionable implementation gap. This should be filtered from TODO audits.

### [ ] 499. `include/circt/Dialect/Arc/ArcOps.td:429`
Same non-actionable ODS type-token hit as surrounding entries. The fix is to exclude such declarations from debt marker extraction.

### [ ] 500. `include/circt/Dialect/Comb/CombOps.h:97`
Power-of-two division/modulo rewrite helpers currently only cover unsigned forms and explicitly omit signed versions. What is missing is signed div/mod power-of-two conversion support. The fix is to add signed rewrite logic with correct rounding/sign semantics.

### [ ] 501. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:754`
Identifier parsing still lacks `RelaxedId` support, so some spec-permitted naming forms are not accepted. What is missing is parser support for relaxed identifier grammar. The fix is to extend token/identifier handling for `RelaxedId` and add import regressions.

### [ ] 502. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2122`
Connect expansion logic is duplicated between parser import and LowerTypes-style behavior, increasing drift risk. What is missing is shared connect-expansion machinery. The fix is to factor expansion into common utilities or delegate to LowerTypes where possible.

### [ ] 503. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2167`
`exp '.' DoubleLit` remains a documented workaround path (issue #470), indicating grammar debt around dotted numeric forms. What is missing is a clean parser/spec-aligned resolution instead of workaround annotation. The fix is to settle grammar behavior and remove workaround-specific handling.

### [ ] 504. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2426`
Same workaround note as entry 503 in a second grammar description location. What is missing is consolidated, resolved treatment of this construct. The fix is to unify grammar comments/implementation after issue #470 is fully addressed.

### [ ] 505. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:3517`
Parser accepts an `else`-info grammar variant that appears unnecessary per spec commentary. What is missing is spec clarification and parser simplification to match actual generated forms. The fix is to confirm spec intent and trim unnecessary optional syntax.

### [ ] 506. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4056`
Parser intentionally accepts `ref_expr` where spec says `static_reference` for `read(probe(x))`, reflecting a permissive compatibility deviation. What is missing is explicit reconciliation between spec wording and practical accepted syntax. The fix is either spec update alignment or stricter parse + migration diagnostics.

### [ ] 507. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4097`
Unsupported reference-source checks are currently parser-local and not fully mirrored in `ref.send` verifier/inference. What is missing is verifier/type-inference parity for these constraints. The fix is to move or duplicate checks into verifier/inferReturnTypes for consistency.

### [ ] 508. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4341`
Agg-of-ref rejection checks exist but lack dedicated regression coverage until feature support lands. What is missing is future-proof tests to lock this behavior. The fix is to add explicit parser/verifier tests once agg-of-ref is introduced.

### [ ] 509. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4514`
Same test-coverage gap as entry 508 in a second connect path. What is missing is matching regression coverage for both variants. The fix is to add paired tests for both parser codepaths.

### [ ] 510. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4722`
`cmem` is parsed despite being undocumented in the FIRRTL spec, leaving spec/implementation mismatch. What is missing is formal spec documentation or deprecation guidance. The fix is spec alignment and explicit parser behavior notes.

### [ ] 511. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4754`
Same mismatch as entry 510 for `smem`: implementation support exists but spec documentation is missing. What is missing is authoritative grammar/semantics in spec text. The fix is to document `smem` or gate it as extension with clear diagnostics.

### [ ] 512. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4900`
Memory-port alphabetical canonicalization is performed in parser instead of op construction/canonicalization layer. What is missing is proper ownership of canonicalization in IR-level facilities. The fix is to move sorting into MemOp canonicalization/build logic.

### [ ] 513. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5038`
Register grammar comment notes `info` placement inconsistency with expected ordering. What is missing is spec-consistent placement rules for register info fields. The fix is align parser order with finalized grammar and keep compatibility handling if needed.

### [ ] 514. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5059`
Reset grammar remains overly complex/ambiguous and indentation-sensitive per TODO note. What is missing is simplified unambiguous reset grammar. The fix is grammar cleanup plus parser update with compatibility diagnostics.

### [ ] 515. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5082`
Importer carries compatibility logic for improperly pretty-printed no-reset registers from Scala implementation. What is missing is clean upstream pretty-printer behavior (or eventual removal timeline for workaround). The fix is to correct pretty-print output at source and retire compatibility path when safe.

### [ ] 516. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5932`
Old `, bound = N` syntax is still accepted alongside new parameter style, keeping legacy parse debt alive. What is missing is migration completion to one canonical syntax. The fix is deprecate/remove old variant after transition window.

### [ ] 517. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:365`
Const-prop does not yet account for `OptimizableExtModuleAnnotation`, limiting annotation-guided optimization behavior. What is missing is annotation-aware extmodule handling. The fix is to interpret this annotation in lattice/replace decisions.

### [ ] 518. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:539`
Aggregate operations (vector/bundle creates and related ops) are still treated conservatively, reducing const-prop effectiveness on aggregates. What is missing is aggregate-aware lattice propagation/folding. The fix is to add aggregate operation semantics to const-prop transfer functions.

### [ ] 519. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:861`
`when` operations are not handled by this propagation path, leaving control-dependent constants under-optimized. What is missing is conditional control-flow support in const-prop. The fix is to model `when` regions and merge lattice states correctly.

### [ ] 520. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1027`
Replacement currently works field-wise and does not replace whole aggregate constants directly. What is missing is entire-aggregate materialization/substitution. The fix is aggregate-level constant emission and direct full-value replacement.

### [ ] 521. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1034`
Pass uses manual type allowlist because `materializeConstant` cannot self-report support and may assert. What is missing is safe capability query/soft-failure API on constant materialization. The fix is to make `materializeConstant` return failure on unsupported types and simplify caller logic.

### [ ] 522. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1056`
Deletion/cleanup handling for `WhenOp` is still incomplete, risking incorrect or missed simplifications after propagation. What is missing is correct post-propagation treatment of conditional regions. The fix is full `WhenOp` rewrite/remove logic with control-flow safety checks.

### [ ] 523. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:832`
Mask-type generation uses a placeholder flip value (`false /* FIXME */`) in bundle element reconstruction, indicating direction/flip fidelity gap. What is missing is correct preservation/computation of element orientation in mask types. The fix is to propagate accurate flip info instead of placeholder.

### [ ] 524. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1698`
Field-ID traversal relies on ad hoc max-field queries and notes potential interface abstraction (`FieldIDTypeInterface`). What is missing is stronger type interface abstraction for field-ID accounting. The fix is to generalize through a dedicated interface and simplify this logic.

### [ ] 525. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1863`
Const-preserving element-type logic lacks a dedicated const trait/interface, causing repetitive type-switch handling. What is missing is reusable `ConstTypeInterface` abstraction. The fix is introducing constness trait/interface to unify const propagation operations.

### [ ] 526. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2137`
Same const-interface gap as entry 525 for open vectors. What is missing is common const propagation API across FIRRTL container types. The fix is shared trait/interface adoption in both bundle and vector implementations.

### [ ] 527. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2369`
Variant validation TODO indicates reference-containing cases are not excluded yet. What is missing is explicit rejection/handling policy for reference-containing variants. The fix is add verifier checks for these invalid compositions.

### [ ] 528. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:66`
Port-direction bitwidth handling still carries legacy workaround assumptions around zero-width APInt support. What is missing is clean zero-port representation without special-casing historical limitations. The fix is to normalize empty-port encoding now that infrastructure supports 0-bit widths.

### [ ] 529. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:86`
Verifier allows empty port-annotation arrays though comment questions semantics, leaving policy ambiguity. What is missing is explicit rule for empty annotations. The fix is to decide policy and enforce it consistently.

### [ ] 530. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:89`
Port annotation structure checks are in op interface verifier but marked for dedicated annotation verifier. What is missing is centralized annotation validation ownership. The fix is to move these checks into a specialized annotation verifier pass/util.

### [ ] 531. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:207`
`toRefType` repeats checks that `RefType::get/verify` already performs, creating duplicated validation logic. What is missing is a single-source validation path for ref-type construction. The fix is to push checks into type construction and simplify callers.

### [ ] 532. `lib/Dialect/FIRRTL/Transforms/AddSeqMemPorts.cpp:463`
`anyInstanceInEffectiveDesign` here is normal analysis filtering logic, not a TODO/debt marker. This is a scanner false positive. The fix is to avoid matching ordinary API identifiers as gaps.

### [ ] 533. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:273`
Metadata emission currently has an acknowledged incongruity for paths outside design scope. What is missing is consistent policy to suppress metadata for non-design paths. The fix is to skip metadata emission for out-of-design paths instead of synthesizing placeholders.

### [ ] 534. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:275`
This line is regular query logic (`anyInstanceInEffectiveDesign`) and not itself unresolved work. The actionable issue is the TODO at line 273. The fix is scanner narrowing.

### [ ] 535. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:321`
Using an unresolvable distinct placeholder for out-of-design paths is flagged as questionable, indicating metadata quality debt. What is missing is a principled representation for absent/unresolvable paths. The fix is to omit or explicitly tag such cases with structured semantics.

### [ ] 536. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:624`
This is routine design-scope filtering and not an implementation gap by itself. It is a scanner false positive from identifier matching. The fix is marker-based scan filtering.

### [ ] 537. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:823`
Same as entry 536: ordinary `anyInstanceInEffectiveDesign` usage, non-actionable. The fix is to suppress this class of API-name hits in audits.

### [ ] 538. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:909`
Again normal analysis query logic, not TODO debt. Non-actionable scanner noise. The fix is scanner refinement.

### [ ] 539. `include/circt/Dialect/Kanagawa/KanagawaInterfaces.td:97`
Nested symbol-table lookup currently requires brute-force scanning due missing nested symbol-table support. What is missing is proper nested symbol table infrastructure and lookup APIs. The fix is to add nested symbol-table support and replace brute-force search.

### [ ] 540. `lib/Dialect/FIRRTL/Transforms/RemoveUnusedPorts.cpp:85`
Inout ports are skipped with TODO, leaving removal logic incomplete for this port class. What is missing is safe inout-aware unused-port elimination semantics. The fix is to implement inout handling with direction-sensitive use checks.

### [ ] 541. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:420`
Inlining replacement logic has unresolved behavior for non-local/unmapped users and currently asserts. What is missing is robust fallback handling for these user classes. The fix is to add explicit diagnostics/handling instead of assert-only behavior.

### [ ] 542. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:739`
There is no clean way to annotate explicit parent scope on instances during inlining. What is missing is first-class parent-scope annotation support. The fix is to add scope annotation mechanism rather than debug-note workaround.

### [ ] 543. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1227`
Symbol rename mapping is not fully propagated across successive `inlineInto` calls, risking stale rename state. What is missing is cross-iteration rename bookkeeping. The fix is to update rename maps after each inlining step.

### [ ] 544. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1316`
Same rename-propagation gap as entry 543 in another path. What is missing is shared rename update logic. The fix is to centralize rename tracking to cover both sites.

### [ ] 545. `lib/Dialect/Pipeline/PipelineOps.cpp:719`
Runoff-stage detection recomputes a property that could be cached/precomputed. What is missing is precomputed stage-kind metadata. The fix is to compute once and reuse.

### [ ] 546. `lib/Dialect/Pipeline/PipelineOps.cpp:980`
Clock-gate storage layout is acknowledged as inefficient and expensive to process. What is missing is better data representation (potentially via properties) for per-register gating info. The fix is to redesign storage to reduce runtime compute.

### [ ] 547. `lib/Dialect/Calyx/Transforms/CompileControl.cpp:34`
Bit-width calculation helper notes possible better built-in support, indicating minor utility-level technical debt. What is missing is canonical utility op/helper for this calculation. The fix is to replace local helper with standardized built-in when available.

### [ ] 548. `lib/Dialect/Calyx/Transforms/CompileControl.cpp:116`
GroupDoneOp guard/source canonicalization is deferred, leaving redundant/non-normalized forms after compilation. What is missing is canonicalization pass or rewrite for these fields. The fix is to normalize guard/source during or after compile-control.

### [ ] 549. `lib/Dialect/FIRRTL/Transforms/MergeConnections.cpp:52`
Constant-detection logic omits several conversion ops (`unrealized_conversion`, `asUInt`, `asSInt`), reducing merge opportunities. What is missing is support for these wrappers in const recognition. The fix is to peel these ops in the matcher.

### [ ] 550. `lib/Dialect/FIRRTL/Transforms/InferReadWrite.cpp:57`
Pass depends on FIRRTL subset constraints because `WhenOp` remains in the same dialect, weakening pass preconditions. What is missing is clearer dialect partitioning or stronger pass input guarantees. The fix is either CHIRRTL move for `WhenOp` or explicit pre-pass normalization.

### [ ] 551. `lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:43`
`anyInstanceUnderLayer` here is ordinary analysis use, not unresolved work. This is scanner false positive noise. The fix is to ignore routine API calls in debt scans.

### [ ] 552. `lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:73`
Same as entry 551: standard analysis query in control flow, non-actionable. The fix is scan precision improvements.

### [ ] 553. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:248`
This pass carries explicit tech debt and deprecation intent (“remove this pass”), indicating architectural cleanup is pending. What is missing is replacement flow that makes pass obsolete. The fix is to migrate responsibilities and retire the pass.

### [ ] 554. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:376`
Defname-based matching is described as a hack, showing fragile extraction criteria. What is missing is robust module classification mechanism beyond name matching. The fix is structured annotations or semantic traits for extraction candidates.

### [ ] 555. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:384`
This is normal `anyInstanceInDesign` filtering, not TODO debt by itself. It is a scanner false positive. The fix is to exclude plain analysis API uses from audit results.

### [ ] 556. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:391`
Extraction prefix is hardcoded (`clock_gate`), reducing configurability and reuse. What is missing is configurable/policy-driven prefix naming. The fix is to parameterize prefix selection.

### [ ] 557. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:416`
Again routine design-scope filtering with `anyInstanceInDesign`, not unresolved implementation. This is scanner noise. The fix is marker-aware scanning.

### [ ] 558. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:423`
Memory extraction prefix is also hardcoded (`mem_wiring`), mirroring entry 556. What is missing is configurable naming policy. The fix is to externalize prefix configuration.

### [ ] 559. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:544`
This is a standard filter predicate using `anyInstanceInDesign`, not a TODO marker. It is non-actionable in gap tracking. The fix is scanner filtering.

### [ ] 560. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:1130`
Same false-positive class as entries 555/557/559: ordinary path-pruning logic using analysis APIs. No implementation gap is implied by this line. The fix is to restrict scans to explicit TODO/FIXME/TBD markers.

### [ ] 561. `lib/Dialect/FIRRTL/Transforms/MemToRegOfVec.cpp:50`
This is normal use of `anyInstanceInEffectiveDesign` for DUT scoping, not unresolved implementation debt. It is a scanner false positive. The fix is to avoid classifying ordinary analysis API calls as gaps.

### [ ] 562. `lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:272`
Dead-code elimination still skips `attach`-class constructs, leaving partial coverage. What is missing is correct liveness/deletion handling for attach and related ops. The fix is to add attach-aware side-effect and use analysis in DCE.

### [ ] 563. `lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:396`
Module list is copied to avoid iterator invalidation during mutation, indicating unresolved structural mutation issue (tracked by issue 3387). What is missing is mutation-safe traversal without extra copy workaround. The fix is to rework traversal/update strategy to avoid invalidation hazards.

### [ ] 564. `lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:1043`
Width inference may be doing redundant second-pass work due conservative cycle/`MinExpr` handling. What is missing is tighter satisfiability-to-solution detection in constraint solving. The fix is improved `MinExpr` cycle reasoning so unnecessary passes are skipped.

### [ ] 565. `lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:2217`
Unknown widths inside zero-length vectors are not recursively forced to zero. What is missing is recursive width update for zero-length container element types. The fix is to recurse into element types even when vector length is zero.

### [ ] 566. `lib/Dialect/FIRRTL/Transforms/CheckCombLoops.cpp:336`
Combinational-loop checking does not handle external modules, leaving coverage holes for cross-module loop analysis. What is missing is external module treatment strategy (modeling or conservative assumptions). The fix is to include extmodule interface behavior in loop detection.

### [ ] 567. `test/Tools/circt-sim/tlul-bfm-a-ready-timeout-short-circuit.sv:2`
This TODO marks a real runtime behavior gap: expected BFM timeout `$display` messages are not emitted in looped task context. What is missing is correct task-level display behavior in that control-flow path. The fix is to repair display scheduling/execution in looped tasks and keep this regression.

### [ ] 568. `lib/Dialect/FIRRTL/Transforms/LowerOpenAggs.cpp:597`
Port add/remove currently uses intermediate cloning plus extra array-attribute handling, indicating inefficient/awkward transformation flow. What is missing is direct add-and-erase port update support. The fix is to implement direct port mutation path without intermediate clones.

### [ ] 569. `lib/Dialect/FIRRTL/Transforms/Lint.cpp:107`
This is ordinary design-scope filtering via `anyInstanceInDesign`, not a TODO marker. It is scanner false positive noise. The fix is marker-based scan narrowing.

### [ ] 570. `lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:497`
Port annotations are not lowered when replacing memory ops with instances. What is missing is annotation mapping from memory ports to generated instance ports. The fix is explicit annotation lowering/remap logic during memory lowering.

### [ ] 571. `lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:557`
This line is routine dedup scoping logic using `anyInstanceInEffectiveDesign`, not unresolved work by itself. It is non-actionable scanner output. The fix is scan heuristic improvement.

### [ ] 572. `lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:232`
Linking logic still carries known mismatch risk when definition/declaration defnames or parameter presence differ. What is missing is robust merge semantics for such mismatches. The fix is explicit reconciliation/diagnostics for defname/parameter differences.

### [ ] 573. `lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:263`
Circuit-attribute merging is incomplete beyond current annotation handling (e.g. `enable_layers`). What is missing is comprehensive merge policy for additional circuit-level attributes. The fix is to enumerate and merge all relevant attributes deterministically.

### [ ] 574. `lib/Dialect/FIRRTL/Transforms/InferResets.cpp:443`
Current inferred-reset port insertion logic is described as brittle and error-prone. What is missing is a robust strategy for inferred-reset port management across modules/instantiations. The fix is to simplify by always adding inferred-reset port with optional reuse optimization.

### [x] 575. `test/Tools/circt-sim/syscall-ungetc.sv:2`
Status update (2026-02-28): this gap is closed and stale in this workspace. `$ungetc` pushback semantics are correct (`A` is re-read after pushback), and the regression now also checks `$ungetc` return value (`65`) in addition to replayed character behavior.

### [ ] 576. `lib/Dialect/Calyx/Export/CalyxEmitter.cpp:974`
`convertToDouble` here is regular float emission code and not an unresolved TODO/FIXME marker. This is a scanner false positive. The fix is to ignore such generic conversion identifiers in debt scans.

### [ ] 577. `lib/Dialect/OM/Transforms/FreezePaths.cpp:170`
FreezePaths does not support instance choices (multi-target referenced module names), currently erroring out. What is missing is instance-choice handling in path freezing. The fix is to add support for multi-choice instance references.

### [ ] 578. `lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:109`
DPI lowering currently omits bundle/enum argument/result support. What is missing is lowering rules for these richer FIRRTL types. The fix is to add bundle/enum lowering and ABI mapping for DPI calls.

### [ ] 579. `lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:206`
DPI call declaration consistency checks are pass-local TODO and intended for verifier ownership once function op support is finalized. What is missing is permanent verifier-level enforcement of call signature consistency. The fix is to move checks into dedicated op verifier.

### [ ] 580. `lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:308`
Unused infer-direction memory ports are dropped mirroring SFC behavior, but annotation sensitivity is unresolved. What is missing is annotation-aware deletion policy for these ports. The fix is to preserve/drop based on annotations explicitly.

### [ ] 581. `lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:454`
Enable inference behavior differs from Scala FIRRTL compiler when address comes from module ports. What is missing is aligned enable inference semantics for this case. The fix is to implement/align address-port enable inference.

### [ ] 582. `include/circt/Conversion/Passes.td:178`
Conversion pass notes possible significant speedup via finer-grained nesting, but current module-scope mutation prevents easy per-module scheduling. What is missing is architecture allowing per-container granularity without violating top-level module constraints. The fix is pass refactor to isolate top-level mutations or stage them separately.

### [ ] 583. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:446`
Value cloning strategy is not recursive and will become insufficient once `FString` ops gain operands. What is missing is recursive clone support in lowering path. The fix is to implement operand-recursive cloning now.

### [ ] 584. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:513`
HierPath insertion currently occurs under lock inside parallel region, adding contention. What is missing is lock-minimized insertion scheduling. The fix is to move insert preparation before parallel region and reduce critical section scope.

### [ ] 585. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:561`
Domain mapping logic is more complex than needed because wires lack domain-kind info (issue 9398). What is missing is domain metadata on wires. The fix is to add wire domain-kind support and simplify this path.

### [ ] 586. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:912`
Pass still has to look through wires to infer domain because wire domain info is absent. What is missing is same wire domain metadata support as entry 585. The fix is to stop wire look-through once metadata lands.

### [ ] 587. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:1361`
HierPath rewrite computes new namepaths eagerly for all ops even when unchanged, wasting work. What is missing is lazy/on-demand namepath recomputation. The fix is to compute new paths only when a modification is required.

### [ ] 588. `lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:31`
Local utility specialization is marked for upstreaming, indicating duplicated infra kept locally. What is missing is upstream shared support for this utility. The fix is to upstream and consume canonical implementation.

### [ ] 589. `lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:790`
Analysis preservation is conservative; could preserve more analyses when no effective specialization changes occurred. What is missing is finer-grained analysis preservation conditions. The fix is to detect empty-change cases and mark specific analyses preserved.

### [ ] 590. `lib/Dialect/FIRRTL/Transforms/Dedup.cpp:307`
Dedup handling of `DistinctAttr` (including path use) remains incomplete and currently skipped (issue 6583). What is missing is correct distinct-attribute semantics during dedup/path transforms. The fix is full `DistinctAttr` support in dedup logic.

### [x] 591. `lib/Runtime/uvm-core/src/base/uvm_component.svh:2569`
Status update (2026-02-28): this gap is closed in this workspace. `uvm_component::suspend()` now uses tracked task-phase process handles (wired from `uvm_task_phase::execute`) and drives runtime `process::suspend()` instead of warning-only stub behavior. Interpreter wakeup flow now also respects explicit process suspension until `resume()`. Follow-up hardening keeps `m_phase_process` tracking run-phase-focused so short-lived non-run task phases do not clobber suspend/resume control under corrected phase-graph associative-array semantics.

### [x] 592. `lib/Runtime/uvm-core/src/base/uvm_component.svh:2577`
Status update (2026-02-28): this gap is closed in this workspace. `uvm_component::resume()` now resumes the tracked phase task process and clears explicit suspension state in the interpreter. Regression coverage was added in `test/Runtime/uvm/uvm_component_suspend_resume_test.sv` and passes with targeted process/UVM suites.

### [x] 593. `lib/Runtime/uvm-core/src/base/uvm_component.svh:3337`
Status update (2026-02-28): this entry is stale duplicate tracking. The active open gap for this TODO remains entry 361.

### [x] 594. `lib/Runtime/uvm-core/src/base/uvm_component.svh:3542`
Status update (2026-02-28): this entry is stale. `m_unsupported_set_local` is identifier-level naming in override plumbing, not an unresolved feature gap.

### [x] 595. `lib/Runtime/uvm-core/src/base/uvm_component.svh:3545`
Status update (2026-02-28): same closure as entry 594. This is routine override implementation code, not a TODO/debt marker.

### [x] 596. `lib/Runtime/uvm-core/src/base/uvm_component.svh:3547`
Status update (2026-02-28): same closure as entry 594. This assignment is ordinary implementation detail, not unresolved feature debt.

### [ ] 597. `lib/Tools/circt-bmc/LowerToBMC.cpp:3126`
`hasUnsupportedUse` tracking indicates this lowering path only handles a constrained signal-use set and skips others. What is missing is broader support for additional LLHD signal user patterns. The fix is to extend use handling (or emit structured diagnostics for skipped cases).

### [ ] 598. `lib/Tools/circt-bmc/LowerToBMC.cpp:3138`
Setting `hasUnsupportedUse = true` on unhandled users enforces conservative bail-out, which can reduce lowering coverage. What is missing is fine-grained support or explicit categorization of the unhandled user classes. The fix is to incrementally legalize common patterns and report residual unsupported classes.

### [ ] 599. `lib/Tools/circt-bmc/LowerToBMC.cpp:3142`
The combined `hasUnsupportedUse || hasEnable` skip gate is a real capability boundary for BMC lowering. What is missing is support for currently excluded enable/unsupported-use scenarios. The fix is to add lowering for enable-bearing cases and additional use patterns where sound.

### [ ] 600. `lib/Tools/circt-bmc/LowerToBMC.cpp:3312`
This repeats the known TODO to avoid LLVM dialect ops at this stage for `main` return value construction. What is missing is cleaner non-LLVM return constant construction in this pass. The fix is to keep generation at higher-level ops until later lowering.

### [ ] 601. `include/circt/Dialect/Calyx/CalyxPrimitives.td:543`
`AnyInteger` here is a normal ODS type constraint, not unresolved feature debt. This is a scanner false positive. The fix is to avoid matching generic type identifiers.

### [ ] 602. `test/Tools/circt-verilog-lsp-server/workspace-symbols.test:3`
`UNSUPPORTED: valgrind` is test-runner environment metadata, not a product feature gap for workspace symbols. The missing issue is valgrind-lane compatibility. The fix is periodic reevaluation of this exclusion.

### [ ] 603. `include/circt/Dialect/Calyx/CalyxPasses.td:27`
Pass documentation still marks unresolved behavior for multiple writes to one signal (read replacement by disjunction). What is missing is implemented semantics for this conflict case. The fix is to implement and verify multi-write read resolution.

### [ ] 604. `test/Tools/circt-verilog-lsp-server/workspace-symbol-project.test:3`
Another `UNSUPPORTED: valgrind` marker, non-actionable as core feature debt. This is environment/test-portability metadata. The fix is ongoing validation of valgrind exclusions.

### [ ] 605. `test/Tools/circt-verilog-lsp-server/uvm-diagnostics.test:3`
Same valgrind exclusion class as entry 604. Not direct implementation debt in UVM diagnostics features. The fix is scanner suppression for this metadata class.

### [ ] 606. `test/Tools/circt-verilog-lsp-server/types.test:3`
Valgrind unsupported test metadata, not product TODO. Non-actionable from gap perspective. The fix is to exclude this marker class from debt reports.

### [ ] 607. `test/Tools/circt-verilog-lsp-server/signature-help.test:3`
Same `UNSUPPORTED: valgrind` false-positive class. The missing capability is test-lane portability, not signature-help implementation. The fix is periodic reevaluation and suppressing this in audits.

### [ ] 608. `include/circt/Dialect/Calyx/CalyxLoweringUtils.h:204`
Use-def ordering invariant for generated scheduleables is not post-checked after insertion. What is missing is explicit invariant validation after lowering. The fix is to add post-insertion consistency checks and fail fast on violations.

### [ ] 609. `test/Tools/circt-verilog-lsp-server/rename-variables.test:3`
`UNSUPPORTED: valgrind` metadata again, not core rename-feature debt. This is scanner noise in test constraints. The fix is marker filtering for valgrind exclusions.

### [ ] 610. `lib/Tools/arcilator/pipelines.cpp:98`
Known pipeline cleanup TODO: possible merge of `RemoveUnusedArcArguments` with `SinkInputs` remains open. What is missing is a consolidated pass strategy. The fix is overlap analysis and pass consolidation.

### [ ] 611. `lib/Tools/arcilator/pipelines.cpp:111`
`LowerClocksToFuncsPass` still has unresolved nested-region/`scf.if` handling risks. What is missing is robust region-aware lowering. The fix is to extend pass logic for nested control-flow contexts.

### [ ] 612. `lib/Tools/arcilator/pipelines.cpp:114`
`InlineArcs` still cannot safely handle `scf.if` patterns in this path, preventing re-enabling the commented conversion. What is missing is resilient inline handling over nested regions. The fix is to harden inliner behavior and re-enable the downstream pass.

### [ ] 613. `test/Tools/circt-verilog-lsp-server/rename-edge-cases.test:3`
Another valgrind exclusion marker, non-actionable implementation-wise. The fix is scanner suppression for `UNSUPPORTED: valgrind` test metadata.

### [ ] 614. `include/circt/Dialect/Calyx/CalyxHelpers.h:49`
Control-leaf classification still lacks `Invoke` support (issue 1679). What is missing is invoke-aware leaf-node logic. The fix is to include `Invoke` handling and add control-graph tests.

### [ ] 615. `test/Tools/circt-verilog-lsp-server/references.test:3`
`UNSUPPORTED: valgrind` test metadata again, not a references-feature implementation gap. The fix is treat as environment exclusion only.

### [ ] 616. `test/Tools/circt-verilog-lsp-server/module-instantiation.test:3`
Same valgrind exclusion class as previous entries. Non-actionable for feature debt. The fix is audit filter refinement.

### [ ] 617. `test/Tools/circt-verilog-lsp-server/interface.test:3`
Valgrind unsupported metadata line, not direct LSP interface-feature gap. The fix is to exclude this marker from debt extraction.

### [ ] 618. `test/Tools/circt-verilog-lsp-server/interface.test:29`
This TODO denotes a real missing LSP capability: hover information for interface signals is not implemented. What is missing is interface-member hover resolution and content generation. The fix is to add hover symbol/type lookup for interface signals.

### [ ] 619. `test/Tools/circt-verilog-lsp-server/interface.test:39`
Another real LSP gap: go-to-definition for interface members is not implemented. What is missing is definition resolution through interface-member symbol tables/usages. The fix is to implement member definition lookup and wire it into LSP definition requests.

### [ ] 620. `test/Tools/circt-verilog-lsp-server/inlay-hints.test:3`
`UNSUPPORTED: valgrind` marker for inlay-hints test is environment metadata, not feature debt. The fix is to suppress this class from actionable scans and periodically revisit portability.

### [ ] 621. `include/circt/Dialect/Comb/CombOps.h:97`
Power-of-two rewrite utilities still do not cover signed division/modulo forms. What is missing is signed div/mod canonicalization support with correct sign/rounding semantics. The fix is to implement signed variants and add exhaustive signed-edge tests.

### [ ] 622. `test/Tools/circt-verilog-lsp-server/goto-definition.test:3`
`UNSUPPORTED: valgrind` is test-environment metadata, not a goto-definition feature gap. This is non-actionable from implementation perspective. The fix is to filter valgrind exclusions from debt scans.

### [ ] 623. `include/circt/Dialect/Arc/ArcOps.td:333`
`AnyInteger` argument typing is normal ODS declaration, not unresolved work. This is a scanner false positive. The fix is marker-only scan matching.

### [ ] 624. `include/circt/Dialect/Arc/ArcOps.td:335`
Same as entry 623: standard ODS result type declaration, non-actionable for gap tracking. The fix is scanner refinement.

### [ ] 625. `include/circt/Dialect/Arc/ArcOps.td:411`
Again an `AnyInteger` constraint in op schema, not a TODO/unsupported marker. This is false-positive audit noise. The fix is to ignore generic type-constraint identifiers.

### [ ] 626. `include/circt/Dialect/Arc/ArcOps.td:413`
Same false-positive class as entry 625: normal result type constraint declaration. The fix is scanner filtering.

### [ ] 627. `include/circt/Dialect/Arc/ArcOps.td:427`
`AnyInteger` in arguments is routine schema text, not unresolved implementation debt. Non-actionable. The fix is explicit marker-context scanning only.

### [ ] 628. `include/circt/Dialect/Arc/ArcOps.td:429`
Same as entry 627: normal ODS type usage, scanner false positive. The fix is no product code change; improve audit heuristics.

### [ ] 629. `lib/Target/ExportSystemC/Patterns/SystemCEmissionPatterns.cpp:140`
Destructor emission hardcodes `override`, which is only safe under current class assumptions (`sc_module` inheritance). What is missing is modeled override semantics tied to class metadata. The fix is to emit `override` based on explicit operation/class attributes.

### [ ] 630. `lib/Target/ExportSystemC/Patterns/SystemCEmissionPatterns.cpp:311`
SystemC emission still errors on unsupported member-access kinds. What is missing is lowering/emission support for additional access forms. The fix is to extend member-access kind handling and provide diagnostics with fallback guidance.

### [ ] 631. `test/Tools/circt-verilog-lsp-server/find-references.test:3`
Valgrind exclusion metadata (`UNSUPPORTED: valgrind`), not a references-feature gap. Non-actionable debt scan hit. The fix is to suppress this metadata class.

### [ ] 632. `test/Tools/circt-verilog-lsp-server/find-references-comprehensive.test:3`
Same valgrind metadata false positive as entry 631. The fix is scanner filtering for test-runner exclusions.

### [ ] 633. `test/Tools/circt-verilog-lsp-server/document-symbols.test:3`
`UNSUPPORTED: valgrind` line is environment constraint metadata, not implementation debt. The fix is to exclude these lines from actionable reports.

### [ ] 634. `test/Tools/circt-verilog-lsp-server/document-highlight.test:3`
Same valgrind unsupported marker class, non-actionable for feature completeness. The fix is audit suppression of these tags.

### [ ] 635. `test/Tools/circt-verilog-lsp-server/document-links.test:3`
Valgrind exclusion metadata again, not direct document-links gap. The fix is scanner refinement.

### [ ] 636. `lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:72`
Template-argument calls are currently excluded from inlinable expression emission. What is missing is template-aware call emission support in EmitC/SystemC pattern matching. The fix is to carry template args through matching and emission.

### [ ] 637. `lib/Target/ExportSystemC/Patterns/EmitCEmissionPatterns.cpp:84`
Statement matching has the same template-argument limitation as entry 636. What is missing is consistent template support in statement emission path. The fix is shared matcher/emitter extension with tests.

### [ ] 638. `test/Tools/circt-verilog-lsp-server/find-package-import-def.test:4`
`UNSUPPORTED: valgrind` in this test is explicit environment exclusion due Slang valgrind issues, not product debt. Non-actionable. The fix is to keep this out of gap audits.

### [ ] 639. `test/Tools/circt-mut-forward-generate-cache-fallback.test:4`
This is regression harness input that intentionally passes `--native-unsupported` to verify argument forwarding/fallback behavior. It does not indicate unresolved implementation debt. The fix is scanner suppression for literal test command lines.

### [ ] 640. `test/Tools/circt-mut-forward-generate-cache-fallback.test:8`
Expected-output check for forwarded `--native-unsupported` argument, also non-actionable from feature-gap perspective. This is test assertion text, not TODO debt. The fix is to ignore FileCheck expectation lines.

### [ ] 641. `test/Tools/circt-verilog-lsp-server/diagnostic.test:4`
Valgrind exclusion metadata line, not a diagnostics-feature implementation gap. The fix is to filter these environment tags from debt scans.

### [ ] 642. `test/Tools/circt-verilog-lsp-server/find-definition.test:3`
`UNSUPPORTED: valgrind` tag, same non-actionable class. The fix is scanner suppression.

### [ ] 643. `test/Tools/circt-verilog-lsp-server/debounce.test:4`
Another valgrind metadata exclusion line. Not direct debounce-feature debt. The fix is to exclude this marker class from gap reports.

### [ ] 644. `test/Tools/circt-verilog-lsp-server/code-lens.test:3`
Same valgrind unsupported test metadata as previous LSP tests. Non-actionable for product gaps. The fix is audit filtering.

### [ ] 645. `test/Tools/circt-verilog-lsp-server/command-files.test:4`
Valgrind exclusion metadata line, not command-file functionality debt. The fix is to suppress these from TODO/unsupported gap scans.

### [ ] 646. `lib/Target/ExportSystemC/ExportSystemC.cpp:39`
Header guard sanitization still does not correct invalid leading-digit cases. What is missing is complete C/C++ identifier normalization for guard names. The fix is to prefix/normalize when first character is not valid.

### [x] 647. `tools/circt-sim/LLHDProcessInterpreter.cpp:26757`
Fixed for the dynamic-selection case: `evaluateFormatString` now handles `comb.mux` over `!sim.fstring` (matching existing `arith.select` support), which removed a real `<unsupported format>` runtime path. Covered by `test/Tools/circt-sim/fmt-mux-dynamic.mlir`. Broader `sim.fmt.*` fallback coverage remains tracked by other open entries.

### [x] 648. `tools/circt-sim/LLHDProcessInterpreter.cpp:27568`
Status update (2026-02-28): this entry is stale. `maybeTraceJoinAnyImmediate` is regular tracing logic and not an implementation gap; this was a scan false positive from nearby text.

### [x] 649. `tools/circt-sim/LLHDProcessInterpreter.cpp:40655`
Status update (2026-02-28): this entry is stale. The referenced comment is defensive behavior documentation (zero-init for failed paths), not unresolved product debt.

### [ ] 650. `tools/circt-sim/LLHDProcessInterpreter.cpp:40717`
This region contains warning plumbing for unsupported trampoline native ABI fallback, reflecting an explicit capability boundary in native trampoline dispatch. What is missing is broader trampoline ABI support beyond current scalar limits. The fix is to extend ABI marshaling support (or richer fallback path) and keep diagnostics for residual unsupported signatures.

### [x] 651. `tools/circt-sim/LLHDProcessInterpreter.cpp:40718`
Status update (2026-02-28): this entry is stale. The cited line is trampoline map initialization plumbing; the real open item is broader unsupported ABI coverage (tracked by entries 650/652/653), not this line.

### [ ] 652. `tools/circt-sim/LLHDProcessInterpreter.cpp:40720`
This corresponds to the warning path for unsupported trampoline native fallback ABI. What is missing is broader native trampoline ABI marshalling beyond current scalar constraints. The fix is to extend argument/result type support or add richer fallback dispatch.

### [ ] 653. `tools/circt-sim/LLHDProcessInterpreter.cpp:40724`
This is part of one-time warning gating plumbing for unsupported trampoline ABI cases. It reflects the same capability boundary as entry 652 rather than a separate issue. The fix is shared ABI-support expansion.

### [ ] 654. `test/Tools/circt-verilog-lsp-server/code-actions.test:3`
`UNSUPPORTED: valgrind` test metadata, not a code-actions feature gap. Non-actionable scanner hit. The fix is to suppress valgrind exclusions in debt reports.

### [ ] 655. `test/Tools/circt-verilog-lsp-server/class-hover.test:3`
Same valgrind exclusion marker class as entry 654. It reflects environment constraints, not missing class-hover implementation. The fix is audit filtering.

### [ ] 656. `test/Tools/circt-verilog-lsp-server/rename-comprehensive.test:3`
Valgrind unsupported metadata line, not product debt for rename functionality. The fix is scanner suppression for this metadata pattern.

### [ ] 657. `test/Tools/circt-verilog-lsp-server/class-definition.test:3`
Another `UNSUPPORTED: valgrind` tag in tests, non-actionable for feature tracking. The fix is to exclude it from gap scans.

### [ ] 658. `lib/Target/ExportSystemC/EmissionPrinter.cpp:32`
Emission printer still falls back to explicit `<<UNSUPPORTED OPERATION>>` placeholders when no pattern exists. What is missing is full op emission coverage (or safer structural fallback lowering). The fix is to add patterns for remaining op kinds and keep hard failures actionable.

### [ ] 659. `lib/Target/ExportSystemC/EmissionPrinter.cpp:50`
Type emission has analogous unsupported placeholder fallback. What is missing is broader type emission pattern support. The fix is to implement missing type emitters and reduce fallback cases.

### [ ] 660. `lib/Target/ExportSystemC/EmissionPrinter.cpp:66`
Attribute emission also relies on unsupported placeholder output when no handler is present. What is missing is complete attribute emission coverage. The fix is to add attribute emitters and diagnostics that point to remediation.

### [ ] 661. `test/Tools/circt-verilog-lsp-server/procedural.test:3`
`UNSUPPORTED: valgrind` test metadata, non-actionable. The fix is scan filtering.

### [ ] 662. `test/Tools/circt-verilog-lsp-server/workspace-symbol.test:3`
Valgrind exclusion metadata line, not workspace-symbol feature debt. The fix is to exclude this class from audits.

### [ ] 663. `test/Tools/circt-verilog-lsp-server/call-hierarchy.test:3`
Same valgrind exclusion marker class, non-actionable. The fix is audit suppression for environment tags.

### [ ] 664. `test/Tools/circt-verilog-lsp-server/workspace-symbol-fuzzy.test:3`
`UNSUPPORTED: valgrind` metadata again, not direct fuzzy-symbol implementation debt. The fix is scanner refinement.

### [ ] 665. `test/Tools/circt-verilog-lsp-server/member-completion.test:3`
Valgrind unsupported test marker, non-actionable from feature-gap perspective. The fix is to filter these test constraints.

### [ ] 666. `test/Tools/circt-verilog-lsp-server/completion.test:3`
Same valgrind metadata false-positive class as entry 665. The fix is suppression in debt scans.

### [ ] 667. `test/Tools/circt-verilog-lsp-server/uvm-completion.test:3`
`UNSUPPORTED: valgrind` line indicates test-lane limitation, not UVM completion feature debt. The fix is to treat as environmental metadata only.

### [ ] 668. `test/Tools/circt-verilog-lsp-server/type-hierarchy.test:3`
Valgrind exclusion metadata, non-actionable implementation-wise. The fix is scanner filtering.

### [ ] 669. `test/Tools/circt-verilog-lsp-server/semantic-tokens.test:3`
Another valgrind unsupported tag in test headers, not semantic-token implementation debt. The fix is to suppress this marker class.

### [ ] 670. `test/Tools/circt-verilog-lsp-server/semantic-tokens-comprehensive.test:3`
Same as entry 669: environment exclusion metadata. Non-actionable in gap list. The fix is scan heuristic improvement.

### [ ] 671. `test/Tools/circt-verilog-lsp-server/rename.test:3`
Valgrind exclusion marker, not rename-feature gap. The fix is to exclude such lines from debt extraction.

### [ ] 672. `test/Tools/circt-verilog-lsp-server/inheritance-completion.test:3`
`UNSUPPORTED: valgrind` metadata line, not inheritance-completion implementation debt. The fix is audit filtering.

### [ ] 673. `test/Tools/circt-verilog-lsp-server/rename-refactoring.test:3`
Same valgrind metadata false positive class as neighboring LSP tests. The fix is scanner suppression.

### [x] 674. `include/circt/Conversion/ImportVerilog.h:231`
Status update (2026-02-28): this entry is closed as non-actionable policy/config documentation. The option describes importer behavior for unsupported SVA recovery mode; underlying SVA feature debt is tracked at implementation-site entries (for example in `AssertionExpr.cpp`).

### [x] 675. `include/circt/Conversion/ImportVerilog.h:233`
Status update (2026-02-28): same closure as entry 674. This default-value line is strict-mode policy, not an unresolved implementation TODO.

### [ ] 676. `test/Tools/circt-verilog-lsp-server/include.test:5`
Valgrind exclusion metadata (with note about Slang internals), not a direct include-feature gap. The fix is to treat as test-environment annotation, not project debt.

### [ ] 677. `test/Tools/circt-verilog-lsp-server/hover.test:3`
`UNSUPPORTED: valgrind` test metadata line, non-actionable. The fix is scanner filtering.

### [ ] 678. `test/Tools/run-sv-tests-bmc-smtlib-no-fallback.test:2`
This is test fixture content describing an unsupported-SMTLIB scenario used to verify no native fallback retry. It is expected harness text, not unresolved implementation debt. The fix is to suppress embedded fixture strings in scans.

### [ ] 679. `test/Tools/circt-verilog-lsp-server/formatting.test:3`
Valgrind exclusion metadata line, not document-formatting feature debt. The fix is to filter this marker class from actionable reports.

### [ ] 680. `include/circt/Conversion/Passes.td:178`
This repeats the known conversion-pass performance TODO around finer-grained nesting/per-module execution. What is missing is architecture allowing granular scheduling without top-level module mutation conflicts. The fix is pass refactor to separate top-level mutation from per-module work.

### [ ] 681. `lib/Support/TruthTable.cpp:217`
Truth-table canonicalization still uses factorial/exponential search, which does not scale. What is missing is a more efficient canonical/semi-canonical algorithm. The fix is to replace brute-force enumeration with a scalable approach.

### [ ] 682. `lib/Support/JSON.cpp:64`
`convertToDouble()` here is ordinary float serialization logic, not TODO/unsupported debt. This is a scanner false positive. The fix is to ignore generic conversion identifiers in gap scans.

### [ ] 683. `lib/Support/CoverageDatabase.cpp:617`
`Unsupported database version` is a strict compatibility diagnostic, not necessarily unfinished functionality. The missing capability is explicit migration/backward-compat handling if multi-version support is desired. The fix is either keep strict rejection with migration tooling or implement version-upgrade readers.

### [ ] 684. `test/Tools/circt-verilog-lsp-server/e2e/lit.local.cfg:2`
This is test harness configuration marking pytest-only e2e directory as unsupported for lit. It is intentional infra behavior, not product debt. The fix is scanner suppression for lit configuration metadata.

### [ ] 685. `test/Tools/circt-verilog-lsp-server/e2e/lit.local.cfg:5`
`config.unsupported = True` in local lit config is expected test-runner wiring, not unresolved feature work. Non-actionable from gap perspective. The fix is to filter test-config markers.

### [ ] 686. `test/Tools/run-sv-tests-circt-lec-drop-remarks.test:4`
This is fixture script text deliberately emitting an “unsupported construct dropped” warning for regression testing. It is not unresolved implementation debt. The fix is to suppress embedded fixture payload lines in scans.

### [ ] 687. `test/Tools/run-sv-tests-circt-lec-drop-remarks.test:16`
This is expected `REASON` check text in a drop-remarks regression test, not a TODO gap. Non-actionable scanner hit. The fix is to ignore FileCheck expectation lines.

### [ ] 688. `test/Tools/run-yosys-sva-circt-lec-drop-remarks.test:3`
Fixture command intentionally emits unsupported-drop warning for policy testing. This is test scaffolding, not a product gap. The fix is scanner filtering for `RUN:` fixture scripts.

### [ ] 689. `test/Tools/run-yosys-sva-circt-lec-drop-remarks.test:21`
Expected `REASON` output assertion in test, non-actionable as implementation debt. The fix is to treat expected-output lines as test metadata.

### [ ] 690. `test/Tools/run-yosys-sva-bmc-drop-remarks.test:3`
Same fixture-warning pattern as entries 686/688: intentional test setup text. Not unresolved work. The fix is scan suppression for fixture payload.

### [ ] 691. `test/Tools/run-yosys-sva-bmc-drop-remarks.test:28`
Expected reason/check line in test output, not product debt. The fix is to exclude expectation assertions from debt extraction.

### [ ] 692. `test/Tools/run-verilator-verification-circt-lec-drop-remarks.test:4`
Fixture command intentionally generates unsupported-drop remark for regression behavior checks. Non-actionable for implementation backlog. The fix is scanner filtering.

### [ ] 693. `test/Tools/run-verilator-verification-circt-lec-drop-remarks.test:16`
Expected `REASON` check content, not unresolved feature gap. This is test assertion text. The fix is to suppress FileCheck lines.

### [ ] 694. `test/Tools/run-formal-all-strict-gate-bmc-drop-remark-cases-verilator.test:4`
Embedded script fixture emitting unsupported-drop warning for strict-gate policy testing. Not implementation debt. The fix is test-fixture suppression in scanning.

### [ ] 695. `test/Tools/run-formal-all-strict-gate-bmc-drop-remark-cases-verilator.test:8`
Same fixture pattern as entry 694, including duplicate warning output for gating behavior. Non-actionable gap signal. The fix is scanner refinement.

### [ ] 696. `test/Tools/circt-verilog-lsp-server/textdocument-didclose.test:4`
`UNSUPPORTED: valgrind` metadata in LSP test, not didClose feature debt. The fix is to exclude valgrind test exclusions from actionable reports.

### [ ] 697. `test/Tools/circt-verilog-lsp-server/textdocument-didchange.test:4`
Same valgrind exclusion marker class, non-actionable. The fix is scanner suppression for environment constraints.

### [ ] 698. `test/Tools/circt-verilog-lsp-server/package-indexing.test:3`
Valgrind unsupported test metadata again, not package-indexing implementation debt. The fix is scan filtering.

### [ ] 699. `test/Tools/run-regression-unified-manifest-profile-internal-space.test:5`
This line is expected diagnostic output in a negative parser test for invalid profile tokens. It is intentional regression text, not unresolved work. The fix is to ignore `CHECK:` expectations in gap scans.

### [ ] 700. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:3`
This is a large fixture script constructing synthetic JIT deopt reasons (including `unsupported_operation`) for policy-gate testing. It is test harness content, not implementation debt. The fix is scanner suppression for embedded script fixtures.

### [ ] 701. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:5`
Allowlist fixture entry using `unsupported_operation` is expected test input, non-actionable as backlog item. The fix is to filter fixture data lines.

### [ ] 702. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:6`
`RUN` line exercising policy gate with fail-on-reason flag is regression orchestration, not unresolved feature. The fix is scanner exclusion for test command lines.

### [ ] 703. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:9`
Negative test invocation (`not env ... fail-on-reason unsupported_operation`) is expected behavior validation text, not project debt. The fix is to suppress command-lines in scans.

### [ ] 704. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:20`
Expected `REASON` table line in test output, non-actionable. The fix is to ignore check-pattern lines.

### [ ] 705. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:23`
Expected failure-log check line, not unresolved functionality. The fix is scanner filtering of `FAILLOG` assertions.

### [ ] 706. `test/Tools/run-avip-circt-sim-jit-policy-gate.test:24`
Same as entry 705: expected `FAILLOG` assertion text in regression, non-actionable. The fix is marker-context refinement.

### [ ] 707. `test/Tools/summarize-circt-sim-jit-reports-policy.test:3`
Fixture command builds JSON with `unsupported_operation` reasons for summarizer policy tests. This is intentional test data, not unresolved gap. The fix is to suppress embedded fixture-generation code in scans.

### [ ] 708. `test/Tools/summarize-circt-sim-jit-reports-policy.test:4`
Second fixture JSON generation line with unsupported reason detail, also non-actionable test scaffolding. The fix is scanner filtering for RUN payload lines.

### [ ] 709. `test/Tools/summarize-circt-sim-jit-reports-policy.test:5`
Allowlist fixture content for policy test, not implementation debt. The fix is to ignore fixture input lines in gap reports.

### [ ] 710. `test/Tools/summarize-circt-sim-jit-reports-policy.test:6`
Additional allowlist fixture setup line in same policy regression, non-actionable. The fix is scan heuristic narrowing to actual TODO/FIXME/unsupported-diagnostic code paths.

### [ ] 711. `test/Tools/summarize-circt-sim-jit-reports-policy.test:8`
This is a negative test command line validating `--fail-on-reason unsupported_operation` policy behavior. It is fixture orchestration, not unresolved implementation debt. The fix is to filter `RUN:` command lines from gap scans.

### [ ] 712. `test/Tools/summarize-circt-sim-jit-reports-policy.test:9`
Same as entry 711: regression command for reason-detail policy gating, intentionally using `unsupported_operation`. Non-actionable as backlog item. The fix is scanner suppression for fixture commands.

### [ ] 713. `test/Tools/summarize-circt-sim-jit-reports-policy.test:10`
This is a positive-run command in the same policy regression, not a TODO gap. It is expected test setup text. The fix is to exclude command fixture content from debt extraction.

### [ ] 714. `test/Tools/summarize-circt-sim-jit-reports-policy.test:16`
Expected `FAIL_REASON` output assertion in test, non-actionable implementation-wise. The fix is to ignore FileCheck expectation lines.

### [ ] 715. `test/Tools/summarize-circt-sim-jit-reports-policy.test:17`
Same expected output assertion class as entry 714. This is test validation text, not unresolved feature work. The fix is scanner filtering.

### [ ] 716. `test/Tools/summarize-circt-sim-jit-reports-policy.test:19`
Expected `FAIL_DETAIL` check line in a regression test, non-actionable as debt marker. The fix is to suppress check-pattern lines.

### [ ] 717. `test/Tools/summarize-circt-sim-jit-reports-policy.test:20`
Same as entry 716: expected failure-detail assertion text. Not a product gap. The fix is scanner refinement.

### [ ] 718. `test/Tools/summarize-circt-sim-jit-reports.test:3`
Fixture JSON generation for summarizer tests using unsupported-operation reasons is intentional test data. It does not represent unresolved functionality. The fix is to exclude embedded fixture generation code from scans.

### [ ] 719. `test/Tools/summarize-circt-sim-jit-reports.test:4`
Second fixture JSON generation line in the same test, likewise non-actionable scaffolding. The fix is scanner suppression for `RUN` payloads.

### [ ] 720. `test/Tools/summarize-circt-sim-jit-reports.test:16`
Expected `LOG` output assertion showing top reasons, not implementation debt. This is regression verification text. The fix is to skip expected-output lines in audits.

### [ ] 721. `test/Tools/summarize-circt-sim-jit-reports.test:19`
Expected detailed log assertion in test output; non-actionable. The fix is marker-context filtering.

### [ ] 722. `test/Tools/summarize-circt-sim-jit-reports.test:20`
Same as entry 721: expected `LOG` check content in regression. Not a missing feature indicator.

### [ ] 723. `test/Tools/summarize-circt-sim-jit-reports.test:23`
Expected reason-table assertion (`REASON`) line in test output. Non-actionable from gap standpoint. The fix is to exclude FileCheck lines.

### [ ] 724. `test/Tools/summarize-circt-sim-jit-reports.test:28`
Expected detail-table assertion (`DETAIL`) line, also non-actionable test expectation text. The fix is scanner suppression.

### [ ] 725. `test/Tools/summarize-circt-sim-jit-reports.test:29`
Another expected detail assertion line; this is test output checking, not debt. The fix is to ignore this category in scans.

### [ ] 726. `test/Tools/summarize-circt-sim-jit-reports.test:32`
Expected process-table assertion (`PROC`) in test output, non-actionable as implementation gap. The fix is check-line filtering.

### [ ] 727. `test/Tools/summarize-circt-sim-jit-reports.test:34`
Same expected process-table assertion class as entry 726. This is regression text, not backlog work. The fix is scanner refinement.

### [ ] 728. `test/Tools/run-regression-unified-manifest-profile-invalid.test:7`
Expected error message assertion for invalid profile token handling. This is deliberate negative-test output and not unresolved debt. The fix is to suppress `CHECK:` expectation lines.

### [ ] 729. `test/Tools/run-verilator-verification-circt-bmc-drop-remarks.test:4`
Fixture command intentionally emits an unsupported-drop warning for drop-remark policy testing. It is test scaffolding, not missing implementation. The fix is to filter fixture script lines.

### [ ] 730. `lib/Support/PrettyPrinter.cpp:43`
This `(TODO)` remains generic documentation-level note without actionable implementation detail. It is low-specificity design debt rather than concrete feature gap. The fix is to replace with specific action items or exclude generic prose TODOs from audits.

### [ ] 731. `include/circt/Analysis/FIRRTLInstanceInfo.h:145`
`anyInstanceUnderDut` is a normal API declaration, not TODO/unsupported marker. This is scanner false positive noise. The fix is to avoid matching routine identifiers.

### [ ] 732. `include/circt/Analysis/FIRRTLInstanceInfo.h:155`
Same as entry 731: ordinary API declaration (`anyInstanceUnderEffectiveDut`), non-actionable. The fix is marker-based scanning only.

### [ ] 733. `include/circt/Analysis/FIRRTLInstanceInfo.h:164`
`anyInstanceUnderLayer` declaration is normal interface surface, not unresolved work. Non-actionable false positive.

### [ ] 734. `include/circt/Analysis/FIRRTLInstanceInfo.h:172`
`anyInstanceInDesign` declaration is standard API text, not a debt marker. The fix is scan heuristic refinement.

### [ ] 735. `include/circt/Analysis/FIRRTLInstanceInfo.h:180`
`anyInstanceInEffectiveDesign` declaration is also routine API code, not unresolved implementation. The fix is to suppress this false-positive class.

### [ ] 736. `include/circt/Analysis/DependenceAnalysis.h:66`
This TODO notes potential upstreaming of the analysis into MLIR AffineAnalysis. What is missing is decision/execution on upstream integration. The fix is to evaluate upstream fit and either upstream or explicitly keep local with rationale.

### [ ] 737. `test/Tools/circt-lec/lec-prune-unreachable-before-smt.mlir:12`
This line explicitly documents an intentionally unsupported construct in a targeted test (`hw.type_scope` before HWToSMT). It is expected test context, not backlog debt. The fix is to classify as intentional unsupported-coverage test annotation.

### [ ] 738. `lib/Scheduling/SimplexSchedulers.cpp:924`
Priority function remains simplistic and flagged for improvement. What is missing is richer scheduling priority heuristics. The fix is to implement a more sophisticated cost function.

### [ ] 739. `lib/Scheduling/SimplexSchedulers.cpp:1062`
`llvm_unreachable("Unsupported objective requested")` is a defensive hard-stop for invalid objective inputs, not itself a TODO. It may indicate limited objective support surface. The missing capability, if desired, is additional objective implementations; otherwise this is intentional guard behavior.

### [ ] 740. `lib/Scheduling/SimplexSchedulers.cpp:1149`
Tie-break fallback still relies on ad hoc condition and notes need for proper graph analysis. What is missing is graph-aware movement decision logic. The fix is to replace fallback condition with dependency/criticality analysis.

### [ ] 741. `lib/Analysis/FIRRTLInstanceInfo.cpp:229`
`anyInstanceUnderDut` is normal analysis API implementation, not TODO/unsupported debt. This is scanner false positive noise. The fix is to avoid matching routine identifier names.

### [ ] 742. `lib/Analysis/FIRRTLInstanceInfo.cpp:239`
Same as entry 741: ordinary API implementation (`anyInstanceUnderEffectiveDut`) and non-actionable for gap tracking. The fix is marker-context filtering.

### [ ] 743. `lib/Analysis/FIRRTLInstanceInfo.cpp:240`
Return expression line in routine analysis code, not unresolved work. This is a false positive from broad token matching.

### [ ] 744. `lib/Analysis/FIRRTLInstanceInfo.cpp:247`
`anyInstanceUnderLayer` implementation is standard analysis logic, not debt marker. Non-actionable scanner hit.

### [ ] 745. `lib/Analysis/FIRRTLInstanceInfo.cpp:257`
Routine `anyInstanceInDesign` API implementation, not unresolved feature work. The fix is to suppress this false-positive category.

### [ ] 746. `lib/Analysis/FIRRTLInstanceInfo.cpp:267`
`anyInstanceInEffectiveDesign` implementation is likewise normal code and non-actionable. The fix is scanner refinement.

### [ ] 747. `lib/Firtool/Firtool.cpp:59`
Pipeline ordering TODO notes instance-choice handling limitations in analysis passes. What is missing is robust instance-choice support across dependent analyses so this pass can move later. The fix is to harden instance-graph consumers and relax ordering constraints.

### [ ] 748. `lib/Firtool/Firtool.cpp:69`
`InjectDUTHierarchy` is marked for deletion, indicating transitional pipeline debt. What is missing is replacement flow that makes this pass unnecessary. The fix is to complete migration and remove the pass.

### [ ] 749. `lib/Firtool/Firtool.cpp:183`
`ExtractInstances` is similarly marked for deletion with `InjectDUTHierarchy`, indicating coupled legacy pipeline debt. What is missing is a cleaner architecture that subsumes both behaviors. The fix is to retire both passes once replacement is in place.

### [ ] 750. `lib/Firtool/Firtool.cpp:250`
LowerLayers currently needs extra canonicalization pass cleanup; TODO calls for improving LowerLayers itself. What is missing is cleaner lowering output that reduces post-canonicalization dependence. The fix is to tighten LowerLayers transformations per issue #7896.

### [ ] 751. `lib/Firtool/Firtool.cpp:382`
“Legalize unsupported operations” is descriptive pipeline comment, not unresolved TODO by itself. It reflects intentional legalization stage behavior. Non-actionable as debt marker.

### [ ] 752. `lib/Firtool/Firtool.cpp:800`
Option default remains conservative pending more testing and `-sv-extract-test-code` removal. What is missing is validation confidence to flip default to true. The fix is expanded testing and eventual default change.

### [ ] 753. `lib/Analysis/CMakeLists.txt:67`
Linting subdirectory remains disabled due Slang header-path issues. What is missing is include-path/dependency cleanup to enable linting build. The fix is to resolve header wiring and re-enable the target.

### [ ] 754. `lib/Bindings/Tcl/circt_tcl.cpp:67`
Tcl binding still has unimplemented FIR loading path. What is missing is FIR file loading support in Tcl bindings. The fix is to implement parser hookup and error handling for FIR mode.

### [ ] 755. `lib/Bindings/Tcl/circt_tcl.cpp:68`
The explicit “loading FIR files is unimplemented” error confirms the same missing capability as entry 754. The fix is shared implementation of FIR loading in Tcl.

### [ ] 756. `lib/Transforms/FlattenMemRefs.cpp:387`
Flattening helper currently targets static shapes and notes dynamic-shape possibility as TODO. What is missing is dynamic memref flattening support. The fix is to extend collapse-shape materialization to dynamic shapes safely.

### [ ] 757. `lib/Analysis/TestPasses.cpp:238`
Expected debug-print label (`anyInstanceUnderDut`) in analysis test pass, not unresolved work. This is scan noise. The fix is to exclude routine print-string lines.

### [ ] 758. `lib/Analysis/TestPasses.cpp:242`
Same false-positive class as entry 757: expected output formatting text. Non-actionable.

### [ ] 759. `lib/Analysis/TestPasses.cpp:243`
Continuation of expected analysis printout, not debt marker. The fix is scanner filtering.

### [ ] 760. `lib/Analysis/TestPasses.cpp:246`
Expected label line in test pass output, non-actionable from gap perspective. The fix is to suppress this class.

### [ ] 761. `lib/Analysis/TestPasses.cpp:247`
Same as entry 760: standard test output line, not unresolved implementation. The fix is marker-only scan matching.

### [ ] 762. `lib/Analysis/TestPasses.cpp:250`
Expected test-pass printing of `anyInstanceInDesign`, not backlog debt. Non-actionable false positive.

### [ ] 763. `lib/Analysis/TestPasses.cpp:254`
Expected print label (`anyInstanceInEffectiveDesign`) in test output. Not a TODO gap. The fix is scan heuristic improvement.

### [ ] 764. `lib/Analysis/TestPasses.cpp:255`
Same print-output continuation as entry 763. Non-actionable for manual gap tracking.

### [ ] 765. `lib/Bindings/Python/support.py:281`
Python support code still uses generic error path instead of dedicated `UnconnectedSignalError`. What is missing is typed exception usage for unconnected backedges. The fix is to raise/use `UnconnectedSignalError` consistently.

### [ ] 766. `lib/Analysis/DependenceAnalysis.cpp:169`
Dependence replacement still scans all results to rewrite sources, with TODO suggesting inverted index. What is missing is efficient reverse lookup structure. The fix is to add source->dependence index and avoid full scans.

### [ ] 767. `test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv:18`
This is literal test stimulus data (`0zxxx0z`) for format-radix behavior, not unresolved implementation debt. It is scanner false positive from `x`/`z` placeholder bits. The fix is to suppress value-literal lines in tests.

### [ ] 768. `test/Tools/circt-sim/format-radix-fourstate-compact-rules.sv:19`
Same as entry 767: expected test vector literal (`xxxxxxxx`) used to validate formatting rules. Non-actionable scanner hit.

### [ ] 769. `lib/Analysis/DebugInfo.cpp:165`
Fallback debug-info builder still does not track instance port assignments. What is missing is port-assignment capture in debug model. The fix is to add assignment tracking fields and populate them during fallback instance handling.

### [ ] 770. `test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir:3`
This is regression test commentary asserting unsupported trampoline ABI must be diagnosed at compile time. It documents intended behavior and existing capability boundary, not unresolved debt by itself. The missing work remains broader ABI support if desired; this line itself is test metadata.

### [ ] 771. `test/Tools/circt-sim/aot-trampoline-unsupported-abi-diagnostic.mlir:6`
This is expected compile-time diagnostic text in a regression test, not unresolved implementation debt. It verifies existing unsupported-ABI detection behavior. The fix is scanner suppression for `CHECK:` expectation lines.

### [ ] 772. `lib/Conversion/ImportLiberty/ImportLiberty.cpp:705`
ImportLiberty currently supports only a subset of Liberty group types. What is missing is broader group-type lowering coverage. The fix is to add parsing/lowering for additional group kinds with validation tests.

### [ ] 773. `lib/Conversion/ImportLiberty/ImportLiberty.cpp:802`
Timing subgroup handling is acknowledged as incomplete. What is missing is proper structured lowering for timing subgroups and related nested semantics. The fix is richer subgroup modeling instead of generic attr capture.

### [ ] 774. `lib/Conversion/ImportLiberty/ImportLiberty.cpp:935`
`define` constructs are not supported yet in ImportLiberty parser/lowering. What is missing is parsing/representation support for Liberty `define`. The fix is to implement define parsing and integrate it into lowering context.

### [ ] 775. `lib/Conversion/ImportLiberty/ImportLiberty.cpp:983`
Timing attributes lack array support in current parser path. What is missing is array-valued timing attribute handling. The fix is to add array parsing/storage for timing attributes.

### [ ] 776. `lib/CAPI/Dialect/FIRRTL.cpp:327`
CAPI wrapper uses `reinterpret_cast` with a FIXME about possible strict-aliasing violation. What is missing is aliasing-safe conversion approach for direction arrays. The fix is to replace cast with safe intermediate storage/copy or API adjustment.

### [ ] 777. `lib/CAPI/Dialect/CMakeLists.txt:1`
Build TODO notes optional-source checks should be configurable in `*_add_library` helpers. What is missing is optional check-source feature control at helper-call sites. The fix is to add argumentized optional-source behavior in CMake helpers.

### [ ] 778. `lib/Bindings/Python/dialects/synth.py:57`
Python synth wrapper objects are not yet associated with backing MLIR value/op handles. What is missing is direct MLIR object linkage for introspection/debugging. The fix is to store and expose corresponding MLIR references in wrapper classes.

### [ ] 779. `test/Tools/circt-sim/aot-trampoline-native-fallback-f64-incompatible-abi.mlir:10`
This is expected runtime warning text in a regression test for incompatible native-fallback ABI. It validates current boundary behavior rather than indicating new unresolved work. The fix is scanner filtering for expected `RUNTIME:` lines.

### [ ] 780. `lib/CAPI/Dialect/OM.cpp:261`
Temporary CAPI helper is marked removable, indicating cleanup debt. What is missing is follow-through removal once callers are migrated. The fix is to delete this legacy helper when compatibility window closes.

### [ ] 781. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:45`
Interop lowering hardcodes verilated module class name. What is missing is deriving class name from config/attributes instead of literal assumptions. The fix is to plumb configurable class-name source from interop metadata.

### [ ] 782. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:125`
Current lowering uses `func::CallIndirectOp` with TODO to move to `systemc::CallIndirectOp` post-PR. What is missing is dialect-native indirect call op usage and reduced func-dialect dependency. The fix is to migrate once target op is available.

### [ ] 783. `lib/Dialect/LLHD/Transforms/Deseq.cpp:501`
Deseq tracing currently skips processes with unsupported terminators. What is missing is terminator handling coverage for those cases. The fix is to extend tracing logic or provide stronger diagnostics for skipped forms.

### [ ] 784. `lib/Dialect/LLHD/Transforms/Deseq.cpp:796`
Trigger-dependence check only considers i1 values and may miss non-i1 dependencies. What is missing is generalized dependence analysis across value widths/types. The fix is to unify boolean/value tables or broaden dependency checks.

### [ ] 785. `lib/Dialect/LLHD/Transforms/Deseq.cpp:824`
Known-value path does not reject values that depend on triggers, as TODO notes. What is missing is explicit rejection/filtering of trigger-dependent values. The fix is to add dependency guard before returning known value.

### [ ] 786. `lib/Dialect/SystemC/SystemCOps.cpp:746`
SystemC op implementation is copy-pasted from `func` dialect due missing reusable upstream abstraction. What is missing is upstream refactoring for shared function-like op logic. The fix is to upstream common functionality and drop local copies.

### [ ] 787. `lib/Dialect/SystemC/SystemCOps.cpp:751`
Verifier symbol-use logic is explicitly an upstream copy. What is missing is deduplicated shared implementation. The fix is to replace exact copy with reusable upstream helper.

### [ ] 788. `lib/Dialect/SystemC/SystemCOps.cpp:816`
`FuncOp` implementation in SystemC remains mostly copied from `func` dialect. What is missing is reusable abstraction to avoid divergence. The fix is shared upstream utility adoption.

### [ ] 789. `lib/Dialect/SystemC/SystemCOps.cpp:878`
Parse function implementation is copied/inlined from upstream internals for SSA-name access needs. What is missing is upstream API that exposes required hooks without copy-paste. The fix is upstream interface extension and local deduplication.

### [ ] 790. `lib/Dialect/SystemC/SystemCOps.cpp:981`
Print function implementation is also copied to customize attribute elision. What is missing is configurable upstream print helper. The fix is to upstream customization points and remove copy.

### [ ] 791. `lib/Dialect/SystemC/SystemCOps.cpp:1012`
Clone operation implementation is an exact upstream copy, representing maintenance debt. What is missing is shared clone helper reuse. The fix is to eliminate duplicate clone logic via common API.

### [ ] 792. `lib/Dialect/SystemC/SystemCOps.cpp:1123`
`ReturnOp` implementation is another copy from `func` dialect. What is missing is reusable return-op infrastructure across dialects. The fix is upstream factoring and local replacement.

### [ ] 793. `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:167`
Loop unrolling currently rejects non-conditional-exit branch shapes as unsupported. What is missing is support for broader exit-branch patterns. The fix is to generalize loop pattern matching beyond current strict form.

### [ ] 794. `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:181`
Unroller supports only narrow exit-condition forms and rejects others. What is missing is richer condition analysis for loop bounds. The fix is to extend recognized exit predicates/expressions.

### [ ] 795. `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:211`
Induction variable initialization forms are constrained and unsupported values are rejected. What is missing is broader induction-value handling. The fix is to support additional SSA patterns for induction initialization.

### [ ] 796. `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:228`
Increment handling is similarly narrow and rejects unsupported increments. What is missing is generalized step-expression support. The fix is to parse/analyze wider increment forms.

### [ ] 797. `lib/Dialect/LLHD/Transforms/UnrollLoops.cpp:250`
Loop-bound inference bails on unsupported bounds, limiting unroll applicability. What is missing is more robust bound analysis. The fix is to support additional bound formulations with safety checks.

### [ ] 798. `test/Tools/circt-sim/aot-native-module-init-hw-struct-create-fourstate.mlir:7`
This `COMPILE-NOT` assertion verifies that `unsupported_op:hw.struct_create` does not appear, i.e., feature is supported in this regression. It is expected test text, not unresolved debt. The fix is scanner suppression for negative-check lines.

### [ ] 799. `test/Tools/circt-sim/aot-native-module-init-scf-if-struct-extract.mlir:7`
Same as entry 798 for `hw.struct_extract`: this is a support-regression assertion, not backlog marker. The fix is to ignore `COMPILE-NOT` expectation lines.

### [ ] 800. `lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:108`
Control-flow decision aggregation is currently eager and may do unnecessary work. What is missing is more targeted aggregation restricted to relevant block set. The fix is to first compute dominator-target region and only evaluate exit-causing decisions.

### [ ] 801. `lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:206`
RemoveControlFlow currently bails out when encountering unsupported terminators. What is missing is broader terminator support in control-flow reduction. The fix is to extend handling for additional terminator ops or provide targeted fallback lowering.

### [ ] 802. `lib/Dialect/RTG/Transforms/ElaborationPass.cpp:1978`
This FIXME notes current reordering assumptions are not aligned with MLIR `MemoryEffects` semantics. What is missing is a dedicated effect model/trait for elaboration-time reordering. The fix is custom trait/interface or stricter ordering preservation.

### [ ] 803. `lib/Dialect/RTG/Transforms/ElaborationPass.cpp:2094`
Elaboration clones sequences even when a single remaining reference may allow reuse. What is missing is reference-aware clone elision. The fix is to detect sole-reference cases and avoid unnecessary cloning.

### [ ] 804. `test/Tools/circt-bmc/drop-unsupported-sva.mlir:2`
This is a regression `RUN` line exercising `--drop-unsupported-sva`, not unresolved debt. It is test orchestration text. The fix is to suppress fixture command lines in scans.

### [ ] 805. `test/Tools/circt-bmc/drop-unsupported-sva.mlir:7`
`circt.unsupported_sva` here is intentional test IR fixture content used to validate drop behavior, not a missing feature marker. Non-actionable as backlog item.

### [ ] 806. `test/Tools/circt-bmc/drop-unsupported-sva.mlir:12`
Expected `DROP` diagnostic check line in test output, not unresolved work. The fix is to exclude `CHECK:` expectations from actionable scans.

### [ ] 807. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:53`
Active-interval tracking uses a simple vector/sort approach with TODO for better data structure. What is missing is more efficient active-set representation. The fix is ordered container/heap keyed by interval end.

### [ ] 808. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:82`
Allocator assumes fully elaborated IR but does not verify precondition. What is missing is explicit elaboration-state validation before allocation. The fix is preflight verifier check.

### [ ] 809. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:119`
Current live-range model does not support labels/jumps/loops, limiting correctness for control-flow-rich programs. What is missing is CFG-aware liveness modeling. The fix is control-flow-sensitive interval construction.

### [ ] 810. `lib/Dialect/RTG/Transforms/LinearScanRegisterAllocationPass.cpp:131`
Interval overlap handling overapproximates conflicts, potentially degrading allocation quality. What is missing is tighter interference analysis. The fix is more precise overlap modeling.

### [ ] 811. `test/Tools/circt-sim/aot-native-module-init-skip-telemetry.mlir:6`
This `CHECK` line is expected telemetry output in a regression test, not unresolved feature debt. It validates skip-reason reporting behavior.

### [ ] 812. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:42`
Constructor parameter `unsupportedInstr` is normal data wiring, not TODO/unsupported debt by itself. This is scanner false positive from identifier naming.

### [ ] 813. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:43`
Same as entry 812: member initialization with `unsupportedInstr` identifier, non-actionable.

### [ ] 814. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:94`
Membership check against `unsupportedInstr` set is routine logic, not unresolved marker. The actionable TODOs in this vicinity are at lines 96 and 120.

### [ ] 815. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:96`
Emitter assumes `//` comment syntax for assembly output. What is missing is target-specific comment delimiter handling. The fix is configurable comment syntax per backend/assembler.

### [ ] 816. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:120`
Binary emission hardcodes `.word`, reducing portability. What is missing is target-aware data directive selection. The fix is backend-configurable directive mapping.

### [ ] 817. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:191`
Member declaration of `unsupportedInstr` set is routine state, not implementation debt. This is scanner noise.

### [ ] 818. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:200`
Function name `parseUnsupportedInstructionsFile` is normal parser utility, not TODO marker. Non-actionable scan hit.

### [ ] 819. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:201`
Parameter `unsupportedInstructionsFile` is ordinary variable naming, not debt signal. The fix is scanner filtering for identifiers.

### [ ] 820. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:202`
Same as entry 819: parameter line with `unsupportedInstrs` is routine code, non-actionable.

### [ ] 821. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:203`
Conditional guard on optional unsupported-instructions file path is ordinary implementation logic. Not backlog debt.

### [ ] 822. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:204`
`ifstream` opening of unsupported-instructions file is routine behavior, not unresolved work.

### [ ] 823. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:209`
Set insertion of parsed unsupported instructions is normal parser functionality, non-actionable.

### [ ] 824. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:228`
Local `unsupportedInstr` set construction is expected pass setup logic, not TODO debt.

### [ ] 825. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:229`
Loop over configured unsupported instructions is routine option handling, non-actionable.

### [ ] 826. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:230`
Insertion into unsupported instruction set is routine code, not unresolved gap.

### [ ] 827. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:231`
Call to parse unsupported-instruction file is expected utility invocation, not debt marker.

### [ ] 828. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:232`
Continuation of same parse call; non-actionable routine code.

### [ ] 829. `lib/Dialect/RTG/Transforms/EmitRTGISAAssemblyPass.cpp:249`
Passing unsupported-instruction set into emitter constructor is routine state wiring, not unresolved work.

### [ ] 830. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:62`
CutRewriter currently supports only AIG `AndInverterOp` in logic-op simulation check and excludes comb and/or/xor ops. What is missing is broader combinational op support in cut simulation. The fix is to extend `isSupportedLogicOp` to comb boolean ops with tests.

### [ ] 831. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:87`
Fatal error path enforces that unsupported simulation ops should have been filtered by `isSupportedLogicOp`. What is missing is wider supported-op coverage so this fatal path is hit less often. The fix is to broaden `isSupportedLogicOp` and associated simulation handlers.

### [ ] 832. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:169`
Truth-table simulation returns an explicit error on unsupported ops, marking a real capability boundary. What is missing is simulation support for additional operation classes. The fix is to implement op semantics for those classes.

### [ ] 833. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:174`
Current truth-table builder assumes single output (`numOutputs == 1`). What is missing is full multi-output truth-table support. The fix is to carry/return all outputs instead of first-only shortcut.

### [ ] 834. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:343`
Merge step notes possible linear merge of already topologically sorted vectors but currently does slower rebuild logic. What is missing is efficient merge-sort style combination. The fix is to merge by operation index directly.

### [ ] 835. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:402`
Merged cut inputs are not sorted by defining operation. What is missing is deterministic input ordering in merged cuts. The fix is to sort merged inputs consistently.

### [ ] 836. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:403`
Area/delay metadata is not updated after cut merges. What is missing is accurate merged-cost recomputation. The fix is to update area/delay while merging cuts.

### [ ] 837. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:552`
Selection currently sorts where a priority queue could reduce runtime overhead. What is missing is more efficient queue-based selection. The fix is to replace repeated sorts with priority queue scheduling.

### [ ] 838. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:684`
Enumeration currently excludes variadic ops and non-single-bit results. What is missing is support for these wider operation/result forms. The fix is to generalize cut enumeration constraints.

### [ ] 839. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:887`
Pattern pipeline still gates out multi-output patterns. What is missing is multi-output pattern support end-to-end. The fix is to remove this guard once multi-output handling lands.

### [ ] 840. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:957`
Arrival-time model ignores global delay information. What is missing is IR-level arrival-time annotation usage in cut timing. The fix is to plumb/capture global `arrivalTime` and include it in timing estimates.

### [ ] 841. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:122`
SOP balancing defaults are intentionally reduced due current CutRewriter inefficiency, indicating performance-related technical debt. What is missing is efficient enough CutRewriter implementation to restore stronger defaults. The fix is performance optimization in CutRewriter and retuning defaults.

### [ ] 842. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:139`
Pipeline still lacks balancing/rewriting/FRAIG stages mentioned in TODO. What is missing is fuller synthesis-stage coverage. The fix is to add these passes with ordering and QoR validation.

### [ ] 843. `lib/Conversion/CalyxNative/CalyxNative.cpp:158`
`XXX` comment indicates awkward block replacement strategy (“baroque”) in lowering implementation. What is missing is cleaner module replacement mechanism. The fix is to replace whole module op directly when feasible.

### [ ] 844. `lib/Dialect/LLHD/Transforms/InlineCalls.cpp:246`
Inlining logic hardcodes object-field offsets rather than deriving from metadata/frontend. What is missing is robust layout-derived offset handling. The fix is to compute offsets from class metadata or pass them explicitly.

### [ ] 845. `lib/Dialect/LLHD/Transforms/InlineCalls.cpp:460`
Recursive function calls remain unsupported for inlining in `--ir-hw` mode. What is missing is recursion-capable inlining strategy or alternative lowering support. The fix is to add recursion handling or structured fallback.

### [ ] 846. `lib/Dialect/LLHD/Transforms/InlineCalls.cpp:488`
Same recursive inlining limitation as entry 845 on another path. What is missing is unified recursive-call support policy. The fix is shared implementation/fallback logic.

### [ ] 847. `lib/Conversion/CombToDatapath/CombToDatapath.cpp:63`
CombToDatapath currently handles only binary multipliers and skips variadic forms. What is missing is variadic multiplier lowering support. The fix is to implement decomposition/lowering for multi-input muls.

### [ ] 848. `lib/Conversion/CombToDatapath/CombToDatapath.cpp:107`
Lowering policy for multi-input multipliers is undecided and currently leaves them legal as comb ops. What is missing is explicit lowering strategy for these ops. The fix is to define and implement multi-input multiplier transformation.

### [ ] 849. `lib/Conversion/HWToSMT/HWToSMT.cpp:305`
This maps to an unsupported aggregate-constant attr/type diagnostic boundary in HWToSMT. What is missing is support for additional aggregate constant combinations. The fix is to extend aggregate constant lowering beyond currently legal pairs.

### [ ] 850. `lib/Conversion/HWToSMT/HWToSMT.cpp:441`
This area includes unsupported bitcast result-type handling. What is missing is broader bitcast lowering support for currently rejected result types. The fix is to add legal conversion paths or explicit normalization rewrites.

### [ ] 851. `lib/Conversion/HWToSMT/HWToSMT.cpp:569`
Array lowering still rejects unsupported array types in some paths. What is missing is comprehensive array-type support in HWToSMT conversion. The fix is to broaden supported array type constraints and conversion rules.

### [ ] 852. `lib/Conversion/HWToSMT/HWToSMT.cpp:601`
Array element type support remains limited, with unsupported element diagnostics. What is missing is support for additional element-type classes in SMT array encoding. The fix is to add element-type lowering coverage.

### [ ] 853. `lib/Conversion/HWToSMT/HWToSMT.cpp:641`
Another unsupported-array-type guard in index/select path indicates residual array-shape limitations. What is missing is end-to-end consistency for array-type handling across all array ops. The fix is to unify and expand array support logic.

### [ ] 854. `lib/Dialect/Synth/Transforms/TechMapper.cpp:135`
Mapped instance naming is still placeholder-level (“mapped”). What is missing is meaningful deterministic instance naming strategy. The fix is to derive names from mapped cell/cut context.

### [ ] 855. `lib/Dialect/Synth/Transforms/TechMapper.cpp:174`
Tech library info is encoded in ad hoc attribute and TODO calls for structured representation. What is missing is dedicated IR construct for technology library metadata. The fix is to introduce a dedicated techlib op/schema.

### [ ] 856. `lib/Dialect/Synth/Transforms/TechMapper.cpp:183`
Mapping currently runs broadly and TODO suggests hierarchy-scoped mapping control. What is missing is selective mapping by hierarchy scope. The fix is scope-aware mapping filters/options.

### [ ] 857. `lib/Dialect/Synth/Transforms/TechMapper.cpp:201`
`convertToDouble()` at this line is routine numeric conversion, not TODO/unsupported debt. This is scanner false positive noise.

### [ ] 858. `lib/Dialect/Synth/Transforms/TechMapper.cpp:207`
Delay parsing assumes integer attributes, pending structured timing units/cell op modeling. What is missing is unit-aware timing attribute representation. The fix is typed timing attributes with units.

### [ ] 859. `lib/Dialect/LLHD/Transforms/HoistSignals.cpp:652`
HoistSignals currently materializes fallback constants where a dedicated `llhd.dontcare`-like value would be preferable. What is missing is explicit dont-care representation. The fix is to introduce/use a dedicated dontcare construct.

### [ ] 860. `lib/Conversion/CombToArith/CombToArith.cpp:290`
Variadic lowering is currently linear fold; TODO notes tree construction would be better. What is missing is balanced-tree lowering for improved depth/QoR. The fix is to build balanced trees for long operand lists.

### [ ] 861. `lib/Conversion/CombToArith/CombToArith.cpp:406`
CombToArith still lacks a dedicated conversion pattern for `comb.parity`. What is missing is parity lowering support in this conversion pipeline. The fix is to add a parity rewrite pattern with type/legalization tests.

### [ ] 862. `lib/Conversion/HWToLLVM/HWToLLVM.cpp:704`
Aggregate lowering currently supports only arrays and structs. What is missing is support for additional aggregate forms beyond those two categories. The fix is to extend aggregate conversion handling and legality checks.

### [ ] 863. `lib/Dialect/Synth/Transforms/LowerWordToBits.cpp:189`
Known-bits fallback path is depth-limited and uncached, as noted in TODO. What is missing is cache-aware, deeper known-bits reuse for this transform. The fix is to thread cached analysis results into this computation path.

### [ ] 864. `lib/Dialect/Synth/Transforms/LowerVariadic.cpp:134`
LowerVariadic currently only handles top-level operations due missing nested-region topological ordering. What is missing is region-aware lowering order. The fix is to add topological traversal across nested regions.

### [ ] 865. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:4242`
Sequence lowering rejects block-argument cases with explicit unsupported error. What is missing is block-argument sequence lowering support in VerifToSMT. The fix is to implement lowering for these argument-driven sequence forms.

### [ ] 866. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:5049`
Init/loop regions are modeled symbolically with TODO to allow concrete handling. What is missing is concrete region support for better performance/precision. The fix is to add conversions between concrete and symbolic where needed.

### [ ] 867. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:6270`
BMC conversion still rejects unsupported boolean type forms. What is missing is broader boolean-type handling in this conversion path. The fix is to extend bool-type lowering coverage or normalize inputs earlier.

### [ ] 868. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:6418`
Register/init lowering rejects certain integer initial values as unsupported. What is missing is support for more integer init-value representations. The fix is to broaden accepted typed-attr/value forms.

### [ ] 869. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:6435`
Similarly, some boolean initial values remain unsupported in BMC conversion. What is missing is complete bool-init value handling. The fix is to accept additional bool-init encodings and normalize them.

### [ ] 870. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:9607`
Model-check loop uses `scf.for` where TODO notes a `while` form could early-exit on property failure. What is missing is early-exit control-flow optimization. The fix is to switch to a loop form permitting short-circuit termination.

### [ ] 871. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:10084`
Current lowering introduces many ITEs and TODO notes performance cost. What is missing is ITE reduction via more concrete region handling or alternative state encoding. The fix is to avoid unnecessary ITE generation in register-state updates.

### [ ] 872. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:10388`
This comment explains why unsupported-op-affecting properties/enables are kept live for diagnostics. It is intentional diagnostic policy, not unresolved implementation debt. Non-actionable for gap tracking.

### [ ] 873. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11001`
This line documents an intentional choice to keep unsupported LLVM constant cases on explicit diagnostic path. It is policy explanation rather than TODO debt. Non-actionable.

### [ ] 874. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11016`
Same as entry 873: intentional routing of non-scalar zeros to unsupported-op diagnostics. This is explanatory policy text, not a missing feature marker.

### [ ] 875. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11763`
This comment describes a deliberate exception to suppress certain unsupported diagnostics for propertyless BMC short-circuit cases. It is intentional behavior, not unresolved debt.

### [ ] 876. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11769`
Local variable `unsupportedOp` declaration is routine implementation detail, not TODO/unsupported debt. This is scanner false positive from identifier naming.

### [ ] 877. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11780`
Assignment to `unsupportedOp` is part of explicit diagnostic detection logic, not unresolved implementation. Non-actionable.

### [ ] 878. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11783`
Control-flow check on `unsupportedOp` is routine diagnostic plumbing, not a gap marker.

### [ ] 879. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11786`
Same as entry 878: loop break condition using `unsupportedOp`, non-actionable implementation detail.

### [ ] 880. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11789`
Conditional branch on `unsupportedOp` before emitting a diagnostic is normal code path, not TODO debt.

### [ ] 881. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11790`
`unsupportedOp->emitError(...)` is intentional explicit diagnostic behavior for unsupported LLVM in SMTLIB export. It indicates current capability boundary, not a TODO at this line.

### [ ] 882. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11793`
Diagnostic message continuation line, non-actionable by itself. It belongs to intentional unsupported-op reporting logic.

### [ ] 883. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11828`
Unsupported register initial-value handling remains a real conversion boundary. What is missing is support for additional register init-value forms. The fix is to expand accepted typed initializers.

### [ ] 884. `lib/Conversion/VerifToSMT/VerifToSMT.cpp:11838`
TODO notes temporary multi-clock register association workaround. What is missing is proper association of register ins/outs with clocks, after which this workaround can be removed. The fix is explicit reg-clock association model.

### [ ] 885. `lib/Conversion/CFToHandshake/CFToHandshake.cpp:726`
Buffer size selection is currently hardcoded/heuristic (`TODO how to size these?`). What is missing is principled buffer sizing strategy from CFG analysis. The fix is path-length or throughput-informed sizing policy.

### [ ] 886. `lib/Dialect/Synth/Transforms/AIGERRunner.cpp:207`
Bit extraction currently recreates `comb.extract` ops without reuse. What is missing is extract-op caching for efficiency. The fix is to cache by `(value, bitPosition)` and reuse existing extracts.

### [ ] 887. `lib/Dialect/RTG/IR/RTGAttributes.cpp:26`
DenseSet hash helper uses weak XOR combination and TODO requests better collision resistance. What is missing is stronger commutative hash mixing. The fix is robust collision-resistant set hash composition.

### [ ] 888. `lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:956`
Default dispatch to `visitUnsupportedOp` is expected fallback mechanism in visitor logic, not itself unresolved debt. This is normal control flow.

### [ ] 889. `lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1025`
`visitUnsupportedOp` function definition is standard unsupported-op handling infrastructure, not a TODO marker by itself.

### [ ] 890. `lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1026`
Comment describing ignored vs unsupported-op handling is explanatory and intentional, not unresolved implementation debt.

### [ ] 891. `lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1048`
This default branch documents a real conversion boundary: only a whitelist of operations is handled for HW-to-BTOR2 and everything else fails. What is missing is broader op coverage or a normalization pipeline that guarantees only legal ops reach this pass. The fix is to either extend BTOR2 lowering support or add a pre-lowering canonicalization pass with strict legality tests.

### [ ] 892. `lib/Conversion/HWToBTOR2/HWToBTOR2.cpp:1051`
The explicit `is an unsupported operation` diagnostic is the concrete failure path for the coverage boundary noted in entry 891. What is missing is implementation support for additional ops currently rejected in this converter. The fix is targeted lowering additions plus regression tests per newly supported op kind.

### [ ] 893. `lib/Dialect/Synth/SynthOps.cpp:68`
`MajorityInverterOp::fold` still lacks constant-fold identities like `maj(x,1,1)=1` and `maj(x,0,0)=0`. What is missing is these simplification rules in the fold path for partial-constant inputs. The fix is to add the identity handling and verify with fold/canonicalization regressions.

### [ ] 894. `lib/Dialect/Synth/SynthOps.cpp:298`
The variadic `AndInverterOp` lowering uses a naive balanced tree and leaves QoR-driven structuring as TODO. What is missing is cost-aware tree construction that can optimize area/critical path. The fix is to use analysis-guided decomposition instead of fixed recursive halving.

### [x] 895. `lib/Conversion/SimToSV/SimToSV.cpp:240`
Status update (2026-02-28): this gap is closed in this workspace. Unclocked, ungated DPI calls now lower directly to `sv.func.call` when the callee has a single explicit return and no output arguments, avoiding unnecessary `always_comb` + temporary-reg wrapping. Fallback lowering remains unchanged for clocked, gated, and output-argument call cases; regression coverage was added in `test/Conversion/SimToSV/dpi.mlir`.

### [ ] 896. `lib/Dialect/LLHD/IR/LLHDOps.cpp:917`
`llvm_unreachable("Not implemented")` here is an intentional sentinel in a type-interface model that advertises integer types as non-destructurable (`getSubelementIndexMap` returns empty). This is not actionable debt unless callers start invoking `getTypeAtIndex` for integers. Scanner false positive for current behavior.

### [ ] 897. `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:389`
This FIXME calls out a portability assumption that `sizeof(intptr_t) == sizeof(size_t)`. What is missing is an explicit, target-correct type strategy that does not rely on that assumption. The fix is to lower with ABI-aware pointer/size types and add cross-target tests.

### [ ] 898. `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:940`
Format lowering currently substitutes placeholders for unsupported format operations. What is missing is complete support for all format fragments expected by Arc print/trace paths. The fix is to implement lowering for remaining format ops and preserve clear diagnostics for truly invalid cases.

### [ ] 899. `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:941`
Appending `"<unsupported>"` confirms partial formatting behavior at runtime when unsupported format pieces are encountered. What is missing is parity with the format semantics users expect from source-level operations. The fix is to close format-operation coverage and convert placeholder cases into full lowering.

### [ ] 900. `lib/Dialect/HW/Transforms/HWAggregateToComb.cpp:340`
`ArraySliceOp` is still excluded from the aggregate-to-comb conversion target setup. What is missing is lowering/legalization support for array-slice aggregates in this pass. The fix is to add a conversion pattern and mark `ArraySliceOp` illegal once implemented.

### [ ] 901. `lib/Conversion/SeqToSV/SeqToSV.cpp:763`
Macro emission here is still stringly and TODO notes missing first-class macro IR modeling. What is missing is macro ops/uses with symbol semantics that allow dead-code elimination of unused defines. The fix is to introduce representational ops and migrate this ad hoc emission logic.

### [ ] 902. `lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:154`
This TODO documents duplicated `AffineToStandard` logic in affine load lowering. What is missing is shared utility reuse to avoid drift between conversion paths. The fix is to factor common expansion helpers and update both users.

### [ ] 903. `lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:188`
Same as entry 902 for affine store lowering: copied logic remains local and unshared. What is missing is centralization of the affine-map expansion/rewrite helper path. The fix is deduplication into reusable conversion utilities.

### [ ] 904. `lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:304`
`Operation *unsupported;` is just local variable setup for later diagnostic emission, not a standalone gap. This is scanner noise from identifier naming.

### [ ] 905. `lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:365`
`unsupported = op;` is internal bookkeeping for the eventual unsupported-op diagnostic and not an independent TODO/debt marker. Non-actionable scanner false positive.

### [ ] 906. `lib/Conversion/AffineToLoopSchedule/AffineToLoopSchedule.cpp:371`
The emitted `unsupported operation` error is a real boundary in loop-schedule conversion: only a subset of ops is currently typed for scheduling. What is missing is broader operator typing/resource modeling for additional affine/memref/arith ops. The fix is to expand `TypeSwitch` coverage and add schedule regression tests.

### [ ] 907. `lib/Conversion/FSMToSV/FSMToSV.cpp:342`
FSM-to-SV conversion rejects operations from dialects outside `comb/hw/fsm` in moved regions. What is missing is either legalization for additional dialect ops or a prerequisite canonicalization pipeline that rewrites them away. The fix is to extend dialect coverage or tighten/automate preconditions.

### [ ] 908. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:211`
Name extraction for block arguments falls back to `<unknown-argument>` outside `hw.module` and TODO notes missing handling. What is missing is naming support for other operation contexts in debug/path reports. The fix is to extend `getNameImpl` to additional parent op types.

### [ ] 909. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:729`
Debug-point tracking is currently always-on in thread-local analysis state. What is missing is optional/flagged debug-point collection to reduce overhead in non-debug runs. The fix is to gate allocation and collection behind an analysis option.

### [ ] 910. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1307`
Memory write endpoint tracking ignores address contribution (`TODO: Add address`). What is missing is address-path participation in timing/path analysis for memory writes. The fix is to incorporate address dependencies into `markRegEndPoint` modeling.

### [ ] 911. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1312`
Same gap as entry 910 for `FirMemReadWriteOp`: write-address effects are not modeled. What is missing is full readwrite-port dependency modeling including address paths. The fix is to thread address operands into endpoint construction.

### [ ] 912. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1383`
Wrapper-module construction drops types with unknown bitwidth as `Unsupported type`. What is missing is handling/flattening for non-integer or dynamic-width types used by analyzed ops. The fix is either richer type lowering or explicit preconditions with diagnostics and tests.

### [ ] 913. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1400`
Unsupported operand types still cause failure while building analyzer wrapper modules. What is missing is operand-type normalization or extension of accepted operand type space. The fix is to add conversions for more operand types or reject earlier with clearer context.

### [ ] 914. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1409`
Unsupported result types similarly stop wrapper construction for operation analysis. What is missing is complete result-type support parity with operations traversed by the pass. The fix is to broaden supported result lowering types and validate with mixed-type regression cases.

### [ ] 915. `lib/Dialect/HW/Transforms/HWStopatSymbolic.cpp:68`
Stopat selector parsing currently enforces a narrow selector grammar and rejects other forms. What is missing is compatibility handling for additional selector syntaxes used by existing tool flows. The fix is to extend parser normalization/grammar support while preserving strict diagnostics for malformed input.

### [ ] 916. `lib/Dialect/Pipeline/PipelineOps.cpp:719`
Runoff-stage detection is recomputed on demand and TODO calls for precomputed property storage. What is missing is cached stage-kind metadata to avoid repeated backwards scans. The fix is to compute once and store as a property/attribute.

### [ ] 917. `lib/Dialect/Pipeline/PipelineOps.cpp:980`
Clock-gate lookup performs repeated prefix-sum walking over `clockGatesPerRegister` and TODO flags data-layout inefficiency. What is missing is a more direct index structure for per-register gate slices. The fix is to store precomputed offsets or a richer property representation.

### [ ] 918. `lib/Conversion/FSMToCore/FSMToCore.cpp:428`
FSM-to-Core has the same dialect-coverage boundary as FSM-to-SV for moved region ops. What is missing is support/legalization for non-`comb/hw/fsm` operations encountered in machine regions. The fix is to extend conversion coverage or enforce earlier canonicalization constraints.

### [ ] 919. `lib/Conversion/SeqToSV/FirRegLowering.cpp:466`
`areEquivalentValues` handles array-index equivalence but leaves struct equivalence as TODO. What is missing is recursive struct field equivalence logic for conditional register-update matching. The fix is to add struct-aware recursion and regression tests for structurally equivalent terms.

### [ ] 920. `lib/Conversion/SeqToSV/FirRegLowering.cpp:793`
Initialization currently asserts on element types outside integer/array/struct in recursive register randomization. What is missing is robust handling or graceful diagnostics for additional aggregate/unsupported element types. The fix is to implement remaining type cases or replace assertion with user-facing failure.

### [ ] 921. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:97`
This TODO tracks a known dialect-conversion infrastructure limitation: safe RAUW via `ConversionPatternRewriter` is not available, forcing direct `replaceAllUsesWith` workarounds. What is missing is supported RAUW semantics in conversion rewriters for this use case. The fix is to migrate once upstream issue #6795 is resolved and remove ad hoc direct-RAUW paths.

### [ ] 922. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:319`
Same infrastructure gap as entry 921 in the forwarding branch: direct RAUW is used because conversion-time RAUW support is incomplete. The fix is shared with 921: adopt supported rewriter RAUW once available and simplify this pattern.

### [ ] 923. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:359`
Same RAUW-support TODO in output-port replacement path. What is missing is consistent conversion-safe replacement APIs. The fix is to replace manual RAUW with canonical conversion rewriter calls after upstream support lands.

### [ ] 924. `lib/Dialect/Kanagawa/Transforms/KanagawaPassPipelines.cpp:62`
Pipeline currently lacks a verification step ensuring temporary `memref.alloca` artifacts are fully eliminated after mem2reg/high-level lowering. What is missing is a guardrail pass that enforces this invariant. The fix is to add a verifier/illegal-op check in the pipeline and test failure on leftover allocas.

### [ ] 925. `lib/Conversion/SeqToSV/FirMemLowering.cpp:42`
Memory collection currently does not filter to DUT hierarchy despite TODO notes. What is missing is DUT-scoped selection to avoid lowering unrelated memories. The fix is to integrate hierarchy-state checks before collecting `FirMemOp`s.

### [ ] 926. `lib/Conversion/SeqToSV/FirMemLowering.cpp:89`
`modName` metadata handling remains unresolved in memory config collection. What is missing is a clear policy for preserving or intentionally dropping this attribute in generated memory artifacts. The fix is to decide semantics, implement propagation/ignore logic explicitly, and test it.

### [ ] 927. `lib/Conversion/SeqToSV/FirMemLowering.cpp:90`
`groupID` handling is likewise unresolved. What is missing is explicit support or documented non-support for this field in lowering/output grouping. The fix is to implement the chosen behavior and add regression coverage for grouped memories.

### [ ] 928. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:55`
Bitcast conversion supports integer/array/struct but still excludes unions, packed arrays, and enums. What is missing is full aggregate-type coverage for bitcast lowering. The fix is to extend recursive packing/unpacking logic for these type classes.

### [ ] 929. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:96`
This assertion is the concrete fallback when recursive integer collection encounters unsupported aggregate kinds. It reflects the type-coverage boundary from entry 928. The fix is to implement missing type cases or replace assertion with graceful diagnostics in all modes.

### [ ] 930. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:167`
Same unsupported-type assertion in aggregate reconstruction path. What is missing is symmetric support between flatten and rebuild phases for additional type kinds. The fix is to extend reconstruction logic in lockstep with input flattening.

### [ ] 931. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:177`
Partial-conversion disabled mode emits unsupported-input diagnostics for non-covered bitcast types. This is a real capability boundary until broader type support is added. The fix is expanded input-type handling and diagnostics tests.

### [ ] 932. `lib/Dialect/HW/Transforms/HWConvertBitcasts.cpp:179`
Same as entry 931 for output type coverage. What is missing is output-side lowering support for currently rejected type forms. The fix is to add output reconstruction for those types and keep partial-conversion behavior well tested.

### [ ] 933. `lib/Conversion/SMTToZ3LLVM/LowerSMTToZ3LLVM.cpp:1387`
This FIXME flags ABI fragility from materializing booleans as LLVM `i1` for Z3 API calls. What is missing is a backend-robust bool representation that matches target ABI lowering expectations. The fix is to standardize bool argument lowering (or explicit extension) and validate across targets.

### [ ] 934. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:293`
Argument/result name gathering is open-coded and TODO notes it should belong in `ModulePortInfo`. What is missing is centralized port metadata ownership. The fix is to move naming responsibilities into `ModulePortInfo` and remove duplicated extraction logic.

### [ ] 935. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:314`
Same RAUW-through-conversion limitation as entries 921–923 appears in output-read replacement. What is missing is proper conversion-time RAUW support, forcing direct use replacement today. The fix is to migrate once upstream rewriter support is available.

### [ ] 936. `lib/Dialect/HW/ModuleImplementation.cpp:157`
This TODO is an upstream-printing workaround: location alias emission can be produced even when locations are not printed. What is missing is upstream `printOptionalLocationSpecifier` behavior that respects effective print usage. The fix is upstream resolution, then removal of local workaround comments/logic.

### [ ] 937. `lib/Dialect/HW/ModuleImplementation.cpp:189`
Same upstream alias-printing issue as entry 936 on result locations. This is not local feature debt but an acknowledged dependency/workaround. The fix remains upstream API behavior correction.

### [ ] 938. `lib/Dialect/HW/ModuleImplementation.cpp:441`
Same location-alias printing TODO in another module-signature printer path. The missing piece is still upstream printer correctness around suppressed debug-location output.

### [ ] 939. `lib/Dialect/Kanagawa/Transforms/KanagawaAddOperatorLibrary.cpp:46`
This TODO is a C++20-era refactor note (templated lambda) rather than a product capability gap. It is low-priority maintainability cleanup, not functional missing support.

### [ ] 940. `lib/Dialect/OM/Transforms/FreezePaths.cpp:170`
`FreezePaths` does not yet support instance choices when resolving instance targets. What is missing is path-freezing support for multi-choice instance references. The fix is to add selection semantics and corresponding access-path generation.

### [ ] 941. `lib/Dialect/OM/Transforms/FreezePaths.cpp:173`
`unsupported instance operation` is the current failure path when multiple referenced modules are present. This is a real boundary tied to missing instance-choice support. The fix is implementation of multi-target instance handling with deterministic path resolution.

### [ ] 942. `lib/Dialect/HW/HWTypes.cpp:595`
`hw.array` parsing currently accepts only integer/param expression/ref dimensions and rejects other dimension forms. What is missing is broader supported dimension kinds if the language surface intends them. The fix is to extend parse/type semantics or clearly document this restricted grammar.

### [ ] 943. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1480`
SCF-to-Calyx rejects empty `scf.yield` outside selected parent ops, so some CFG/region contexts are not handled. What is missing is generalized empty-yield handling across additional supported region parents. The fix is to broaden parent-op coverage and keep explicit diagnostics for truly invalid placements.

### [ ] 944. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1683`
`llvm_unreachable("unsupported comparison predicate")` is a defensive sink after enumerating known `CmpIPredicate` values, not a currently observed unsupported feature by itself. This is non-actionable unless predicate space expands without corresponding lowering updates.

### [ ] 945. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:1833`
This TODO is design-decision debt: whether the `ExecuteRegion` inlining pattern should remain in the lowering strategy. What is missing is a clear keep/remove decision backed by performance/correctness evidence. The fix is to evaluate and either retire or fully commit with dedicated tests.

### [ ] 946. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2029`
Do-while-like `scf.while` forms (mutated iter args in `before`) are explicitly unsupported. What is missing is support for transformed iter-args and broader while argument/result typing semantics. The fix is to model these variants in while-group lowering.

### [ ] 947. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2494`
CFG scheduling currently assumes at most two successors (conditional branches) and punts richer branching forms such as switch-style control flow. What is missing is multi-way branch lowering support or mandatory prior canonicalization enforcement. The fix is to support N-way branching or require/verify pre-lowering.

### [ ] 948. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2861`
This TODO references issue #7764 in caller/callee memref-arg rewriting logic. What is missing is completion of that known restructuring path so this transformation no longer needs a marked caveat. The fix is to resolve the issue and lock behavior with targeted functional tests.

### [ ] 949. `lib/Conversion/SCFToCalyx/SCFToCalyx.cpp:2878`
The `TypeSwitch` fatal-error default is currently unreachable given pre-filtered op kinds (`alloca/alloc/get_global`) and acts as a defensive guard. This is not standalone implementation debt at this line unless upstream op-set assumptions change.

### [x] 950. `test/Tools/circt-sim/syscall-save-restart-warning.sv:5`
Status update (2026-02-28): this entry is stale. Focused regression execution now passes (`test/Tools/circt-sim/syscall-save-restart-warning.sv`) and the test explicitly checks for warning output, so the silent-drop behavior is no longer current.

### [ ] 951. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:586`
This is explanatory comment text in an annotation allowlist path, not a TODO or unsupported boundary by itself. It describes why certain annotations can be silently dropped after earlier passes. Non-actionable scanner hit.

### [ ] 952. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1136`
Handling of force-name annotations assumes non-local form and leaves local-annotation behavior unresolved. What is missing is defined compatibility behavior for local force-name annotations. The fix is to align with SFC semantics (or document divergence) and add verification/regression coverage.

### [ ] 953. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1151`
Missing-NLA handling is deferred to later annotation verification instead of being guaranteed earlier. What is missing is a stronger verification contract that catches broken non-local references before lowering. The fix is to enforce annotation verification ordering and keep this path as defensive diagnostics.

### [ ] 954. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1165`
Conflicting force-name behavior is currently strict-error, with TODO noting possible overwrite or module-duplication semantics for compatibility. What is missing is a settled policy for duplicate/competing force names. The fix is to choose deterministic semantics and test for compatibility with upstream expectations.

### [ ] 955. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1565`
Unconnected output-like cases currently return without synthesizing explicit poison/constant behavior. What is missing is a concrete lowering artifact for unconnected cases (`sv.constant` noted). The fix is to materialize explicit fallback values and validate downstream semantics.

### [ ] 956. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:1858`
Inner symbol lowering does not currently guard against symbol collisions or rewrite dependent `InnerRefAttr`s. What is missing is collision-safe renaming with back-reference updates. The fix is to introduce namespace collision handling plus remap bookkeeping.

### [ ] 957. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:2611`
Bundle connect lowering rejects partial-connect cases by requiring equal element counts. What is missing is partial bundle connect semantics in recursive lowering. The fix is field-wise partial connect support (or explicit pre-expansion) with targeted tests.

### [ ] 958. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:2856`
This TODO is naming/ownership cleanup: printf-specific state flag is now used for broader file-descriptor lowering behavior. What is missing is generalized naming/config separation for FD-related lowering. The fix is to refactor naming without breaking existing macro users.

### [ ] 959. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:3700`
Memory lowering still requires prior aggregate-lowering for bundle memories. What is missing is preservation/lowering support for aggregate memories directly in this phase. The fix is to remove the restriction by extending memory-type lowering coverage.

### [ ] 960. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4224`
Elementwise logical FIRRTL ops are lowered via bitcast-to-int workaround instead of first-class HW elementwise ops. What is missing is dedicated elementwise operations in HW/SV-level IR. The fix is to introduce proper ops and retire bitcast abuse.

### [ ] 961. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4560`
Invalid-value lowering currently hardcodes zero while TODO notes potential randomized semantics. What is missing is spec-aligned policy for invalid initialization value materialization. The fix is to reconcile with FIRRTL spec/tooling behavior and encode it as an explicit lowering option.

### [ ] 962. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:4576`
`Invalid` lowering still rejects some aggregate/type forms with `unsupported type`. What is missing is complete type coverage for invalid-value materialization. The fix is to extend invalid lowering beyond currently supported ground/bitcastable cases.

### [ ] 963. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5099`
Format substitution lowering supports only selected fstring substitution ops (`Time`, `HierarchicalModuleName`) and errors on others. What is missing is full substitution-op lowering coverage. The fix is to implement remaining substitutions and keep precise diagnostics for unknown forms.

### [ ] 964. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5102`
This note location is the companion diagnostic attachment for entry 963, pointing to the unsupported substitution op. It reflects a real capability boundary but is not a separate root cause.

### [ ] 965. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5292`
Verification flavor is currently partially controlled via per-op format attributes, with TODO noting this should be pass-level lowering configuration. What is missing is clear separation between IR semantics and lowering-policy knobs. The fix is to move flavor selection to pass options.

### [ ] 966. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp:5444`
UNR-only assume lowering relies on stringly guard inspection rather than structured metadata. What is missing is a dedicated attribute/flag that marks UNR-only intent. The fix is to formalize this as typed IR metadata and remove string matching.

### [ ] 967. `lib/Dialect/OM/Evaluator/Evaluator.cpp:208`
Diagnostic streaming for actual parameters requires manual loop formatting because direct stream insertion is broken. What is missing is a reliable printer/formatter path for parameter lists in diagnostics. The fix is either operator overload support or shared helper formatting utilities.

### [x] 968. `lib/Dialect/Sim/SimOps.cpp:97`
Status update (2026-02-28): this duplicate of entry 336 is closed as scanner false-positive noise.

### [x] 969. `lib/Dialect/Sim/SimOps.cpp:100`
Status update (2026-02-28): same closure as entry 968.

### [ ] 970. `lib/Dialect/Handshake/Transforms/Materialization.cpp:83`
Pass currently infers “already erased/replaced” ops by dialect/op-kind checks, with TODO questioning robustness of that indicator. What is missing is explicit marking or tracking for replaced ops during materialization. The fix is to use a stronger state marker instead of heuristic op-type filtering.

### [ ] 971. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:322`
Driver tracing stops at unary primops unless explicitly extended. What is missing is optional look-through for additional single-driver unary ops when needed by analyses. The fix is controlled extension of look-through rules with safety constraints.

### [ ] 972. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:333`
`walkDrivers` is asserted/tested only for passive types and leaves flipped-field behavior unresolved. What is missing is explicit semantics for traversing aggregates with reverse-flow fields. The fix is to define and implement flip-aware driver filtering/traversal.

### [ ] 973. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:452`
Same limitation as entry 971 in the `FieldRef`-based traversal path: unary look-through is intentionally conservative and incomplete. The fix is to extend this path consistently if downstream analyses require it.

### [ ] 974. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:688`
`llvm_unreachable("unsupported type")` in field-name path construction indicates unsupported aggregate class handling beyond bundle/vector/class. What is missing is support (or graceful fallback) for additional aggregate kinds that can carry field IDs. The fix is broader type-case coverage.

### [ ] 975. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:726`
`getValueByFieldID` uses fatal error on unrecognized types and TODO notes missing error plumbing. What is missing is recoverable error propagation to callers. The fix is to return failure/diagnostic status instead of hard-failing.

### [ ] 976. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:768`
Enum treatment in `walkGroundTypes` remains semantically ambiguous (aggregate vs ground traversal behavior). What is missing is a clear enum traversal contract for fieldID generation and callers. The fix is to define enum semantics and align traversal implementation accordingly.

### [ ] 977. `lib/Dialect/FIRRTL/FIRRTLUtils.cpp:827`
Inner symbol property lists are appended then sorted each time; TODO calls out better sorted storage/search behavior. What is missing is insertion strategy that preserves sortedness and avoids full re-sort. The fix is ordered insertion/binary-search maintenance.

### [ ] 978. `lib/Dialect/Handshake/Transforms/LockFunctions.cpp:61`
The RAUW pattern (`replaceAllUsesExcept`) has an explicit “is this UB?” concern in lock-function rewriting. What is missing is proof of IR mutation safety or a safer rewrite idiom. The fix is to validate dominance/use constraints and replace with a guaranteed-safe rewrite pattern if needed.

### [ ] 979. `lib/Dialect/Handshake/HandshakeUtils.cpp:276`
`NoneType` wrapper logic has a temporary bridge comment tied to future handshake move to `i0`. What is missing is finalized canonical zero-width/none representation across handshake/ESI boundary utilities. The fix is to switch helper behavior when the handshake type migration lands.

### [ ] 980. `test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:7`
This is test-fixture script text containing fingerprint token `xxxx9999...`, not unresolved implementation debt. The scan hit is from generic marker matching in expected-output setup lines. Non-actionable scanner false positive.

### [ ] 981. `test/Tools/run-formal-all-opentitan-connectivity-contract-parity-fail.test:24`
This line is expected-output fixture content containing a synthetic fingerprint token (`xxxx9999...`), not unresolved product debt. Non-actionable scanner false positive.

### [ ] 982. `lib/Dialect/Handshake/HandshakeOps.cpp:95`
Handshake index verification currently supports only integer/index-typed index values and rejects other index carrier types. What is missing is broader index-type handling (or canonicalization) for ops that use non-standard index forms. The fix is to extend accepted types or normalize before verification.

### [ ] 983. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:4`
This `RUN: printf ... xxxx9999...` line is test setup data, not an unsupported-feature marker. It is a scanner false positive from token matching inside fixture payloads.

### [ ] 984. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:5`
Same as entry 983: allowlist test input text intentionally includes synthetic fingerprint IDs and is non-actionable for implementation-gap tracking.

### [ ] 985. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-allowlist.test:12`
This `PARITY:` expectation line is fixture output matching, not unresolved work. Scanner false positive due generic marker matching in test expectations.

### [ ] 986. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:4`
Same class as entries 983/984: synthetic fingerprint test input (`xxxx9999...`) in setup script, non-actionable.

### [ ] 987. `test/Tools/check-opentitan-connectivity-contract-fingerprint-parity-fail.test:12`
Expected parity-output check line using synthetic fingerprint token, not TODO/unsupported debt. Non-actionable scanner hit.

### [ ] 988. `test/Tools/circt-sim/tlul-bfm-a-ready-timeout-short-circuit.sv:2`
This test documents a real circt-sim behavior gap: timeout-path task-level `$display` in looped BFM code is not printed as expected. What is missing is correct task/loop display execution semantics in this path. The fix is simulator runtime semantics correction plus functional regression validation.

### [ ] 989. `lib/Dialect/MSFT/ExportQuartusTcl.cpp:80`
Quartus Tcl export logic is currently device/toolchain-specific (Stratix 10 + QuartusPro assumptions). What is missing is generalized device-family/tool edition support. The fix is parameterized emission keyed by device/tool capabilities.

### [ ] 990. `lib/Dialect/Moore/Transforms/LowerConcatRef.cpp:75`
ConcatRef lowering assumes one packed range orientation and does not account for alternate index direction (`[0:7]` vs `[7:0]`). What is missing is range-direction-aware slice extraction. The fix is to compute extraction offsets from declared packed range order.

### [ ] 991. `lib/Dialect/FIRRTL/FIRRTLReductions.cpp:1821`
Namespace use here is retained for compatibility because some FIRRTL passes still assume unique port names. What is missing is consistent downstream behavior that removes this uniqueness crutch. The fix is pass cleanup to eliminate hidden uniqueness assumptions.

### [ ] 992. `lib/Dialect/Moore/Transforms/CreateVTables.cpp:151`
This comment documents intentional handling: pure-virtual/unimplemented methods are skipped when populating own vtable entries. It is policy/explanatory text, not unresolved TODO at this location.

### [ ] 993. `lib/Dialect/Moore/Transforms/CreateVTables.cpp:168`
Same as entry 992: leaving unimplemented slots null with runtime warning is described intentional behavior, not standalone gap debt at this line.

### [ ] 994. `lib/Dialect/MSFT/DeviceDB.cpp:357`
Placement walk currently performs filter-and-optional-sort because backing data structures are not order-optimized. What is missing is indexed/sorted storage for efficient bounded traversal. The fix is data-structure redesign for ordered queries.

### [x] 995. `test/Tools/circt-sim/syscall-ungetc.sv:2`
Status update (2026-02-28): this gap is closed and stale in this workspace. Current runtime behavior correctly implements `$ungetc` pushback semantics, and the regression verifies both pushback replay and return code.

### [x] 996. `lib/Dialect/Sim/Transforms/ProceduralizeSim.cpp:106`
Status update (2026-02-28): this gap is closed in this workspace. `--sim-proceduralize` now accepts format-string block arguments by threading them as `hw.triggered` region arguments and using mapped values directly during procedural print emission. Regression coverage was added in `test/Dialect/Sim/proceduralize-sim.mlir` (`@print_blockarg_fstring`).

### [x] 997. `lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:80`
Status update (2026-02-28): this duplicate of entry 347 is closed for scalar floating-point support. `LowerDPIFunc` now allows integer and float port types, validated by `test/Dialect/Sim/lower-dpi-float.mlir`.

### [x] 998. `lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:83`
Status update (2026-02-28): this line-level boundary changed with the float-support update. The transform no longer enforces an integer-only restriction; it now emits `unsupported DPI argument type` only for still-unsupported non-scalar type classes.

### [x] 999. `lib/Dialect/Sim/Transforms/LowerDPIFunc.cpp:100`
Status update (2026-02-28): this duplicate of entry 348 is now closed. `LowerDPIFunc` enforces function-type compatibility when binding to an existing `func.func` by `verilogName`, with regression coverage in `test/Dialect/Sim/lower-dpi-errors.mlir`.

### [ ] 1000. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:298`
ExportVerilog type-dimension collection still rejects some verilog-emittable type forms with unsupported-type diagnostics. What is missing is complete type-shape support in dimension extraction. The fix is to extend `getTypeDims` coverage.

### [ ] 1001. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:726`
Bit-select inlining legality is conservative and skips operators like concat that could be handled. What is missing is broader composable-expression support in this legality check. The fix is to whitelist additional safe expression forms.

### [ ] 1002. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:1426`
Temporary-name inference only handles limited structural patterns. What is missing is additional common expression-pattern naming heuristics. The fix is to extend inference rules for more operations.

### [ ] 1003. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:1864`
Packed-type printer falls back to unsupported verilog type diagnostics for unhandled types. What is missing is print support for remaining type variants that should be representable in Verilog output. The fix is to add those type branches in printer/type emission logic.

### [ ] 1004. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:1964`
Parameter float emission currently relies on default float-to-string formatting precision. What is missing is deterministic/round-trip-safe numeric formatting policy. The fix is to use controlled precision formatting and add reproducibility tests.

### [ ] 1005. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:1977`
Signedness handling for parameter references is TODO and currently defaults unsigned treatment in this path. What is missing is signed parameter reference semantics in expression printing. The fix is to track/propagate parameter signedness metadata.

### [ ] 1006. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2082`
Param expression printer inserts sign casts conservatively and may emit redundant wrappers. What is missing is simplification/cast-elision comparable to mainline expression emitter. The fix is targeted redundancy elimination in param-emission path.

### [ ] 1007. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2111`
Add-expression pretty-print simplification handles negative integer literals but not symbolic `x * -1` forms. What is missing is broader algebraic normalization in parameter expression output. The fix is to fold additional subtract-equivalent patterns.

### [ ] 1008. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2504`
Expression binary emitter does not support SV attribute emission on ops and currently errors. What is missing is attribute-preserving emission for attributed expressions. The fix is SV attribute serialization support in expression emitters.

### [ ] 1009. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2512`
Binary expression grouping still uses local heuristics and TODO notes missing precedence/fixity tree construction. What is missing is robust tree-level pretty-print grouping to avoid awkward wrapping. The fix is precedence-aware grouping across same-level associative chains.

### [ ] 1010. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2529`
`MLIR should have general Associative trait` is an upstream infrastructure wish, not a direct local unsupported feature marker. It is non-local cleanup debt rather than immediate product-gap work in this file.

### [ ] 1011. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2564`
Expression emitter still rejects SV attributes on unary expressions (`hasSVAttributes(op)` hard-errors). What is missing is attribute-preserving emission for expression-level ops. The fix is to integrate `emitSVAttributes` support across all relevant expression visitors.

### [ ] 1012. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2781`
Same root gap as entry 1011 in `ExtractOp` emission: SV attributes on this expression form are currently unimplemented.

### [ ] 1013. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2809`
Same root gap as entry 1011 for `GetModportOp` expression emission with SV attributes.

### [ ] 1014. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2819`
Same root gap as entry 1011 for `SystemFunctionOp` expression emission with SV attributes.

### [ ] 1015. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2833`
Same root gap as entry 1011 for `ReadInterfaceSignalOp` expression emission with SV attributes.

### [ ] 1016. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2844`
Same root gap as entry 1011 for `XMROp` expression emission with SV attributes.

### [ ] 1017. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2854`
`XMRRefOp` name-resolution logic duplicates code from remote-name lookup path. What is missing is shared helper abstraction for XMR naming to avoid drift/bugs. The fix is refactor and deduplicate this logic.

### [ ] 1018. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2858`
Same root gap as entry 1011 for `XMRRefOp` expression emission with SV attributes.

### [ ] 1019. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2885`
Same root gap as entry 1011 for verbatim expression emission with SV attributes.

### [ ] 1020. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2897`
Same root gap as entry 1011 for macro-call expression emission with SV attributes.

### [ ] 1021. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2925`
Same root gap as entry 1011 for `ConstantXOp` expression emission with SV attributes.

### [ ] 1022. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2934`
Same root gap as entry 1011 for `ConstantStrOp` expression emission with SV attributes.

### [ ] 1023. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2942`
Same root gap as entry 1011 for `ConstantZOp` expression emission with SV attributes.

### [ ] 1024. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2981`
Same root gap as entry 1011 for integer `ConstantOp` expression emission with SV attributes.

### [ ] 1025. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:2989`
General expression emitter treats zero-width constants as unsupported outside special-case paths. What is missing is robust generic zero-width constant emission policy. The fix is either representational fallback or full zero-width-aware expression handling.

### [ ] 1026. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3018`
Struct constant emission currently elides zero-width fields and TODO notes comment-style emission would be preferable. What is missing is richer pretty-printing that preserves zero-width field context without invalid syntax. The fix is optional comment-annotated zero-width field emission.

### [ ] 1027. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3082`
Same root gap as entry 1011 for `AggregateConstantOp` expression emission with SV attributes.

### [ ] 1028. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3094`
Same root gap as entry 1011 for `ParamValueOp` expression emission with SV attributes.

### [ ] 1029. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3107`
Same root gap as entry 1011 for `ArraySliceOp` expression emission with SV attributes.

### [ ] 1030. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3135`
Same root gap as entry 1011 for `ArrayCreateOp` expression emission with SV attributes.

### [ ] 1031. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3158`
Same root gap as entry 1011 for `UnpackedArrayCreateOp` expression emission with SV attributes.

### [ ] 1032. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3168`
Same root gap as entry 1011 for `ArrayConcatOp` expression emission with SV attributes.

### [ ] 1033. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3176`
Same root gap as entry 1011 for `ArrayIndexInOutOp` expression emission with SV attributes.

### [ ] 1034. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3191`
Same root gap as entry 1011 for `IndexedPartSelectInOutOp` expression emission with SV attributes.

### [ ] 1035. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3207`
Same root gap as entry 1011 for `IndexedPartSelectOp` expression emission with SV attributes.

### [ ] 1036. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3223`
Same root gap as entry 1011 for `StructFieldInOutOp` expression emission with SV attributes.

### [ ] 1037. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3233`
Same root gap as entry 1011 for `SampledOp` expression emission with SV attributes.

### [ ] 1038. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3243`
Same root gap as entry 1011 for `SFormatFOp` expression emission with SV attributes.

### [ ] 1039. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3248`
This TODO is formatting-quality debt: line breaking around `$sformatf` substitution lists is suboptimal. What is missing is targeted break control after commas without forcing over-grouping. The fix is pretty-printer break-policy improvement for variadic call-like emissions.

### [ ] 1040. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3265`
Same root gap as entry 1011 for `TimeOp` expression emission with SV attributes.

### [ ] 1041. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3273`
Same root gap as entry 1011: SV attributes are not emitted for `STimeOp` expressions and currently trigger a hard error.

### [ ] 1042. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3320`
Same root gap as entry 1011 for `ReverseOp` expression emission with SV attributes.

### [ ] 1043. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3368`
Same root gap as entry 1011 for `StructCreateOp` expression emission with SV attributes.

### [ ] 1044. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3370`
Unpacked-struct emission is still pending in this expression path; TODO notes current pattern/concatenation choice is temporary until unpacked struct support exists. What is missing is full unpacked-struct constant/expression emission semantics.

### [ ] 1045. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3386`
Same root gap as entry 1011 for `StructExtractOp` expression emission with SV attributes.

### [ ] 1046. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3396`
Same root gap as entry 1011 for `StructInjectOp` expression emission with SV attributes.

### [ ] 1047. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3398`
Same underlying TODO as entry 1044: unpacked-struct-aware pattern emission remains unimplemented.

### [ ] 1048. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3423`
Same root gap as entry 1011 for `EnumCmpOp` expression emission with SV attributes.

### [ ] 1049. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3433`
Same root gap as entry 1011 for `UnionCreateOp` expression emission with SV attributes.

### [ ] 1050. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3475`
Same root gap as entry 1011 for `UnionExtractOp` expression emission with SV attributes.

### [ ] 1051. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3498`
This is the generic fallback for unhandled combinational expressions (`<<unsupported expr...>>`). What is missing is visitor coverage for remaining expression ops that reach this path.

### [ ] 1052. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:3719`
Property emitter has a similar generic unsupported fallback (`<<unsupported: ...>>`) for unhandled LTL/property ops. What is missing is broader property/sequence op emission coverage.

### [ ] 1053. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4300`
Statement emitter still rejects SV attributes on `ForceOp`. What is missing is statement-level SV attribute emission support for this op class.

### [ ] 1054. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4307`
Same root gap as entry 1053 for `ReleaseOp` statement emission with SV attributes.

### [ ] 1055. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4325`
Same root gap as entry 1053 for `AliasOp` statement emission with SV attributes.

### [ ] 1056. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4351`
Same root gap as entry 1053 for `InterfaceInstanceOp` statement emission with SV attributes.

### [ ] 1057. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4456`
Same root gap as entry 1053 for `TypedeclOp` statement emission with SV attributes.

### [ ] 1058. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4606`
Same root gap as entry 1053 for `FFlushOp` statement emission with SV attributes.

### [ ] 1059. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4625`
Same root gap as entry 1053 for `FWriteOp` statement emission with SV attributes.

### [ ] 1060. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4639`
Formatting-quality TODO: `$fwrite` substitution arguments lack ideal break behavior when long expressions wrap. What is missing is comma-aware wrap control for this call-like emission pattern.

### [ ] 1061. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4658`
Same root gap as entry 1053 for `VerbatimOp` statement emission with SV attributes.

### [ ] 1062. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4700`
Same root gap as entry 1053 for `MacroRefOp` statement emission with SV attributes.

### [ ] 1063. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4732`
Same root gap as entry 1053 for simulation-control task emission (`$stop/$finish/$exit`) with SV attributes.

### [ ] 1064. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4770`
Same root gap as entry 1053 for severity message task emission (`$fatal/$error/$warning/$info`) with SV attributes.

### [ ] 1065. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4794`
Formatting-quality TODO similar to entry 1060: severity-message argument lists need better comma/wrap behavior.

### [ ] 1066. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4872`
Generate statement emission currently omits location-info handling (`TODO: location info?`). What is missing is consistent location propagation/printing for generated constructs.

### [ ] 1067. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4889`
Same location-info gap as entry 1066 for `GenerateCaseOp` emission.

### [ ] 1068. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:4908`
Generate-case name legalization is currently tracked locally with TODO noting broader storage may be needed for verbose formatting. What is missing is persistent/legalized-name mapping infrastructure across recursive emission contexts.

### [ ] 1069. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5010`
Formatting-quality TODO in assertion message emission: break/box behavior for interpolation args is still under-specified.

### [ ] 1070. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5022`
Same root gap as entry 1053 for immediate assertion emission with SV attributes.

### [ ] 1071. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5071`
Same statement-level root gap as entry 1053: concurrent assertion emission still rejects attached SV attributes.

### [ ] 1072. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5116`
Same statement-level root gap as entry 1053 for property-assertion emission with SV attributes.

### [ ] 1073. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5180`
Same statement-level root gap as entry 1053 for `ifdef/ifndef` emission helper with SV attributes.

### [ ] 1074. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5411`
This TODO is formatting/layout debt in reset-branch emission (`if` grouping consistency). What is missing is unified grouping policy with normal `if` emission paths.

### [ ] 1075. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5484`
Case bit-pattern printing currently emits only bit-level form; TODO notes possible hex emission optimization. What is missing is smarter compact literal formatting when nibble-aligned and safe.

### [ ] 1076. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5745`
Interface emission lacks source-location info plumbing. What is missing is consistent location/comment emission for interface-level statements.

### [ ] 1077. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5750`
Current interface body emission reuses generic statement emission, with FIXME indicating semantic mismatch. What is missing is dedicated interface-body emission discipline instead of generic statement block traversal.

### [ ] 1078. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5798`
Modport emission has TODO around break/group behavior. What is missing is stable pretty-printer policy for long modport port lists.

### [ ] 1079. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5816`
Assign-interface-signal emission duplicates assign-like formatting logic. What is missing is refactoring to shared assign emission helper for consistency.

### [ ] 1080. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5836`
Macro definition emission still lacks source-info attachment (`TODO: source info!`). What is missing is consistent location propagation in macro-def output.

### [ ] 1081. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:5989`
Dominance checking currently uses a conservative same-block heuristic instead of MLIR `DominanceInfo`. What is missing is proper dominance analysis integration for declaration-assignment inlining decisions.

### [ ] 1082. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:6098`
Unpacked-array declaration assignment inlining is deliberately disabled due downstream tool support limitations (issue 6363). This is a known portability constraint; enabling requires tool-compatibility strategy.

### [ ] 1083. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:6123`
Same portability constraint as entry 1082 in procedural `logic` declaration inlining: unpacked-array inline syntax remains disabled for compatibility reasons.

### [ ] 1084. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:6230`
Same statement-level root gap as entry 1053 for `bind` emission with SV attributes.

### [ ] 1085. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:6339`
Same statement-level root gap as entry 1053 for interface-bind emission with SV attributes.

### [ ] 1086. `lib/Conversion/ExportVerilog/ExportVerilog.cpp:6951`
Split-output handling for `TypeScopeOp`/typedef placement is still undecided. What is missing is explicit policy for where typedef scopes live when output is partitioned across files.

### [ ] 1087. `lib/Dialect/Seq/SeqOps.cpp:610`
Seq canonicalization still depends on missing HW aggregate constant support and keeps workaround logic in canonicalization instead of folding. What is missing is aggregate-constant support sufficient to move this into the folder.

### [ ] 1088. `lib/Dialect/Seq/SeqOps.cpp:693`
Seq register canonicalization currently only supports simple 1D array cases. What is missing is nested array/bundle support in this optimization path.

### [ ] 1089. `lib/Dialect/Seq/SeqOps.cpp:743`
`FirRegOp::fold` still does not handle preset values. What is missing is preset-aware folding semantics for constant/reset cases.

### [ ] 1090. `test/Tools/circt-sim/syscall-getpattern.sv:2`
This test documents current non-implementation of legacy `$getpattern` (returns 0). It is a real runtime capability boundary if parity with legacy system tasks is desired.

### [x] 1091. `test/Tools/circt-sim/syscall-randomize-with.sv:2`
Status update (2026-02-28): same closure as entry 420. This test passes in focused execution and no longer demonstrates an active inline-constraint gap.

### [x] 1092. `test/Tools/circt-sim/syscall-generate.sv:2`
Status update (2026-02-28): same closure as entry 440. Focused regression now passes and does not reproduce the documented width/padding mismatch.

### [x] 1093. `test/Tools/circt-sim/syscall-random.sv:2`
Status update (2026-02-28): same closure as entry 422. Focused regression now passes and does not reproduce the documented seed-mutation mismatch.

### [x] 1094. `test/Tools/circt-sim/syscall-fread.sv:2`
Status update (2026-02-28): same closure as entry 441. Focused regression now passes, so this test is no longer evidence of missing `$fread` support.

### [x] 1095. `test/Tools/circt-sim/syscall-queue-stochastic.sv:8`
Status update (2026-02-28): this line-level entry is stale. It is expected diagnostic coverage for intentionally unsupported legacy stochastic queue tasks, and focused regression still passes.

### [x] 1096. `test/Tools/circt-sim/syscall-feof.sv:2`
Status update (2026-02-28): same closure as entry 455. Focused regression now passes and does not reproduce the previously documented integration failure.

### [ ] 1097. `lib/Dialect/Seq/Transforms/LowerSeqHLMem.cpp:67`
LowerSeqHLMem currently rejects unsupported memory handle user port kinds. What is missing is broader port-type lowering support beyond read/write ports.

### [ ] 1098. `lib/Dialect/Seq/Transforms/LowerSeqHLMem.cpp:120`
Read-latency lowering TODO notes missing optimization tradeoff for latency > 2 (address buffering vs data buffering). What is missing is area/timing-aware buffering strategy.

### [ ] 1099. `lib/Conversion/ExportVerilog/PrepareForEmission.cpp:218`
Temporary wire naming still uses result index suffixes; TODO suggests using result names for better readability/stability. What is missing is result-name-aware naming policy.

### [ ] 1100. `lib/Conversion/ExportVerilog/PrepareForEmission.cpp:728`
This TODO is implementation-style cleanup (`consider virtual functions`) in heuristic dispatch, not a direct user-visible feature gap.

### [ ] 1101. `lib/Conversion/ExportVerilog/PrepareForEmission.cpp:787`
Prettification/wire-spilling currently skips procedural regions entirely. What is missing is equivalent post-legalization cleanup for procedural blocks.

### [ ] 1102. `lib/Conversion/ExportVerilog/PrepareForEmission.cpp:1210`
Balanced-tree lowering currently keys off `Commutative`, which is broader than the intended “fully associative” condition. What is missing is a correct trait/eligibility criterion for safe variadic balancing.

### [x] 1103. `test/Tools/circt-sim/syscall-q-full.sv:8`
Status update (2026-02-28): this line-level entry is stale. It is expected-error coverage for intentionally unsupported legacy stochastic queue tasks, and focused regression still passes.

### [x] 1104. `test/Tools/circt-sim/syscall-pld-sync-array.sv:14`
Status update (2026-02-28): this line-level entry is stale. It is expected diagnostic coverage for unsupported legacy PLD array tasks, and focused regression still passes.

### [x] 1105. `test/Tools/circt-sim/syscall-pld-array.sv:14`
Status update (2026-02-28): this line-level entry is stale. It is expected unsupported-legacy diagnostic coverage, and focused regression still passes.

### [x] 1106. `test/Tools/circt-sim/syscall-monitor.sv:2`
Status update (2026-02-28): same closure as entry 432. Focused regression now passes and no longer demonstrates one-shot-only monitor behavior.

### [ ] 1107. `lib/Dialect/Seq/Transforms/HWMemSimImpl.cpp:652`
Seq memory simulation lowering duplicates logic from LowerToHW. What is missing is shared helper infrastructure to avoid behavior drift between these paths.

### [x] 1108. `test/Tools/circt-sim/syscall-isunbounded.sv:2`
Status update (2026-02-28): same closure as entry 434. Focused regression now passes and does not reproduce the documented class/type-parameter failure.

### [x] 1109. `test/Tools/circt-sim/syscall-shortrealtobits.sv:2`
Status update (2026-02-28): same closure as entry 418. Focused regression now passes and does not reproduce the documented shortreal conversion mismatch.

### [x] 1110. `test/Tools/circt-sim/syscall-strobe.sv:2`
Status update (2026-02-28): same closure as entry 409. Focused regression now passes and no longer demonstrates immediate-value `$strobe` behavior.

### [ ] 1111. `lib/Dialect/SV/SVOps.cpp:285`
`ConstantXOp` verifier rejects zero/unknown-width constants by design. This is dialect invariant enforcement, not unresolved implementation debt.

### [ ] 1112. `lib/Dialect/SV/SVOps.cpp:300`
Same as entry 1111 for `ConstantZOp`: intentional verifier guard, non-actionable from gap perspective.

### [ ] 1113. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:211`
`llvm_unreachable("Unsupported Flow type.")` is an enum-exhaustiveness sentinel in `swapFlow`, not a known user-facing unsupported feature.

### [ ] 1114. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:226`
Same as entry 1113 for `toString(Flow)`: defensive unreachable path, non-actionable unless Flow enum expands without updates.

### [ ] 1115. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:304`
`getDeclarationKind` and `foldFlow` duplicate structural walks. What is missing is shared traversal returning combined flow/decl-kind data.

### [ ] 1116. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:1331`
This is the recurring upstream printer-location alias issue. Missing behavior is upstream `printOptionalLocationSpecifier` correctness, not local FIRRTL logic.

### [ ] 1117. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:1676`
Parser path still routes visibility through attributes instead of op properties. What is missing is full properties-based representation.

### [ ] 1118. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:2100`
Same properties-migration debt as entry 1117 in class-like parsing.

### [ ] 1119. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:3461`
Memory-port verification leaves missing-read-flip handling unresolved. What is missing is explicit verifier erroring/semantics for missing expected flip on read data fields.

### [ ] 1120. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4076`
Connect flow verifier remains stricter than desired for some read-from-output/input-port cases. What is missing is relaxed-but-safe flow rules for those declarations.

### [ ] 1121. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4321`
Ref-define verifier has unresolved policy about `ref.sub` destination flow constraints. What is missing is finalized source/sink rules for sub-reference destinations.

### [ ] 1122. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4949`
`BundleCreateOp` verification still skips flow checking. What is missing is full flow validation for aggregate construction operands.

### [ ] 1123. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:4962`
Same as entry 1122 for `VectorCreateOp`: missing flow checks in verifier.

### [ ] 1124. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6791`
`ref.sub` with rwprobe behavior remains unresolved and explicitly noted for further semantics/testing. What is missing is a finalized policy plus tests.

### [ ] 1125. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6923`
`XMRRefOp` symbol-use verification does not validate referenced target type compatibility. What is missing is type matching between op result and hierpath target.

### [ ] 1126. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:6935`
Same as entry 1125 for `XMRDerefOp`: missing target type compatibility verification.

### [ ] 1127. `lib/Dialect/FIRRTL/FIRRTLOps.cpp:7023`
Layer-block verifier intentionally leaves a hole for non-passive connect-like flow validation. What is missing is a complete verifier once flip-removal/canonicalization assumptions are not required.

### [ ] 1128. `lib/Dialect/ESI/ESIServices.cpp:98`
ESI cosim service lowering currently explodes bundles manually per channel and notes missing direct bundle-capable cosim op semantics. What is missing is first-class bundle support in cosim op + downstream lowering.

### [ ] 1129. `lib/Dialect/SV/Transforms/SVExtractTestCode.cpp:82`
Backward-slice extraction has unresolved recursion policy at block arguments across multi-block regions. What is missing is a defined traversal contract for cross-block parent-region backtracking.

### [ ] 1130. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:512`
FIR emitter still falls back to `<unsupported-attr ...>` for unhandled parameter attributes. What is missing is serialization support for the remaining parameter attribute kinds.

### [ ] 1131. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:513`
FIR emitter still falls back to `<unsupported-attr ...>` for unhandled parameter attribute kinds. What is missing is full parameter-attribute serialization coverage.

### [ ] 1132. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:956`
Format-string emission supports only selected substitution op kinds and errors on others. What is missing is complete fstring substitution lowering in FIREmitter.

### [ ] 1133. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1229`
Memory export currently rejects FIRRTL memory debug ports. What is missing is emission support for debug-port semantics in textual FIRRTL output.

### [ ] 1134. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1282`
`MemoryPortAccessOp` emission duplicates assign-like formatting logic. What is missing is a shared `emitAssignLike`-style helper for consistency and maintenance.

### [ ] 1135. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1386`
`InvalidValueOp` emission path has the same assign-like formatting debt as entry 1134. What is missing is shared statement-emission plumbing.

### [ ] 1136. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1457`
Expression emission has a generic unsupported fallback (`<unsupported-expr-...>`). What is missing is expression visitor coverage for remaining operation kinds.

### [ ] 1137. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1465`
Constant expression printing lacks configurable literal base policy (bin/oct/dec/hex). What is missing is user-selectable numeric base emission options.

### [ ] 1138. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1477`
Type alias declarations are not emitted in this special-constant path. What is missing is type-decl emission for aliased types.

### [ ] 1139. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1743`
Same alias-declaration gap as entry 1138 in general type emission: alias-aware type decl output remains TODO.

### [ ] 1140. `lib/Dialect/FIRRTL/Export/FIREmitter.cpp:1843`
Location printing does not yet handle fused locations or location uniquing, leading to redundant filename output. What is missing is richer location canonicalization during emission.

### [ ] 1141. `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:463`
InOut port elimination currently bails when encountering unsupported access-use operations on an inout path. What is missing is broader operation support in access collection/rewrite.

### [ ] 1142. `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:527`
Multiple-writer inout cases are still unsupported in some configurations (unless explicitly resolvable/same-value). What is missing is full multi-writer resolution semantics.

### [ ] 1143. `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp:998`
`hw.inout` outputs are guarded by an assert/FIXME path and not genuinely supported in this conversion flow. What is missing is explicit output-inout lowering support without relying on fragile direction assumptions.

### [ ] 1144. `lib/Conversion/DatapathToComb/DatapathToComb.cpp:286`
Datapath multiplier lowering notes missed opportunity to use concatenation forms that better aid longest-path analysis. What is missing is analysis-friendly structural lowering here.

### [ ] 1145. `lib/Conversion/DatapathToComb/DatapathToComb.cpp:293`
Booth preprocessing does not yet choose operand ordering based on sparsity/non-zero bits. What is missing is operand-selection heuristic for smaller/faster encoding.

### [ ] 1146. `lib/Conversion/DatapathToComb/DatapathToComb.cpp:466`
Pos partial-product lowering still lacks Booth implementation in this path. What is missing is Booth-mode lowering support.

### [ ] 1147. `lib/Conversion/DatapathToComb/DatapathToComb.cpp:571`
Greedy rewrite with timing info does not enforce topological op processing order first. What is missing is explicit topo ordering to guarantee dependency-first handling.

### [ ] 1148. `lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:167`
Array-assignment splitting is limited and TODO notes missing per-field decomposition generalization. What is missing is fuller assignment normalization for array structures.

### [ ] 1149. `lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:184`
Concat/slice recognition is not generalized to ranges and arbitrary concatenations. What is missing is broader pattern support in prettification rewrite logic.

### [ ] 1150. `lib/Dialect/SV/Transforms/PrettifyVerilog.cpp:351`
Prettify pass uses ad hoc conditions where TODO suggests reusing ExportVerilog’s event-control inlining criteria. What is missing is unified policy between these passes.

### [ ] 1151. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:541`
Some hand-written fold patterns are still marked for migration to DRR. What is missing is declarative rewrite coverage for these canonical folds.

### [ ] 1152. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:854`
`EQ` folding does not fully support `SInt<1>` and related edge cases on LHS. What is missing is complete one-bit signed folding coverage.

### [ ] 1153. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:904`
Same as entry 1152 for `NEQ` folding: incomplete `SInt<1>` coverage.

### [ ] 1154. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:950`
`IntegerAddOp` fold path still lacks constant folding implementation. What is missing is basic constant-fold support for integer add.

### [ ] 1155. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:956`
`IntegerMulOp` fold path likewise lacks constant folding. What is missing is integer multiply constant-fold support.

### [ ] 1156. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1657`
Mux fold TODO: `x ? ~0 : 0 -> sext(x)` optimization is not yet implemented. What is missing is this signed-extension fold.

### [ ] 1157. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1658`
Mux fold TODO tracks additional constant-pattern rewrites (`x ? c1 : c2`). What is missing is broader mux constant simplification set.

### [ ] 1158. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1660`
Mux fold TODO: `x ? a : 0 -> sext(x) & a` is not implemented. What is missing is this canonical masked-form rewrite.

### [ ] 1159. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:1663`
Mux fold TODO: canonical swap/rewrite forms like `x ? c1 : y -> ~x ? y : c1` remain unimplemented.

### [ ] 1160. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2042`
`MultibitMuxOp` canonicalization to two-way mux is currently restricted to `uint<1>` index types. What is missing is equivalent canonicalization for wider-compatible index representations.

### [ ] 1161. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2155`
`MatchingConnectOp` canonicalization still lacks normalization toward explicit extension/flip forms. What is missing is a stronger canonical form that reduces downstream pattern variance in connect rewrites.

### [ ] 1162. `lib/Dialect/FIRRTL/FIRRTLFolds.cpp:2207`
Attach canonicalization drops single-use wires but still has unresolved annotation-policy nuance beyond current `dontTouch` checks. What is missing is finalized annotation-safety rules for wire-elimination in attach folding.

### [ ] 1163. `lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:1`
This is a file-banner phrase ("lower unsupported IR features away"), not unresolved work. It is scanner noise from header comments, not an actionable implementation gap.

### [ ] 1164. `lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:10`
Same as entry 1163: this is descriptive pass documentation about tool constraints, not a TODO/debt marker by itself.

### [ ] 1165. `lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:357`
Real legalization coverage boundary: packed-array users outside the recognized lowering patterns error out. What is missing is broader packed-array expression lowering coverage in `HWLegalizeModules`.

### [ ] 1166. `lib/Dialect/SV/Transforms/HWLegalizeModules.cpp:438`
Same underlying gap as entry 1165 at result-type validation time: operations still producing packed arrays after legalization are rejected. What is missing is full lowering of residual packed-array producers under disallow settings.

### [ ] 1167. `lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:227`
Real annotation resolution debt: path checks use a temporary reference-type guard and still defer robust `containsReference()` handling. What is missing is generalized reference-containing target validation.

### [ ] 1168. `lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:268`
Same as entry 1167 for instance-port target resolution: reference-type rejection is incomplete until `containsReference()`-style checks are implemented uniformly.

### [ ] 1169. `lib/Dialect/FIRRTL/FIRRTLAnnotationHelper.cpp:467`
MemTap import currently only accepts `CombMem` sources. What is missing is support (or explicit canonical conversion) for additional memory source forms where MemTap semantics are valid.

### [ ] 1170. `lib/Conversion/DCToHW/DCToHW.cpp:217`
This `@todo` is a refactor note (move helper into support), not a direct user-visible feature gap.

### [ ] 1171. `lib/Dialect/ESI/runtime/python/CMakeLists.txt:76`
Real platform-support gap: nanobind stub generation is disabled on Windows pending DLL-path handling. What is missing is a working Windows stubgen path for the Python runtime wheel flow.

### [ ] 1172. `test/Tools/select-opentitan-connectivity-cfg-invalid-row-kind.test:7`
This is an intentional negative test asserting diagnostics for malformed CSV row kinds, not unresolved implementation debt.

### [ ] 1173. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:578`
This is a deliberate version guard: string-encoded integer literals are rejected for FIRRTL >= 3.0.0. It is policy/spec enforcement, not a parser TODO.

### [ ] 1174. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:754`
Real parser coverage gap: `parseFieldId` still lacks `RelaxedId` handling. What is missing is complete grammar acceptance for relaxed identifiers in field-id positions.

### [ ] 1175. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2122`
This is duplication debt between parser-side invalidate expansion and `LowerTypes` connect expansion logic. What is missing is shared infrastructure or a single canonical expansion owner.

### [ ] 1176. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2167`
Parser still carries a workaround grammar path (`exp '.' DoubleLit`) for historical issue `#470`. What is missing is resolution/removal of this compatibility workaround with clear spec-conformant behavior.

### [ ] 1177. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:2426`
Same as entry 1176 in postfix parsing: workaround-specific grammar handling remains and should be retired once the underlying issue is fully resolved.

### [ ] 1178. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:3517`
This is a FIRRTL grammar-cleanup/spec-alignment TODO around `else :` info handling. What is missing is a finalized parser policy matching the intended spec grammar.

### [ ] 1179. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4056`
This is an intentional compatibility extension (`read` accepts `ref_expr` where spec says `static_reference`), not a raw unsupported-feature marker. The remaining work is documenting or gating this spec divergence explicitly.

### [ ] 1180. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4096`
Real layering gap: unsupported probe-source cases are enforced ad hoc in parser code. What is missing is moving these constraints into op verifier/type inference so all construction paths are consistent.

### [ ] 1181. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4097`
Same underlying gap as entry 1180: `ref.send` verifier/inferReturnTypes still does not encode these source restrictions.

### [ ] 1182. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4341`
This TODO tracks missing agg-of-ref support plus missing regression coverage for this specific connect rejection. The feature gap is aggregate-of-reference typing/connect semantics.

### [ ] 1183. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4514`
Same as entry 1182 in the `<=` statement path: agg-of-ref support is absent and this guard lacks the future positive regression it references.

### [ ] 1184. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4722`
This is primarily a spec-documentation ambiguity note (`cmem` undocumented), not direct implementation debt in this parser path.

### [ ] 1185. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4754`
Same as entry 1184 for `smem`: spec/documentation ambiguity marker rather than a concrete missing code path.

### [ ] 1186. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:4900`
Port sorting for memories is implemented in parser code instead of `MemOp` construction/canonicalization. What is missing is centralization of this canonicalization rule in IR construction/canonicalization layers.

### [ ] 1187. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5038`
Real spec-conformance drift: `info` placement around register parsing follows current parser behavior instead of preferred grammar order. What is missing is parser/spec alignment (or explicit compatibility mode definition).

### [ ] 1188. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5059`
Register reset parsing still relies on permissive/ambiguous grammar handling. What is missing is a simplified, unambiguous grammar implementation for reset blocks.

### [ ] 1189. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5082`
This is a compatibility workaround for Scala FIRRTL pretty-printer output of reset-less registers. What is missing is eventual upstream/downstream convergence so this special-case can be removed.

### [ ] 1190. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5339`
`Invalid/unsupported annotation format` is a generic failure path in JSON annotation import. This is mostly a boundary diagnostic, not standalone TODO debt at this location.

### [ ] 1191. `lib/Dialect/FIRRTL/Import/FIRParser.cpp:5932`
Formal-like parsing still supports both legacy `, bound = N` and the newer parameter-block form. What is missing is retirement of the legacy syntax path once migration is complete.

### [ ] 1192. `test/Tools/circt-bmc/commandline.mlir:17`
`CHECK-DAG: --drop-unsupported-sva` is CLI help-text regression coverage, not an unresolved implementation gap by itself.

### [ ] 1193. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:73`
This `assert False` is an abstract-method placeholder in the `ESIType` base class, not a direct missing feature for concrete types. The real cleanup is using explicit abstract-method/`NotImplementedError` style APIs.

### [ ] 1194. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:78`
Same as entry 1193 for `bit_width`: base-class placeholder behavior, primarily API-hardening/refactor debt.

### [ ] 1195. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:90`
Same as entry 1193 for `serialize`: intentional abstract placeholder pattern, not a standalone runtime capability gap.

### [ ] 1196. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:95`
Same as entry 1193 for `deserialize`: abstract-base placeholder semantics that should be tightened to explicit abstract interfaces.

### [ ] 1197. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:464`
`unsupported type` here enforces host-communication constraints (`supports_host`), so this is a deliberate runtime boundary rather than a hidden TODO.

### [ ] 1198. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:527`
Real maintainability gap: service-port specialization still relies on `isinstance` branching in `__new__`. What is missing is a first-class registration/dispatch mechanism for service port wrappers.

### [ ] 1199. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:565`
Real behavior gap: `MessageFuture.result(timeout=...)` ignores the provided timeout. What is missing is timeout-aware waiting/error semantics matching Python `Future` expectations.

### [ ] 1200. `lib/Dialect/ESI/runtime/python/esiaccel/types.py:574`
Real API gap: `add_done_callback` is unimplemented for ESI message futures. What is missing is callback registration and completion notification behavior.

### [ ] 1201. `test/Tools/circt-bmc/circt-bmc-prune-unreachable-hw-before-smt.mlir:11`
This is an intentional negative/behavioral test documenting that `hw.type_scope` is unsupported by HWToSMT unless pruned. It references a real capability boundary but is not new debt at this test line.

### [ ] 1202. `lib/Dialect/FIRRTL/Import/FIRParserAsserts.cpp:13`
This is a file-level policy comment explaining retained detection for intentionally unsupported legacy printf-encoded verif forms. It is descriptive, not a fresh TODO.

### [ ] 1203. `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp:241`
This TODO is algorithmic/perf cleanup in post-order traversal bookkeeping. What is missing is pruning consumed elements during arc extraction to reduce avoidable scanning overhead.

### [ ] 1204. `lib/Dialect/ESI/runtime/python/esiaccel/codegen.py:302`
Real codegen boundary: integer type emission rejects widths above 64 bits. What is missing is wide-integer C++ representation/emission support for larger bit widths.

### [ ] 1205. `lib/Dialect/ESI/runtime/python/esiaccel/codegen.py:373`
Struct bitfield emission currently relies on implementation-defined C++ layout. What is missing is deterministic cross-compiler packing semantics (or explicit pack/unpack scheme).

### [ ] 1206. `lib/Dialect/FIRRTL/Import/FIRLexer.cpp:99`
Real lexer completeness gap: octal and unicode escapes are not handled in string decoding. What is missing is full escape-sequence support matching FIRRTL expectations.

### [ ] 1207. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:365`
IMConstProp still skips `OptimizableExtModuleAnnotation` handling. What is missing is annotation-aware constant-propagation behavior for eligible extmodules.

### [ ] 1208. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:462`
This assert is primarily a defensive invariant (`IntegerAttr` expected for non-property constants), not a standalone TODO marker. The broader gap is clearer typed-lattice handling when new constant kinds appear.

### [ ] 1209. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:539`
Real propagation coverage gap: aggregate operations (e.g. vector/bundle create and related ops) are still conservatively overdefined. What is missing is aggregate-aware lattice propagation.

### [ ] 1210. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:861`
Real precision gap: `when` operations are not modeled in this analysis path. What is missing is control-flow-sensitive handling for conditional regions.

### [ ] 1211. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:957`
Folding results with unsupported constant attribute kinds are conservatively marked overdefined. What is missing is broader constant-attribute interpretation to retain precision.

### [ ] 1212. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1027`
Replacement logic currently works per leaf and explicitly skips whole-aggregate replacement. What is missing is direct aggregate-level constant materialization/rewrite.

### [ ] 1213. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1034`
This is an API layering TODO: capability probing for constant materialization is duplicated locally. What is missing is a queryable `materializeConstant` support contract.

### [ ] 1214. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1035`
Same root as entry 1213: current path guards against materialization asserts instead of using typed support introspection.

### [ ] 1215. `lib/Dialect/FIRRTL/Transforms/IMConstProp.cpp:1056`
Same root as entry 1210 in rewrite phase: `WhenOp`-aware rewriting is incomplete, limiting precision/correctness for conditionally constant values.

### [ ] 1216. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:113`
Real runtime quality gap: known leak paths remain in Python callback integration, with leak warnings suppressed. What is missing is leak-safe lifecycle ownership in callback bindings.

### [ ] 1217. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:209`
Binding completeness gap: `ModuleInfo.extra` is not exposed in the Python API. What is missing is field parity between C++ metadata and Python wrapper surface.

### [ ] 1218. `lib/Dialect/ESI/runtime/python/esiaccel/esiCppAccel.cpp:373`
Real stability gap: callback bridge notes deterministic crash scenarios under certain conditions. What is missing is robust callback lifetime/thread/GIL handling that eliminates these crash modes.

### [ ] 1219. `lib/Dialect/SSP/Transforms/Schedule.cpp:73`
ASAP scheduler currently hard-gates on a single problem type name. What is missing is broader problem-type dispatch (or normalized registration) for SSP scheduling backends.

### [ ] 1220. `lib/Dialect/SSP/Transforms/Schedule.cpp:163`
Same as entry 1219 for simplex scheduling dispatch: unsupported problem-name branches reveal incomplete scheduler coverage/registration for available SSP problem kinds.

### [ ] 1221. `lib/Dialect/SSP/Transforms/Schedule.cpp:201`
LP scheduling currently supports only selected SSP problem kinds and rejects others by name. What is missing is broader problem-type coverage (or extensible registration) for LP backend dispatch.

### [ ] 1222. `lib/Dialect/SSP/Transforms/Schedule.cpp:223`
CPSAT scheduling is similarly hard-gated to a narrow problem set (`SharedOperatorsProblem`). What is missing is wider CPSAT problem support where model translation is valid.

### [ ] 1223. `lib/Dialect/SSP/Transforms/Schedule.cpp:254`
Scheduler selection rejects unknown scheduler strings. This is expected validation, but the capability gap is limited scheduler backend surface unless additional schedulers are implemented/registered.

### [ ] 1224. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:832`
`getMaskType` for bundle elements still carries a `FIXME` around flip handling (`false` forced). What is missing is a finalized policy for direction/flip treatment in mask-type construction.

### [ ] 1225. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1698`
This is a type-system refactor TODO: field-id computation depends on ad hoc `FieldIdImpl` calls instead of a tighter interface contract on element types.

### [ ] 1226. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:1863`
Const-preserving element-type logic still uses manual type switching. What is missing is a shared const-type interface/trait to remove duplicated ad hoc handling.

### [ ] 1227. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2137`
Same as entry 1226 for open vectors: const propagation should move to a reusable type-interface mechanism.

### [ ] 1228. `lib/Dialect/FIRRTL/FIRRTLTypes.cpp:2369`
Enum verification still leaves unresolved policy for excluding reference-containing element types. What is missing is explicit verifier enforcement for this constraint.

### [ ] 1229. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:66`
Module-like verifier still carries an edge-case TODO for zero-port direction bitwidth representation. What is missing is clean zero-port handling without historical APInt assumptions.

### [ ] 1230. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:86`
Port-annotation verification currently tolerates empty annotation arrays with unclear intent. What is missing is a finalized verifier policy for empty-vs-sized port annotation lists.

### [ ] 1231. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:89`
Annotation structure checks are embedded in module-like verification. What is missing is dedicated annotation verifier plumbing to centralize these checks.

### [ ] 1232. `lib/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp:207`
Forceable-result type construction duplicates checks that `RefType` already verifies. What is missing is de-duplication to keep one source of truth for validity constraints.

### [ ] 1233. `lib/Conversion/CombToSynth/CombToSynth.cpp:178`
Unknown-bit analysis for concat trees treats many ops as opaque unknowns. What is missing is additional handling for ops like replicate/extract to improve precision.

### [ ] 1234. `lib/Conversion/CombToSynth/CombToSynth.cpp:488`
Adder-architecture selection remains heuristic-only. What is missing is analysis-driven architecture choice (or cost-model-backed synthesis) under timing/area goals.

### [ ] 1235. `lib/Conversion/CombToSynth/CombToSynth.cpp:703`
Lazy prefix-tree support is currently specific to Kogge-Stone. What is missing is generalization to other parallel-prefix families.

### [ ] 1236. `lib/Conversion/CombToSynth/CombToSynth.cpp:1038`
Real lowering gap: signed division lowering lacks dedicated implementation and falls back to emulation paths.

### [ ] 1237. `lib/Conversion/CombToSynth/CombToSynth.cpp:1056`
Real lowering gap: signed modulus lowering likewise lacks dedicated implementation beyond emulation.

### [ ] 1238. `lib/Conversion/CombToSynth/CombToSynth.cpp:1102`
Prefix-comparison lowering still computes more intermediate prefix values than necessary outside lazy Kogge-Stone path. What is missing is lazy computation across architectures.

### [ ] 1239. `lib/Conversion/CombToSynth/CombToSynth.cpp:1243`
`i0` signed comparison is explicitly rejected. This is a known corner-case capability boundary in signed compare lowering.

### [ ] 1240. `lib/Conversion/CombToSynth/CombToSynth.cpp:1361`
`i0` signed shift is explicitly rejected. This is the analogous corner-case boundary in signed shift lowering.

### [ ] 1241. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:273`
Real metadata-policy TODO: handling of out-of-design memory paths is inconsistent and currently uses sketchy fallback behavior. What is missing is a consistent rule for metadata emission of non-DUT paths.

### [ ] 1242. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:275`
`anyInstanceInEffectiveDesign(...)` here is normal design-membership filtering logic, not unresolved work. This is scanner noise.

### [ ] 1243. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:321`
Real follow-on to entry 1241: unresolvable distinct path placeholders are acknowledged as questionable. What is missing is a principled representation for optimized-away/non-DUT paths.

### [ ] 1244. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:624`
Same as entry 1242: this is ordinary DUT filtering (`anyInstanceInEffectiveDesign`) and not a TODO/debt marker.

### [ ] 1245. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:823`
Same scanner false-positive class as entries 1242/1244: routine effective-design filter logic, non-actionable by itself.

### [ ] 1246. `lib/Dialect/FIRRTL/Transforms/CreateSiFiveMetadata.cpp:909`
Same scanner false-positive class as entries 1242/1244/1245: ordinary in-DUT classification logic.

### [ ] 1247. `lib/Dialect/FIRRTL/Transforms/RemoveUnusedPorts.cpp:85`
Real transform gap: inout ports are not handled by unused-port pruning. What is missing is safe inout-aware port removal logic.

### [ ] 1248. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:420`
Inner-ref user rewriting currently assumes all relevant users are locally mapped. What is missing is robust handling/reporting for non-local or unmapped users during inlining.

### [ ] 1249. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:739`
This is a debug/metadata gap: there is no good mechanism to preserve explicit parent scope annotations on instances through inlining.

### [ ] 1250. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:979`
Inlining supports only a limited set of region-containing ops (`LayerBlockOp/WhenOp/MatchOp`) and errors on others. What is missing is broader region-op inlining support.

### [ ] 1251. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1227`
Real NLA bookkeeping gap: symbol renames created during recursive retop/inlining are not fully propagated for subsequent inline steps. What is missing is end-to-end rename propagation across chained inline operations.

### [ ] 1252. `lib/Dialect/FIRRTL/Transforms/ModuleInliner.cpp:1316`
Same as entry 1251 in the non-recursive path: symbol-rename carry-forward remains incomplete for later `inlineInto` calls.

### [ ] 1253. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:248`
This is explicit architectural tech debt: behavior is coupled to legacy `InjectDUTHierarchyAnnotation` handling while the pass is slated for removal. What is missing is cleanup/removal of this transitional path.

### [ ] 1254. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:376`
Clock-gate extraction still relies on suffix-based defname matching (`EICG_wrapper`). What is missing is a principled annotation- or interface-driven identification mechanism.

### [ ] 1255. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:384`
`anyInstanceInDesign(module)` here is regular DUT-membership filtering logic, not unresolved work. This is scanner noise.

### [ ] 1256. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:391`
Extraction configuration hardcodes the `clock_gate` prefix. What is missing is configurable prefix policy instead of embedded literals.

### [ ] 1257. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:416`
Same as entry 1255: `anyInstanceInDesign(module)` here is standard filtering, non-actionable by itself.

### [ ] 1258. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:423`
Same root as entry 1256: memory extraction uses hardcoded `mem_wiring` prefix. What is missing is configurable naming.

### [ ] 1259. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:544`
`anyInstanceInDesign(parent)` in this termination condition is expected hierarchy filtering, not a TODO/debt marker.

### [ ] 1260. `lib/Dialect/FIRRTL/Transforms/ExtractInstances.cpp:1130`
Same scanner false-positive class as entries 1255/1257/1259: this is routine non-DUT path trimming logic.

### [ ] 1261. `lib/Dialect/FIRRTL/Transforms/InferReadWrite.cpp:57`
InferReadWrite currently relies on a precondition that `WhenOp`s are eliminated first, with TODO noting dialect-structure mismatch. What is missing is stronger pipeline/dialect separation so this pass need not defensively reject whens.

### [ ] 1262. `lib/Dialect/FIRRTL/Transforms/InferReadWrite.cpp:62`
Real capability boundary: pass errors out in the presence of `WhenOp` because driver tracing is not implemented for conditional regions. What is missing is conditional-control-aware driver analysis.

### [ ] 1263. `lib/Dialect/FIRRTL/Transforms/MergeConnections.cpp:52`
Constant-like detection in merge logic is incomplete (missing cases like `unrealized_conversion`, `asUInt`, `asSInt`). What is missing is broader constant-equivalence recognition.

### [ ] 1264. `lib/Dialect/FIRRTL/Transforms/MemToRegOfVec.cpp:50`
`anyInstanceInEffectiveDesign(moduleOp)` is expected module filtering and not a standalone implementation gap.

### [ ] 1265. `lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:272`
Real IMDCE coverage gap: side-effect handling still leaves constructs like `attach` unmodeled. What is missing is complete liveness treatment for these ops.

### [ ] 1266. `lib/Dialect/FIRRTL/Transforms/IMDeadCodeElim.cpp:396`
Known implementation debt: module list is copied to avoid iterator invalidation while mutating instance graph. What is missing is a safer traversal/update strategy that avoids this workaround.

### [ ] 1267. `test/Target/ExportSystemC/errors.mlir:3`
This is an intentional negative test expecting unsupported `hw.module` emission in ExportSystemC, not new debt at the test line itself.

### [ ] 1268. `test/Target/ExportSystemC/errors.mlir:9`
Same as entry 1267 for unsupported `!hw.inout` type emission: expected negative coverage, while the underlying product gap is missing inout type support in ExportSystemC.

### [ ] 1269. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:165`
HandshakeToHW type-name lowering supports only a narrow data-type set and errors otherwise. What is missing is broader handshake data-type coverage in conversion.

### [ ] 1270. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:189`
Constant naming logic only handles integer attributes and errors on other constant forms. What is missing is generalized constant-type handling (or stricter op invariants if non-integer constants are disallowed).

### [ ] 1271. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:456`
This `@todo` is refactor debt (move `RTLBuilder` utilities to shared support), not a direct feature gap.

### [ ] 1272. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:665`
Real i0-related cleanup boundary: one-hot mux initialization path still carries workaround logic pending handshake/i0 support semantics.

### [ ] 1273. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:703`
Zero-value construction only supports selected types and errors on others. What is missing is complete zero-initialization support for remaining lowered handshake data types.

### [ ] 1274. `lib/Conversion/HandshakeToHW/HandshakeToHW.cpp:1667`
Same i0-related boundary as entry 1272 in sequential buffer stage setup: behavior remains temporary until i0 support is fully defined.

### [ ] 1275. `test/Target/ExportSystemC/basic.mlir:159`
This is a test TODO placeholder noting missing precedence-stress coverage because suitable inlinable lower-precedence ops are not yet available. It reflects test-coverage debt rather than immediate code debt at this line.

### [ ] 1276. `test/Target/ExportSystemC/basic.mlir:167`
Same as entry 1275: placeholder for pending precedence test case once enabling operations exist.

### [ ] 1277. `test/Target/ExportSystemC/basic.mlir:193`
Same class of test-coverage TODO: COMMA-precedence emission case cannot yet be tested due missing applicable operation support.

### [ ] 1278. `test/Target/ExportSystemC/basic.mlir:196`
Same class of test-coverage TODO: CAST-precedence parenthesization stress case awaits suitable lower-precedence inlinable op support.

### [ ] 1279. `test/Target/ExportSystemC/basic.mlir:230`
Same class of test-coverage TODO in `systemc.cpp.new` precedence handling.

### [ ] 1280. `test/Target/ExportSystemC/basic.mlir:237`
Same class of test-coverage TODO in `systemc.cpp.delete` precedence handling.

### [ ] 1281. `lib/Dialect/FIRRTL/Transforms/LowerOpenAggs.cpp:597`
LowerOpenAggs still performs cloned intermediate instance ops to add/remove ports. What is missing is direct port mutation/update flow without temporary clone churn and manual array-attr handling.

### [ ] 1282. `lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:232`
Linking extmodule declarations/definitions still has unresolved `defname`/parameter reconciliation semantics. What is missing is robust merge logic when declaration and definition metadata differ.

### [ ] 1283. `lib/Dialect/FIRRTL/Transforms/LinkCircuits.cpp:263`
Circuit-linking currently merges only selected circuit attributes and leaves others (`enable_layers`, etc.) as TODO. What is missing is comprehensive circuit-attribute merge policy.

### [ ] 1284. `test/Tools/run-opentitan-fpv-circt-bmc-stopat-selector-validation.test:11`
This is intentional negative test coverage for `unsupported_stopat_selector` classification in OpenTitan FPV tooling, not standalone implementation debt at the test line.

### [ ] 1285. `lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:497`
Real lowering gap: memory port annotations are dropped/empty in generated replacement instance. What is missing is explicit lowering/mapping of port annotations.

### [ ] 1286. `lib/Dialect/FIRRTL/Transforms/LowerMemory.cpp:557`
`anyInstanceInEffectiveDesign(moduleOp)` here is normal dedup-scope filtering, not unresolved work.

### [ ] 1287. `lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:43`
`anyInstanceUnderLayer(moduleOp)` is expected analysis gating logic in layer checks, not a TODO/debt marker.

### [ ] 1288. `lib/Dialect/FIRRTL/Transforms/CheckLayers.cpp:73`
Same as entry 1287: `anyInstanceUnderLayer(parent)` is routine analysis use, non-actionable by itself.

### [ ] 1289. `lib/Conversion/MooreToCore/MooreToCore.cpp:1969`
Class lowering currently skips vtable generation from `ClassMethodDeclOp` in this path. What is missing is complete vtable layout/population for method declarations.

### [ ] 1290. `lib/Conversion/MooreToCore/MooreToCore.cpp:2100`
Module port conversion rejects ports whose Moore types cannot be converted. What is missing is broader Moore-to-core type conversion coverage for module ports.

### [ ] 1291. `lib/Conversion/MooreToCore/MooreToCore.cpp:2107`
Real port-model gap: special directions (`inout`/`ref`) are temporarily treated as input until net/ref-capable core type representation is available.

### [ ] 1292. `lib/Conversion/MooreToCore/MooreToCore.cpp:3761`
Zero-initialization logic notes missing four-valued core support (cannot emit all-X defaults). What is missing is full four-state literal/constant support in core dialects.

### [ ] 1293. `lib/Conversion/MooreToCore/MooreToCore.cpp:6031`
Constraint `foreach` lowering currently erases op and defers complex validation. What is missing is generated loop-based runtime validation for nontrivial element constraints.

### [ ] 1294. `lib/Conversion/MooreToCore/MooreToCore.cpp:6054`
`dist` constraints are currently erased with TODO for weighted generation. What is missing is weighted random distribution lowering/runtime calls.

### [ ] 1295. `lib/Conversion/MooreToCore/MooreToCore.cpp:7126`
Net lowering supports a fixed set of net kinds and rejects others. What is missing is support for remaining Moore net kinds in conversion.

### [ ] 1296. `lib/Conversion/MooreToCore/MooreToCore.cpp:7132`
Same four-valued-core dependency as entry 1292, now in net initialization/handling path.

### [x] 1297. `lib/Conversion/MooreToCore/MooreToCore.cpp:7819`
Status update (2026-02-28): this entry is stale and closed. Four-state extract lowering already models out-of-bounds portions via unknown-mask fill, and runtime regression `test/Tools/circt-sim/dyn-bit-select-oob-read-x.sv` confirms dynamic OOB reads produce `X` for `logic` and `0` for `bit`.

### [x] 1298. `lib/Conversion/MooreToCore/MooreToCore.cpp:8108`
Status update (2026-02-28): this entry is closed in this workspace for static `ExtractRef` lowering. Full/partial out-of-bounds static extract-ref reads now lower to explicit fallback references (`X` for four-state, `0` for two-state) instead of aliasing truncated in-range bits; covered by `test/Conversion/MooreToCore/fourstate-bit-extract.mlir` OOB cases.

### [x] 1299. `lib/Conversion/MooreToCore/MooreToCore.cpp:8878`
Status update (2026-02-28): this entry is closed in this workspace for packed `DynExtractRef` lowering. Dynamic packed extract-ref now performs explicit OOB detection (plus unknown-index invalidation) and yields fallback refs (`X` for four-state, `0` for two-state) instead of aliasing truncated in-range bits. Regression coverage added in `test/Conversion/MooreToCore/fourstate-bit-extract.mlir`.

### [ ] 1300. `lib/Conversion/MooreToCore/MooreToCore.cpp:11233`
CaseX/CaseZ equality lowering is still constant-centric until four-valued core integers are available for non-constant X/Z mask extraction. What is missing is full non-constant four-state casex/casez semantics.

### [x] 1301. `lib/Conversion/MooreToCore/MooreToCore.cpp:12736`
Status update (2026-02-28): this entry is stale/non-actionable. The matched content is descriptive implementation text, not an unresolved MooreToCore TODO marker.

### [x] 1302. `lib/Conversion/MooreToCore/MooreToCore.cpp:12737`
Status update (2026-02-28): same closure as entry 1301.

### [x] 1303. `lib/Conversion/MooreToCore/MooreToCore.cpp:12739`
Status update (2026-02-28): same closure as entry 1301.

### [x] 1304. `lib/Conversion/MooreToCore/MooreToCore.cpp:12811`
Status update (2026-02-28): same closure as entry 1301.

### [x] 1305. `lib/Conversion/MooreToCore/MooreToCore.cpp:13009`
Status update (2026-02-28): same closure as entry 1301.

### [ ] 1306. `lib/Conversion/MooreToCore/MooreToCore.cpp:14637`
Real semantic gap: conditional lowering is only sound for two-valued conditions; X/Z requires dual-branch evaluate-and-merge semantics not yet implemented.

### [x] 1307. `lib/Conversion/MooreToCore/MooreToCore.cpp:15148`
Status update (2026-02-28): this line-level tracker is stale. Current `moore.fmt.int` enum cases map directly to supported lowering branches (`decimal`, `binary`, `octal`, `hex_lower`, `hex_upper`) and existing string-format regressions pass.

### [ ] 1308. `lib/Conversion/MooreToCore/MooreToCore.cpp:16035`
Format-string conversion falls back to empty string for unsupported fragment types. What is missing is complete format-fragment lowering to preserve semantics instead of silent empty fallback.

### [ ] 1309. `lib/Conversion/MooreToCore/MooreToCore.cpp:17884`
Queue sort-with lowering accepts only queue/open/fixed unpacked array containers and rejects others. What is missing is broader container-type support (or stricter op typing to make this unreachable).

### [ ] 1310. `lib/Conversion/MooreToCore/MooreToCore.cpp:18190`
Same as entry 1309 for reverse sort-with lowering: unsupported queue/container type path remains.

### [ ] 1311. `lib/Conversion/MooreToCore/MooreToCore.cpp:19004`
Stream-concat lowering currently accepts only queue/dynamic-array/fixed-array inputs and rejects others. What is missing is broader input-container coverage (or stricter typing to make this unreachable).

### [ ] 1312. `lib/Conversion/MooreToCore/MooreToCore.cpp:19149`
Stream-unpack lowering requires destination `ref` to queue/open/fixed array and rejects other nested ref kinds. What is missing is destination ref-type coverage beyond current container set.

### [ ] 1313. `lib/Conversion/MooreToCore/MooreToCore.cpp:19925`
Field-offset computation uses a simplified type walk and bails on unsupported path nodes. What is missing is complete layout/path handling (including non-struct path components) for robust offset derivation.

### [ ] 1314. `lib/Conversion/MooreToCore/MooreToCore.cpp:21188`
Array-locator string helper returns null for unsupported operations. What is missing is full lowering support for remaining locator operations in this path.

### [ ] 1315. `lib/Conversion/MooreToCore/MooreToCore.cpp:21584`
Array-locator conversion still rejects unsupported array container kinds. What is missing is broader container-type support for locator lowering.

### [ ] 1316. `lib/Conversion/MooreToCore/MooreToCore.cpp:22223`
Same container-coverage boundary as entry 1315 in the simple-predicate lowering path: unsupported array types are rejected.

### [ ] 1317. `lib/Conversion/MooreToCore/MooreToCore.cpp:22610`
Associative-array operations reject unsupported key representations. What is missing is key-type coverage and conversion support for additional key forms.

### [ ] 1318. `lib/Conversion/MooreToCore/MooreToCore.cpp:23960`
`sscanf` lowering only writes back to a restricted set of destination ref types. What is missing is destination-type coverage for additional legal scan targets.

### [ ] 1319. `lib/Conversion/MooreToCore/MooreToCore.cpp:24064`
Same as entry 1318 for `fscanf`: unsupported destination-ref types remain unhandled.

### [x] 1320. `lib/Conversion/MooreToCore/MooreToCore.cpp:24335`
Status update (2026-02-28): this line-level tracker is stale. The matched content is explanatory implementation/comment text and not an unresolved MooreToCore TODO.

### [ ] 1321. `lib/Conversion/MooreToCore/MooreToCore.cpp:30654`
Type conversion notes that unpacked arrays have semantics broader than packed arrays, while current mapping is naive. What is missing is semantics-faithful unpacked-array representation/layout in core lowering.

### [ ] 1322. `lib/Conversion/MooreToCore/MooreToCore.cpp:30683`
Unpacked-struct conversion is similarly flagged as naive relative to packed semantics. What is missing is correct handling of memory layout and simulation granularity differences.

### [ ] 1323. `lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:109`
LowerDPI conversion helper currently supports integers/arrays but not bundle or enum payloads. What is missing is conversion support for these aggregate/symbolic types.

### [ ] 1324. `lib/Dialect/FIRRTL/Transforms/LowerDPI.cpp:206`
DPI signature consistency checks are currently pass-local logic. What is missing is op-level verifier coverage once FIRRTL function constructs are available.

### [ ] 1325. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:57`
Manifest service lookup is reused for engine creation via an acknowledged hack. What is missing is separated service-vs-engine construction plumbing.

### [ ] 1326. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:178`
Manifest constant extraction does not strongly validate/coerce value payloads against declared type metadata. What is missing is typed value conversion/validation during manifest parse.

### [ ] 1327. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:290`
Engine instantiation is supported only at top-level design nodes. What is missing is recursive/lower-level engine handling.

### [ ] 1328. `lib/Dialect/ESI/runtime/cpp/lib/Manifest.cpp:324`
Same root as entry 1325: `createEngine` is a hacky wrapper over service construction and needs dedicated implementation.

### [ ] 1329. `lib/Conversion/HWToSystemC/HWToSystemC.cpp:71`
HWToSystemC currently hardcodes generated inner logic function naming. What is missing is better symbol extraction and robust uniquing policy.

### [ ] 1330. `lib/Conversion/HWToSystemC/HWToSystemC.cpp:77`
Inlining graph-region logic into `systemc.func` lacks dominance/use-before-def/cycle analysis. What is missing is legality checks (and repairs where possible) for SSA CFG emission.

### [ ] 1331. `lib/Dialect/FIRRTL/Transforms/CheckCombLoops.cpp:336`
Comb-loop analysis path explicitly skips external modules. What is missing is external-module modeling/handling in loop checks.

### [ ] 1332. `lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:308`
CHIRRTL memory-port cleanup mirrors SFC behavior with unresolved policy around dropping inferred/unused ports (especially with annotations). What is missing is finalized policy and explicit annotation-preservation semantics.

### [ ] 1333. `lib/Dialect/FIRRTL/Transforms/LowerCHIRRTL.cpp:454`
Enable inference for sequential memory read ports remains heuristic and has known gap when address comes from module ports. What is missing is robust enable inference independent of operand-defining-op form.

### [ ] 1334. `lib/Dialect/FIRRTL/Transforms/Dedup.cpp:307`
Dedup hash/equality logic still punts on `DistinctAttr` and path interactions. What is missing is full `DistinctAttr` semantics in dedup decisions.

### [ ] 1335. `lib/Dialect/FIRRTL/Transforms/Dedup.cpp:589`
This is minor diagnostics polish debt: dedup mismatch reporting assumes named ports and lacks fallback port-index printing.

### [ ] 1336. `lib/Dialect/FIRRTL/Transforms/Dedup.cpp:733`
Same root as entry 1334: `DistinctAttr` remains incompletely modeled in attribute comparison/path handling.

### [ ] 1337. `lib/Dialect/FIRRTL/Transforms/Lint.cpp:107`
`anyInstanceInDesign(fModule)` here is expected lint gating (non-DUT modules are exempt in this check), not unresolved implementation work.

### [ ] 1338. `lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:167`
This is intentional compatibility gating: runtime rejects unknown ESI header versions. It is a version-support boundary, not a hidden TODO at this line.

### [ ] 1339. `lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:422`
ServiceRegistry creation still relies on hardcoded typeid branches. What is missing is a real registration/dispatch mechanism.

### [ ] 1340. `lib/Dialect/ESI/runtime/cpp/lib/Services.cpp:435`
Same root as entry 1339 for service-name lookup: registry mapping is hardcoded rather than extensible registration-driven.

### [ ] 1341. `test/Tools/run-formal-all-strict-gate-bmc-lec-contract-fingerprint-parity-defaults.test:8`
This is expected-value fixture text (`xxxx9999...`) used to force a strict-gate parity mismatch in test flow, not unresolved implementation debt.

### [ ] 1342. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcClient.cpp:180`
Real runtime robustness gap: manifest fetch uses polling-loop workaround for startup race. What is missing is a proper synchronization/availability contract after DPI API changes.

### [ ] 1343. `lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:736`
`llvm_unreachable("unsupported comparison predicate")` here is an enum-exhaustiveness sentinel after covering known predicates, not a standalone feature TODO.

### [ ] 1344. `lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:994`
Pipeline lowering assumes integer-typed stage results and asserts otherwise. What is missing is support (or explicit prevalidation) for non-integer pipeline result types.

### [ ] 1345. `lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:1256`
Pipeline-body scheduling only handles group scheduleables in this branch and errors on other block scheduleables. What is missing is broader schedulable-kind support.

### [ ] 1346. `lib/Conversion/LoopScheduleToCalyx/LoopScheduleToCalyx.cpp:1316`
CFG scheduling currently assumes at most conditional branches and punts on richer branch forms (`std.switch` etc.). What is missing is multi-successor branching support or mandatory pre-lowering contract enforcement.

### [ ] 1347. `lib/Dialect/FIRRTL/Transforms/AnnotateInputOnlyModules.cpp:58`
`anyInstanceInEffectiveDesign(module)` is expected gating logic for annotation scope and not a debt marker.

### [ ] 1348. `lib/Conversion/LTLToCore/LTLToCore.cpp:1102`
LTL lowering currently supports only selected `abort_on` actions for `and`; other actions error out. What is missing is full action semantic support.

### [ ] 1349. `lib/Conversion/LTLToCore/LTLToCore.cpp:1133`
Same as entry 1348 for `or`: only selected `abort_on` action is implemented.

### [ ] 1350. `lib/Conversion/LTLToCore/LTLToCore.cpp:1265`
Generic property lowering still has unsupported-defining-op fallback. What is missing is expanded lowering coverage for remaining property op forms.

### [ ] 1351. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:446`
Capture cloning for “special” operands is not recursive yet; current workaround notes future FString operand dependencies. What is missing is recursive clone support.

### [ ] 1352. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:513`
HierPath creation currently uses a mutexed critical section in parallel flow and notes avoidable lock placement. What is missing is lock-free/earlier setup strategy.

### [ ] 1353. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:561`
Domain inference still peeks through intermediary wires due missing wire-level domain metadata. What is missing is domain-kind info on wires to simplify/strengthen analysis.

### [ ] 1354. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:912`
Same root as entry 1353 in domain define handling: wire look-through remains until wire domain info exists.

### [ ] 1355. `lib/Dialect/FIRRTL/Transforms/LowerLayers.cpp:1361`
HierPath rewrite currently rebuilds namepaths eagerly for every op. What is missing is change-driven incremental rewrite to avoid unnecessary recomputation.

### [ ] 1356. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:427`
Real type-correctness gap: Cosim HostMem channel types are acknowledged as wrong (missing channel wrapping). What is missing is corrected channel-typed wiring.

### [ ] 1357. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:489`
HostMem read service path lacks mapping validation before dereference. What is missing is mapped-memory safety checks.

### [ ] 1358. `lib/Dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp:517`
Same as entry 1357 for HostMem write path: missing mapped-memory validation.

### [ ] 1359. `lib/Dialect/FIRRTL/Transforms/LowerIntmodules.cpp:164`
EICG wrapper migration still drops `dedupGroup` annotation with warning as temporary workaround. What is missing is proper annotation-preserving migration into intrinsic flow.

### [ ] 1360. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:305`
Real concurrency debt: `InstanceGraph` mutation in `LowerDomains` is documented non-thread-safe. What is missing is thread-safe mutation strategy before parallelization.

### [ ] 1361. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:311`
`indexToDomain` declaration is normal per-module bookkeeping state, not unresolved debt by itself.

### [ ] 1362. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:360`
`indexToDomain[i] = ...` is ordinary domain-map population logic, not a TODO marker.

### [ ] 1363. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:376`
`indexToDomain[i].op = object` is routine state assignment in lowering bookkeeping, non-actionable.

### [ ] 1364. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:377`
`indexToDomain[i].temp = ...` here is ordinary temporary value wiring state, not unresolved work.

### [ ] 1365. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:441`
Association push into `indexToDomain` is expected bookkeeping for domain associations, non-actionable scanner hit.

### [ ] 1366. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:457`
Iteration over `indexToDomain` is normal lowering traversal logic, not a gap marker.

### [ ] 1367. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:644`
Domain define lowering rejects source forms other than expected conversion/domain-create cases. What is missing is support for additional source op forms or stricter pre-normalization.

### [ ] 1368. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:671`
Cleanup TODO: lowerInstances path duplicates module-kind gating logic already present elsewhere. What is missing is structural refactor to make no-op conditions centralized.

### [ ] 1369. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:683`
LowerDomains has unimplemented lowering for non-`InstanceOp` instance-graph uses. What is missing is handling for these additional use kinds.

### [ ] 1370. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:690`
`indexToDomain[i].temp = ...` in instance rewrite loop is routine bookkeeping/wiring and not a standalone debt marker.

### [ ] 1371. `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp:696`
`for (auto &[i, info] : indexToDomain)` is normal iteration over lowering bookkeeping state, not unresolved work.

### [ ] 1372. `lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:59`
Trace backend constructor still has unimplemented mode branch guarded by `assert(false)`. What is missing is complete implementation (or removal) of non-write/discard modes.

### [ ] 1373. `lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:211`
Trace read-port message generation currently supports only fixed-width bit-vector types. What is missing is support for additional runtime types.

### [ ] 1374. `lib/Dialect/ESI/runtime/cpp/lib/backends/Trace.cpp:212`
Same root as entry 1373: unsupported read type throws at runtime.

### [ ] 1375. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:153`
RPC server still uses insecure credentials on localhost by policy/expedience. What is missing is secure credential support/configuration.

### [ ] 1376. `lib/Dialect/ESI/runtime/cpp/lib/backends/RpcServer.cpp:310`
Write-reactor loop still relies on polling/sleep instead of proper notifications. What is missing is migration to the newer notification mechanism.

### [ ] 1377. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:91`
GrandCentral YAML description handling depends on verbatim/comment workaround due missing `firrtl.DocStringAnnotation` support. What is missing is first-class docstring annotation handling.

### [ ] 1378. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:106`
Nested interface representation still relies on workaround struct because interfaces cannot instantiate interfaces. What is missing is a proper solution to issue `#1464`.

### [ ] 1379. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:358`
GrandCentral YAML conversion explicitly drops constructs marked as unsupported. What is missing is lowering support for those construct classes instead of drop behavior.

### [ ] 1380. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:359`
Same as entry 1379: `"unsupported"` typed YAML placeholders represent currently unlowered constructs.

### [ ] 1381. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:715`
This is architecture cleanup debt: pass stores `InstancePathCache` via pointer workaround because of analysis object constraints. Not directly user-visible but technical debt.

### [ ] 1382. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:760`
Description text lowering still relies on verbatim comment-string munging. What is missing is a dedicated comment op lowering path.

### [ ] 1383. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:761`
Tracking issue reference for comment-op lowering (`#1677`), same root as entry 1382.

### [ ] 1384. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1716`
Interface field-path construction uses field names instead of stable signal symbols due current nested-interface limitations. What is missing is symbol-based pathing to avoid brittle renaming behavior.

### [ ] 1385. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1749`
Bundle traversal still requires ad hoc runtime checks for required attribute fields (`defName`, `elements`). What is missing is structural attribute/schema enforcement.

### [ ] 1386. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1762`
`IntegerAttr() /* XXX */` placeholder indicates incomplete interface metadata population in builder state.

### [ ] 1387. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1775`
Same as entry 1384 in view-bundle traversal: name-based path append is a brittle workaround for missing symbol-based references.

### [ ] 1388. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1906`
GrandCentral run method still contains a TODO around options/lifecycle handling when constructing instance-path analysis state.

### [ ] 1389. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:1998`
Unresolved handling decision for unexpected annotated `InstanceOp` path in annotation stripping logic. What is missing is finalized behavior/policy for this case.

### [ ] 1390. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2187`
`anyInstanceInEffectiveDesign(...)` here is expected module filtering logic in companion processing, not standalone debt.

### [ ] 1391. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2477`
`yaml.type = "unsupported"` is marker emission corresponding to the unsupported-construct drop path (entries 1379/1380), not a separate new gap.

### [ ] 1392. `lib/Dialect/FIRRTL/Transforms/GrandCentral.cpp:2633`
Same as entry 1391 in alternate YAML path: marker emission for already-known unsupported construct class.

### [ ] 1393. `lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:58`
`TODO: check all argument types` reflects incomplete validation coverage in LowerTypes setup.

### [ ] 1394. `lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:130`
Read-only `RefType` preservation is intentionally disabled as workaround (issue `4479`). What is missing is correct preservation semantics without causing MemTap mismatches.

### [ ] 1395. `lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:644`
Symbol partitioning rejects unsupported parent types. What is missing is partitioning support across remaining FIRRTL type forms.

### [ ] 1396. `lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:696`
Same as entry 1395 for alternate type-switch branch: unsupported types cannot be partitioned for inner symbols.

### [ ] 1397. `lib/Dialect/FIRRTL/Transforms/LowerTypes.cpp:1622`
Real maintenance gap: annotation updates are flagged FIXME during instance recreation in type lowering.

### [ ] 1398. `lib/Dialect/ESI/runtime/cpp/lib/Values.cpp:382`
`BitVector::toString` rejects non-{2,8,10,16} bases by design. This is API boundary enforcement, not unresolved implementation debt.

### [x] 1399. `lib/Conversion/ImportVerilog/ImportVerilog.cpp:215`
Status update (2026-02-28): this line-level tracker is stale. The marker points to descriptive implementation text for already-landed format width/alignment compatibility rewriting, not an unresolved TODO at this site. Existing focused format compatibility regressions pass (`format-width-ignored-compat.sv`, `format-n-compat.sv`, `format-vuz-compat.sv`).

### [x] 1400. `lib/Conversion/ImportVerilog/ImportVerilog.cpp:1838`
Status update (2026-02-28): this gap is closed. `SimplifyProcedures` is re-enabled in the ImportVerilog module pipeline, and the pass now avoids shadowing read-only globals and preserves reads inside `moore.wait_event` regions so event-control observation remains tied to the correct module-level signals.

### [ ] 1401. `lib/Dialect/FIRRTL/Transforms/AddSeqMemPorts.cpp:463`
`anyInstanceInEffectiveDesign(op)` here is expected design-scope filtering for pass application, not unresolved implementation work.

### [ ] 1402. `lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:31`
This is local upstreaming debt for a utility specialization (`PointerLikeTypeTraits<ArrayAttr>`), not direct user-facing feature debt.

### [ ] 1403. `lib/Dialect/FIRRTL/Transforms/SpecializeLayers.cpp:790`
Minor analysis-preservation optimization TODO: pass could preserve more specific analyses in no-op specialization cases.

### [ ] 1404. `lib/Dialect/ESI/Passes/ESIBuildManifest.cpp:32`
General maintainability note: pass implementation is acknowledged as working but structurally messy. This is refactor debt rather than feature gap.

### [ ] 1405. `lib/Dialect/FIRRTL/Transforms/LowerAnnotations.cpp:131`
Non-local annotation path scattering still has FIXME about unique chain links. What is missing is robust uniquing of annotation-chain anchors.

### [ ] 1406. `lib/Dialect/Datapath/DatapathFolds.cpp:154`
Fold currently implemented as `comb.add` rewrite though it conceptually belongs in `compress` canonicalization. What is missing is proper canonicalization ownership without current flag/workaround constraints.

### [ ] 1407. `lib/Dialect/Datapath/DatapathFolds.cpp:316`
KnownBits-based extraction of constant-one runs in `oneExt` patterns remains TODO. What is missing is stronger bit-level analysis-driven folding.

### [ ] 1408. `lib/Dialect/Datapath/DatapathFolds.cpp:426`
`PartialProductOp` lacks constant-multiplication-specific folding support.

### [ ] 1409. `lib/Dialect/Datapath/DatapathFolds.cpp:484`
Sign-extension partial-product reduction currently requires equal input widths. What is missing is support for mixed-width inputs.

### [ ] 1410. `lib/Dialect/FIRRTL/Transforms/InjectDUTHierarchy.cpp:13`
This is explanatory header comment text describing terminology/diagram, not unresolved implementation debt.

### [ ] 1411. `lib/Dialect/FIRRTL/Transforms/InjectDUTHierarchy.cpp:340`
Policy TODO: NLA root rewrite behavior may need tightening to `moveDut=true` mode depending on annotation semantic expectations.

### [ ] 1412. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:121`
Platform support is explicitly limited to Linux/Windows in this utility path. What is missing is non-Linux/Windows implementation (and this branch currently uses a typoed `#eror` directive).

### [ ] 1413. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:147`
Same as entry 1412 for shared-library path discovery: unsupported non-Linux/Windows platforms.

### [ ] 1414. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:199`
Same as entry 1412 for backend plugin naming/selection: unsupported non-Linux/Windows platforms.

### [ ] 1415. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:285`
Same as entry 1412 for dynamic-load error handling branch: unsupported non-Linux/Windows platforms.

### [ ] 1416. `lib/Dialect/ESI/runtime/cpp/lib/Accelerator.cpp:396`
Service-thread loop currently uses busy-yield polling with TODO for better wake strategy. What is missing is efficient notification-driven scheduling.

### [ ] 1417. `lib/Conversion/ImportVerilog/CrossSelect.cpp:97`
Cross-select intersect value-range lowering requires constant range center/tolerance; non-constant expressions are rejected.

### [ ] 1418. `lib/Conversion/ImportVerilog/CrossSelect.cpp:104`
Same as entry 1417: non-integer/non-constant intersect tolerance values remain unsupported.

### [ ] 1419. `lib/Conversion/ImportVerilog/CrossSelect.cpp:121`
Intersect range bounds that overflow `int64` are rejected. What is missing is wider-range handling semantics for cross-select intersections.

### [ ] 1420. `lib/Conversion/ImportVerilog/CrossSelect.cpp:150`
`matches` policy in cross-select set expressions currently requires constant/evaluable values; non-constant policies are rejected.

### [ ] 1421. `lib/Conversion/ImportVerilog/CrossSelect.cpp:159`
Same as entry 1420 for invalid or nonpositive/nonconstant `matches` policy forms.

### [ ] 1422. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1027`
Negation of cross identifier (`!cross_id`) in cross-select DNF building is unsupported.

### [ ] 1423. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1069`
Nested `with` clause inside cross-select expression is unsupported.

### [ ] 1424. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1072`
Nested cross-set expression inside cross-select expression is unsupported.

### [ ] 1425. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1109`
Cross-select lowering cannot represent always-false case when no cross targets are available; this path errors.

### [ ] 1426. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1132`
Cross-set tuple extraction requires tuple/unpacked shape; non-tuple elements are unsupported.

### [ ] 1427. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1137`
Cross-set tuple elements must match cross arity; mismatched tuple arity is unsupported.

### [ ] 1428. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1144`
Cross-set tuple elements must be integer-constant values; non-integer tuple values are unsupported.

### [ ] 1429. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1165`
Cross-set expression must be compile-time constant/evaluable; non-constant sets are unsupported.

### [ ] 1430. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1187`
Cross-set expression currently supports only queue or unpacked tuple-list constants; other container forms are unsupported.

### [ ] 1431. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1273`
Additional cross-select path with the same constant-evaluable restriction as entries 1417/1418: non-constant intersect ranges are unsupported.

### [ ] 1432. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1307`
Same root as entry 1431 for singleton intersect values: non-constant intersect values are unsupported.

### [ ] 1433. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1357`
`with`-clause iterator local binding currently supports only finite widths in a narrow range (`1..64`). Wider/zero-width iterators are unsupported.

### [ ] 1434. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1457`
Finite coverpoint-bin domain expansion is capped; oversized domains are rejected to bound compile-time/state blowup.

### [ ] 1435. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1469`
Same as entry 1434 for range expansion path: large finite coverpoint-bin domains are unsupported.

### [ ] 1436. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1482`
Empty finite coverpoint bins are rejected in cross-select lowering.

### [ ] 1437. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1500`
Coverpoint bin ranges in cross-select explicit values must be constant-evaluable; non-constant ranges are unsupported.

### [ ] 1438. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1512`
Same as entry 1437 for scalar bin values: non-constant bin values are unsupported.

### [ ] 1439. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1545`
Coverpoint-bin `with` iterator typing has bounded-width constraints; unsupported iterator widths are rejected.

### [ ] 1440. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1585`
Set-based coverpoint bins must evaluate to constant sets; non-constant set expressions are unsupported.

### [ ] 1441. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1592`
Set-based coverpoint bin elements must be integer constants; non-integer values are unsupported.

### [ ] 1442. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1630`
Transition-bin ranges in cross-select handling must be constant-evaluable; non-constant ranges are unsupported.

### [ ] 1443. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1642`
Same as entry 1442 for transition-bin scalar values: non-constant values are unsupported.

### [ ] 1444. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1668`
Default/default-sequence coverpoint bins are not valid cross-select targets in this lowering path.

### [ ] 1445. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1702`
Automatic finite-domain construction for `with` clauses requires integral coverpoints; non-integral types are unsupported.

### [ ] 1446. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1709`
Automatic finite-domain construction also imposes a max coverpoint width cap; larger widths are unsupported.

### [ ] 1447. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1757`
`unsupportedBinShape` flag tracks unsupported default-sequence bin forms. This is tied to the real default-bin-shape limitation, not scanner noise.

### [ ] 1448. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1758`
Same root as entry 1447: unsupported bin-shape branch for default-sequence bins.

### [ ] 1449. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1830`
If no finite bins can be built for a coverpoint under current constraints, cross-select lowering fails for that coverpoint.

### [ ] 1450. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1835`
Cross-space cardinality guard: overly large finite Cartesian spaces are rejected.

### [ ] 1451. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1898`
Condition-target lowering only supports certain target kinds; unsupported target kinds are rejected.

### [ ] 1452. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1904`
Detached coverage-bin references in conditions are unsupported.

### [ ] 1453. `lib/Conversion/ImportVerilog/CrossSelect.cpp:1908`
Coverage bins whose parent cannot be resolved to a coverpoint are unsupported.

### [ ] 1454. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2013`
Same empty-bin guard as entry 1436 encountered during tuple/value expansion path.

### [ ] 1455. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2030`
`with`-clause evaluation caps candidate value-tuple checks; excessive tuple counts are rejected.

### [ ] 1456. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2067`
Same non-constant cross-set limitation as entry 1429, in bin-tuple evaluation path.

### [ ] 1457. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2105`
Same container-form limitation as entry 1430, in bin-tuple evaluation path (expects queue/unpacked tuple list).

### [ ] 1458. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2165`
`with`-clause selection caps number of selected tuples; oversized selections are rejected.

### [ ] 1459. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2356`
General cross-select selection also caps selected tuple count to bound expansion.

### [ ] 1460. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2444`
Finite intersect value-set expansion has a hard cap; oversized finite value sets are unsupported.

### [ ] 1461. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2457`
Cross-select intersect-range handling rejects invalid range forms. What is missing is richer symbolic/late-bound handling for malformed or non-finite range cases instead of hard failure.

### [ ] 1462. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2499`
Same constant-evaluable requirement as earlier intersect entries: non-constant intersect value ranges are unsupported in cross-select lowering.

### [ ] 1463. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2523`
Same root as entry 1462 for scalar intersect values: non-constant intersect values are unsupported.

### [ ] 1464. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2562`
Negating cross-set expressions is unsupported in this lowering path (`!set_expr` form).

### [ ] 1465. `lib/Conversion/ImportVerilog/CrossSelect.cpp:2570`
Negation of cross-select expressions that include a `with` clause is unsupported.

### [ ] 1466. `lib/Dialect/FIRRTL/Transforms/BlackBoxReader.cpp:223`
Real TODO: when multiple blackbox annotations converge on the same emitted file, text equivalence is not strictly checked. What is missing is deterministic conflict detection on content, not only path-prefix reconciliation.

### [ ] 1467. `lib/Dialect/FIRRTL/Transforms/BlackBoxReader.cpp:374`
`anyInstanceInEffectiveDesign(...)` here is routine design-membership gating for output-dir placement, not an unresolved feature gap.

### [ ] 1468. `lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:1043`
This TODO flags potential solver-pass redundancy in cycle handling. The gap is optimization/maintainability (extra solving work), not semantic correctness.

### [ ] 1469. `lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:1719`
Width inference for `rwprobe` currently supports only a subset of resolved target forms; unsupported target shapes still fail hard.

### [ ] 1470. `lib/Dialect/FIRRTL/Transforms/InferWidths.cpp:2217`
Real edge-case TODO: zero-length vectors do not recurse into element types when solving unknown widths, leaving incomplete width normalization for that corner case.

### [ ] 1471. `lib/Dialect/FIRRTL/Transforms/ProbesToSignals.cpp:163`
Force/release probe operations are explicitly rejected. What is missing is lowering semantics for force/release in the probes-to-signals pipeline.

### [ ] 1472. `lib/Dialect/FIRRTL/Transforms/ProbesToSignals.cpp:393`
Memory debug ports (`memtap`) remain unsupported in probe lowering; this is a real capability boundary.

### [ ] 1473. `lib/Dialect/Datapath/DatapathOps.cpp:129`
TODO indicates missed canonicalization: constant-`1` bits are not folded during compressor-tree construction, leaving avoidable logic/perf overhead.

### [ ] 1474. `lib/Dialect/Datapath/DatapathOps.cpp:216`
Real algorithmic TODO: current reduction still follows Dadda assumptions despite timing-aware intent. What is missing is a fully timing-driven compression strategy.

### [ ] 1475. `lib/Dialect/ESI/ESIOps.cpp:478`
Type-matching support for ESI containers is incomplete; additional container kinds beyond current handled set still fall through as unsupported.

### [ ] 1476. `lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:50`
This is default abstract-base behavior (`Type::serialize`) throwing “not implemented” when subclasses do not override. It is intentional API boundary behavior, not direct product debt at this line.

### [ ] 1477. `lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:57`
Same as entry 1476 for base `Type::deserialize`: intentional virtual-default throw path, scanner false positive as TODO debt.

### [ ] 1478. `lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:71`
Same as entry 1476 for base `Type::ensureValid`: intentional default throw path for unimplemented subtype validation.

### [ ] 1479. `lib/Dialect/ESI/runtime/cpp/include/esi/Types.h:98`
Documentation TODO in generated ESI runtime headers: bundle channel direction semantics need clearer host-API description.

### [ ] 1480. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:41`
`emitUnsupportedSvaDiagnostic` is helper infrastructure for warn/error policy and not itself a missing feature; scanner matched identifier naming.

### [ ] 1481. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:43`
`continueOnUnsupportedSVA` is policy gating (warn vs error) for unsupported features, not an implementation gap by itself.

### [ ] 1482. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:118`
This callsite is part of unsupported sampled-value-type diagnostics. The underlying capability gap is incomplete sampled-value type coverage in SVA lowering.

### [ ] 1483. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:119`
Real ImportVerilog gap: sampled-value functions still reject some operand type classes (“unsupported sampled value type”).

### [ ] 1484. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:394`
Same as entry 1483: additional sampled-value lowering path still rejects unsupported operand types.

### [ ] 1485. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:395`
Same as entry 1483; this line is the paired diagnostic payload for that unsupported sampled-value-type path.

### [ ] 1486. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:405`
Same root gap as entry 1483 in another sampled-value conversion branch.

### [ ] 1487. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:406`
Same as entry 1486: diagnostic text line for unsupported sampled-value-type handling.

### [ ] 1488. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:633`
Same sampled-value type-coverage gap as entry 1483 in a later lowering path.

### [ ] 1489. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:634`
Same as entry 1488; paired diagnostic emission line.

### [ ] 1490. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:649`
Same sampled-value type-coverage boundary as entry 1483 in another callsite.

### [ ] 1491. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:650`
Same as entry 1490; this is the paired diagnostic payload for unsupported sampled-value types.

### [ ] 1492. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:656`
Same sampled-value type coverage boundary as entry 1483 in the edge-operand conversion path.

### [ ] 1493. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:657`
Same as entry 1492; paired diagnostic payload for unsupported sampled-value type.

### [ ] 1494. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:985`
Same sampled-value type coverage gap as entry 1483 in the main sampled-function lowering path.

### [ ] 1495. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:986`
Same as entry 1494; this line emits the corresponding unsupported sampled-type diagnostic text.

### [ ] 1496. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1076`
Same root as entry 1483: additional sampled-value branch rejects unsupported operand types.

### [ ] 1497. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1077`
Same as entry 1496; paired diagnostic text line.

### [ ] 1498. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1081`
Same sampled-value type boundary as entry 1483 in another type-check branch.

### [ ] 1499. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1082`
Same as entry 1498; diagnostic payload line.

### [ ] 1500. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1088`
Same sampled-value type gap as entry 1483 in another conversion guard.

### [ ] 1501. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1089`
Same as entry 1500; paired diagnostic payload.

### [ ] 1502. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1104`
Same root sampled-value type limitation as entry 1483.

### [ ] 1503. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1105`
Same as entry 1502; diagnostic payload line for unsupported sampled-value type.

### [ ] 1504. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1119`
Same sampled-value type coverage boundary as entry 1483 in a later check path.

### [ ] 1505. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1120`
Same as entry 1504; paired diagnostic payload.

### [ ] 1506. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1332`
Real ImportVerilog SVA gap: `$past` with sampled-value controls still rejects unsupported input type classes.

### [ ] 1507. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1333`
Same as entry 1506; this line carries detailed unsupported `$past` type diagnostics.

### [ ] 1508. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1336`
`continueOnUnsupportedSVA && !inAssertionExpr` here is fallback-policy gating (warn/placeholder behavior), not a distinct implementation gap by itself.

### [ ] 1509. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1402`
Same `$past` sampled-value-controls gap as entry 1506, here for current/storage type mismatch during lowering.

### [ ] 1510. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1403`
Same as entry 1509; paired diagnostic payload text.

### [ ] 1511. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1411`
Same `$past` sampled-value-controls type limitation as entry 1506 in another current-type check path.

### [ ] 1512. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1412`
Same as entry 1511; diagnostic payload line.

### [ ] 1513. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1618`
Real sequence match-item capability gap: non-unpacked local assertion variable types are unsupported in compound assignment history access.

### [ ] 1514. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1619`
Same as entry 1513; paired diagnostic payload for unsupported local assertion variable type.

### [ ] 1515. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1631`
Real match-item lowering gap: assignment typing is restricted; unsupported LHS type shapes are rejected.

### [ ] 1516. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1632`
Same as entry 1515; diagnostic payload line for unsupported match-item assignment type.

### [ ] 1517. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1650`
Same root as entry 1515 for RHS typing: unsupported assignment result types in match items are rejected.

### [ ] 1518. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1651`
Same as entry 1517; paired diagnostic payload line.

### [ ] 1519. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1672`
Real match-item gap: unary operator support is limited to increment/decrement forms; other unary operators are unsupported.

### [ ] 1520. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:1673`
Same as entry 1519; diagnostic payload line for unsupported match-item unary operator.

### [ ] 1521. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:2449`
Real runtime gap: assertion match-item handling warns that `$save/$restart/$incsave/$reset` are unsupported because checkpoint/restart is not implemented in `circt-sim`.

### [ ] 1522. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:2456`
Real runtime gap: `$sdf_annotate` remains unsupported due missing SDF timing-annotation support in `circt-sim`.

### [ ] 1523. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:2747`
Same root as entry 1519/1515 cluster: only a subset of sequence match-item expression kinds is handled; others diagnose as unsupported.

### [ ] 1524. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:2748`
Same as entry 1523; paired diagnostic payload line for unsupported match-item expression.

### [ ] 1525. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3562`
Generic assertion-expression visitor fallback still diagnoses unsupported expression kinds, indicating incomplete assertion expression coverage.

### [ ] 1526. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3563`
Same as entry 1525; payload line for unsupported assertion expression kind.

### [ ] 1527. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3713`
`emitUnsupportedNonConcurrentSampledPlaceholder` is continue-mode fallback scaffolding and not a standalone implementation gap; it exists because sampled-value type support is incomplete.

### [ ] 1528. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3714`
Same policy-gating pattern as entry 1508: `continueOnUnsupportedSVA && !inAssertionExpr` controls fallback behavior, not a separate feature gap.

### [ ] 1529. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3718`
This warning text confirms a real semantic gap: unsupported sampled-value types in non-concurrent contexts currently degrade to placeholder `0` in continue mode.

### [ ] 1530. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3737`
Fallback invocation line; same root as entry 1529 (sampled-value type coverage gap), with policy-controlled placeholder behavior.

### [ ] 1531. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3739`
Same sampled-value type limitation as entry 1483, emitted when continue-mode fallback is not taken.

### [ ] 1532. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3740`
Same as entry 1531; paired diagnostic payload line.

### [ ] 1533. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3747`
Same fallback path as entry 1530 for unsupported sampled-value-type handling.

### [ ] 1534. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3749`
Same sampled-value type limitation as entry 1531 in an adjacent branch.

### [ ] 1535. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3750`
Same as entry 1534; paired diagnostic payload.

### [ ] 1536. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3765`
Same sampled-value type coverage boundary as entry 1483 in the final post-conversion type check.

### [ ] 1537. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:3766`
Same as entry 1536; diagnostic payload line.

### [ ] 1538. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:4197`
Real SVA feature gap: unsupported assertion system subroutines still hit generic unsupported-system-call diagnostics.

### [ ] 1539. `lib/Conversion/ImportVerilog/AssertionExpr.cpp:4198`
Same as entry 1538; paired diagnostic payload line naming the unsupported system call.

### [ ] 1540. `lib/Dialect/FIRRTL/Transforms/InferResets.cpp:443`
Real maintainability TODO: inferred-reset port-reuse logic is explicitly called brittle/error-prone and needs simplification or safer normalization.

### [ ] 1541. `lib/Dialect/FIRRTL/Transforms/InferResets.cpp:746`
`llvm_unreachable("unsupported type")` here is defensive exhaustiveness for internal aggregate traversal, not direct unresolved feature debt.

### [ ] 1542. `lib/Conversion/ImportVerilog/TimingControls.cpp:42`
Real TODO: `SV 16.16` handling for edge-unspecified events in LTL conversion remains policy-based (`Both` fallback) rather than fully specified semantics.

### [ ] 1543. `lib/Conversion/ImportVerilog/TimingControls.cpp:175`
Global clocking support is restricted to specific symbol kinds; unsupported `$global_clock` symbol forms are rejected.

### [ ] 1544. `lib/Conversion/ImportVerilog/TimingControls.cpp:245`
Generic event-control lowering still rejects unhandled timing-control kinds, indicating incomplete timing-control coverage.

### [ ] 1545. `lib/Conversion/ImportVerilog/TimingControls.cpp:1584`
Sequence event-list lowering only accepts signal-event entries; other event kinds remain unsupported.

### [ ] 1546. `lib/Conversion/ImportVerilog/TimingControls.cpp:1644`
Global clocking expansion in sequence event lists supports only canonical signal-event forms; other event kinds are unsupported.

### [ ] 1547. `lib/Conversion/ImportVerilog/TimingControls.cpp:1649`
Same as entry 1543 in sequence-event-list path: unsupported global clocking symbol kinds are rejected.

### [ ] 1548. `lib/Conversion/ImportVerilog/TimingControls.cpp:1669`
Clocking-block expansion in sequence event lists supports only specific canonical event shapes; unsupported kinds are rejected.

### [ ] 1549. `lib/Conversion/ImportVerilog/TimingControls.cpp:1991`
Delay-control lowering remains partial: unhandled delay-control kinds still diagnose as unsupported.

### [ ] 1550. `lib/Conversion/ImportVerilog/TimingControls.cpp:2083`
Same root as entry 1543 for cycle-delay clock-source resolution: unsupported global clocking symbol kinds are rejected.

### [ ] 1551. `lib/Conversion/ImportVerilog/TimingControls.cpp:2274`
LTL clock-control lowering remains partial; unhandled clock-control kinds still fail with unsupported diagnostics.

### [ ] 1552. `lib/Conversion/ImportVerilog/TimingControls.cpp:2407`
Cycle-delay lowering only accepts default clocking symbols that resolve to supported clocking-block forms; other symbol kinds are unsupported.

### [ ] 1553. `lib/Conversion/ImportVerilog/TimingControls.cpp:2417`
Same as entry 1552 for global clocking fallback in cycle delay: unsupported symbol kinds are rejected.

### [ ] 1554. `lib/Conversion/ImportVerilog/TimingControls.cpp:2427`
Cycle-delay lowering requires canonical signal-event forms; unsupported clocking event kinds remain unhandled.

### [ ] 1555. `lib/Conversion/ImportVerilog/TimingControls.cpp:2500`
Generic timing-control conversion still has a default unsupported branch, indicating incomplete coverage across `slang::ast::TimingControlKind`.

### [ ] 1556. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:134`
`SkipLoopWithUnsupportedExitBranch` is intentional regression-test naming for a pass limitation case, not unresolved implementation debt at this line.

### [ ] 1557. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:135`
Same as entry 1556: module name in a test fixture for unsupported-shape behavior, not a standalone code gap.

### [ ] 1558. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:176`
`CHECK-LABEL` for unsupported-exit-condition coverage is test expectation text, not actionable debt.

### [ ] 1559. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:177`
Same as entry 1558: test module label for known unsupported loop-exit pattern.

### [ ] 1560. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:197`
`SkipLoopWithUnsupportedInductionVariable1` is expected test naming for limitation coverage, not a TODO marker.

### [ ] 1561. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:198`
Same as entry 1560: fixture module declaration, non-actionable from gap perspective.

### [ ] 1562. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:244`
`SkipLoopWithUnsupportedInductionVariable2` is another test-coverage label, not unresolved implementation at this location.

### [ ] 1563. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:245`
Same as entry 1562: fixture module declaration for unsupported-case regression coverage.

### [ ] 1564. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:317`
`SkipLoopWithUnsupportedInitialInductionVariableValue` is test naming text and should be filtered from actionable scans.

### [ ] 1565. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:318`
Same as entry 1564: fixture module declaration, not project debt.

### [ ] 1566. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:360`
`SkipLoopWithUnsupportedIncrement` is a regression label documenting pass behavior, not an unresolved implementation line.

### [ ] 1567. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:361`
Same as entry 1566: module fixture line, non-actionable.

### [ ] 1568. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:382`
`SkipLoopWithUnsupportedBounds` is test nomenclature for a known limitation case and should be filtered as scan noise.

### [ ] 1569. `test/Dialect/LLHD/Transforms/unroll-loops.mlir:383`
Same as entry 1568: fixture module declaration, not a codebase TODO.

### [ ] 1570. `lib/Conversion/ImportVerilog/Statements.cpp:191`
`kUnsupportedSvaAttr` is metadata key definition for continue-mode placeholders, not an implementation gap by itself.

### [ ] 1571. `lib/Conversion/ImportVerilog/Statements.cpp:192`
Same as entry 1570 for `kUnsupportedSvaReasonAttr` declaration scaffolding.

### [ ] 1572. `lib/Conversion/ImportVerilog/Statements.cpp:193`
Same as entry 1571; string constant value is part of diagnostic metadata plumbing, not unresolved feature work.

### [ ] 1573. `lib/Conversion/ImportVerilog/Statements.cpp:435`
`emitUnsupportedConcurrentAssertionPlaceholder` callsite reflects real unsupported-concurrent-assertion fallback behavior, but this line itself is support scaffolding.

### [ ] 1574. `lib/Conversion/ImportVerilog/Statements.cpp:438`
`continueOnUnsupportedSVA` guard is policy gating (fallback enabled/disabled), not a standalone feature gap.

### [ ] 1575. `lib/Conversion/ImportVerilog/Statements.cpp:473`
Setting `circt.unsupported_sva` attribute is annotation plumbing for placeholder ops, not unresolved implementation debt.

### [ ] 1576. `lib/Conversion/ImportVerilog/Statements.cpp:474`
Same as entry 1575 for reason-attribute assignment.

### [ ] 1577. `lib/Conversion/ImportVerilog/Statements.cpp:476`
Warning text confirms a real gap class: concurrent assertion forms still require skip/placeholder behavior in continue mode.

### [ ] 1578. `lib/Conversion/ImportVerilog/Statements.cpp:1919`
Real pattern-matching gap: case-pattern lowering still rejects some `PatternKind` forms as unsupported.

### [ ] 1579. `lib/Conversion/ImportVerilog/Statements.cpp:1973`
Real case `inside` pattern gap: set-membership pattern matching is explicitly unsupported.

### [ ] 1580. `lib/Conversion/ImportVerilog/Statements.cpp:2555`
Real immediate-assertion gap: only a subset of `AssertionKind` values is lowered; unsupported kinds still hard-error.

### [ ] 1581. `lib/Conversion/ImportVerilog/Statements.cpp:2843`
`tolerateUnsupportedSVA` helper lambda is continue-mode fallback scaffolding, not a direct feature gap by itself.

### [ ] 1582. `lib/Conversion/ImportVerilog/Statements.cpp:2844`
Same as entry 1581: this call routes through placeholder emission for unsupported concurrent assertions, but is not itself missing functionality.

### [ ] 1583. `lib/Conversion/ImportVerilog/Statements.cpp:2894`
Real concurrent-assertion lowering gap: some properties still fail lowering and must fall back to continue-mode placeholders.

### [ ] 1584. `lib/Conversion/ImportVerilog/Statements.cpp:3005`
Same as entry 1583 in a second conversion path: property lowering can still fail and rely on unsupported-SVA fallback.

### [ ] 1585. `lib/Conversion/ImportVerilog/Statements.cpp:3164`
Same fallback pattern as entry 1583 for unsupported concurrent assertion kinds.

### [ ] 1586. `lib/Conversion/ImportVerilog/Statements.cpp:3166`
Real gap: unsupported concurrent assertion kinds still hard-error when continue mode cannot absorb them.

### [ ] 1587. `lib/Conversion/ImportVerilog/Statements.cpp:3557`
Same runtime gap as entry 1521: checkpoint/restart tasks remain unsupported in `circt-sim`.

### [ ] 1588. `lib/Conversion/ImportVerilog/Statements.cpp:3632`
Real (legacy) coverage gap: PLD gate-array tasks are intentionally unimplemented/deprecated and still rejected.

### [ ] 1589. `lib/Conversion/ImportVerilog/Statements.cpp:3649`
Same as entry 1588; explicit diagnostic path for unsupported legacy PLD array tasks.

### [ ] 1590. `lib/Conversion/ImportVerilog/Statements.cpp:3655`
Real (legacy) coverage gap: abstract stochastic queue tasks are deprecated and not implemented.

### [ ] 1591. `lib/Conversion/ImportVerilog/Statements.cpp:3660`
Same as entry 1590; explicit diagnostic for unsupported legacy stochastic queue tasks.

### [ ] 1592. `lib/Conversion/ImportVerilog/Statements.cpp:3669`
Same runtime gap as entry 1522: `$sdf_annotate` support is missing (SDF timing annotation not implemented).

### [ ] 1593. `lib/Conversion/ImportVerilog/Statements.cpp:4304`
Delayed event triggering remains partial: delayed blocking trigger forms are unsupported (only selected delayed nonblocking behavior is handled).

### [ ] 1594. `lib/Conversion/ImportVerilog/Statements.cpp:5087`
Generic statement visitor fallback still rejects unhandled statement kinds, indicating incomplete statement conversion coverage.

### [ ] 1595. `lib/Conversion/ImportVerilog/Types.cpp:350`
Type conversion remains partial: unhandled Slang type kinds still fail through generic unsupported-type diagnostics.

### [ ] 1596. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/Trace.h:44`
Real ESI backend gap: full trace replay/compare mode is still TODO and not yet supported.

### [ ] 1597. `lib/Conversion/ImportVerilog/FormatStrings.cpp:220`
Format-string lowering supports only a subset of SystemVerilog specifiers; unsupported format specifiers still hard-error.

### [ ] 1598. `test/Dialect/Synth/tech-mapper.mlir:88`
This is test commentary describing missing area-flow optimization, not a code TODO site; actionable work belongs in Synth tech-mapper implementation.

### [ ] 1599. `test/Dialect/Synth/tech-mapper.mlir:92`
Same as entry 1598: fixture `FIXME` in test expectations reflecting area-flow absence, not standalone debt in test text itself.

### [ ] 1600. `test/Dialect/LLHD/Transforms/inline-calls-errors.mlir:30`
Expected-error text for recursion inliner limitations (`unsupported in --ir-hw`) is regression fixture content, not unresolved work at this line.

### [ ] 1601. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:25`
Real architectural TODO: RPC server wrapper is not yet a fully proper backend abstraction.

### [ ] 1602. `lib/Dialect/ESI/runtime/cpp/include/esi/backends/RpcServer.h:36`
Real concurrency/API gap: manifest-setting race is documented; DPI/API ordering needs redesign to guarantee manifest-before-connect.

### [ ] 1603. `test/Dialect/Synth/tech-mapper-error.mlir:59`
Expected-error text in a negative test (`Unsupported operation for truth table simulation`) is fixture content, not direct implementation debt at this location.

### [ ] 1604. `lib/Dialect/ESI/runtime/cosim_dpi_server/CMakeLists.txt:14`
Real packaging/linkage TODO: current install path ships a dummy simulator library (`MtiPli`) to avoid runtime link errors; build integration remains fragile.

### [ ] 1605. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:133`
Real DPI API debt: endpoint registration API should be split by direction and return handles rather than string lookup paths.

### [ ] 1606. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:357`
Real low-level cosim gap: MMIO entry-point path was left broken during gRPC conversion and remains unrevived.

### [ ] 1607. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:384`
Concrete unimplemented path: MMIO read-response DPI function currently hard-aborts via `assert(false && "unimplemented")`.

### [ ] 1608. `lib/Dialect/ESI/runtime/cosim_dpi_server/DpiEntryPoints.cpp:396`
Concrete unimplemented path: MMIO write-response DPI function also hard-aborts via `assert(false && "unimplemented")`.

### [ ] 1609. `lib/Conversion/ImportVerilog/HierarchicalNames.cpp:1591`
Real TODO: hierarchical-name traversal currently skips unhandled symbol kinds silently; missing diagnostics/coverage for skipped symbols.

### [ ] 1610. `lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:86`
Real enhancement TODO: cosim driver lacks a max-speed throttling option for interactive debugging workflows.

### [ ] 1611. `lib/Dialect/ESI/runtime/cosim_dpi_server/driver.cpp:94`
Real TODO: cosim driver does not yet model an ESI reset handshake sequence.

### [ ] 1612. `lib/Dialect/ESI/runtime/cpp/include/esi/Utils.h:78`
Real runtime perf TODO: queue callback path copies front data defensively; zero-copy/safer reference ownership is missing.

### [ ] 1613. `test/Dialect/Synth/lut-mapper.mlir:1`
This is test commentary describing a known optimization gap (non-minimal cut filtering), not a direct implementation site.

### [ ] 1614. `lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:335`
Real API TODO: callback mode lacks a completion/notification handle for caller-observable flow control.

### [ ] 1615. `lib/Dialect/ESI/runtime/cpp/include/esi/Ports.h:458`
Real API-cleanliness TODO: `BundlePort::getAs()` constness is acknowledged as likely wrong due current const access patterns.

### [ ] 1616. `lib/Dialect/Comb/CombFolds.cpp:1091`
Real canonicalization TODO: `and` folding only combines trailing constants and misses general constant coalescing opportunities.

### [ ] 1617. `lib/Dialect/Comb/CombFolds.cpp:1109`
Real fold-coverage TODO: replicate-plus-mask optimization is limited to single-bit operands and not generalized.

### [ ] 1618. `lib/Dialect/Comb/CombFolds.cpp:1205`
Real simplification TODO: complement pattern `and(..., x, not(x)) -> 0` is not implemented.

### [ ] 1619. `lib/Dialect/Comb/CombFolds.cpp:1421`
Real simplification TODO: complement pattern `or(..., x, not(x)) -> all-ones` is not implemented.

### [ ] 1620. `lib/Dialect/Comb/CombFolds.cpp:2194`
Real fold TODO: mux-chain analysis does not yet decompose relational predicates such as `x < 2` into equivalent constant cases.

### [ ] 1621. `lib/Dialect/Comb/CombFolds.cpp:2507`
Real generalization TODO: common-mux-op reduction currently targets concat only and misses recurring and/or/xor/icmp forms.

### [ ] 1622. `lib/Dialect/Comb/CombFolds.cpp:3254`
Known FIXME: zero-bit concat handling is still constrained due upstream/merge limitations.

### [ ] 1623. `lib/Conversion/ImportVerilog/Structure.cpp:554`
Real interface-lowering gap: unsupported interface port types still fail in input-port connection path.

### [ ] 1624. `lib/Conversion/ImportVerilog/Structure.cpp:570`
Same as entry 1623 for output-port connection path: unsupported interface port types remain unhandled.

### [ ] 1625. `lib/Conversion/ImportVerilog/Structure.cpp:590`
Same as entry 1623 for inout interface connection path.

### [ ] 1626. `lib/Conversion/ImportVerilog/Structure.cpp:641`
Interface port lowering supports only specific `ArgumentDirection` forms; other directions still diagnose as unsupported.

### [ ] 1627. `lib/Conversion/ImportVerilog/Structure.cpp:671`
Same as entry 1626 in multi-port expansion path.

### [ ] 1628. `lib/Conversion/ImportVerilog/Structure.cpp:680`
Interface port symbol-kind coverage is partial; unsupported symbol kinds for interface ports still fail.

### [ ] 1629. `lib/Conversion/ImportVerilog/Structure.cpp:875`
Top-level structure conversion still has a generic unsupported-construct fallback, indicating incomplete member-kind coverage.

### [ ] 1630. `lib/Conversion/ImportVerilog/Structure.cpp:904`
Package conversion remains partial: unsupported package member kinds still error via generic fallback.

### [ ] 1631. `lib/Conversion/ImportVerilog/Structure.cpp:1269`
Interface instantiation handling cannot represent some unconnected interface-port scenarios and currently errors out.

### [ ] 1632. `lib/Conversion/ImportVerilog/Structure.cpp:1343`
Interface connection resolution supports only a subset of connection expression/symbol forms; unsupported forms still fail.

### [ ] 1633. `lib/Conversion/ImportVerilog/Structure.cpp:1445`
Unconnected-port materialization supports only net/variable internal symbols; other internal symbol kinds are unsupported.

### [ ] 1634. `lib/Conversion/ImportVerilog/Structure.cpp:1456`
Real TODO: inout unconnected-port handling is deferred and explicitly marked for later support.

### [ ] 1635. `lib/Conversion/ImportVerilog/Structure.cpp:1459`
Same root as entry 1634: unsupported port kinds in unconnected-port branch still error.

### [ ] 1636. `lib/Conversion/ImportVerilog/Structure.cpp:1465`
Same root as entry 1635 for non-`PortSymbol` cases.

### [ ] 1637. `lib/Conversion/ImportVerilog/Structure.cpp:1517`
Instance-port lowering still rejects unsupported port symbol kinds outside handled regular/multi-port cases.

### [ ] 1638. `lib/Conversion/ImportVerilog/Structure.cpp:1837`
`hasAnyIff` is a local boolean variable name in covergroup sampling logic, not a TODO or unsupported marker.

### [ ] 1639. `lib/Conversion/ImportVerilog/Structure.cpp:1841`
Same as entry 1638: this is routine variable assignment in control flow, not actionable gap debt.

### [ ] 1640. `lib/Conversion/ImportVerilog/Structure.cpp:1858`
Same as entry 1638: conditional use of `hasAnyIff` is normal logic and scanner noise.

### [ ] 1641. `lib/Conversion/ImportVerilog/Structure.cpp:1901`
Real net-lowering gap: user-defined/unknown net kinds are rejected as unsupported.

### [ ] 1642. `lib/Conversion/ImportVerilog/Structure.cpp:1948`
Continuous-assignment lowering only supports specific lvalue container types (ref/class-handle); other lvalue type shapes are unsupported.

### [ ] 1643. `lib/Conversion/ImportVerilog/Structure.cpp:1976`
Continuous assignments with unhandled timing-control kinds still fail with unsupported diagnostics.

### [ ] 1644. `lib/Conversion/ImportVerilog/Structure.cpp:2441`
Real UDP lowering gap: unsupported sequential UDP initial-value symbols are dropped.

### [ ] 1645. `lib/Conversion/ImportVerilog/Structure.cpp:2498`
`unsupportedSymbol` here is a helper parameter used for UDP-symbol decoding flow, not a standalone unsupported feature marker.

### [ ] 1646. `lib/Conversion/ImportVerilog/Structure.cpp:2522`
Same as entry 1645: local control flag assignment in parser helper, scanner false positive.

### [ ] 1647. `lib/Conversion/ImportVerilog/Structure.cpp:2530`
Same as entry 1645 for function signature plumbing of UDP-symbol decode state.

### [ ] 1648. `lib/Conversion/ImportVerilog/Structure.cpp:2537`
Same as entry 1645: local temporary flag for branch control, non-actionable.

### [ ] 1649. `lib/Conversion/ImportVerilog/Structure.cpp:2538`
Same as entry 1645: helper invocation carrying an `unsupported` temporary, not a gap on its own.

### [ ] 1650. `lib/Conversion/ImportVerilog/Structure.cpp:2541`
Same as entry 1645: branch on temporary decode flag, scanner-level noise.

### [ ] 1651. `lib/Conversion/ImportVerilog/Structure.cpp:2542`
Same as entry 1645: temporary flag propagation assignment.

### [ ] 1652. `lib/Conversion/ImportVerilog/Structure.cpp:2546`
Same as entry 1649: second helper invocation using local unsupported flag.

### [ ] 1653. `lib/Conversion/ImportVerilog/Structure.cpp:2549`
Same as entry 1650: control-flow check over temporary decode state.

### [ ] 1654. `lib/Conversion/ImportVerilog/Structure.cpp:2550`
Same as entry 1651: temporary flag propagation, not direct implementation debt.

### [ ] 1655. `lib/Conversion/ImportVerilog/Structure.cpp:2611`
Real UDP lowering gap: unsupported UDP row token shapes are dropped.

### [ ] 1656. `lib/Conversion/ImportVerilog/Structure.cpp:2624`
Same as entry 1655 for alternative row-shape path.

### [ ] 1657. `lib/Conversion/ImportVerilog/Structure.cpp:2634`
`unsupportedSymbol` local variable in UDP row loop is parser plumbing, not standalone debt.

### [ ] 1658. `lib/Conversion/ImportVerilog/Structure.cpp:2636`
Same as entry 1657: helper-call argument wiring for temporary unsupported-state tracking.

### [ ] 1659. `lib/Conversion/ImportVerilog/Structure.cpp:2639`
Same as entry 1657: temporary flag check controlling fallback/error behavior.

### [ ] 1660. `lib/Conversion/ImportVerilog/Structure.cpp:2640`
Real UDP lowering gap: unsupported UDP input-row symbols cause row drop/failure.

### [ ] 1661. `lib/Conversion/ImportVerilog/Structure.cpp:2684`
Real UDP lowering gap: unsupported shorthand edge-transition symbols are rejected.

### [ ] 1662. `lib/Conversion/ImportVerilog/Structure.cpp:2688`
Same parser-plumbing class as entry 1657: temporary `unsupportedSymbol` variable, not direct gap.

### [ ] 1663. `lib/Conversion/ImportVerilog/Structure.cpp:2692`
Same as entry 1662: helper wiring for unsupported-edge-symbol detection state.

### [ ] 1664. `lib/Conversion/ImportVerilog/Structure.cpp:2694`
Same as entry 1662: control-flow check on temporary symbol-support state.

### [ ] 1665. `lib/Conversion/ImportVerilog/Structure.cpp:2695`
Same root as entry 1661: unsupported UDP edge transition symbols are rejected.

### [ ] 1666. `lib/Conversion/ImportVerilog/Structure.cpp:2735`
Real UDP lowering gap: unsupported current-state row symbols in sequential UDP tables are rejected.

### [ ] 1667. `lib/Conversion/ImportVerilog/Structure.cpp:2742`
Real UDP lowering gap: unsupported UDP output-row symbols are rejected.

### [ ] 1668. `lib/Conversion/ImportVerilog/Structure.cpp:2747`
Same as entry 1667 for alternate output-symbol path.

### [ ] 1669. `lib/Conversion/ImportVerilog/Structure.cpp:3075`
Real primitive-modeling TODO: tristate/high-Z behavior for bufif/notif family is approximated and not modeled faithfully.

### [ ] 1670. `lib/Conversion/ImportVerilog/Structure.cpp:3132`
Same root as entry 1669 for MOS switch primitives: high-Z/tristate semantics are simplified and not fully implemented.

### [ ] 1671. `lib/Conversion/ImportVerilog/Structure.cpp:3188`
Real primitive-modeling TODO: complementary MOS (`cmos`/`rcmos`) semantics are currently simplified and lack proper tristate behavior.

### [ ] 1672. `lib/Conversion/ImportVerilog/Structure.cpp:3228`
Real primitive-modeling TODO: bidirectional switch primitives (`tran`/`rtran`) are approximated with symmetric continuous assigns, not full bidirectional semantics.

### [ ] 1673. `lib/Conversion/ImportVerilog/Structure.cpp:3289`
Real primitive-modeling TODO: controlled bidirectional switches (`tranif*`/`rtranif*`) ignore control semantics in current simplified lowering.

### [ ] 1674. `lib/Conversion/ImportVerilog/Structure.cpp:3311`
Primitive coverage remains incomplete: unhandled primitive names fall through to generic “unsupported primitive type” errors.

### [ ] 1675. `lib/Conversion/ImportVerilog/Structure.cpp:3324`
Module-member conversion has a generic unsupported fallback, indicating incomplete symbol-kind handling in module bodies.

### [ ] 1676. `lib/Conversion/ImportVerilog/Structure.cpp:3601`
Top-level definition conversion supports only selected definition kinds (module/program here); other definition kinds are still unsupported.

### [ ] 1677. `lib/Conversion/ImportVerilog/Structure.cpp:3657`
Generic interface ports without resolvable concrete interface definitions remain unsupported unless top-level generic interface ports are explicitly allowed.

### [ ] 1678. `lib/Conversion/ImportVerilog/Structure.cpp:3730`
Module port-list lowering is partial: unsupported non-port/non-multiport/non-interface symbols still hard-error.

### [ ] 1679. `lib/Conversion/ImportVerilog/Structure.cpp:4100`
Real port-mapping gap: some lowered ports cannot be mapped back to a `PortSymbol` and fail as unsupported.

### [ ] 1680. `lib/Conversion/ImportVerilog/Structure.cpp:4111`
Real port-mapping gap: some ports do not resolve to an internal symbol/expression target for wiring and currently error out.

### [ ] 1681. `lib/Conversion/ImportVerilog/Structure.cpp:7138`
Real class-lowering gap: virtual/interface method support in class forward-declaration handling is explicitly unimplemented.

### [ ] 1682. `lib/Conversion/ImportVerilog/Structure.cpp:7328`
Class member conversion still has a generic unsupported fallback for unhandled class-member construct kinds.

### [ ] 1683. `test/Dialect/Emit/Reduction/pattern-registration.mlir:1`
`UNSUPPORTED: system-windows` is test-runner metadata due platform-specific issues, not direct product feature debt at this line.

### [ ] 1684. `test/Dialect/Emit/Reduction/emit-op-eraser.mlir:1`
Same as entry 1683: platform exclusion metadata in tests, non-actionable from implementation-gap perspective.

### [ ] 1685. `test/Dialect/Kanagawa/Transforms/scoperef_tunneling.mlir:198`
This TODO documents a known Kanagawa pass limitation in a test fixture (`hw.module` placement constraints), signaling a real feature boundary but not a code TODO location itself.

### [ ] 1686. `lib/Dialect/Comb/Transforms/IntRangeAnnotations.cpp:110`
Real pass-coverage TODO: integer-range overflow annotation currently omits subtraction support.

### [ ] 1687. `test/Dialect/HW/svEmitErrors.mlir:3`
Expected-error text for an intentionally unsupported Verilog type is regression fixture content, not unresolved debt at this line.

### [ ] 1688. `test/Analysis/firrtl-test-instance-info.mlir:15`
`CHECK` expectation text (`anyInstanceUnderDut`) in analysis tests is output-fixture data, not a TODO/unsupported implementation marker.

### [ ] 1689. `test/Analysis/firrtl-test-instance-info.mlir:17`
Same as entry 1688 for `anyInstanceUnderEffectiveDut` check text.

### [ ] 1690. `test/Analysis/firrtl-test-instance-info.mlir:19`
Same as entry 1688 for `anyInstanceUnderLayer` check text.

### [ ] 1691. `test/Analysis/firrtl-test-instance-info.mlir:24`
Same check-fixture class as entry 1688; non-actionable scanner hit.

### [ ] 1692. `test/Analysis/firrtl-test-instance-info.mlir:26`
Same as entry 1689 in another expectation block.

### [ ] 1693. `test/Analysis/firrtl-test-instance-info.mlir:28`
Same as entry 1690 in another expectation block.

### [ ] 1694. `test/Analysis/firrtl-test-instance-info.mlir:33`
Same check-fixture false positive pattern as entry 1688.

### [ ] 1695. `test/Analysis/firrtl-test-instance-info.mlir:35`
Same check-fixture false positive pattern as entry 1689.

### [ ] 1696. `test/Analysis/firrtl-test-instance-info.mlir:37`
Same check-fixture false positive pattern as entry 1690.

### [ ] 1697. `test/Analysis/firrtl-test-instance-info.mlir:42`
Same check-fixture false positive pattern as entry 1688.

### [ ] 1698. `test/Analysis/firrtl-test-instance-info.mlir:44`
Same check-fixture false positive pattern as entry 1689.

### [ ] 1699. `test/Analysis/firrtl-test-instance-info.mlir:46`
Same check-fixture false positive pattern as entry 1690.

### [ ] 1700. `test/Analysis/firrtl-test-instance-info.mlir:51`
Same check-fixture false positive pattern as entry 1688.

### [ ] 1701. `test/Analysis/firrtl-test-instance-info.mlir:53`
`CHECK` expectation text (`anyInstanceUnderEffectiveDut`) in analysis regression output is fixture content, not unresolved implementation work.

### [ ] 1702. `test/Analysis/firrtl-test-instance-info.mlir:55`
Same as entry 1701 for `anyInstanceUnderLayer` check text in test expectations.

### [ ] 1703. `test/Analysis/firrtl-test-instance-info.mlir:67`
Same check-fixture false positive class as entry 1688 (`anyInstanceUnderDut` output assertion).

### [ ] 1704. `test/Analysis/firrtl-test-instance-info.mlir:69`
Same check-fixture false positive class as entry 1701 (`anyInstanceUnderEffectiveDut` assertion text).

### [ ] 1705. `test/Analysis/firrtl-test-instance-info.mlir:71`
Same check-fixture false positive class as entry 1702 (`anyInstanceUnderLayer` assertion text).

### [ ] 1706. `test/Analysis/firrtl-test-instance-info.mlir:95`
Same as entry 1703: `FileCheck` output assertion line, non-actionable from gap perspective.

### [ ] 1707. `test/Analysis/firrtl-test-instance-info.mlir:97`
Same as entry 1704: test expectation line, not product TODO debt.

### [ ] 1708. `test/Analysis/firrtl-test-instance-info.mlir:99`
Same as entry 1705: test expectation line, scanner noise.

### [ ] 1709. `test/Analysis/firrtl-test-instance-info.mlir:110`
Same `CHECK-NEXT` fixture category as entry 1703 for DUT-membership output.

### [ ] 1710. `test/Analysis/firrtl-test-instance-info.mlir:112`
Same `CHECK-NEXT` fixture category as entry 1704 for effective-DUT output.

### [ ] 1711. `test/Analysis/firrtl-test-instance-info.mlir:114`
Same `CHECK-NEXT` fixture category as entry 1705 for layer-membership output.

### [ ] 1712. `test/Analysis/firrtl-test-instance-info.mlir:153`
Same check-fixture false positive pattern as prior entries (`anyInstanceUnderDut` assertion text).

### [ ] 1713. `test/Analysis/firrtl-test-instance-info.mlir:155`
Same check-fixture false positive pattern as entry 1712 (`anyInstanceUnderEffectiveDut` assertion text).

### [ ] 1714. `test/Analysis/firrtl-test-instance-info.mlir:173`
Same regression-output expectation line class; not actionable feature debt.

### [ ] 1715. `test/Analysis/firrtl-test-instance-info.mlir:175`
Same as entry 1714 for adjacent `effectiveDut` expectation text.

### [ ] 1716. `test/Analysis/firrtl-test-instance-info.mlir:177`
Same as entry 1714 for adjacent `underLayer` expectation text.

### [ ] 1717. `test/Analysis/firrtl-test-instance-info.mlir:179`
`CHECK-NEXT: anyInstanceInDesign` is expected analysis output text in test fixtures, not unresolved implementation.

### [ ] 1718. `test/Analysis/firrtl-test-instance-info.mlir:181`
`CHECK-NEXT: anyInstanceInEffectiveDesign` is also fixture expectation text and scanner noise in TODO audits.

### [ ] 1719. `test/Analysis/firrtl-test-instance-info.mlir:186`
Same check-fixture false positive pattern as entry 1714.

### [ ] 1720. `test/Analysis/firrtl-test-instance-info.mlir:188`
Same check-fixture false positive pattern as entry 1715.

### [ ] 1721. `test/Analysis/firrtl-test-instance-info.mlir:190`
Same check-fixture false positive pattern as entry 1716.

### [ ] 1722. `test/Analysis/firrtl-test-instance-info.mlir:192`
Same check-fixture false positive pattern as entry 1717.

### [ ] 1723. `test/Analysis/firrtl-test-instance-info.mlir:194`
Same check-fixture false positive pattern as entry 1718.

### [ ] 1724. `test/Analysis/firrtl-test-instance-info.mlir:209`
Same expected-output assertion-line class as entry 1703; non-actionable.

### [ ] 1725. `test/Analysis/firrtl-test-instance-info.mlir:211`
Same expected-output assertion-line class as entry 1704; non-actionable.

### [ ] 1726. `test/Analysis/firrtl-test-instance-info.mlir:213`
Same expected-output assertion-line class as entry 1705; non-actionable.

### [ ] 1727. `test/Analysis/firrtl-test-instance-info.mlir:215`
Same expected-output assertion-line class as entry 1717; non-actionable.

### [ ] 1728. `test/Analysis/firrtl-test-instance-info.mlir:217`
Same expected-output assertion-line class as entry 1718; non-actionable.

### [ ] 1729. `test/Analysis/firrtl-test-instance-info.mlir:231`
`CHECK: anyInstanceInDesign` line is test oracle text, not a TODO marker.

### [ ] 1730. `test/Analysis/firrtl-test-instance-info.mlir:233`
`CHECK-NEXT: anyInstanceInEffectiveDesign` line is test oracle text, not unresolved implementation debt.

### [ ] 1731. `test/Analysis/firrtl-test-instance-info.mlir:237`
`CHECK: anyInstanceInDesign` is regression output expectation text, not an implementation TODO marker.

### [ ] 1732. `test/Analysis/firrtl-test-instance-info.mlir:239`
`CHECK-NEXT: anyInstanceInEffectiveDesign` is test oracle text and non-actionable in gap scans.

### [ ] 1733. `test/Analysis/firrtl-test-instance-info.mlir:247`
Same check-fixture false positive class as entry 1731.

### [ ] 1734. `test/Analysis/firrtl-test-instance-info.mlir:249`
Same check-fixture false positive class as entry 1732.

### [ ] 1735. `test/Analysis/firrtl-test-instance-info.mlir:262`
`CHECK` expectation for `anyInstanceInEffectiveDesign` is test output text, not a code gap.

### [ ] 1736. `test/Analysis/firrtl-test-instance-info.mlir:284`
Same check-fixture false positive pattern as entry 1731.

### [ ] 1737. `test/Analysis/firrtl-test-instance-info.mlir:286`
Same check-fixture false positive pattern as entry 1732.

### [ ] 1738. `test/Analysis/firrtl-test-instance-info.mlir:290`
Same expected-output fixture line class as entry 1731.

### [ ] 1739. `test/Analysis/firrtl-test-instance-info.mlir:292`
Same expected-output fixture line class as entry 1732.

### [ ] 1740. `test/Analysis/firrtl-test-instance-info.mlir:296`
Same expected-output fixture line class as entry 1731.

### [ ] 1741. `test/Analysis/firrtl-test-instance-info.mlir:298`
Same expected-output fixture line class as entry 1732.

### [ ] 1742. `test/Analysis/firrtl-test-instance-info.mlir:302`
Same expected-output fixture line class as entry 1731.

### [ ] 1743. `test/Analysis/firrtl-test-instance-info.mlir:304`
Same expected-output fixture line class as entry 1732.

### [ ] 1744. `test/Analysis/firrtl-test-instance-info.mlir:308`
Same expected-output fixture line class as entry 1731.

### [ ] 1745. `test/Analysis/firrtl-test-instance-info.mlir:310`
Same expected-output fixture line class as entry 1732.

### [ ] 1746. `test/Analysis/firrtl-test-instance-info.mlir:314`
Same expected-output fixture line class as entry 1731.

### [ ] 1747. `test/Analysis/firrtl-test-instance-info.mlir:316`
Same expected-output fixture line class as entry 1732.

### [ ] 1748. `test/Analysis/firrtl-test-instance-info.mlir:320`
Same expected-output fixture line class as entry 1731.

### [ ] 1749. `test/Analysis/firrtl-test-instance-info.mlir:322`
Same expected-output fixture line class as entry 1732.

### [ ] 1750. `test/Analysis/firrtl-test-instance-info.mlir:385`
Same expected-output fixture line class as entry 1731.

### [ ] 1751. `test/Analysis/firrtl-test-instance-info.mlir:387`
Same expected-output fixture line class as entry 1732.

### [ ] 1752. `test/Analysis/firrtl-test-instance-info.mlir:391`
Same expected-output fixture line class as entry 1731.

### [ ] 1753. `test/Analysis/firrtl-test-instance-info.mlir:393`
Same expected-output fixture line class as entry 1732.

### [ ] 1754. `test/Analysis/firrtl-test-instance-info.mlir:397`
Same expected-output fixture line class as entry 1731.

### [ ] 1755. `test/Analysis/firrtl-test-instance-info.mlir:399`
Same expected-output fixture line class as entry 1732.

### [ ] 1756. `test/Analysis/firrtl-test-instance-info.mlir:403`
Same expected-output fixture line class as entry 1731.

### [ ] 1757. `test/Analysis/firrtl-test-instance-info.mlir:405`
Same expected-output fixture line class as entry 1732.

### [ ] 1758. `test/Analysis/firrtl-test-instance-info.mlir:409`
Same expected-output fixture line class as entry 1731.

### [ ] 1759. `test/Analysis/firrtl-test-instance-info.mlir:411`
Same expected-output fixture line class as entry 1732.

### [ ] 1760. `test/Analysis/firrtl-test-instance-info.mlir:415`
Same expected-output fixture line class as entry 1731.

### [ ] 1761. `test/Analysis/firrtl-test-instance-info.mlir:417`
`CHECK-NEXT: anyInstanceInEffectiveDesign` is regression expectation text, not unresolved implementation.

### [ ] 1762. `test/Analysis/firrtl-test-instance-info.mlir:421`
Same check-fixture false positive class as entry 1731.

### [ ] 1763. `test/Analysis/firrtl-test-instance-info.mlir:423`
Same check-fixture false positive class as entry 1761.

### [ ] 1764. `test/Analysis/firrtl-test-instance-info.mlir:471`
`CHECK` output assertion (`anyInstanceUnderLayer`) in analysis tests is fixture text, not TODO debt.

### [ ] 1765. `test/Analysis/firrtl-test-instance-info.mlir:473`
`CHECK` output assertion (`anyInstanceInEffectiveDesign`) is fixture text, not TODO debt.

### [ ] 1766. `test/Analysis/firrtl-test-instance-info.mlir:478`
Same check-fixture false positive pattern as entry 1764.

### [ ] 1767. `test/Analysis/firrtl-test-instance-info.mlir:480`
Same check-fixture false positive pattern as entry 1765.

### [ ] 1768. `test/Analysis/firrtl-test-instance-info.mlir:501`
Same check-fixture false positive class as entry 1703 (`anyInstanceUnderDut` oracle output).

### [ ] 1769. `test/Analysis/firrtl-test-instance-info.mlir:510`
Same check-fixture false positive class as entry 1768.

### [ ] 1770. `test/Analysis/firrtl-test-instance-info.mlir:522`
Same check-fixture false positive class as entry 1768.

### [ ] 1771. `lib/Dialect/Comb/Transforms/BalanceMux.cpp:216`
Real optimization TODO: balanced-priority-mux split point currently uses a simple midpoint, not timing-arrival-aware partitioning.

### [ ] 1772. `test/Dialect/HW/parameters.mlir:33`
`#hw.param.verbatim<"xxx">` here is placeholder text in test expectations, not unresolved work.

### [ ] 1773. `test/Dialect/HW/parameters.mlir:34`
Same as entry 1772: test fixture instance with verbatim `"xxx"` parameter, scanner false positive.

### [ ] 1774. `test/Dialect/HW/parameters.mlir:128`
Same as entry 1772 for `CHECK-SAME` expectation text containing `"xxx"`.

### [ ] 1775. `test/Dialect/HW/parameters.mlir:129`
Same as entry 1773 for actual test IR line using verbatim `"xxx"` placeholder.

### [ ] 1776. `lib/Conversion/ImportVerilog/Expressions.cpp:213`
`hasAnyIff` is a local control variable in covergroup sampling logic, not a TODO/unsupported marker.

### [ ] 1777. `lib/Conversion/ImportVerilog/Expressions.cpp:217`
Same as entry 1776: local variable assignment, non-actionable scanner hit.

### [ ] 1778. `lib/Conversion/ImportVerilog/Expressions.cpp:233`
Same as entry 1776: conditional on local `hasAnyIff`, not implementation debt.

### [ ] 1779. `lib/Conversion/ImportVerilog/Expressions.cpp:1670`
Real expression-lowering gap: element select is only supported for a restricted set of base types; other base types hard-error.

### [ ] 1780. `lib/Conversion/ImportVerilog/Expressions.cpp:2480`
Real assertion-expression gap: local assertion variable history access still rejects unsupported/non-unpacked variable types.

### [ ] 1781. `lib/Conversion/ImportVerilog/Expressions.cpp:2937`
Real symbol-resolution gap: unsupported arbitrary symbol-reference forms still fail with diagnostics.

### [ ] 1782. `lib/Conversion/ImportVerilog/Expressions.cpp:3132`
Assignment lowering supports only selected lvalue categories; unsupported lvalue types in assignment still error.

### [ ] 1783. `lib/Conversion/ImportVerilog/Expressions.cpp:3396`
Unary-expression lowering is incomplete: unhandled unary operators still fall through generic unsupported diagnostics.

### [ ] 1784. `lib/Conversion/ImportVerilog/Expressions.cpp:3830`
Real equality-operator gap: some unpacked aggregate equality operand combinations remain unsupported.

### [ ] 1785. `lib/Conversion/ImportVerilog/Expressions.cpp:3844`
Real equality-operator gap: some dynamic unpacked (queue/assoc/open-array) equality operand combinations remain unsupported.

### [ ] 1786. `lib/Conversion/ImportVerilog/Expressions.cpp:3991`
Real inequality-operator gap: some unpacked aggregate inequality operand combinations remain unsupported.

### [ ] 1787. `lib/Conversion/ImportVerilog/Expressions.cpp:4005`
Real inequality-operator gap: some dynamic unpacked inequality operand combinations remain unsupported.

### [ ] 1788. `lib/Conversion/ImportVerilog/Expressions.cpp:4143`
Real case-equality gap: some dynamic unpacked case-equality operand combinations remain unsupported.

### [ ] 1789. `lib/Conversion/ImportVerilog/Expressions.cpp:4153`
Real case-equality gap: some unpacked aggregate case-equality operand combinations remain unsupported.

### [ ] 1790. `lib/Conversion/ImportVerilog/Expressions.cpp:4182`
Real case-inequality gap: some dynamic unpacked case-inequality operand combinations remain unsupported.

### [ ] 1791. `lib/Conversion/ImportVerilog/Expressions.cpp:4192`
Real case-inequality gap: some unpacked aggregate case-inequality operand combinations remain unsupported.

### [ ] 1792. `lib/Conversion/ImportVerilog/Expressions.cpp:4261`
Binary-expression lowering remains incomplete: unhandled binary operators still hit generic unsupported diagnostics.

### [ ] 1793. `lib/Conversion/ImportVerilog/Expressions.cpp:4486`
Conditional-expression lowering currently supports only a single condition arm; multi-condition conditional forms are unsupported.

### [ ] 1794. `lib/Conversion/ImportVerilog/Expressions.cpp:4945`
Real coverpoint-method coverage gap: unimplemented coverpoint methods currently warn and fall back to regular function-call lowering.

### [ ] 1795. `lib/Conversion/ImportVerilog/Expressions.cpp:5030`
Real cross-method coverage gap: unimplemented covercross methods currently warn and fall back to regular function-call lowering.

### [ ] 1796. `lib/Conversion/ImportVerilog/Expressions.cpp:5072`
`hasAnyIff` is a local helper variable for covergroup iff collection, not a TODO/unsupported marker.

### [ ] 1797. `lib/Conversion/ImportVerilog/Expressions.cpp:5077`
Same as entry 1796: local variable assignment in control flow, non-actionable scanner hit.

### [ ] 1798. `lib/Conversion/ImportVerilog/Expressions.cpp:5082`
Same as entry 1796: branch on local helper state, not implementation debt.

### [ ] 1799. `lib/Conversion/ImportVerilog/Expressions.cpp:5253`
Real covergroup-method coverage gap: additional covergroup methods beyond supported set currently warn and fall back to regular calls.

### [ ] 1800. `lib/Conversion/ImportVerilog/Expressions.cpp:5450`
Real process-method coverage gap: unhandled `process` class methods currently warn and fall back to regular function-call lowering.

### [ ] 1801. `lib/Conversion/ImportVerilog/Expressions.cpp:5717`
Real mailbox-method coverage gap: unhandled mailbox built-in methods remain not yet implemented.

### [ ] 1802. `lib/Conversion/ImportVerilog/Expressions.cpp:5839`
Real semaphore-method coverage gap: unhandled semaphore built-in methods remain not yet implemented.

### [ ] 1803. `lib/Conversion/ImportVerilog/Expressions.cpp:6986`
`$cast` handling supports selected destination categories; unsupported destination lvalue types still error.

### [ ] 1804. `lib/Conversion/ImportVerilog/Expressions.cpp:8503`
Real system-call coverage gap: unrecognized system calls in expression lowering still fail through generic unsupported-system-call diagnostics.

### [ ] 1805. `lib/Conversion/ImportVerilog/Expressions.cpp:8762`
Assignment-pattern lowering remains partial: unsupported target type forms still hard-error.

### [ ] 1806. `lib/Conversion/ImportVerilog/Expressions.cpp:8897`
Streaming-concatenation lowering for queue/mixed dynamic forms has element-type restrictions; unsupported element types still error.

### [ ] 1807. `lib/Conversion/ImportVerilog/Expressions.cpp:9692`
Expression visitor still has a generic unsupported fallback for unhandled expression kinds.

### [ ] 1808. `lib/Conversion/ImportVerilog/Expressions.cpp:9763`
Real assertion-local-variable gap: default-init/materialization path still rejects unsupported local assertion variable types.

### [ ] 1809. `lib/Conversion/ImportVerilog/Expressions.cpp:9789`
Same local assertion variable type-coverage gap as entry 1808 in history/offset handling path.

### [ ] 1810. `lib/Conversion/ImportVerilog/Expressions.cpp:9798`
Same local assertion variable type-coverage gap as entry 1808 in final type check path.

### [ ] 1811. `lib/Conversion/ImportVerilog/Expressions.cpp:10449`
Packed-to-SBV conversion intentionally rejects packed aggregates containing `time` fields; broader time-aware aggregate conversion is unsupported.

### [ ] 1812. `lib/Conversion/ImportVerilog/Expressions.cpp:10491`
SBV-to-packed conversion has the same `time`-field restriction as entry 1811; such aggregate conversions remain unsupported.

### [ ] 1813. `lib/Conversion/ImportVerilog/Expressions.cpp:10809`
Real TODO: conversion materialization still relies on generic `ConversionOp` for many cases; dedicated conversion ops are missing for broader coverage/precision.

### [ ] 1814. `lib/Conversion/ImportVerilog/Expressions.cpp:10977`
Real system-call coverage gap: unsupported arity-0 system calls still fail through default unsupported diagnostics.

### [ ] 1815. `lib/Conversion/ImportVerilog/Expressions.cpp:11813`
Real system-call coverage gap: unsupported arity-1 system calls still fail through default unsupported diagnostics.

### [ ] 1816. `lib/Conversion/ImportVerilog/Expressions.cpp:11930`
Legacy stochastic queue functions are explicitly marked deprecated/unimplemented in the arity-2 path.

### [ ] 1817. `lib/Conversion/ImportVerilog/Expressions.cpp:11934`
Same as entry 1816: `$q_exam` is explicitly unsupported as a legacy stochastic queue function.

### [ ] 1818. `lib/Conversion/ImportVerilog/Expressions.cpp:11941`
Same as entry 1816: `$q_full` is explicitly unsupported as a legacy stochastic queue function.

### [ ] 1819. `lib/Conversion/ImportVerilog/Expressions.cpp:11952`
Real system-call coverage gap: unsupported arity-2 system calls still fail through default unsupported diagnostics.

### [ ] 1820. `lib/Conversion/ImportVerilog/Expressions.cpp:11969`
Real system-call coverage gap: unsupported arity-3 system calls still fail through default unsupported diagnostics.

### [ ] 1821. `test/Dialect/Arc/lower-clocks-to-funcs-errors.mlir:30`
Expected-error fixture line for a known Arc limitation (multiple `InitialOp`s unsupported in this pass path), not direct implementation debt at this test line.

### [ ] 1822. `test/Dialect/Arc/lower-clocks-to-funcs-errors.mlir:31`
Same as entry 1821 for multiple `FinalOp`s: regression expectation text, scanner false positive at this location.

### [ ] 1823. `test/Dialect/Seq/canonicalization.mlir:106`
Test TODO comment about future constant-aggregate-attribute usage is fixture/planning text, not a production-code gap line.

### [ ] 1824. `test/Dialect/Arc/latency-retiming.mlir:13`
Test comment notes potential canonicalization opportunity; this is fixture discussion, not direct unresolved implementation at this line.

### [ ] 1825. `test/Dialect/Arc/canonicalizers.mlir:135`
Test TODO about preserving a non-folded op for coverage is test-maintenance note, not product implementation debt.

### [ ] 1826. `test/Dialect/FIRRTL/infer-domains-infer-all-errors.mlir:122`
Test TODO describing verifier reliance is fixture commentary, not direct code gap at this line.

### [ ] 1827. `test/Dialect/Arc/mux-to-control-flow.mlir:40`
Test TODO about statement-coverage quality is test-quality commentary, not unresolved feature implementation.

### [ ] 1828. `test/Dialect/HW/hw-convert-bitcasts.mlir:49`
“Don’t crash on unsupported types” is test intent documentation, not a TODO marker in implementation code.

### [ ] 1829. `test/Dialect/HW/hw-convert-bitcasts.mlir:50`
`@unsupported` module label in a negative test is fixture naming, not unresolved work.

### [ ] 1830. `test/Dialect/HW/hw-convert-bitcasts.mlir:51`
Same as entry 1829: test module declaration for unsupported-type behavior coverage.

### [ ] 1831. `test/Dialect/Arc/infer-state-properties.mlir:248`
Test TODO about adding a usage-shape case is regression-suite planning, not a production code gap line.

### [ ] 1832. `test/Dialect/Arc/infer-state-properties.mlir:249`
Same as entry 1831 for reset/enable gating coverage note.

### [ ] 1833. `test/Dialect/HW/hw-convert-bitcasts-errors.mlir:3`
`@unsupported` module in an errors test is fixture naming, not unresolved implementation debt.

### [ ] 1834. `test/Dialect/HW/hw-convert-bitcasts-errors.mlir:4`
Expected-error assertion for unsupported output type is test oracle text, not actionable debt at this line.

### [ ] 1835. `test/Dialect/HW/hw-convert-bitcasts-errors.mlir:6`
Expected-error assertion for unsupported input type is test oracle text, not actionable debt at this line.

### [ ] 1836. `test/Dialect/FIRRTL/annotations-errors.fir:6`
Expected-error line for invalid/unsupported annotation format is negative-test fixture text, not a TODO site.

### [ ] 1837. `test/Dialect/FIRRTL/annotations-errors.fir:17`
Same as entry 1836 for a second invalid annotation fixture case.

### [ ] 1838. `test/Dialect/Arc/Reduction/state-elimination.mlir:1`
`UNSUPPORTED: system-windows` is platform test-runner metadata, not product feature debt.

### [ ] 1839. `test/Dialect/RTG/Transform/linear-scan-register-allocation.mlir:69`
`@unsupportedUser` here is test naming for a scenario, not an unresolved implementation marker.

### [ ] 1840. `test/Dialect/HW/errors.mlir:418`
Expected-error assertion for unsupported `hw.array` dimension kind is negative-test oracle text, not unresolved work at this line.

### [ ] 1841. `test/Dialect/Arc/Reduction/pattern-registration.mlir:1`
`UNSUPPORTED: system-windows` is test metadata, not implementation debt.

### [ ] 1842. `lib/Dialect/Calyx/Transforms/CompileControl.cpp:34`
Real Calyx TODO: helper for state-bit-width calculation notes need for a better built-in operation/utility abstraction.

### [ ] 1843. `lib/Dialect/Calyx/Transforms/CompileControl.cpp:116`
Real Calyx TODO: `GroupDoneOp` guard/source canonicalization is deferred and currently handled ad hoc in the pass.

### [ ] 1844. `test/Dialect/RTG/Transform/emit-rtg-isa-assembly-errors.mlir:1`
`unsupported-instructions=` in a test `RUN` line is CLI option usage for negative testing, not unresolved feature debt.

### [ ] 1845. `lib/Dialect/Calyx/Transforms/CalyxLoweringUtils.cpp:798`
Calyx lowering currently assumes integer block arguments; non-integer block argument types are unsupported and guarded by assert.

### [ ] 1846. `lib/Dialect/Calyx/Transforms/CalyxLoweringUtils.cpp:822`
Calyx lowering currently assumes integer/float function return types; other return types are unsupported and guarded by assert.

### [ ] 1847. `test/Dialect/RTG/Transform/emit-rtg-isa-assembly.mlir:2`
`unsupported-instructions` in a `RUN` invocation is test configuration text, not a codebase TODO marker.

### [ ] 1848. `test/Dialect/FIRRTL/simplify-mems.mlir:471`
Test TODO references a real optimization idea (deduplicating generated pipelining artifacts), but this line itself is fixture commentary.

### [ ] 1849. `test/Dialect/Handshake/errors.mlir:12`
`invalid_mux_unsupported_select` is negative-test naming, not unresolved implementation debt at this line.

### [ ] 1850. `test/Dialect/Handshake/errors.mlir:13`
Expected-error assertion for unsupported indexing type is test oracle text, not a product TODO location.

### [ ] 1851. `test/Dialect/Handshake/errors.mlir:28`
`invalid_cmerge_unsupported_index` is negative-test naming, not unresolved implementation debt at this line.

### [ ] 1852. `test/Dialect/Handshake/errors.mlir:29`
Expected-error assertion for unsupported indexing type is test oracle text, not an actionable TODO marker.

### [ ] 1853. `test/Dialect/FIRRTL/SFCTests/ExtractSeqMems/Simple2.fir:1`
`UNSUPPORTED: system-windows` is platform test-runner metadata, not product feature debt.

### [ ] 1854. `test/Dialect/HW/Reduction/pattern-registration.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1855. `test/Dialect/FIRRTL/SFCTests/ExtractSeqMems/Compose.fir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1856. `test/Dialect/HW/Reduction/hw-operand-forwarder.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1857. `test/Dialect/SV/hw-legalize-modules-packed-arrays.mlir:14`
Expected-error text for unsupported packed-array expression is regression fixture content, not unresolved implementation at this location.

### [ ] 1858. `test/Dialect/HW/Reduction/hw-module-output-pruner.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1859. `test/Dialect/FIRRTL/ref.mlir:211`
This test TODO references a real idea (inferring forceable ref result existence/type), but this line is fixture commentary rather than production-code debt.

### [ ] 1860. `test/Dialect/RTG/Reduction/virtual-register-constantifier.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1861. `test/Dialect/HW/Reduction/hw-module-input-pruner.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1862. `lib/Dialect/Calyx/Export/CalyxEmitter.cpp:974`
`convertToDouble()` here is normal constant emission logic and does not itself indicate unsupported/unimplemented behavior; this appears to be a scanner false positive.

### [ ] 1863. `test/Dialect/FIRRTL/Reduction/root-extmodule-port-pruner.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1864. `test/Dialect/HW/Reduction/hw-module-externalizer.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1865. `test/Dialect/FIRRTL/Reduction/module-port-pruner.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1866. `test/Dialect/HW/Reduction/hw-constantifier.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1867. `test/Dialect/Moore/vtable-partial-impl.mlir:5`
Commentary describing prior unimplemented-method handling in a regression narrative is test documentation, not active unresolved debt at this line.

### [ ] 1868. `test/Dialect/Moore/vtable-partial-impl.mlir:8`
Same as entry 1867: explanatory test comment about vtable behavior, non-actionable in debt scans.

### [ ] 1869. `test/Dialect/SV/errors.mlir:253`
Expected-error text for unsupported type is negative-test oracle content, not direct implementation TODO at this location.

### [ ] 1870. `test/Dialect/SV/errors.mlir:260`
Same as entry 1869: expected-error fixture line.

### [ ] 1871. `test/Dialect/FIRRTL/Reduction/extmodule-port-pruner.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1872. `test/Dialect/FIRRTL/Reduction/list-create-element-remover.mlir:4`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1873. `test/Dialect/FIRRTL/Reduction/layer-disable.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1874. `test/Dialect/FIRRTL/Reduction/extmodule-instance-remover.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1875. `test/Dialect/FIRRTL/inferRW-errors.mlir:9`
Expected-error assertion (“unsupported by InferReadWrite”) is negative-test oracle text, not unresolved implementation at this test line.

### [ ] 1876. `test/Dialect/FIRRTL/Reduction/instance-stubber.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1877. `test/Dialect/FIRRTL/Reduction/connect-forwarder.mlir:1`
Same as entry 1853: platform exclusion metadata in test harness.

### [ ] 1878. `test/Dialect/FIRRTL/parse-errors.fir:848`
`UnsupportedRadixSpecifiedIntegerLiterals` is a test circuit name in parser negative tests, not a TODO marker.

### [ ] 1879. `test/Dialect/FIRRTL/parse-errors.fir:849`
Same as entry 1878 for module naming in parser negative-test fixture.

### [ ] 1880. `test/Dialect/FIRRTL/parse-errors.fir:893`
`UnsupportedStringEncodedIntegerLiterals` is a parser negative-test circuit name, not unresolved implementation text at this line.

### [ ] 1881. `test/Dialect/FIRRTL/parse-errors.fir:894`
Same as entry 1880 for module naming inside a negative parser test fixture.

### [ ] 1882. `test/Dialect/FIRRTL/parse-errors.fir:896`
Expected-error assertion (“string-encoded integer literals unsupported after FIRRTL 3.0.0”) is parser negative-test oracle text.

### [ ] 1883. `test/Dialect/FIRRTL/parse-errors.fir:929`
`UnsupportedVersionDeclGroups` is a negative-test circuit name for version-gating behavior, not a TODO marker.

### [ ] 1884. `test/Dialect/FIRRTL/parse-errors.fir:943`
`UnsupportedVersionLayer` is a negative-test circuit name for version-gating behavior, not a TODO marker.

### [ ] 1885. `test/Dialect/FIRRTL/parse-errors.fir:950`
`UnsupportedVersionGroups` is a negative-test circuit name, not unresolved implementation text at this line.

### [ ] 1886. `test/Dialect/FIRRTL/parse-errors.fir:951`
Same as entry 1885 for module naming in the same negative parser fixture.

### [ ] 1887. `test/Dialect/FIRRTL/parse-errors.fir:966`
`UnsupportedLayerBlock` is negative-test fixture naming for version-gated parsing, not a TODO marker.

### [ ] 1888. `test/Dialect/FIRRTL/parse-errors.fir:967`
Same as entry 1887 for module naming in the same fixture.

### [ ] 1889. `test/Dialect/FIRRTL/parse-errors.fir:974`
`UnsupportedLayerConvention` is negative-test fixture naming, not unresolved implementation text at this location.

### [ ] 1890. `test/Dialect/FIRRTL/parse-errors.fir:1396`
`PublicModuleUnsupported` is a negative-test circuit name for version gating, not a TODO marker.

### [ ] 1891. `test/Dialect/FIRRTL/parse-errors.fir:1398`
Same as entry 1890 for module naming in the corresponding parser error test.

### [ ] 1892. `test/Dialect/FIRRTL/lower-domains.mlir:327`
Test TODO comment about stronger ExpandWhens verification is test-maintenance commentary, not production-code debt at this line.

### [ ] 1893. `test/Dialect/FIRRTL/Reduction/simplify-resets.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not product feature debt.

### [ ] 1894. `test/Dialect/Moore/canonicalizers.mlir:94`
`bXXXXX101` in `CHECK` output is expected-value fixture text for unknown-bit behavior, not unresolved implementation.

### [ ] 1895. `test/Dialect/Moore/canonicalizers.mlir:146`
`hXXXXXX` in `CHECK-DAG` is expected-value fixture text, not unresolved implementation.

### [ ] 1896. `test/Dialect/Moore/canonicalizers.mlir:164`
Same as entry 1895: expected unknown-bit pattern in test oracle text.

### [ ] 1897. `test/Dialect/Moore/canonicalizers.mlir:191`
Same as entry 1895: expected unknown-bit pattern in test oracle text.

### [ ] 1898. `test/Dialect/FIRRTL/Reduction/port-pruner.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1899. `lib/Dialect/Calyx/CalyxOps.cpp:100`
Real Calyx TODO: verification strategy (`verifyNotComplexSource`) is temporary until native Calyx wire declarations support lands.

### [ ] 1900. `lib/Dialect/Calyx/CalyxOps.cpp:1544`
This parser comment about optional guard parsing (“accompanying `?`”) is normal implementation text; scanner false positive.

### [ ] 1901. `test/Dialect/FIRRTL/Reduction/pattern-registration.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1902. `test/Dialect/FIRRTL/Reduction/object-inliner.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1903. `test/Dialect/FIRRTL/Reduction/node-symbol-remover.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1904. `test/Dialect/OM/Reduction/unused-class-remover.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1905. `test/Dialect/FIRRTL/Reduction/module-swapper.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1906. `test/Dialect/FIRRTL/lower-chirrtl.mlir:53`
Test TODO about `FileCheck` quoting (`[[[DATA]]]`) is fixture/tooling note, not production feature debt.

### [ ] 1907. `test/Dialect/OM/Reduction/object-to-unknown-replacer.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1908. `test/Dialect/FIRRTL/Reduction/module-externalizer.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1909. `test/Dialect/FIRRTL/parse-basic.fir:1570`
Test TODO notes missing/incorrect object flow-checking behavior; this is a real area to improve, but this line is test-maintenance commentary.

### [ ] 1910. `test/Dialect/FIRRTL/canonicalization.mlir:2606`
`FIXME` about zero-width memory elimination is test-suite planning text pointing to a real optimization/testing gap, not production code at this line.

### [ ] 1911. `test/Dialect/FIRRTL/canonicalization.mlir:2983`
Test TODO (“Move to an appropriate place”) is suite-organization commentary, not unresolved product implementation at this line.

### [ ] 1912. `test/Dialect/FIRRTL/canonicalization.mlir:3036`
Same as entry 1911: test-suite organization TODO, non-actionable from feature-gap perspective.

### [ ] 1913. `test/Dialect/OM/Reduction/list-element-pruner.mlir:6`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1914. `test/Dialect/FIRRTL/Reduction/memory-stubber.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1915. `test/Dialect/OM/Reduction/class-parameter-pruner.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1916. `test/Dialect/FIRRTL/Reduction/issue-3555.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1917. `test/Dialect/FIRRTL/infer-resets.mlir:140`
Test TODO gating disabled checks on issue `#1303` is regression-maintenance commentary, not a production code TODO line.

### [ ] 1918. `test/Dialect/FIRRTL/infer-resets.mlir:605`
Test TODO about checking extra reset-port absence is test-coverage commentary, not direct implementation debt.

### [ ] 1919. `test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:3`
`@unsupported` module naming in a negative test is fixture text, not unresolved implementation.

### [ ] 1920. `test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:4`
Expected-error assertion for unsupported inout-port consumer op is negative-test oracle text, not a TODO marker.

### [ ] 1921. `test/Dialect/SV/EliminateInOutPorts/hw-eliminate-inout-ports-errors.mlir:10`
Expected-error assertion for multiple inout writers is negative-test oracle text, not a TODO marker.

### [ ] 1922. `test/Dialect/OM/Reduction/class-field-pruner.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1923. `test/Dialect/FIRRTL/Reduction/force-dedup.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1924. `test/Dialect/OM/Reduction/anycast-of-unknown-simplifier.mlir:3`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1925. `lib/Dialect/Arc/ArcDialect.cpp:29`
Real Arc TODO: dialect inliner legality for region-to-region inlining is left unimplemented (currently always `false`).

### [ ] 1926. `lib/Dialect/Arc/ArcDialect.cpp:34`
Real Arc TODO: dialect inliner legality for op-to-region inlining is likewise unimplemented (currently always `false`).

### [ ] 1927. `test/Dialect/FIRRTL/Reduction/eager-inliner.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1928. `test/Dialect/FIRRTL/inliner-errors.mlir:58`
Expected-error assertion for unsupported inlining of `sv.ifdef` is negative-test oracle text, not unresolved debt at this line.

### [ ] 1929. `lib/Dialect/Arc/ArcFolds.cpp:241`
Real optimization TODO: canonicalization only hoists constant-like outputs today; broader safe hoisting of non-clocked/side-effect-free ops is deferred.

### [ ] 1930. `test/Dialect/FIRRTL/Reduction/constantifier.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1931. `test/Dialect/FIRRTL/Reduction/connect-source-operand-forward.mlir:1`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1932. `lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:58`
Real Arc cost-model TODO: `ReplicateOp` runtime cost estimate is coarse and may not match eventual lowering (sext/mul alternatives).

### [ ] 1933. `lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:90`
Real Arc cost-model TODO: HW `ArraySliceOp`/`ArrayConcatOp` cost estimation is missing.

### [ ] 1934. `lib/Dialect/Arc/Interfaces/RuntimeCostEstimateInterfaceImpl.cpp:104`
Real Arc cost-model TODO: `scf.if` estimate is acknowledged as arbitrary and needs calibration.

### [ ] 1935. `lib/Dialect/Arc/ArcCostModel.cpp:19`
Real Arc FIXME: operation cost constants are provisional and need refinement with better cost data.

### [ ] 1936. `test/Dialect/FIRRTL/Reduction/annotation-remover.mlir:4`
`UNSUPPORTED: system-windows` is platform test metadata, not implementation debt.

### [ ] 1937. `lib/Dialect/Arc/Transforms/InferMemories.cpp:95`
Real capability boundary: Arc InferMemories currently supports only write-latency `1`; other memory write latencies hard-fail as unsupported.

### [ ] 1938. `test/Conversion/VerifToSMT/bmc-for-smtlib-no-property-live-llvm-call.mlir:4`
Comment describing unsupported LLVM ops in a propertyless-BMC test is fixture rationale, not unresolved implementation debt.

### [ ] 1939. `test/Conversion/VerifToSMT/bmc-concat-unknown-bounds.mlir:3`
`CHECK` expectation for unsupported sequence lowering is negative-test oracle text, not a TODO marker.

### [ ] 1940. `lib/Dialect/Arc/Transforms/Dedup.cpp:204`
Real Arc TODO: structural-equivalence dedup does not yet bail on non-terminator fan-in extraneous ops; robustness/precision improvement is deferred.

### [ ] 1941. `test/Dialect/FIRRTL/lower-types.mlir:579`
Test TODO (“Enable this”) is regression-maintenance commentary, not direct production implementation debt at this line.

### [ ] 1942. `test/Dialect/FIRRTL/lower-types.mlir:602`
Same as entry 1941: test-gating TODO in fixture text.

### [ ] 1943. `test/Dialect/FIRRTL/lower-types.mlir:621`
Same as entry 1941: test-gating TODO in fixture text.

### [ ] 1944. `lib/Dialect/Arc/Transforms/LowerState.cpp:582`
`setInsertionPoint(ifClockOp.thenYield())` is normal builder-placement code, not an unsupported/todo marker; scanner false positive.

### [ ] 1945. `lib/Dialect/Arc/Transforms/LowerState.cpp:609`
Same as entry 1944 for reset path insertion-point placement.

### [ ] 1946. `lib/Dialect/Arc/Transforms/LowerState.cpp:641`
Same as entry 1944 for enable path insertion-point placement.

### [ ] 1947. `lib/Dialect/Arc/Transforms/LowerState.cpp:731`
Same as entry 1944 for memory-write lowering insertion-point placement.

### [ ] 1948. `lib/Dialect/Arc/Transforms/LowerState.cpp:749`
Same as entry 1944 for nested enable-guard insertion-point placement.

### [ ] 1949. `lib/Dialect/Arc/Transforms/AllocateState.cpp:97`
Real Arc TODO: trace-tap annotation assumes `StateWriteOp` is model-local; sharing across models is unresolved.

### [ ] 1950. `lib/Dialect/Arc/Transforms/AllocateState.cpp:174`
Real Arc capability boundary: unsupported allocation op kinds hit an assert-fail path instead of graceful diagnostics.

### [ ] 1951. `test/Conversion/SeqToSV/error.mlir:3`
Test TODO about improving the error message is test-quality commentary, not a production-code TODO site.

### [ ] 1952. `lib/Dialect/Arc/Transforms/ArcCanonicalizer.cpp:130`
Real Arc TODO: listener bookkeeping for inserted symbol-defining ops can miss existing users when symbols pre-exist.

### [ ] 1953. `lib/Dialect/Arc/Transforms/ArcCanonicalizer.cpp:143`
Same as entry 1952 for insertion callback path.

### [ ] 1954. `test/Dialect/FIRRTL/errors.mlir:325`
Expected-error line with placeholder `"xxx"` is negative-test oracle text, not unresolved implementation debt.

### [ ] 1955. `test/Dialect/FIRRTL/errors.mlir:326`
Fixture line intentionally using wrong port name `"xxx"` to trigger diagnostics; non-actionable in gap scans.

### [ ] 1956. `test/Dialect/FIRRTL/errors.mlir:2169`
Expected-error assertion for wrong class-port name `"xxx"` is test oracle text, not production TODO debt.

### [ ] 1957. `test/Dialect/FIRRTL/errors.mlir:2170`
Fixture line intentionally using wrong port name `"xxx"` for error testing; non-actionable in gap scans.

### [ ] 1958. `test/Dialect/FIRRTL/errors.mlir:2978`
Expected-error assertion using placeholder symbols (`@XXX::@YYY`) is negative-test oracle text.

### [ ] 1959. `test/Dialect/FIRRTL/errors.mlir:2979`
Fixture `firrtl.bind <@XXX::@YYY>` line is intentional unresolved-target test input, not project debt.

### [ ] 1960. `lib/Dialect/Arc/Transforms/SplitLoops.cpp:240`
Real Arc TODO: loop splitting currently over-approximates and splits broadly; minimal loop-involved split selection is not implemented.

### [ ] 1961. `lib/Dialect/Arc/Transforms/LowerLUT.cpp:318`
Real Arc TODO: LUT calculator is pass-local and could be promoted to reusable analysis.

### [ ] 1962. `test/Dialect/FIRRTL/grand-central-view-errors.mlir:146`
Comment (“Invalid / unsupported class in element”) is test documentation for a negative case, not direct production-code debt.

### [ ] 1963. `lib/Dialect/Arc/Transforms/LowerClocksToFuncs.cpp:87`
Real Arc limitation: models containing multiple `InitialOp`s are currently unsupported in this lowering pass.

### [ ] 1964. `lib/Dialect/Arc/Transforms/LowerClocksToFuncs.cpp:93`
Real Arc limitation: models containing multiple `FinalOp`s are currently unsupported in this lowering pass.

### [ ] 1965. `lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:249`
Real Arc TODO: conversion front-end currently handles mux only; support for additional select-like ops (e.g. `arith.select`) is deferred.

### [ ] 1966. `lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:256`
Real Arc TODO: conversion-benefit heuristic needs tuning/calibration.

### [ ] 1967. `lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:289`
Real Arc FIXME: pass assumes topological ordering in regions where muxes are transformed.

### [ ] 1968. `lib/Dialect/Arc/Transforms/MuxToControlFlow.cpp:291`
Real Arc FIXME: side effects are not yet considered in mux-to-control-flow conversion safety.

### [ ] 1969. `lib/Dialect/Arc/Transforms/MergeIfs.cpp:340`
`prevIfOp.thenYield().erase()` is ordinary merge-rewrite implementation detail, not an unsupported/todo marker; scanner false positive.

### [ ] 1970. `lib/Dialect/Arc/Transforms/IsolateClocks.cpp:44`
`moveToDomain(Operation *op)` is a method declaration, not a TODO/unsupported marker; scanner false positive.

### [ ] 1971. `lib/Dialect/Arc/Transforms/IsolateClocks.cpp:64`
`ClockDomain::moveToDomain` function definition itself is normal implementation, not a TODO/unsupported marker.

### [ ] 1972. `lib/Dialect/Arc/Transforms/IsolateClocks.cpp:85`
Real Arc TODO: when encountering an existing clock domain with the same clock, domains are not merged yet.

### [ ] 1973. `lib/Dialect/Arc/Transforms/IsolateClocks.cpp:104`
`if (moveToDomain(op))` is ordinary control flow in domain sinking, not an unresolved gap marker.

### [ ] 1974. `lib/Dialect/Arc/Transforms/IsolateClocks.cpp:210`
`if (domain.moveToDomain(op))` is ordinary control flow in pass logic, not an unresolved gap marker.

### [ ] 1975. `lib/Dialect/Arc/Transforms/InlineArcs.cpp:190`
Real Arc TODO: call-tracking in arc bodies is marked “make safe,” indicating unresolved safety/robustness handling in nested-call analysis.

### [ ] 1976. `lib/Dialect/Arc/Transforms/InlineArcs.cpp:343`
Real Arc TODO: region selection for inlining is hardcoded today; pass should ideally use legality/query interfaces instead.

### [ ] 1977. `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:113`
Real Arc TODO: reset extraction would ideally split arcs per reset kind; currently deferred due cost-model concerns.

### [ ] 1978. `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:120`
Real Arc limitation: `arc.state` reset handling currently supports only zero-reset semantics.

### [ ] 1979. `lib/Dialect/Arc/Transforms/InferStateProperties.cpp:176`
Real Arc TODO: enable extraction would ideally split arcs per enable kind; currently deferred due cost-model concerns.

### [ ] 1980. `test/Conversion/FIRRTLToHW/errors.mlir:46`
Comment “unknown widths are unsupported” in a conversion error test is fixture rationale, not a production TODO location.

### [ ] 1981. `test/Conversion/ArcToLLVM/lower-arc-to-llvm.mlir:151`
Test FIXME about placement (“does not really belong here”) is suite-organization commentary, not direct implementation debt.

### [ ] 1982. `test/Conversion/ExportAIGER/errors.mlir:3`
Test comment for unsupported variadic AND gates is negative-test description, not unresolved implementation at this line.

### [ ] 1983. `test/Conversion/ExportAIGER/errors.mlir:23`
Test comment for unsupported operation handling is negative-test description, not unresolved implementation at this line.

### [ ] 1984. `test/Conversion/ExportVerilog/pretty.mlir:243`
`@ForStatement` test module declaration is fixture content; scanner false positive from placeholder-style names.

### [ ] 1985. `test/Conversion/ExportVerilog/pretty.mlir:246`
Long identifier names (`xxxxxxxx...`) in test IR are intentional formatting/pretty-printer fixtures, not TODO debt.

### [ ] 1986. `test/Conversion/ExportVerilog/pretty.mlir:247`
Same as entry 1985: fixture identifier text, non-actionable.

### [ ] 1987. `test/Conversion/ExportVerilog/pretty.mlir:250`
Same as entry 1985: fixture identifier text, non-actionable.

### [ ] 1988. `test/Conversion/ExportVerilog/pretty.mlir:255`
`CHECK` line with long placeholder identifiers is expected-output fixture text, not unresolved implementation.

### [ ] 1989. `test/Conversion/ExportVerilog/name-legalize.mlir:84`
`%xxx` naming-conflict test fixture line is intentional regression input, not TODO debt.

### [ ] 1990. `test/Conversion/ExportVerilog/name-legalize.mlir:89`
`CHECK: .p1 (xxx)` is expected-output fixture text, not unresolved implementation.

### [ ] 1991. `test/Conversion/ExportVerilog/name-legalize.mlir:91`
Instance using `%xxx` is intentional conflict fixture input, not TODO debt.

### [ ] 1992. `test/Conversion/ExportVerilog/name-legalize.mlir:95`
`CHECK: reg ... xxx_0` is expected legalizer output fixture text, not unresolved implementation.

### [ ] 1993. `test/Conversion/ExportVerilog/name-legalize.mlir:96`
`sv.reg name "xxx"` is intentional test input for renaming/legalization behavior.

### [ ] 1994. `test/Conversion/ExportVerilog/hw-dialect.mlir:274`
Test TODO “Specify parameter declarations” points at a potential ExportVerilog enhancement, but this line is fixture commentary.

### [ ] 1995. `test/Conversion/ExportVerilog/hw-dialect.mlir:365`
Test FIXME notes an upstream MLIR parser limitation for `i0`; this is an external parser issue referenced by fixture comments.

### [ ] 1996. `test/Conversion/ExportVerilog/hw-dialect.mlir:1257`
Test FIXME (“Decl word should be localparam”) points to an ExportVerilog quality gap, but this line is fixture commentary.

### [ ] 1997. `test/Conversion/ExportVerilog/verilog-errors.mlir:3`
Expected-error text for unsupported Verilog type `f32` is negative-test oracle content, not TODO debt at this line.

### [ ] 1998. `test/Conversion/ExportVerilog/sv-dialect.mlir:1959`
`sv.macro.error "my message xxx yyy"` is test payload text; scanner hit on `xxx/yyy` is non-actionable.

### [ ] 1999. `test/Conversion/ExportVerilog/sv-dialect.mlir:1961`
`CHECK` expectation for `_ERROR_my_message_xxx_yyy` is fixture output text, not unresolved implementation.

### [x] 2000. `test/Conversion/MooreToCore/errors.mlir:7`
Status update (2026-02-28): this entry is stale. `@unsupportedConversion` is negative-test fixture naming, not unresolved MooreToCore implementation debt.

### [x] 2001. `test/Conversion/MooreToCore/interface-timing-after-inlining.sv:8`
Status update (2026-02-28): this item is stale. The test now serves as a regression that asserts `moore.wait_event`/`moore.detect_event` are lowered away after inlining; there is no active `FIXME`/`XFAIL` debt remaining for this case.

### [x] 2002. `test/Conversion/MooreToCore/basic.mlir:173`
Status update (2026-02-28): this entry is stale. `moore.constant bXXXXXX` is expected unknown-value regression payload (not a TODO marker), and current MooreToCore regression coverage remains passing.

### [x] 2003. `test/Conversion/MooreToCore/basic.mlir:628`
Status update (2026-02-28): this entry is stale. `always_comb` coverage exists in dedicated MooreToCore regression tests (for example `procedure-always-comb-latch.mlir` and related tests), and current suite passes.

### [x] 2004. `test/Conversion/MooreToCore/basic.mlir:629`
Status update (2026-02-28): same as entry 2003. `always_latch` lowering is covered in dedicated regressions, so this should no longer be tracked as an open TODO gap.

### [x] 2005. `test/Conversion/FSMToSV/test_errors.mlir:4`
Status update (2026-02-28): this entry is stale. The matched text is negative-test diagnostic oracle content, not unresolved product debt.

### [x] 2006. `test/Conversion/FSMToCore/errors.mlir:4`
Status update (2026-02-28): same closure as entry 2005.

### [ ] 2007. `test/Conversion/ImportAIGER/basic-binary.mlir:3`
This TODO is a test-maintenance note (regenerate binary AIG input from MLIR once exporter is upstreamed). It points to workflow/tooling follow-up, not immediate product debt on this line.

### [x] 2008. `test/Conversion/ImportVerilog/global-variable-init.sv:6`
Status update (2026-02-28): this entry is stale. `UNSUPPORTED: valgrind` is environment metadata and not actionable ImportVerilog feature debt.

### [x] 2009. `test/Conversion/ImportVerilog/procedures.sv:6`
Status update (2026-02-28): same closure as entry 2008. This is valgrind metadata, not a product-gap marker.

### [x] 2010. `test/Conversion/ImportVerilog/assoc_arrays.sv:6`
Status update (2026-02-28): same closure as entry 2008. This is valgrind metadata and not unresolved ImportVerilog semantics.

### [x] 2011. `test/Conversion/ImportVerilog/assoc_arrays.sv:32`
Status update (2026-02-28): this entry is stale. This is explanatory regression-comment text, not unresolved ImportVerilog feature debt.

### [x] 2012. `test/Conversion/ImportVerilog/assoc_arrays.sv:345`
Status update (2026-02-28): same closure as entry 2011.

### [x] 2013. `test/Conversion/ImportVerilog/constraint-solve.sv:5`
Status update (2026-02-28): same closure as entry 2008. This is valgrind metadata and not unresolved ImportVerilog debt.

### [x] 2014. `test/Conversion/ImportVerilog/constraint-method-call.sv:5`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2015. `test/Conversion/ImportVerilog/constraint-implication.sv:5`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2016. `test/Conversion/ImportVerilog/queues.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2017. `test/Conversion/ImportVerilog/types.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2018. `test/Conversion/ImportVerilog/basic.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2019. `test/Conversion/ImportVerilog/basic.sv:773`
Status update (2026-02-28): this entry is stale. `CHECK: ... hXXXXXXXX` is regression oracle payload for unknown-bit behavior, not unresolved implementation debt.

### [x] 2020. `test/Conversion/ImportVerilog/hierarchical-names.sv:6`
Status update (2026-02-28): same closure as entry 2008. This is valgrind metadata and not a feature gap.

### [x] 2021. `test/Conversion/ImportVerilog/errors.sv:5`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2022. `test/Conversion/ImportVerilog/inherited-virtual-methods.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2023. `test/Conversion/ImportVerilog/classes.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2024. `test/Conversion/ImportVerilog/four-state-constants.sv:45`
Status update (2026-02-28): this entry is stale. `4'bxxxx` is intentional four-state regression stimulus, not a TODO marker.

### [x] 2025. `test/Conversion/ImportVerilog/class-e2e.sv:5`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2026. `test/Conversion/ImportVerilog/builtins.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2027. `test/Conversion/ImportVerilog/queue-max-min.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2028. `test/Conversion/ImportVerilog/queue-delete-index.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2029. `test/Conversion/ImportVerilog/pre-post-randomize.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2030. `test/Conversion/ImportVerilog/continuous-assign-delay-one-step-supported.sv:11`
Status update (2026-02-28): this entry is stale. This `DIAG-NOT` line is a support oracle, not unresolved implementation debt.

### [x] 2031. `test/Conversion/ImportVerilog/static-property-fixes.sv:6`
Status update (2026-02-28): same closure as entry 2008. This is valgrind metadata and not active ImportVerilog debt.

### [x] 2032. `test/Conversion/ImportVerilog/delay-cycle-supported.sv:17`
Status update (2026-02-28): this entry is stale. This `DIAG-NOT` line is a support oracle, not unresolved implementation debt.

### [x] 2033. `test/Conversion/ImportVerilog/delay-one-step-supported.sv:8`
Status update (2026-02-28): same closure as entry 2032.

### [x] 2034. `test/Conversion/ImportVerilog/randomize.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2035. `test/Conversion/ImportVerilog/runtime-randomization.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2036. `test/Conversion/ImportVerilog/randomize-inline-control.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2037. `test/Conversion/ImportVerilog/time-type-handling.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2038. `test/Conversion/ImportVerilog/cross-select-intersect-open-range-wide-supported.sv:3`
Status update (2026-02-28): this entry is stale. The matched token is a test module identifier, not a TODO/debt marker.

### [x] 2039. `test/Conversion/ImportVerilog/uvm_classes.sv:6`
Status update (2026-02-28): same closure as entry 2008.

### [x] 2040. `test/Conversion/ImportVerilog/system-calls-complete.sv:5`
Status update (2026-02-28): this entry is stale. This is test-intent commentary, not unresolved implementation debt.

### [x] 2041. `test/Conversion/ImportVerilog/system-calls-complete.sv:8`
Status update (2026-02-28): same closure as entry 2040.

### [x] 2042. `test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv:3`
Status update (2026-02-28): this entry is stale. The matched token is a fixture module identifier, not unresolved implementation debt.

### [x] 2043. `test/Conversion/ImportVerilog/cross-select-intersect-plusminus-unsupported.sv:17`
Status update (2026-02-28): this entry is stale as a line-level tracker. It is expected-error oracle text; underlying cross-select capability gaps remain tracked at implementation-site entries.

### [ ] 2044. `test/Conversion/ImportVerilog/sva-sequence-match-item-coverage-sdf-static-subroutine.sv:18`
This diagnostic text captures a real runtime gap: `$sdf_annotate` is intentionally warned as unsupported because SDF timing annotation is not implemented in `circt-sim`.

### [x] 2045. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:2`
Status update (2026-02-28): this entry is stale. `RUN:` configuration text is fixture metadata, not unresolved implementation debt.

### [x] 2046. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:3`
Status update (2026-02-28): same closure as entry 2045.

### [x] 2047. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:6`
Status update (2026-02-28): this entry is stale. The module identifier is fixture naming, not a TODO marker.

### [x] 2048. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:10`
Status update (2026-02-28): this entry is stale. This fixture no longer tracks `$past` unsupported behavior at this line.

### [x] 2049. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:15`
Status update (2026-02-28): this entry is stale. It is strict-mode oracle content, not unresolved implementation debt.

### [x] 2050. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:17`
Status update (2026-02-28): same closure as entry 2049.

### [x] 2051. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:18`
Status update (2026-02-28): same closure as entry 2049.

### [x] 2052. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:21`
Status update (2026-02-28): this entry is stale. IR oracle checks are fixture assertions, not unresolved debt markers.

### [x] 2053. `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv:22`
Status update (2026-02-28): same closure as entry 2052.

### [ ] 2054. `test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:22`
This warning text reflects a real runtime capability gap: `$save` is unsupported because checkpoint/restart is not implemented in `circt-sim`.

### [ ] 2055. `test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:23`
Same real gap class as entry 2054 for `$restart` (checkpoint/restart not implemented).

### [ ] 2056. `test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:24`
Same real gap class as entry 2054 for `$incsave`.

### [ ] 2057. `test/Conversion/ImportVerilog/sva-sequence-match-item-debug-checkpoint-subroutine.sv:25`
Same real gap class as entry 2054 for `$reset` in checkpoint/restart flows.

### [x] 2058. `test/Conversion/ImportVerilog/sva-sequence-match-item-rewind-function.sv:20`
Status update (2026-02-28): this entry is stale. `DIAG-NOT` oracle text is not a debt marker.

### [x] 2059. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:2`
Status update (2026-02-28): this entry is stale. `RUN:` configuration text is fixture metadata.

### [x] 2060. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:3`
Status update (2026-02-28): same closure as entry 2059.

### [x] 2061. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:5`
Status update (2026-02-28): this entry is stale. The module identifier is fixture naming, not unresolved debt.

### [x] 2062. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:12`
Status update (2026-02-28): this entry is stale. Strict-mode oracle text is not a debt marker.

### [x] 2063. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:14`
Status update (2026-02-28): same closure as entry 2062.

### [x] 2064. `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv:16`
Status update (2026-02-28): same closure as entry 2062.

### [x] 2065. `test/Conversion/ImportVerilog/cross-select-with-wide-auto-domain-supported.sv:3`
Status update (2026-02-28): this entry is stale. The matched token is a fixture module identifier.

### [x] 2066. `test/Conversion/ImportVerilog/sva-sequence-match-item-stacktrace-function.sv:20`
Status update (2026-02-28): this entry is stale. `DIAG-NOT` oracle text is not a debt marker.

### [x] 2067. `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:2`
Status update (2026-02-28): this entry is stale. `RUN:` configuration text is fixture metadata.

### [x] 2068. `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:3`
Status update (2026-02-28): same closure as entry 2067.

### [x] 2069. `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:19`
Status update (2026-02-28): this entry is stale. This strict-mode check is an oracle assertion and no longer tracks an active unsupported path at this line.

### [x] 2070. `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:20`
Status update (2026-02-28): same closure as entry 2069.

### [x] 2071. `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv:21`
Status update (2026-02-28): same closure as entry 2069.

### [x] 2072. `test/Runtime/uvm/uvm_sequence_test.sv:12`
Status update (2026-02-28): this gap is closed in this workspace. The test now has execution-backed semantic coverage (`circt-verilog --ir-hw` + `circt-sim`) with a report-phase pass marker (`UVM_SEQUENCE_PATTERNS_PASS`). Runtime hardening in `uvm_sequence_item::set_id_info` now falls back to parent-sequence ID when request `sequence_id` is unavailable, so response routing in the sequence-pattern flow no longer fails silently.

### [x] 2073. `test/Runtime/uvm/uvm_ral_test.sv:11`
Status update (2026-02-28): this gap is closed in this workspace. The test was upgraded from parse-only to semantic runtime coverage (`circt-verilog --ir-hw` + `circt-sim`) with `FileCheck` assertions for `test_reg_field_ops`. Semantic expectations were corrected to reflect UVM RAL behavior (`set()` updates desired value, `predict()` updates mirrored value), so mirrored-value regressions are now validated by execution rather than silently passing parse-only checks.

### [x] 2074. `test/Runtime/uvm/uvm_coverage_test.sv:11`
Status update (2026-02-28): this gap is closed in this workspace. The test now runs semantically (`circt-verilog --ir-hw` + `circt-sim`) for both `mam_test` and `coverage_db_test`, with explicit pass markers (`UVM_COVERAGE_MAM_PASS`, `UVM_COVERAGE_DB_PASS`) and `UVM_ERROR` exclusion checks. During conversion, two silent semantic mismatches were fixed in test logic: `my_coverage` now derives from `uvm_object` (the runtime does not provide a concrete `uvm_coverage` base object), and MAM allocation checks now use deterministic `reserve_region` flows instead of policy-randomized `request_region`, which is not stable in this runtime path.
