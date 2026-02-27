# Project Gaps Manual Writeup

This file is a manual pass over `out/project-gap-todo.all.txt`, in source order, with one judgment paragraph per entry.

## Entries 1-40

### 1. `utils/check_avip_circt_sim_mode_parity.py:94`
This is mostly a strict input-validation path, not a product feature gap by itself: the script rejects allowlist rows whose mode is not `exact`, `prefix`, or `regex`. What is missing is either broader syntax support for mode aliases (for example `contains` or `glob`) or clearer user-facing documentation that only those three modes are valid. The practical fix is to choose one direction explicitly: either extend the parser and matching engine with additional modes plus tests, or keep the current grammar and document it prominently in the allowlist format spec and error help text.

### 2. `utils/run_mutation_cover.sh:406`
This guard currently limits mutation input format to `il|v|sv`, which blocks experiments that might naturally emit other formats (for example direct MLIR/FIRRTL artifacts) before conversion. The missing piece is either native support for extra formats or a normalized pre-conversion step so the script can accept broader inputs safely. A good fix is to add a format adapter layer that maps new formats into one of the existing internal representations, then add end-to-end CLI tests proving each accepted format reaches execution and is not silently misparsed.

### 3. `utils/summarize_circt_sim_jit_reports.py:113`
Like entry 1, this is a strict parser decision point: unknown allowlist modes are rejected rather than interpreted. The gap is not correctness in current behavior, but missing flexibility and possibly inconsistent expectations across tools that use similar allowlist files. The fix should be shared with the other allowlist consumers via a common parser module so mode semantics are centralized, and tests should assert identical behavior for accepted and rejected modes across all scripts.

### 4. `unittests/Runtime/MooreRuntimeTest.cpp:10074`
`unimplemented_test` here is intentionally used to verify warning behavior for an unregistered UVM test, so this line is not itself a missing implementation. The underlying gap signal is that grep-based scans treat this expected test fixture string as a real product TODO, which creates noise in project-level audits. The fix is to keep the behavior test, but tag the scan pipeline to suppress known test-fixture literals (or support a local suppression annotation) so audit output better tracks true implementation debt.

### 5. `unittests/Runtime/MooreRuntimeTest.cpp:10081`
This assertion confirms that the runtime warning output contains the string `unimplemented_test`, which again is deliberate negative-case testing and not an unfinished runtime feature. The missing capability is better audit precision: today, mechanical scanning cannot distinguish “testing an unimplemented path” from “this codebase still needs implementation.” The fix is to add scanner exclusions for known test assertions, or move such sentinel names to values that don’t trigger TODO-gap regexes while preserving test intent.

### 6. `unittests/Analysis/CMakeLists.txt:1`
This is a real TODO: linting tests are disabled pending “slang header fixes,” so analysis-lint coverage is currently absent from regular test execution. What is missing is the dependency hygiene or include structure needed to compile and run that test subtree reliably. The concrete fix is to resolve the header incompatibilities, re-enable `add_subdirectory(Linting)`, and add CI coverage so the linting target stays live instead of regressing back to a commented-out state.

### 7. `utils/run_opentitan_fpv_circt_lec.py:414`
The option text explicitly says cover evidence emission is off by default because case-level LEC checks cannot yet model native cover reachability semantics, so this is a genuine capability gap. What is missing is a trustworthy semantic bridge between FPV cover objectives and the case-level LEC health model. The fix is to define and implement that mapping first (including edge cases like vacuous/unreachable cover points), then enable cover evidence by default only after parity tests with the reference flow confirm interpretation is correct.

### 8. `utils/formal/lib/runner_common.py:117`
This is another allowlist-mode parser with strict rejection for unknown modes, mirroring entries 1 and 3. The gap is ecosystem consistency and maintainability: three separate parsers can drift over time and create confusingly different user behavior. The best fix is to consolidate allowlist parsing into one shared library API used by all formal/analysis scripts, with a single test suite that validates supported modes and diagnostics.

### 9. `utils/run_sv_tests_circt_sim.sh:516`
The script infers `UNSUPPORTED` by grepping free-form log text for phrases like `unsupported` or `not yet implemented`, which is brittle and can misclassify failures. What is missing is a structured machine-readable failure taxonomy from `circt-verilog`/`circt-sim` (for example diagnostic IDs or explicit status codes). The right fix is to key classification on structured diagnostics rather than substring matching, then keep a compatibility fallback only for legacy logs.

### 10. `utils/run_sv_tests_circt_sim.sh:517`
Setting `result="UNSUPPORTED"` from the grep heuristic encodes policy in a way that can hide real regressions if unrelated errors contain matching words. The missing piece is confidence that this bucket means “known feature gap” instead of “random failure with unfortunate wording.” The fix is to gate this assignment behind explicit diagnostic classes (or allowlisted IDs) and add regression tests showing true unsupported cases and true failures are separated deterministically.

### 11. `tools/circt-sim-compile/circt-sim-compile.cpp:630`
This rejection reason for `builtin.unrealized_conversion_cast:unsupported_arity` indicates the compilability check only handles a restricted cast arity shape and bails on others. What is missing is legalization support for multi-input/multi-result conversion-cast patterns that can appear after upstream transformations. The fix is to implement arity-general handling (or normalize such casts earlier), and add tests with representative multi-arity casts to ensure they are either lowered correctly or rejected with precise actionable diagnostics.

### 12. `tools/circt-sim-compile/circt-sim-compile.cpp:1417`
The `unsupported` tracking flag marks a conservative synthesis strategy for native module-init generation: unsupported constructs cause full module-init fallback. The gap is functional coverage, because currently many modules are skipped even when most of their init body is otherwise compatible. The fix is to incrementally widen the accepted op subset and dependency forms while preserving safety guarantees, then measure emitted-module ratio improvements with regression metrics.

### 13. `tools/circt-sim-compile/circt-sim-compile.cpp:1427`
`unsupported_call:<callee>` shows direct calls outside the allowlisted native-init set are treated as blockers, which prevents native init emission for modules with otherwise simple call usage. What is missing is a call-lowering path for a vetted subset of callees (for example pure helper functions with legal argument/result types). The fix is to add call eligibility analysis plus lowering support for that subset, and keep explicit skip reasons for remaining unsafe callees.

### 14. `tools/circt-sim-compile/circt-sim-compile.cpp:1429`
`unsupported_call:indirect` is a hard stop for indirect calls in module-init synthesis, which is understandable for safety but leaves capability on the table. What is missing is at least a restricted strategy for cases where the target set is statically known or single-target after analysis. The fix is to add conservative devirtualization/constant-target detection and support that narrow case, while retaining skip behavior for truly dynamic indirect calls.

### 15. `tools/circt-sim-compile/circt-sim-compile.cpp:1431`
The generic `unsupported_op:<name>` reason is broad and currently catches any non-native-init op, which can make diagnosis coarse and make feature expansion less targeted. The missing piece is finer-grained capability accounting per op family so developers can prioritize highest-impact additions. The fix is to split this into explicit op-category checks and maintain per-category counters/tests, so newly supported ops can be added deliberately without broad behavior changes.

### 16. `tools/circt-sim-compile/circt-sim-compile.cpp:1434`
Setting `unsupported = true` here causes an immediate break and module skip once an unsupported op is found, even if remaining ops are harmless. What is missing is partial extraction capability that could still compile a safe prefix/slice of init behavior. The fix is to decide whether partial semantics are acceptable for module-init; if yes, implement safe slicing plus explicit residual handling, and if not, keep full-skip semantics but improve reporting so users can quickly see the first blocking op.

### 17. `tools/circt-sim-compile/circt-sim-compile.cpp:1439`
`unsupported_op:hw.struct_extract` indicates struct extraction support is limited to a constrained form (`isSupportedNativeModuleInitStructExtract`). The missing behavior is broader handling of struct extraction shapes that are semantically safe for init-time evaluation. The fix is to expand the support predicate and corresponding lowering paths stepwise, with focused tests for each newly admitted struct-extract pattern.

### 18. `tools/circt-sim-compile/circt-sim-compile.cpp:1440`
This flag assignment is the control point that enforces the struct-extract limitation from entry 17. The capability gap is not this line itself but the narrow admissible struct-extract set behind it. The fix is to broaden that admissible set and keep this guard as the final safety net, while recording distinct skip reasons for every still-unsupported variant.

### 19. `tools/circt-sim-compile/circt-sim-compile.cpp:1446`
`unsupported_op:hw.struct_create` shows struct construction is also only partially supported during native module-init synthesis. The missing feature is handling for more struct-create forms (especially those composed from already-supported operands and deterministic control flow). The fix is to define legal struct-create criteria, implement lowering/mapping for those cases, and add regression tests that prove both acceptance and correct emitted values.

### 20. `tools/circt-sim-compile/circt-sim-compile.cpp:1447`
As with entry 18, this is the switch that enforces the struct-create support boundary. What is missing remains wider verified struct-create support rather than changes to the boolean itself. The fix is to evolve `isSupportedNativeModuleInitStructCreate` with test-first additions, then keep this fast-fail behavior for residual unsupported shapes.

### 21. `tools/circt-sim-compile/circt-sim-compile.cpp:1453`
Region-bearing ops (other than `scf.if`) are currently rejected for native init, which excludes loops/switch-like constructs even when they may be statically evaluable. The missing capability is a safe structural subset of region ops for init-time compilation. The fix is to admit additional region ops one by one with strict constraints (boundedness, side-effect profile, supported operand types) and introduce semantic tests to verify equivalence with interpreter behavior.

### 22. `tools/circt-sim-compile/circt-sim-compile.cpp:1467`
This blocker rejects module body operands that are block arguments unless they match a narrow probe exception. The missing support is richer handling of block-argument-fed values during native init synthesis, especially when provenance is still deterministic and read-only. The fix is to extend block-argument admissibility analysis (including explicit source categories), then add tests for accepted and rejected cases so accidental over-acceptance does not creep in.

### 23. `tools/circt-sim-compile/circt-sim-compile.cpp:1480`
Dependencies on “skipped ops” currently force unsupported status for dependent ops, which can cascade and eliminate otherwise valid module-init candidates. The missing behavior is dependency rewriting for specific skipped-op patterns that can be normalized into supported forms. The fix is to add targeted rewrites (as already done for some probe patterns) and verify that transformed dependencies preserve runtime semantics before being admitted to native init emission.

### 24. `tools/circt-sim-compile/circt-sim-compile.cpp:1484`
`operand_dep_unsupported:<op>` captures unsupported producers in the same block, but the diagnostic granularity remains low for triage because it reports only the producer op name. The missing piece is richer diagnostic context such as consumer op, operand index, and maybe a compact value trail. The fix is to enrich skip reason metadata and surface it in summary reports, enabling developers to prioritize the most common dependency blockers.

### 25. `tools/circt-sim-compile/circt-sim-compile.cpp:1486`
This is the enforcement point for unsupported dependencies from entry 24. What is missing is not the guard itself, but support for more producer ops or conversion paths that would avoid tripping it. The fix is to progressively legalize high-frequency producer patterns and keep this fallback for genuinely unsupported dependencies.

### 26. `tools/circt-sim-compile/circt-sim-compile.cpp:1492`
Breaking out as soon as `unsupported` is observed gives deterministic control flow, but it also means only the first blocker is discovered per module. The missing capability is multi-reason collection for better developer feedback and faster gap reduction. The fix is to add an optional “collect all blockers” analysis mode for diagnostics while preserving first-failure short-circuit in the fast path.

### 27. `tools/circt-sim-compile/circt-sim-compile.cpp:1497`
`if (unsupported || opsToClone.empty())` skips module-init emission even for empty-op modules, conflating “unsupported” and “nothing to emit.” The missing clarity is separate accounting for each outcome so optimization and diagnostics can distinguish a true feature gap from a no-op case. The fix is to split counters/reasons for `unsupported` versus `empty`, and reflect that separation in emitted stats and regression expectations.

### 28. `tools/circt-sim-compile/circt-sim-compile.cpp:1498`
The nested `if (unsupported)` branch only increments skip reasons when unsupported, but still shares outer control flow with empty-op skips. The missing piece is cleaner semantics for reporting and downstream tooling that consumes these stats. The fix is to isolate unsupported handling into its own branch and ensure every skip path has an explicit, semantically meaningful reason code.

### 29. `tools/circt-sim-compile/circt-sim-compile.cpp:1500`
`unsupported:unknown` is a useful fallback but also a quality-of-diagnostics smell: it means a blocker occurred without a specific reason string. What is missing is complete reason coverage for every early-exit path. The fix is to audit all branches that set `unsupported`/`cloneUnsupported`, guarantee they populate structured reason codes, and add a regression assertion that unknown skip reasons remain at zero in normal test suites.

### 30. `tools/circt-sim-compile/circt-sim-compile.cpp:1514`
`cloneUnsupported` starts a second unsupported-tracking phase during op cloning, indicating that pre-scan acceptance does not fully guarantee clone-time success. The missing capability is a single unified admissibility model that makes clone failure rare and predictable. The fix is to push clone-time constraints back into the pre-scan predicates where possible, reducing double-phase failure and making skip reasons easier to interpret.

### 31. `tools/circt-sim-compile/circt-sim-compile.cpp:1521`
This clone-time unsupported branch fires when probing a block argument that fails supported probe constraints, even after earlier filtering. What is missing is either stronger up-front filtering or broader clone-time handling of probe forms. The fix is to align `isSupportedNativeModuleInitBlockArgProbe` checks across both scan and clone stages and add dedicated tests for every rejected probe subtype.

### 32. `tools/circt-sim-compile/circt-sim-compile.cpp:1550`
Here clone-time support is limited to integer and pointer probe result types, rejecting other result kinds outright. The missing feature is typed conversion support for additional legal probe result types that could be faithfully represented in the generated init function. The fix is to enumerate and implement safe conversions for additional types (with explicit width/sign semantics), and retain rejection for types lacking a clear ABI-safe mapping.

### 33. `tools/circt-sim-compile/circt-sim-compile.cpp:1575`
This branch erases a partially built init function whenever cloning encountered unsupported content, preserving correctness but losing any work that might have been salvageable. The missing capability is either transactional partial emission with verified semantics or richer artifact retention for debugging. The fix is to keep correctness-first erase behavior, but optionally emit debug metadata or temporary IR traces explaining exactly what blocked finalization.

### 34. `tools/circt-sim-compile/circt-sim-compile.cpp:1577`
`cloneSkipReason = "unsupported:unknown"` mirrors the earlier unknown-reason fallback and has the same triage weakness. The missing part is complete reason initialization for clone-time failure branches. The fix is to force every `cloneUnsupported = true` site to assign a specific code and to add tests or runtime asserts that unknown clone reasons are not produced in covered paths.

### 35. `tools/circt-sim-compile/circt-sim-compile.cpp:1592`
The comment admits that support for post-SCF block-argument `hw.struct_extract` forms is still incomplete, and the code works around that with a conservative rewrite. What is missing is first-class handling of those forms without requiring this special-case rewrite path. The fix is to generalize struct-extract lowering across SCF boundaries, then simplify or retire this compensating pass once direct support is robust.

### 36. `tools/circt-sim-compile/circt-sim-compile.cpp:1859`
This comment describes a deliberate degradation strategy in format-string lowering: unsupported fragments become placeholders instead of causing hard failure. The missing capability is complete `sim.fmt.*` coverage for printable fragments so placeholder text rarely appears in production logs. The fix is to expand fragment support by op/type combination and keep placeholders only as a last-resort escape hatch with explicit diagnostics when triggered.

### 37. `tools/circt-sim-compile/circt-sim-compile.cpp:1872`
If a format value has no defining op, lowering injects `<unsupported>`, which avoids crashes but loses user-visible fidelity. The missing behavior is robust handling of block arguments or externally produced format values in this path. The fix is to add explicit support for those value origins (or an upstream normalization guaranteeing local definers), then validate with tests where format values flow through control/data boundaries.

### 38. `tools/circt-sim-compile/circt-sim-compile.cpp:1899`
For decimal formatting, failure to convert the integer into a `printf`-compatible width currently emits `<unsupported>`. The missing capability is conversion support for more integer-like value kinds that can still be represented safely in native printing. The fix is to widen `convertIntegerForPrintf` coverage (or insert legalizing casts before formatting), and add tests proving decimal output stays correct across widths and signedness.

### 39. `tools/circt-sim-compile/circt-sim-compile.cpp:1911`
The same placeholder fallback appears for hexadecimal formatting when conversion fails, so hex rendering coverage is still incomplete for some value types. What is missing is a reliable path from supported IR integer forms to hex-printable operands without dropping content. The fix is to close remaining conversion gaps and add targeted tests for hex formatting on boundary widths/types that currently fall through to `<unsupported>`.

### 40. `tools/circt-sim-compile/circt-sim-compile.cpp:1923`
Octal formatting has the same fallback behavior as decimal and hex, which signals shared conversion limitations rather than an octal-specific bug. The missing piece is comprehensive conversion normalization before selecting radix-specific format verbs. The fix is to centralize integer normalization once, reuse it across radix handlers, and add cross-radix parity tests so one conversion bug cannot silently affect only one format mode.

### 41. `tools/circt-sim-compile/circt-sim-compile.cpp:1936`
Binary formatting currently falls back to `<unsupported>` when argument conversion fails, even though the operation has already selected a hex-based fallback verb for binary output. What is missing is robust conversion support that guarantees the value reaches a print-compatible integer representation in this path. The fix is to harden conversion of integer-like operands before formatting and add tests where `sim.fmt.bin` exercises edge-width and cast-heavy values.

### 42. `tools/circt-sim-compile/circt-sim-compile.cpp:1948`
Character formatting inserts `<unsupported>` when the value cannot be converted to a 32-bit integer for `%c`, which silently degrades output fidelity. The missing capability is reliable narrowing/normalization for char-printable sources that are semantically valid but not already in the expected type. The fix is to extend conversion coverage for char operands and verify non-ASCII and boundary-value behavior with explicit runtime checks.

### 43. `tools/circt-sim-compile/circt-sim-compile.cpp:1961`
Scientific formatting currently supports `f64` and widening from `f32`, but all other float-like or numeric forms degrade to `<unsupported>`. What is missing is a broader numeric-to-float bridge for cases that are semantically safe to print in scientific notation. The fix is to decide which additional source types are legal (for example integer promotion rules), implement those casts explicitly, and add semantic comparison tests against reference simulator output.

### 44. `tools/circt-sim-compile/circt-sim-compile.cpp:1975`
The `%f` path has the same conversion boundary as scientific mode, meaning unsupported source types are dropped to placeholders instead of being printed. The missing piece is consistent floating-point print coverage across all float format ops. The fix is to unify float conversion policy across `%f/%e/%g`, implement it once, and lock behavior with a shared test matrix across these three formatting operations.

### 45. `tools/circt-sim-compile/circt-sim-compile.cpp:1989`
General floating formatting (`%g`) still emits `<unsupported>` for types outside the narrow accepted set, so behavior depends on frontend typing details instead of user intent. What is missing is tolerant but principled coercion for values that can be represented as `double` without ambiguity. The fix is to add a controlled coercion ladder and add parity tests proving `%g` output aligns with `%f/%e` conversions where mathematically equivalent.

### 46. `tools/circt-sim-compile/circt-sim-compile.cpp:2009`
Dynamic-string formatting requires a specific `{ptr,len}` struct shape and still falls back to `<unsupported>` if `len` cannot be converted for `%.*s`. The missing capability is broader acceptance of length field representations that are common in lowered IR. The fix is to broaden integer-length normalization (and fail loudly when unsafe), then add tests for multiple length widths/signs so string truncation behavior is deterministic.

### 47. `tools/circt-sim-compile/circt-sim-compile.cpp:2026`
This final catch-all `<unsupported>` means any unrecognized `sim.fmt.*` producer silently appears as a placeholder instead of hard-failing lowering. The missing part is visibility: placeholder rendering is useful for liveness, but without explicit diagnostics developers may not realize formatting semantics were dropped. The fix is to keep the placeholder fallback but emit structured diagnostics/counters whenever this path is hit, and add a regression that asserts those counters remain zero on supported suites.

### 48. `tools/circt-sim-compile/circt-sim-compile.cpp:3487`
This `unsupported` flag is used in call-indirect lowering to LLVM calls, indicating the lowering path is intentionally limited to ABI-compatible argument/return types. The gap is that any unsupported type causes the rewrite to be skipped entirely, leaving potential performance or functionality on the table for partially legal calls. The fix is to incrementally extend the supported ABI surface with test-first additions and retain explicit skips for truly non-lowerable cases.

### 49. `tools/circt-sim-compile/circt-sim-compile.cpp:3493`
Setting `unsupported = true` when argument type lowering fails means there is no fallback conversion for that argument beyond the current compatibility helper. The missing capability is richer ABI adaptation for argument types that are representable but not currently recognized as LLVM-compatible. The fix is to expand type conversion support (or pre-normalize signatures) and add tests where previously skipped indirect calls are now lowered correctly.

### 50. `tools/circt-sim-compile/circt-sim-compile.cpp:3499`
The return-type path only proceeds when arguments were all supported, so return handling is gated behind earlier success and cannot provide independent diagnostics for mixed failures. What is missing is better reason reporting when both arguments and return types are problematic. The fix is to collect independent arg/ret compatibility outcomes for diagnostics while preserving a single skip decision for lowering.

### 51. `tools/circt-sim-compile/circt-sim-compile.cpp:3504`
Return types that cannot be lowered to LLVM-compatible form trigger unsupported status, preventing indirect-call conversion. The missing behavior is return-type bridging for additional legal forms that can be packed/unpacked safely. The fix is to add result-type adaptation rules (or trampoline-based fallback for this rewrite site) and include regression tests for representative formerly-unsupported return signatures.

### 52. `tools/circt-sim-compile/circt-sim-compile.cpp:3508`
`if (unsupported) continue;` silently drops the rewrite opportunity without recording why a specific call_indirect stayed unlowered. The missing part is observability for skipped transformations in optimization/debug workflows. The fix is to add optional skip diagnostics or counters keyed by failure reason so users can understand why indirect calls remain in the IR.

### 53. `tools/circt-sim-compile/circt-sim-compile.cpp:3526`
Argument materialization failures during operand adaptation set `unsupported` and abandon the conversion, indicating gaps in value-level casting even when signature-level type checks passed. What is missing is robust per-operand bridging for hard cases (for example mixed dialect/value forms). The fix is to strengthen `materializeLLVMValue` coverage and add tests that exercise operand conversions under call-indirect rewriting.

### 54. `tools/circt-sim-compile/circt-sim-compile.cpp:3532`
This second `if (unsupported) continue;` repeats the silent-skip pattern after operand rewriting, which can make transformed coverage hard to reason about. The missing capability is structured reporting of rewrite success/failure rates at pass scope. The fix is to track conversion attempts and skip reasons in pass statistics and expose them in logs so unsupported cases are actionable.

### 55. `tools/circt-sim-compile/circt-sim-compile.cpp:3825`
Returning zero slots for a nested unsupported field type in trampoline ABI flattening blocks trampoline generation for composite signatures that include that field. The missing capability is flattening support for more nested type shapes that can still map to the `uint64_t` slot convention. The fix is to extend `countTrampolineSlots` and corresponding pack/unpack logic together, with round-trip tests on nested arrays/structs.

### 56. `tools/circt-sim-compile/circt-sim-compile.cpp:3830`
The base-case `return 0; // unsupported` marks any unrecognized type as ABI-inexpressible for trampoline dispatch. The gap is breadth: many types may be semantically supportable but are currently excluded by this conservative default. The fix is to explicitly enumerate additional supported kinds and keep this zero-slot fallback only for genuinely unsafe/unmapped types.

### 57. `tools/circt-sim-compile/circt-sim-compile.cpp:4171`
This comment documents a real architectural gap: trampolines are needed specifically because some functions still contain unsupported ops and cannot be compiled directly. What is missing is broader direct compilation coverage that reduces reliance on interpreted fallback boundaries. The fix is to continue expanding unsupported-op coverage in the compiled path while keeping trampoline fallback as a compatibility safety net.

### 58. `tools/circt-sim-compile/circt-sim-compile.cpp:4215`
`sawUnsupportedReferencedExternal` tracks whether referenced externals could not be trampolined, which currently forces failure for that generation pass. The missing capability is partial progress handling where unsupported externals are isolated without aborting all trampoline generation. The fix is to decide policy (strict vs best-effort) explicitly and, if best-effort is allowed, emit precise diagnostics while still generating valid trampolines for supported externals.

### 59. `tools/circt-sim-compile/circt-sim-compile.cpp:4253`
Vararg externals are currently rejected for trampoline generation and set the global unsupported flag. The missing behavior is a defined vararg ABI strategy, which is difficult but sometimes necessary for real-world C library interop. The fix is either to implement a constrained vararg trampoline ABI or to preserve strict rejection but improve tooling guidance so users can route such calls through direct extern linkage paths.

### 60. `tools/circt-sim-compile/circt-sim-compile.cpp:4259`
`hasUnsupported` begins ABI compatibility checks for function parameters/returns, and any unsupported part currently rejects trampoline generation for that function. The missing piece is finer-grained ABI support and clearer distinction between “unsupported forever” and “not implemented yet.” The fix is to turn this check into a capability table with explicit per-type rationale and to grow it in a controlled, tested manner.

### 61. `tools/circt-sim-compile/circt-sim-compile.cpp:4260`
This dedicated `unsupportedReason` string is useful, but only captures a single first failure reason and can hide secondary incompatibilities. What is missing is multi-reason reporting that helps prioritize the dominant blockers across a codebase. The fix is to collect all incompatible ABI features for a symbol in diagnostic mode, while still selecting one canonical reason for concise normal-mode output.

### 62. `tools/circt-sim-compile/circt-sim-compile.cpp:4265`
Setting `hasUnsupported = true` when a parameter flattens to zero slots is the exact point where non-flattenable parameter types are excluded. The missing capability is type flattening support for those parameter types or a fallback passing convention that can represent them. The fix is to extend slot-packing semantics for additional legal types and validate with interpreter trampoline round-trip tests.

### 63. `tools/circt-sim-compile/circt-sim-compile.cpp:4266`
The parameter-type reason string is informative but still plain text and not machine-typed, making automated triage across many failures harder. The missing part is structured diagnostic metadata for unsupported ABI reasons. The fix is to emit stable reason codes plus human text so dashboards and scripts can aggregate top blockers reliably.

### 64. `tools/circt-sim-compile/circt-sim-compile.cpp:4271`
Return-type checking is deferred until parameter checks pass, which is efficient but can obscure full incompatibility profiles for a function signature. The missing behavior is optional comprehensive validation for diagnostic workflows. The fix is to add a non-fast-path mode that inspects both params and return unconditionally and reports combined ABI incompatibilities.

### 65. `tools/circt-sim-compile/circt-sim-compile.cpp:4275`
Unsupported return type detection currently depends on zero-slot flattening and collapses all failure causes into one branch. The missing detail is differentiation between specific return-type incompatibility classes (for example nested aggregate vs unsupported primitive kind). The fix is to expose typed reason codes at this decision point and prioritize implementation based on observed frequency.

### 66. `tools/circt-sim-compile/circt-sim-compile.cpp:4276`
Guarding reason assignment behind `if (hasUnsupported)` is correct, but it currently overwrites context with a broad “return type …” message when return checking fails. The missing capability is retaining richer causal context for debugging (e.g., exact nested field causing flatten failure). The fix is to thread inner flatten diagnostics up to this point and include them in the final reason text/code.

### 67. `tools/circt-sim-compile/circt-sim-compile.cpp:4277`
The return-type `unsupportedReason` payload is human-readable but not normalized, which limits automated bucketing across runs. What is missing is stable ABI error coding that still preserves rich type text for humans. The fix is to emit both a compact reason code and full rendered type string, then use the code in summaries and the text in detailed logs.

### 68. `tools/circt-sim-compile/circt-sim-compile.cpp:4279`
This branch rejects trampoline generation whenever unsupported ABI traits are found, prioritizing correctness over degraded execution. The missing capability is any alternative execution path for those specific extern signatures besides failing the overall trampoline-generation step. The fix is to either add fallback dispatch mechanisms for more signatures or fail earlier with clearer user guidance on how to avoid unsupported extern shapes.

### 69. `tools/circt-sim-compile/circt-sim-compile.cpp:4282`
The emitted diagnostic text `unsupported trampoline ABI (...)` is accurate but still relies on free-form messaging. The missing piece is integration with structured diagnostics so tooling can classify this failure without string matching. The fix is to attach a stable diagnostic ID/category to this error and add tests that assert both message clarity and ID stability.

### 70. `tools/circt-sim-compile/circt-sim-compile.cpp:4284`
Marking `sawUnsupportedReferencedExternal = true` ensures failures are propagated, but it also means one unsupported external can fail a larger compile scope. The missing capability is policy control: some workflows may prefer warnings and partial output over hard failure. The fix is to introduce a strictness option (strict by default) that can downgrade selected unsupported-external cases to non-fatal diagnostics when explicitly requested.

### 71. `tools/circt-sim-compile/circt-sim-compile.cpp:4283`
The line that prints `unsupportedReason` into the trampoline ABI error is useful but still too coarse when debugging nested aggregate ABI failures. What is missing is deeper reason decomposition so users can see exactly which nested field or shape made flattening impossible. The fix is to preserve and print nested flatten context (for example field index paths) and add regression tests that assert actionable diagnostics for complex unsupported signatures.

### 72. `tools/circt-sim-compile/circt-sim-compile.cpp:4289`
`if (sawUnsupportedReferencedExternal) return failure();` enforces all-or-nothing behavior for trampoline generation. The missing capability is best-effort output when only a subset of externals are problematic. The fix is to make this policy configurable and, in non-strict mode, emit supported trampolines while reporting unsupported externals as structured warnings.

### 73. `tools/circt-sim-compile/circt-sim-compile.cpp:7498`
This unsupported flag appears in a separate declaration-cloning path where `func::FuncOp` types are converted into `LLVMFuncOp` declarations for referenced symbols. The gap is that only a narrow type subset (integers, pointers, floats, index) is accepted, so valid-but-richer function signatures are silently skipped. The fix is to broaden signature lowering support in lockstep with trampoline ABI support, and surface skip statistics so dropped declarations are visible.

### 74. `tools/circt-sim-compile/circt-sim-compile.cpp:7506`
Unsupported is set when any input type falls outside the currently handled scalar/index set, which makes declaration availability dependent on type shape rather than call reachability needs. What is missing is a fallback declaration strategy for additional argument kinds. The fix is to add explicit lowering rules for additional legal MLIR function input types and keep strict rejection only for truly unrepresentable signatures.

### 75. `tools/circt-sim-compile/circt-sim-compile.cpp:7510`
The `if (!unsupported)` gate means one unsupported input type suppresses declaration creation entirely, even if downstream code could still handle partial cases via interpreter fallback. The missing piece is clearer coordination between declaration emission and trampoline generation. The fix is to document and enforce a unified policy: either declarations are required only for trampoline-able signatures, or a broader stub strategy is introduced with explicit runtime fallback.

### 76. `tools/circt-sim-compile/circt-sim-compile.cpp:7522`
Unsupported is set for return types outside the small accepted set, so multi-value or aggregate returns are excluded from this declaration path. The missing capability is return-type lowering parity with the rest of the call boundary machinery. The fix is to support additional return encodings where safe, or require trampoline wrapping for those signatures and emit explicit reasoned diagnostics when declaration emission is skipped.

### 77. `tools/circt-sim-compile/circt-sim-compile.cpp:7525`
This branch rejects functions with more than one return value for declaration conversion, reflecting a current ABI simplification. The missing behavior is support for multi-result function signatures (or canonical tuple lowering) at this interface boundary. The fix is to introduce a canonical multi-result flattening approach and validate it with call-site and trampoline integration tests.

### 78. `tools/circt-sim-compile/circt-sim-compile.cpp:7527`
The final `if (!unsupported)` check controls whether an external declaration gets emitted at all, so unsupported signatures become invisible entries in the table. What is missing is explicit accounting of skipped declarations for developer feedback. The fix is to record skipped symbol names/reasons and expose them in compile summaries so missing declarations are discoverable without debugging IR manually.

### 79. `tools/circt-sim-compile/circt-sim-compile.cpp:7536`
The comment admits skipped entries are tolerated with null table slots when symbols are unsupported or absent, which avoids crashes but can delay failure to runtime dispatch. The missing capability is earlier user feedback for null-entry use paths. The fix is to report null-table symbol coverage at compile time and add runtime assertions that identify unresolved entries with clear symbol names when invoked.

### 80. `utils/run_mutation_mcy_examples.sh:1006`
This is a scan false positive: `mktemp ...XXXXXX` was matched because the audit regex includes `XXX`, but the line is normal temporary-file creation rather than an implementation gap. What is missing is scanner precision, not product functionality. The fix is to tighten TODO token matching to word boundaries (or require `TODO|FIXME|...` as standalone markers) so mktemp templates do not pollute the gap list.

### 81. `utils/run_mutation_mcy_examples.sh:2379`
This line is the same false-positive pattern as entry 80 (`mktemp` with `XXXXXX`), not a true unsupported/TODO gap. The missing part is the quality of the gap-discovery query. The fix is to refine scanning rules and optionally add a suppression list for known lexical false positives in shell scripts.

### 82. `utils/run_mutation_mcy_examples.sh:2380`
Again, this `mktemp ...XXXXXX` match is not technical debt in behavior; it is an artifact of using `XXX` as a broad scan token. What is missing is better discrimination between deliberate placeholders and canonical random-suffix templates. The fix is to change the audit pattern to avoid matching inside all-uppercase temp suffix literals.

### 83. `utils/run_mutation_mcy_examples.sh:2381`
This is another `XXXXXX` false positive and should not be treated as an implementation gap. The missing capability is a robust audit filter that can distinguish housekeeping shell idioms from actionable TODO markers. The fix is to post-process scan output through a false-positive filter and keep only semantically meaningful matches.

### 84. `utils/run_mutation_mcy_examples.sh:2382`
This is the fourth consecutive `mktemp` false positive in the same block and confirms the scan regex is currently over-inclusive. The missing piece is audit hygiene: without filtering, these matches dilute attention from real unsupported paths. The fix is to update the scanner and regenerate the audit so these noise entries disappear from future manual passes.

### 85. `frontends/PyRTG/src/pyrtg/support.py:60`
`assert False, "Unsupported value"` indicates the conversion helper has no fallback for unknown CIRCT value wrapper types and will hard-stop in debug-oriented style. What is missing is either comprehensive value-type coverage or an exception path that reports unsupported inputs with actionable context instead of an assertion. The fix is to replace assertion-based termination with structured exceptions and extend `_FromCirctValue` mappings for newly introduced RTG value classes.

### 86. `frontends/PyRTG/src/pyrtg/support.py:117`
`raise ValueError("unsupported type")` in `_FromCirctType` is a hard boundary for type conversion coverage in PyRTG. The missing behavior is support for additional CIRCT/RTG type nodes that users can encounter from evolving dialect features. The fix is to add new type cases as dialect support grows and include round-trip tests that confirm Python wrappers are generated for each supported type form.

### 87. `test/Runtime/uvm/uvm_phase_wait_for_state_test.sv:2`
`// UNSUPPORTED: true` here intentionally disables the test in the current harness, so this line marks a real testing coverage gap rather than code semantics ambiguity. What is missing is an execution path in CI that can run UVM runtime simulation tests instead of only compile-only checks. The fix is to stand up a dedicated runtime-capable test lane for these UVM tests and remove blanket `UNSUPPORTED` once infrastructure is available.

### 88. `test/Runtime/uvm/uvm_phase_aliases_test.sv:2`
This is the same infrastructure-driven testing gap as entry 87: the test exists, but the harness does not yet execute it. The missing capability is runtime UVM test integration in the standard test workflow. The fix is to add that runtime lane and convert this test from permanently unsupported to actively validated behavior.

### 89. `utils/run_avip_circt_sim.sh:249`
Rejecting `AVIP_SET` values outside `core8|all9` is likely an intentional guard, but it also hard-codes set names and blocks extension without script edits. What is missing is a configurable registry for AVIP sets so users can add curated subsets without patching the script. The fix is to load sets from a config file (or env-provided manifest) and keep these strict checks as validation against that dynamic registry.

### 90. `utils/run_avip_circt_sim.sh:269`
Likewise, limiting `CIRCT_SIM_MODE` to `interpret|compile` enforces current known modes but prevents smooth introduction of new execution modes. The missing behavior is mode extensibility with clear capability discovery. The fix is to centralize mode definitions (with help text and validation) and source this script from that single definition so new modes can be added once without scattering guard updates.

### 91. `frontends/PyRTG/src/pyrtg/control_flow.py:92`
This FIXME is a real maintainability gap in PyRTG control-flow lowering: it rebuilds `scf.if` ops in a workaround because Python MLIR bindings do not support deleting region blocks cleanly in this flow. What is missing is a robust structural edit path that preserves block semantics without cloning-and-append gymnastics. The fix is to move to an API pattern that avoids block deletion requirements (or upgrade bindings and refactor), then add regression tests for nested `If/Else` rewrites to ensure SSA/value mapping stays correct.

### 92. `frontends/PyRTG/src/pyrtg/control_flow.py:174`
Setting deleted locals to `None` instead of removing them is a pragmatic workaround, but it leaks temporary names and can subtly alter user frame behavior. The missing behavior is proper local cleanup semantics after control-flow capture. The fix is to safely delete locals where possible (or isolate execution scope to avoid touching caller locals directly) and add tests asserting no stray loop/if locals remain visible after block exit.

### 93. `frontends/PyRTG/src/pyrtg/contexts.py:73`
This TODO identifies a concrete scalability issue: context capture currently hoovers up all locals, not just values actually used inside the sequence body. What is missing is dependency-minimal capture, which impacts readability and can bloat generated sequence signatures. The fix is to compute used-value sets (AST analysis or tracing) and only pass needed values into context sequences, then verify generated argument lists shrink without semantic regression.

### 94. `frontends/PyRTG/src/pyrtg/contexts.py:87`
Assuming `_context_seq` is a reserved prefix is fragile and can collide with user symbols, so this is a legitimate naming-hygiene gap. What is missing is proper symbol uniquing integrated with module symbol tables. The fix is to switch to a deterministic uniquing utility (or MLIR symbol utilities) and add tests covering collisions with user-provided names to prove generated sequence symbols remain conflict-free.

### 95. `utils/run_avip_arcilator_sim.sh:103`
Restricting `AVIP_SET` to `core8|all9` is likely intentional, but it hard-codes policy and makes extension cumbersome. The missing piece is a configurable AVIP-set registry so new curated sets can be added without editing script logic. The fix is to externalize set definitions to data (TSV/JSON/env list), validate them at startup, and keep strict erroring for unknown set names.

### 96. `utils/create_mutated_yosys.sh:95`
Rejecting design extensions outside `.il/.v/.sv` is a clear boundary that may be intentional, but it limits pipeline interoperability when upstream tools emit other forms. What is missing is either richer format support or a documented canonical pre-conversion step. The fix is to add adapters for additional common inputs where feasible, or keep strict validation but provide a companion conversion helper so users can normalize inputs automatically.

### 97. `utils/create_mutated_yosys.sh:106`
The output-extension restriction mirrors entry 96 and can block downstream consumers expecting other file types. The missing behavior is pluggable output backend selection rather than fixed extension branching. The fix is to parameterize output mode independently from filename extension and add tests that verify each backend emits the expected syntactically valid artifact.

### 98. `utils/check_opentitan_connectivity_cover_parity.py:101`
This is the same recurring allowlist grammar boundary: unsupported mode names are rejected. The missing piece is shared parsing infrastructure so all parity scripts behave identically and evolve together. The fix is to move allowlist parsing to one common helper module and cover it with shared tests used by every script that consumes allowlists.

### 99. `utils/check_opentitan_connectivity_status_parity.py:101`
This entry repeats the allowlist-mode consistency issue, and the real gap is duplication, not correctness at this call site. What is missing is de-duplication of parser logic across parity checkers. The fix is to import a shared parser API and keep script-level behavior focused on domain checks, not format parsing.

### 100. `test/CMakeLists.txt:53`
This TODO marks a real build/test coverage gap: `circt-verilog-lsp-server` is excluded from test dependencies due to slang API compatibility issues. What is missing is compatibility glue (or versioned API handling) that allows the LSP server target to build and test with the current slang frontend. The fix is to resolve API drift and re-enable the target in `CIRCT_TEST_DEPENDS`, then add CI coverage to prevent silent re-breakage.

### 101. `utils/run_opentitan_connectivity_circt_bmc.py:120`
Again this is a duplicated allowlist parser rejecting unknown modes. The missing behavior is centralized parser governance, especially because this script participates in high-volume OpenTitan workflows where consistency matters. The fix is the same shared helper approach with a single compliance test suite for `exact/prefix/regex` semantics and diagnostics.

### 102. `utils/run_opentitan_connectivity_circt_bmc.py:412`
Failing on unknown connectivity rule types in the manifest is correct for data integrity, but it also means new manifest schema variants cannot roll out incrementally. The missing capability is versioned rule-type handling and forward-compatibility strategy. The fix is to add schema-version negotiation with explicit support tables (and optional strict mode), so known-new rule types can be introduced without hard-breaking older runners.

### 103. `frontends/PyCDE/test/test_esi.py:170`
`# TODO: fixme` is a low-information TODO and therefore a process-quality gap: it flags debt but gives no actionable scope or acceptance criteria. What is missing is a precise statement of why `PureTest` is disabled and what condition would allow re-enablement. The fix is to replace this placeholder with a concrete issue description and expected behavior, then add/enable a regression once that behavior is implemented.

### 104. `include/circt/Runtime/MooreRuntime.h:4208`
This is a scanner false positive (`super.XXX_phase()` in docs), not a TODO/unsupported implementation gap. What is missing is scan precision for tokens like `XXX` when used as literal examples in comments. The fix is to tighten audit regexes (for example word-boundary TODO markers only) or maintain an allowlist for known documentation phrases.

### 105. `frontends/PyCDE/test/test_polynomial.py:120`
This TODO is real: the test notes that IR verification fails before all modules are generated because `hw.instance` references are unresolved. What is missing is staged generation/verification semantics that tolerate forward references or enforce topological emission order. The fix is to either support deferred symbol resolution during generation or reorder module emission so the intermediate IR is always verifiable.

### 106. `utils/refactor_continue.sh:6`
This is another scan false positive: `TODO_PATH` is just a variable name and not unfinished implementation. The missing issue is audit specificity, not script behavior. The fix is to exclude identifiers containing `TODO` unless they appear in recognized marker syntax (`# TODO`, `TODO:`) or known diagnostic strings.

### 107. `utils/refactor_continue.sh:14`
`plan/todo` in usage text is documentation wording, not implementation debt. The gap is scanner overreach into ordinary prose. The fix is to constrain scans to marker prefixes or code comments, not arbitrary help text.

### 108. `utils/refactor_continue.sh:18`
This `--todo` option description is similarly non-actionable and should not be treated as a gap. What is missing is classification hygiene in the audit pipeline. The fix is to add filtering rules for command-line option names that happen to contain `todo`.

### 109. `utils/refactor_continue.sh:41`
The `--todo)` case label is functional CLI parsing and not technical debt. The gap is again false-positive identification in the scan results. The fix is to filter shell `case` labels that merely include the token text.

### 110. `utils/refactor_continue.sh:42`
Assigning `TODO_PATH="$2"` is normal option processing, not a TODO marker. What is missing is semantic filtering during audit generation. The fix is to require TODO matches in comments/messages rather than variable assignments unless explicitly configured.

### 111. `utils/refactor_continue.sh:61`
Checking for file existence of `TODO_PATH` is expected runtime validation and not a feature gap. The missing piece is separating “string contains TODO” from actual TODO markers. The fix is scanner refinement rather than code changes here.

### 112. `utils/refactor_continue.sh:62`
Printing `todo file not found` is a normal error path, not unsupported functionality. The gap is audit noise introduced by matching lowercase `todo` text in user diagnostics. The fix is to restrict scan rules to upper-case marker conventions or comment annotations.

### 113. `utils/refactor_continue.sh:66`
The canonical prompt string includes `TODO_PATH` as content and is not evidence of unfinished code. What is missing is a more context-aware scanner that understands operational text versus debt markers. The fix is to run token scans on comments/diagnostics only, or add per-file suppression for known helper scripts.

### 114. `utils/refactor_continue.sh:81`
This reference to `"$TODO_PATH"` in an awk invocation is operational plumbing, not a TODO item. The gap remains scanner precision. The fix is to post-filter entries where the matched substring is part of a variable name rather than a marker.

### 115. `utils/refactor_continue.sh:92`
Same as entry 114: use of `TODO_PATH` as a parameter is functional code, not missing implementation. The fix is no code change here; it is refining how audit matches are interpreted and filtered.

### 116. `utils/refactor_continue.sh:122`
`printf 'TODO: %s\n'` is user-facing status output, not a debt marker about this script. The missing piece is distinguishing display labels from unresolved tasks during scanning. The fix is to mark this script as a known false-positive hotspot or tighten marker detection.

### 117. `utils/run_yosys_sva_circt_bmc.sh:142`
`UNSUPPORTED_SVA_POLICY` is a configuration knob name, and its appearance in the scan does not itself indicate a gap. The real underlying gap is elsewhere: policy exists because unsupported SVA constructs are still present in the flow. The fix is to keep the policy control, but track and reduce unsupported-SVA incidence so `lenient` mode becomes less necessary over time.

### 118. `utils/run_yosys_sva_circt_bmc.sh:298`
Validating policy values (`strict|lenient`) is good hygiene, but it also reflects that only two policies are currently expressible. What is missing is finer policy granularity (for example warn-only, per-diagnostic overrides) if workflows need it. The fix is to define a richer policy model only if demanded by users; otherwise keep strict validation and document semantics clearly.

### 119. `utils/run_yosys_sva_circt_bmc.sh:299`
The invalid-policy diagnostic is not a feature gap on its own; it is a boundary message for entry 118’s guard. The missing capability, if any, is policy extensibility rather than message text. The fix is to keep this explicit diagnostic but drive accepted values from a central enum/table to avoid drift with future policy additions.

### 120. `utils/run_yosys_sva_circt_bmc.sh:2747`
Rejecting unsupported JSONL line formats in history files is intentional data validation, but it can be brittle for slightly malformed legacy artifacts. What is missing is resilient migration support for a broader set of legacy line shapes. The fix is to extend migration parsing with explicit compatibility cases and keep hard-fail behavior only when recovery would be ambiguous.

### 121. `utils/run_yosys_sva_circt_bmc.sh:2867`
Unsupported drop-event hash modes are currently rejected, which is safe but rigid. The missing capability is pluggable hash-provider support when teams need stronger/stabler IDs than built-ins. The fix is to define a vetted extension point for hash algorithms and retain strict rejection for unknown modes by default.

### 122. `utils/run_yosys_sva_circt_bmc.sh:3026`
Rejecting unknown lock backends is correct for synchronization safety, but it means backend selection logic is closed to extension. The missing piece is a clean backend abstraction layer with explicit capabilities/timeouts. The fix is to formalize backend interfaces (`flock`, `mkdir`, future backends), then validate each with lock-contention tests before enabling.

### 123. `utils/run_yosys_sva_circt_bmc.sh:3186`
Failing on unsupported history-TSV headers is a strict schema guard that protects data quality, but it blocks gradual schema evolution. What is missing is explicit versioned header compatibility with migration paths. The fix is to track schema versions in-band and add deterministic migrators so older headers can be upgraded instead of hard-failed when possible.

### 124. `utils/run_yosys_sva_circt_bmc.sh:8522`
The comment states that smoke mode treats sim-only tests as unsupported and skips them, which is a real coverage trade-off. What is missing is lightweight bounded checking for sim-only cases in smoke that preserves determinism without complete omission. The fix is to introduce a cheap smoke-safe fallback (or a separate tiny sim lane) so these tests still get minimal signal.

### 125. `utils/run_yosys_sva_circt_bmc.sh:8616`
`lenient` policy injects `--sva-continue-on-unsupported`, which is useful operationally but confirms unsupported constructs are still common enough to require bypass. The missing capability is broader SVA lowering support so lenient mode is exceptional rather than routine. The fix is to track which diagnostics trigger this path and prioritize implementation against the highest-frequency unsupported constructs.

### 126. `utils/run_yosys_sva_circt_bmc.sh:8617`
This flag insertion is the mechanism for entry 125 and carries the same semantic gap: it trades strict correctness for throughput under unsupported SVA. What is missing is confidence that skipped assertions are transparent to users. The fix is to ensure every continue-on-unsupported event is surfaced in structured reports and tied to explicit pass/fail policy decisions.

### 127. `utils/run_yosys_sva_circt_bmc.sh:8676`
This second insertion point for lenient policy in another execution branch suggests duplicated policy plumbing. The missing maintainability piece is centralized argument synthesis so policy behavior cannot drift between branches. The fix is to factor policy-to-arg mapping into a helper function and test both code paths for parity.

### 128. `utils/run_yosys_sva_circt_bmc.sh:8677`
Same as entry 127: this is duplicated flag wiring for lenient mode. The missing capability is single-source policy mapping with consistent diagnostics across all run modes. The fix is refactoring plus regression tests that diff generated command lines across branches.

### 129. `utils/run_yosys_sva_circt_bmc.sh:8761`
Lenient policy also affects `circt-bmc` arguments here, which is correct but again duplicated branch logic. What is missing is policy cohesion across frontend and bmc tool invocation assembly. The fix is to generate both frontend and bmc policy args from one shared function/object so future policy changes stay synchronized.

### 130. `utils/run_yosys_sva_circt_bmc.sh:8762`
Adding `--drop-unsupported-sva` in lenient mode is a real semantic compromise that can hide assertion coverage holes if not tracked. The missing behavior is robust accountability for dropped properties during result interpretation. The fix is to require per-case drop accounting in outputs and optionally fail when dropped-SVA count exceeds configured thresholds.

### 131. `lib/Runtime/MooreRuntime.cpp:2481`
This TODO is a genuine runtime capability gap: `__moore_wait_condition` is a placeholder and does not integrate with an actual simulation scheduler to suspend/resume processes. What is missing is simulation-aware blocking semantics tied to signal/event re-evaluation. The fix is to wire this call into the process scheduler/event queue infrastructure and add temporal regression tests proving conditions block and wake correctly.

### 132. `lib/Runtime/MooreRuntime.cpp:12227`
Array-element signal lookup currently strips indices and returns the base signal handle with a TODO for index calculation, so element-precise access is not implemented. What is missing is mapping from parsed index vectors to actual element offsets/handles for packed/unpacked arrays. The fix is to implement index resolution against registered signal metadata and add tests for multidimensional and out-of-range cases.

### 133. `lib/Runtime/MooreRuntime.cpp:14402`
The comment says test creation via UVM factory is TODO, but the function now contains concrete factory-create logic; this is stale documentation debt rather than missing executable code at this exact site. What is missing is comment-to-implementation consistency, which matters for maintainability. The fix is to update the comment to describe current behavior accurately and reserve TODO tags for truly unfinished work.

### 134. `lib/Runtime/MooreRuntime.cpp:17781`
Backdoor register read currently does not integrate with HDL path access and simply returns mirror state, which is a real functional gap for UVM backdoor semantics. What is missing is a bridge from register metadata/HDL path to live design state readback. The fix is to implement HDL path callbacks/resolution for backdoor reads and validate parity with frontdoor/mirror behavior under synchronized and desynchronized scenarios.

### 135. `lib/Runtime/MooreRuntime.cpp:17823`
Backdoor register write has the same missing HDL path integration as entry 134, so writes update mirror state but do not propagate through configured HDL access mechanisms. What is missing is actual design-state writeback when backdoor paths are present. The fix is to implement HDL path write hooks, define error behavior for unresolved paths, and add tests that verify mirrored and physical states converge correctly.

### 136. `frontends/PyCDE/test/test_instances.py:124`
This TODO marks an explicit test-coverage gap: physical region support tests were removed/disabled and not restored. What is missing is either feature readiness in the implementation or stable test scaffolding around region APIs. The fix is to re-enable region creation/bounds tests once the API is stable and ensure failures are clearly attributable to implementation regressions, not test harness fragility.

### 137. `frontends/PyCDE/test/test_instances.py:156`
Anonymous reservation tests are similarly commented out, so reservation behavior currently lacks active regression coverage in this file. What is missing is verification that placedb reservation semantics still work end-to-end. The fix is to reinstate these tests (or replace with updated equivalents) and assert reservation conflict/lookup behavior explicitly.

### 138. `utils/check_opentitan_fpv_bmc_evidence_parity.py:111`
This is another duplicated allowlist parser with strict `exact|prefix|regex` mode handling; the implementation is fine but duplicated across many scripts. The missing piece is a shared parser to eliminate drift and repeated bug fixes. The fix is to centralize allowlist parsing and migrate these parity scripts to the shared helper with a common test suite.

### 139. `utils/check_opentitan_target_manifest_drift.py:87`
Same pattern as entry 138: unsupported mode rejection is acceptable behavior, but parser duplication is the real maintainability gap. What is missing is single-source allowlist semantics across target-manifest and parity tooling. The fix is shared library extraction plus migration tests to guarantee unchanged user-visible behavior.

### 140. `utils/run_avip_circt_verilog.sh:157`
This `mktemp ...XXXXXX` hit is a scan false positive caused by matching `XXX` inside the random-suffix template. There is no implementation gap at this line. The fix is to tighten scanner patterns so `XXXXXX` templates are excluded from TODO-style matches.

### 141. `utils/run_avip_circt_verilog.sh:184`
This is the same `mktemp ...XXXXXX` false positive as entry 140. The missing issue is audit precision, not script functionality. The fix is identical: refine marker regexes or apply false-positive suppression for mktemp suffix literals.

### 142. `include/circt/Transforms/Passes.td:86`
The word “unsupported” here appears in pass documentation describing why index conversion exists; it is not itself a TODO in this file. The true gap, if any, is upstream pass dependence on this conversion to avoid unsupported index arithmetic in downstream mapping. The fix is scanner filtering for descriptive prose and, separately, continued reduction of downstream unsupported index operations so this pass becomes less critical.

### 143. `unittests/Support/TestReportingTest.cpp:54`
`tc.skip("Not implemented")` is intentional test data for skip-state behavior, not an implementation gap in reporting. What is missing is scanner discrimination between fixture messages and real code debt. The fix is to suppress known test-fixture phrases like “Not implemented” in audit output.

### 144. `unittests/Support/TestReportingTest.cpp:58`
This assertion checks the same fixture string and is not actionable product debt. The missing capability is better audit signal-to-noise around unit-test literals. The fix is to filter assertions that compare expected fixture messages from TODO/unsupported scans.

### 145. `unittests/Support/TestReportingTest.cpp:195`
Again this is a deliberate skipped-test fixture setup, not a runtime/reporting feature gap. The scanner is over-attributing gap semantics to test literals. The fix is to classify unit-test string fixtures separately and remove them from actionable gap lists.

### 146. `unittests/Support/TestReportingTest.cpp:203`
This output substring check intentionally looks for “Not implemented” and should not be treated as missing implementation. What is missing is context-aware scanning. The fix is to refine audit tooling to ignore expected-output assertions in unit tests by default.

### 147. `utils/check_opentitan_connectivity_contract_fingerprint_parity.py:90`
This is the same repeated allowlist mode parser pattern, with duplication being the real debt. What is missing is a canonical parsing utility to keep behavior aligned across all connectivity/fingerprint checkers. The fix is to migrate this script to shared parser code and delete local copies.

### 148. `utils/mutation_mcy/lib/native_mutation_plan.py:57`
Rejecting unknown native mutation ops is correct for safety, but it also means operator extensibility depends on code edits rather than declarative registration. What is missing is a pluggable op registry or capability discovery path. The fix is to keep strict validation but source allowed ops from a centralized registry definition so extending mutation ops is controlled and testable.

### 149. `utils/mutation_mcy/lib/drift.sh:646`
This is another `mktemp ...XXXXXX` false positive and not a TODO/unsupported code gap. The missing issue is scan pattern quality. The fix is to exclude mktemp suffix templates from TODO token matching.

### 150. `utils/mutation_mcy/lib/drift.sh:647`
Same false-positive category as entry 149: routine temp-file creation matched by `XXX` token scan. The fix is scanner filtering, not code change.

### 151. `utils/mutation_mcy/lib/drift.sh:648`
Again a false positive from mktemp suffix text. What is missing is a robust noise filter in the audit toolchain. The fix is to add lexical exclusions for uppercase temp patterns.

### 152. `utils/mutation_mcy/lib/drift.sh:649`
Fourth false-positive in the same block, confirming this category should be filtered globally. The fix is to implement and keep a false-positive suppression rule for `mktemp` templates before future manual review passes.

### 153. `utils/run_regression_unified.sh:120`
Rejecting unsupported profile tokens is expected schema validation, but supported values are hard-coded and may drift across tooling layers. What is missing is centralized profile schema definition shared by producers and consumers. The fix is to define profile enums in one source of truth and validate manifests against that shared definition.

### 154. `utils/check_opentitan_fpv_objective_parity.py:170`
This is another duplicated allowlist parser boundary. The missing behavior is not at this branch itself, but in ecosystem-level parser reuse and consistency. The fix is to consolidate parser logic and migrate objective-parity scripts to the shared implementation.

### 155. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm_time.svh:105`
`// ToDo: Check resolution` in vendored UVM core indicates an unresolved semantic detail in time scaling logic. What is missing is explicit validation that conversion respects expected timescale resolution behavior under all legal inputs. The fix is to add targeted UVM time-resolution tests and either resolve this TODO upstream or patch locally with clear divergence notes.

### 156. `utils/run_opentitan_connectivity_circt_lec.py:130`
This is the same allowlist mode parser duplication seen in BMC/connectivity scripts. The missing piece is centralization to prevent behavior drift between BMC and LEC wrappers. The fix is shared helper adoption with cross-tool parity tests.

### 157. `utils/run_opentitan_connectivity_circt_lec.py:366`
Failing on unsupported connectivity rule types is robust for strict manifests, but it prevents staged rollout of new rule kinds in LEC flows. What is missing is versioned manifest schema handling with explicit compatibility policy. The fix is to implement schema-version gates and supported-kind negotiation so new rule types can be introduced safely.

### 158. `utils/internal/checks/wasm_cxx20_contract_check.sh:45`
This “accepted unsupported C++ standard override” string is a negative test assertion, not a product TODO. The script is intentionally verifying that unsupported standards are rejected. The missing issue is scan-context awareness for test/check scripts. The fix is to suppress these contract-check expectation messages from actionable gap inventories.

### 159. `utils/run_opentitan_fpv_bmc_policy_workflow.sh:323`
`unsupported mode: $MODE` is a strict CLI guard with a closed mode set; functionally valid, but extensibility requires touching this script. What is missing is declarative workflow mode registration and shared validation with callers. The fix is to define allowed modes in one table and reuse it across argument parsing/help so adding a mode is a data change, not branch surgery.

### 160. `utils/select_opentitan_connectivity_cfg.py:276`
Rejecting unknown connectivity CSV row kinds (`CONNECTION|CONDITION`) enforces manifest hygiene, but it hard-fails future schema extension. What is missing is explicit schema versioning and controlled forward compatibility in CSV ingestion. The fix is to pair row-kind validation with schema-version checks and migration logic, keeping hard failures only for unknown kinds under the active schema.

### 161. `utils/run_avip_xcelium_reference.sh:94`
This is the same constrained AVIP-set guard seen in other AVIP runner scripts: only `core8` and `all9` are accepted. What is missing is a shared, data-driven AVIP set definition so support doesn’t diverge between scripts. The fix is to centralize AVIP set metadata and have all runner scripts validate against that shared source.

### 162. `utils/check_opentitan_fpv_objective_parity_drift.py:83`
Again this is duplicated allowlist mode parsing (`exact|prefix|regex`) rather than a unique local gap. The missing capability is parser reuse across parity and drift tools to avoid subtle behavior drift. The fix is to consolidate parser logic and move this script to the shared helper.

### 163. `utils/generate_mutations_yosys.sh:696`
Rejecting unsupported design extensions is safe but rigid; it requires users to pre-normalize files into `.il/.v/.sv`. What is missing is either broader front-end format support or an integrated conversion step. The fix is to add optional conversion/adaptation for common additional formats while preserving strict validation for unknown types.

### 164. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:76`
This macro defines a “not implemented” error for TLM-2 interface tasks in base interface scaffolding, which reflects incomplete method bodies by design. The missing behavior is concrete task implementations in derived interfaces/components where these APIs are expected to function. The fix is to ensure runtime integrations override these stubs and to add tests that fail if base “not implemented” paths are reached in supported flows.

### 165. `lib/Runtime/uvm-core/src/tlm2/uvm_tlm2_ifs.svh:81`
Same as entry 164, but for function paths: the macro exists as a default error for unimplemented TLM-2 functions. The missing capability is validated functional overrides for all required API points used by CIRCT-supported UVM workloads. The fix is coverage tests that exercise each required TLM-2 function and assert no default “not implemented” macro fires.

### 166. `test/firtool/firtool.fir:77`
This TODO is real and useful: the test currently depends on brittle behavior around aggressive port removal, so it can fail for non-semantic reasons. What is missing is a more robust assertion strategy that checks intended semantics rather than incidental port shape. The fix is to rewrite the test with narrower, behavior-focused checks and minimize sensitivity to unrelated canonicalization changes.

### 167. `unittests/Support/PrettyPrinterTest.cpp:75`
This `xxxxxxxx...` identifier is stress-test fixture data for long-token wrapping, not a TODO/unsupported gap. The missing issue is scan noise from `XXX`-like substrings inside test literals. The fix is to exclude long literal fixtures from marker scans.

### 168. `unittests/Support/PrettyPrinterTest.cpp:204`
Same fixture pattern as entry 167: this line intentionally validates pretty-print wrapping with long names. There is no implementation debt signaled here. The fix is audit filtering for known fixture strings.

### 169. `unittests/Support/PrettyPrinterTest.cpp:237`
Another intentional expected-output line using long `x...` tokens for formatter tests. What is missing is not product functionality but scan precision. The fix is to suppress these fixture regions from TODO/unsupported audits.

### 170. `unittests/Support/PrettyPrinterTest.cpp:247`
This is still fixture text in expected pretty-printed output, not a gap marker. The fix remains scanner refinement to avoid matching generic `xxx` sequences in test literals.

### 171. `unittests/Support/PrettyPrinterTest.cpp:279`
Same category: expected-output fixture, not unresolved work. The missing part is audit quality. The fix is to filter test expectation string blocks in this file.

### 172. `unittests/Support/PrettyPrinterTest.cpp:285`
Again this is intentionally long token text used for line-breaking tests and should not be treated as TODO debt. The fix is no code change here, only scan-rule tightening.

### 173. `unittests/Support/PrettyPrinterTest.cpp:304`
This match comes from expected formatting output and is non-actionable from a feature-gap perspective. The missing issue is false positives in static scans. The fix is to update scan heuristics.

### 174. `unittests/Support/PrettyPrinterTest.cpp:317`
This line is test fixture content for nested-call wrapping behavior, not missing functionality. The fix is scanner suppression for this literal family.

### 175. `unittests/Support/PrettyPrinterTest.cpp:321`
Again fixture text, not gap. The missing work is in tooling around audits, not in pretty printer implementation at this location.

### 176. `unittests/Support/PrettyPrinterTest.cpp:335`
Expected-output literal with long `x` token; no TODO or unsupported path implied. The fix is to avoid matching these literals in debt scans.

### 177. `unittests/Support/PrettyPrinterTest.cpp:339`
Same as entries 167–176: intentional fixture string. This should be filtered from actionable gap tracking.

### 178. `unittests/Support/PrettyPrinterTest.cpp:350`
Long prototype string in a margin-2048 test is deliberate stress data, not unresolved functionality. The fix is better scan discrimination, not production code changes.

### 179. `unittests/Support/PrettyPrinterTest.cpp:356`
Another long fixture prototype for nested call formatting; non-actionable as a gap. The fix is scanner filtering.

### 180. `unittests/Support/PrettyPrinterTest.cpp:362`
Same fixture pattern in test expectations; no missing feature indicated. The fix is to treat this as scan noise.

### 181. `unittests/Support/PrettyPrinterTest.cpp:671`
`StringToken("xxxxxxxxxxxxxxx")` is test input to force line wrapping behavior. It is not a TODO/unsupported marker. The fix is exclusion of synthetic token literals from scans.

### 182. `unittests/Support/PrettyPrinterTest.cpp:690`
This expected output line with repeated x/y strings is formatter test data, not debt. The missing item is audit precision only.

### 183. `unittests/Support/PrettyPrinterTest.cpp:697`
Again expected formatter output containing long x/y words; not an implementation gap. The fix is scanner suppression for this test section.

### 184. `unittests/Support/PrettyPrinterTest.cpp:702`
Same non-actionable fixture output as previous entries. No code change needed in this location for gap closure.

### 185. `unittests/Support/PrettyPrinterTest.cpp:709`
This is indentation fixture output (`>>>>>>xxxxxxxx...`) for pretty-printer behavior validation. It should not be interpreted as TODO debt.

### 186. `unittests/Support/PrettyPrinterTest.cpp:716`
Same fixture-category false positive as entry 185. The fix is audit filtering.

### 187. `unittests/Support/PrettyPrinterTest.cpp:723`
Expected output text in tests; not unresolved implementation. The missing issue is scan noise reduction.

### 188. `unittests/Support/PrettyPrinterTest.cpp:730`
Again a test fixture string used for printer layout checks, not a real gap marker. The fix is no code change, only scan heuristic refinement.

### 189. `frontends/PyCDE/src/pycde/types.py:576`
This TODO is real: the >63-bit `Bits` constant workaround for Python binding limits has not been generalized to `UInt` and `SInt`. What is missing is equivalent large-constant construction paths for signed/unsigned typed wrappers. The fix is to factor the chunked-constant creation logic into shared code and apply type-correct adaptation for `UInt`/`SInt` with dedicated tests.

### 190. `unittests/Support/FVIntTest.cpp:26`
`"XXXX1"` here is a four-valued logic expectation in a unit test, not a TODO marker. The missing problem is scanner confusion between X-state test vectors and “XXX” debt tags. The fix is to exclude FVInt test literals from `XXX`-based scans.

### 191. `unittests/Support/FVIntTest.cpp:104`
This `XXXXZZZZ` value is deliberate test data for unknown/high-impedance logic handling, not an implementation gap. The fix is scanner filtering for four-valued literal patterns.

### 192. `unittests/Support/FVIntTest.cpp:108`
`000001XX0XXX0XXX` is expected result data in logical-ops tests and should not be treated as TODO debt. The missing item is audit precision around literal matching.

### 193. `unittests/Support/FVIntTest.cpp:109`
Same as entry 192: this is expected-value text in tests, not unresolved work. The fix is to narrow marker scans to actual comments/messages.

### 194. `unittests/Support/FVIntTest.cpp:110`
`01XX10XXXXXXXXXX` is again intentional expected output for FVInt behavior tests. This is non-actionable from a gap perspective and should be filtered.

### 195. `utils/check_opentitan_connectivity_objective_parity.py:151`
This is the repeated allowlist mode parser pattern seen throughout parity tooling. The missing piece is still parser deduplication and shared governance. The fix is to replace local parsing code with a shared helper and keep one conformance test suite.

### 196. `test/firtool/spec/refs/read_subelement_add.fir:9`
`XXX: ADDED` is annotation text in spec-reference test input, not a TODO debt marker. The missing issue is scanner inability to distinguish editorial comments in test fixtures. The fix is to exclude `; XXX: ADDED` style spec-note comments from actionable scans.

### 197. `test/firtool/spec/refs/read_subelement_add.fir:10`
Same as entry 196: this is a fixture comment to indicate spec-example augmentation, not unresolved implementation. The fix is scanner filtering for this annotation pattern.

### 198. `test/firtool/spec/refs/read_subelement_add.fir:11`
Again `XXX: ADDED` in test fixture text, not a project gap. The fix is no code change here; improve scan heuristics.

### 199. `test/firtool/dedup-modules-with-output-dirs.fir:7`
`"dirname": "XXX"` is a literal directory token in a test and not a TODO/unsupported marker. The missing issue is false positives from generic `XXX` matching. The fix is to ignore quoted literal values in tests for debt scans.

### 200. `test/firtool/dedup-modules-with-output-dirs.fir:18`
`CHECK: FILE "XXX..."` is expected-output pattern text in a test, not an implementation debt signal. The fix is to suppress FileCheck pattern lines from TODO/unsupported scanning, since they routinely use placeholder strings.

### 201. `test/firtool/dedup-modules-with-output-dirs.fir:72`
`"dirname": "ZZZ/XXX"` is a test fixture value used to verify output-directory dedup behavior, not a TODO/unsupported marker. The missing issue is scanner overmatching of placeholder directory names. The fix is to ignore quoted fixture values in FIRRTL test refs during debt scans.

### 202. `test/firtool/dedup-modules-with-output-dirs.fir:95`
`CHECK: FILE "ZZZ...XXX...A.sv"` is again expected pattern text and non-actionable as implementation debt. The fix is to treat FileCheck directives as test assertions, not TODO indicators.

### 203. `tools/circt-mut/circt-mut.cpp:2368`
Rejecting unsupported mutant formats for native prequalification is reasonable, but it limits native coverage to `.il/.v/.sv` and forces fallback workflows otherwise. What is missing is broader native format ingestion or integrated conversion. The fix is to expand native prequalification input support (or automate conversion before native path) while keeping strict erroring for truly unknown formats.

### 204. `tools/circt-mut/circt-mut.cpp:6369`
The TOML parser supports only a small escape subset and reports other escapes as unsupported, which can reject valid-ish user config expectations. What is missing is fuller TOML string escape support consistent with documented config grammar. The fix is to either implement the missing escapes or document the restricted subset clearly and validate configs with precise diagnostics.

### 205. `tools/circt-mut/circt-mut.cpp:14642`
This comment is a real migration-state gap: unknown generate options trigger fallback to the script backend because native option parity is incomplete. What is missing is full native CLI compatibility with legacy script options. The fix is to close option-by-option parity, add compatibility tests, and eventually remove fallback for unsupported options once native coverage is complete.

### 206. `tools/circt-mut/circt-mut.cpp:15299`
Native generate mode still rejects non-`.il/.v/.sv` design extensions, which mirrors tooling gaps elsewhere. What is missing is format flexibility or builtin conversion for additional source forms. The fix is to extend native loader capabilities or provide a first-class conversion stage before read command synthesis.

### 207. `test/firtool/spec/refs/read_subelement.fir:10`
`XXX: ADDED` is editorial annotation in spec reference material, not unresolved implementation. The fix is to filter this annotation style from actionable scans.

### 208. `test/firtool/spec/refs/read_subelement.fir:11`
Same annotation false positive as entry 207. No product gap is implied at this line.

### 209. `test/firtool/spec/refs/read_subelement.fir:12`
Again `XXX: ADDED` fixture text, not technical debt. Scanner filtering is the correct remedy.

### 210. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:34`
This macro marks default “not implemented” behavior for base TLM1 interface tasks, indicating stub semantics remain in foundational interfaces. What is missing is guaranteed override coverage in supported runtime flows. The fix is to verify all required TLM1 API paths are implemented by concrete classes used in CIRCT-supported environments and to fail tests if base stubs are reached.

### 211. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_ifs.svh:35`
Same as entry 210 for function APIs: base macro indicates unimplemented default behavior by design. The missing part is enforcement that supported workflows do not depend on these defaults. The fix is regression coverage for concrete function-path overrides and clear diagnostics when stubs are accidentally invoked.

### 212. `frontends/PyCDE/src/pycde/system.py:122`
This TODO is real and architectural: cf-to-handshake lowering is disabled in PyCDE because required dialects are not registered through current Python hooks. What is missing is a reliable dialect-loading/registration path from Python side. The fix is to expose stronger bindings (likely pybind11 module support as noted), register required dialects, and re-enable the handshake pass pipeline with tests.

### 213. `frontends/PyCDE/src/pycde/system.py:249`
Symbolref handling across potentially renamed symbols is still TODO, so imported MLIR objects can lose robust cross-reference fidelity after renaming. What is missing is symbol remapping through the import pipeline. The fix is to build a symbol-translation map during import and resolve symbolrefs through it, with regression tests around renamed modules/ops.

### 214. `frontends/PyCDE/src/pycde/system.py:574`
This TODO documents a broken MLIR live-operation cleanup hook and leaves PyCDE without an equivalent replacement, which can obscure leaked op references during debugging. What is missing is a supported lifecycle/introspection mechanism for op liveness in Python. The fix is to adopt a replacement API (or add one) and restore warnings/metrics for leaked operation references.

### 215. `test/circt-verilog/roundtrip-register-enable.sv:4`
`// UNSUPPORTED: valgrind` is test metadata due external tool/runtime constraints, not a product feature gap in register-enable lowering itself. What is missing is valgrind-lane compatibility or a reliable suppression rationale audit. The fix is to periodically reassess these UNSUPPORTED tags and retire them when upstream tool issues are resolved.

### 216. `test/firtool/spec/refs/read.fir:10`
`XXX: ADDED` is spec-fixture annotation, not unresolved implementation. The fix is scanner suppression for this editorial marker.

### 217. `test/firtool/spec/refs/read.fir:11`
Same fixture annotation false positive as entry 216. No actionable product gap at this line.

### 218. `test/firtool/spec/refs/read.fir:12`
Again fixture/comment marker, not technical debt. Scanner filtering should remove this.

### 219. `utils/run_opentitan_fpv_circt_bmc.py:231`
Unsupported stopat selector formats are rejected, which enforces strict selector grammar but can surprise users with legacy/informal selector shapes. What is missing is either richer selector normalization or clearer migration tooling. The fix is to document accepted grammar precisely and optionally add compatibility normalization for common legacy selector variants.

### 220. `utils/run_opentitan_fpv_circt_bmc.py:373`
This is the repeated allowlist parser mode gate found across many OpenTitan scripts. The missing capability remains shared parser infrastructure and single-point maintenance. The fix is parser deduplication into a common module.

### 221. `utils/run_opentitan_fpv_circt_bmc.py:1607`
`unsupported_stopat_selector` is an internal error classification when normalization fails; the gap is that selector-language support is narrower than some contract inputs. What is missing is support for additional selector forms or richer user guidance for correction. The fix is to either extend selector parsing safely or provide precise remediation hints tied to the failing row/value.

### 222. `test/circt-verilog/registers.sv:4`
Another `UNSUPPORTED: valgrind` test metadata line, not direct implementation debt. The missing piece is environmental test portability under valgrind, not register semantics support per se. The fix is to track and periodically revalidate valgrind exclusions.

### 223. `test/firtool/spec/refs/probe_export_simple.fir:7`
`XXX: Added width.` is explanatory fixture commentary in spec refs and not a TODO gap. The fix is scanner suppression for `XXX:` annotations in reference tests.

### 224. `test/firtool/spec/refs/nosubaccess.fir:8`
`XXX: Modified ...` is similarly a fixture note documenting adaptation of a spec example, not unresolved implementation. The fix is to ignore this annotation class in actionable debt reports.

### 225. `utils/check_opentitan_compile_contract_drift.py:95`
This is the same allowlist mode parser duplication pattern. The missing behavior is shared parser reuse across contract drift/parity tools. The fix is to migrate to one common parser and drop per-script copies.

### 226. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:35`
Default “task not implemented” macro in TLM FIFO base indicates stub behavior for unimplemented FIFO task APIs. What is missing is assurance that supported runtime paths bind to concrete implementations instead of stub defaults. The fix is targeted runtime tests and explicit override verification.

### 227. `lib/Runtime/uvm-core/src/tlm1/uvm_tlm_fifo_base.svh:36`
Same as entry 226 for function APIs: this macro is a default error path, not full implementation. The missing capability is validated coverage of concrete FIFO function behavior in CIRCT-supported UVM flows. The fix is to add/strengthen tests that ensure these defaults are never hit in expected scenarios.

### 228. `test/circt-verilog/memories.sv:4`
`UNSUPPORTED: valgrind` here is another environmental test exclusion marker, not a direct memory-lowering TODO. The missing piece is tool/environment compatibility to run this test under valgrind reliably. The fix is to revisit exclusions as toolchain issues improve.

### 229. `tools/circt-sim/AOTProcessCompiler.h:211`
The comment notes that functions with unsupported ops are skipped during AOT function-body compilation, which is a real capability boundary. What is missing is wider AOT compilability coverage and/or better fallback accounting for skipped functions. The fix is to expand `isFuncBodyCompilable()` support set and produce structured skip diagnostics so unsupported-op hotspots are visible.

### 230. `test/circt-verilog/redundant-files.sv:4`
This `UNSUPPORTED: valgrind` marker is the same class as entries 215/222/228: test-runner environment exclusion, not core feature debt at this location. The fix is ongoing periodic validation of whether valgrind exclusions are still necessary.

### 231. `lib/Dialect/LLHD/Transforms/Deseq.cpp:824`
This TODO is a real correctness gap in deseq simplification: values that depend on trigger signals are currently accepted and folded through `getKnownValue`, which can hide trigger-coupled behavior. What is missing is explicit dependence rejection for trigger-derived values. The fix is to run a backward dependence walk against trigger SSA roots and refuse simplification when dependence is detected.

### 232. `lib/Runtime/uvm-core/src/base/uvm_root.svh:1280`
This `TBD` is a real lifecycle-design concern: the run-phase zero-time check is currently piggybacking on assumptions about `uvm_root` behavior. What is missing is a callback location with explicit ordering guarantees. The fix is to move the check into a phase callback with defined semantics (as noted by the nearby comment) and add phase-order regression tests.

### 233. `lib/Runtime/uvm-core/src/base/uvm_root.svh:1281`
Same issue as entry 232: this TODO marks uncertainty about where run-phase initialization validation belongs. The missing piece is a formally correct hook point for this policy. The fix is to migrate to `phase_started` (or equivalent authoritative callback), then lock behavior with UVM phase-timing tests.

### 234. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:86`
Expanded macros are explicitly not handled in this range-processing path, which means symbol indexing/navigation can silently miss or misplace macro-origin entities. What is missing is macro-expansion-aware source mapping. The fix is to resolve expansion vs spelling locations through Slang source manager APIs and add macro-heavy LSP index tests.

### 235. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:449`
This parser only supports one serialized location style (`@[...]`), creating a compatibility gap when location formatting settings differ. What is missing is multi-format location parsing or normalization. The fix is to support alternate location-info styles (or a shared parser abstraction) with roundtrip tests per style.

### 236. `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogIndex.cpp:472`
This path also skips expanded macros, so reference mapping can be incomplete for macro-generated code. What is missing is robust handling of expanded macro source ranges in reference indexing. The fix is macro-aware range translation and targeted tests for macro definition/usage references.

### 237. `lib/Dialect/LLHD/Transforms/RemoveControlFlow.cpp:108`
This TODO is a performance/scalability gap: branch-decision aggregation is eager across control-flow, potentially doing unnecessary work on large regions. What is missing is a bounded region-aware decision analysis. The fix is to restrict analysis to blocks between dominator and target (as the comment suggests) and benchmark compile-time improvements.

### 238. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:45`
Hardcoding the Verilated module class name is a real configurability gap and risks mismatches with CLI/user-specified Verilator naming. What is missing is attribute/config-driven class resolution. The fix is to derive class names from interop config attributes (or pass options) and validate with integration tests over custom module names.

### 239. `lib/Dialect/SystemC/Transforms/SystemCLowerInstanceInterop.cpp:125`
Indirect evaluation still uses a temporary `func::CallIndirectOp` fallback, leaving a dialect-boundary cleanup debt. What is missing is native `systemc::CallIndirectOp` use and removal of extra func-dialect coupling. The fix is to migrate once the dependency lands, then prune the fallback dependency chain and add interop call-lowering tests.

### 240. `include/circt/Dialect/HWArith/HWArithOps.td:170`
`AnyInteger` here is a type constraint declaration in ODS, not a TODO/unsupported marker. This is a scan false positive. The fix is to refine scanners to ignore ordinary type-constraint tokens in TableGen op definitions.

### 241. `include/circt/Dialect/HWArith/HWArithOps.td:171`
Same as entry 240: `AnyInteger` result typing is expected ODS content and not unresolved work. The fix is scanner filtering for non-comment, non-diagnostic TableGen tokens.

### 242. `include/circt/Dialect/SV/SVStatements.td:68`
This TODO marks an ergonomic/tooling gap in ODS: custom builders are required only to ensure implicit region terminator installation. What is missing is a declarative ODS mechanism to request this behavior directly. The fix is upstream ODS support (or local helper abstraction) to eliminate repetitive custom-builder boilerplate.

### 243. `include/circt/Dialect/SV/SVStatements.td:124`
Same pattern as entry 242 for another op: boilerplate builders exist due ODS terminator limitations, increasing maintenance surface. The missing piece is shared declarative terminator support. The fix is centralizing builder generation once ODS capability exists and removing duplicated custom code.

### 244. `include/circt/Dialect/SV/SVStatements.td:169`
Again, this is the same ODS limitation on implicit terminator insertion. What is missing is declarative region-terminator control at ODS level. The fix is an ODS enhancement and follow-up cleanup to standard builders.

### 245. `include/circt/Dialect/SV/SVStatements.td:335`
This is the same terminator-builder workaround repeated for loop-body region ops. The missing capability remains ODS-native region terminator provisioning. The fix is to adopt such support and remove custom builders across SV statement ops.

### 246. `lib/Dialect/SystemC/SystemCOps.cpp:746`
Copy-pasting from the func dialect is a real maintainability gap: behavior parity depends on manual sync with upstream changes. What is missing is reusable shared implementation hooks for call-like op verification/parsing/printing. The fix is to factor common function-interface utilities upstream or into shared CIRCT helpers.

### 247. `lib/Dialect/SystemC/SystemCOps.cpp:751`
This explicit `FIXME` confirms an exact upstream copy for symbol-use verification. What is missing is deduplicated function-interface logic. The fix is to replace copied verification code with shared utility calls and add parity tests to avoid drift.

### 248. `lib/Dialect/SystemC/SystemCOps.cpp:816`
The FuncOp implementation is largely copied from upstream, creating drift risk and larger review surface for future updates. What is missing is a composable function-op implementation layer reusable by SystemC ops. The fix is to upstream/factor common pieces and minimize local forks.

### 249. `lib/Dialect/SystemC/SystemCOps.cpp:878`
Parser code is copied because SSA argument-name access requires custom handling. What is missing is an upstream parser extension point that exposes argument-name handling without full copy-paste. The fix is to add that hook upstream (or local utility wrapper) and delete duplicated parser logic.

### 250. `lib/Dialect/SystemC/SystemCOps.cpp:981`
Printing logic is inlined from upstream to customize attribute elision, which again creates divergence risk. What is missing is configurable upstream print helpers. The fix is to expose attribute-elision policy hooks and switch SystemC ops back to shared printers.

### 251. `lib/Dialect/SystemC/SystemCOps.cpp:1012`
Clone-operation code is copied from upstream; this is maintainability debt and a parity hazard when upstream clone semantics evolve. What is missing is reusable cloning helpers for function-interface ops. The fix is to consolidate on shared clone utility code and add behavior lock tests.

### 252. `lib/Dialect/SystemC/SystemCOps.cpp:1123`
ReturnOp implementation is also copy-pasted, repeating the same drift issue. What is missing is shared return-op validation/building infrastructure for function-like dialects. The fix is helper extraction and replacement of local copies.

### 253. `lib/Runtime/uvm-core/src/base/uvm_resource_base.svh:554`
`record_xxx_access` in this comment is part of an API name and not a TODO/unsupported marker. This is a scanner false positive caused by `xxx` token matching. The fix is to ignore identifier fragments like `xxx` when they are inside established API names/comments with no debt semantics.

### 254. `lib/Bindings/Tcl/circt_tcl.cpp:67`
This TODO is a real user-facing feature gap: Tcl `load FIR` is explicitly unimplemented. What is missing is FIR file ingestion through the Tcl binding path. The fix is to implement FIR loader plumbing (or remove/guard the command with clear capability docs) and add Tcl integration tests.

### 255. `lib/Dialect/LLHD/Transforms/InlineCalls.cpp:246`
Hardcoded C++ class-field offsets (`m_parent`, `m_name`) are a real correctness/portability gap and can break with ABI/layout changes. What is missing is metadata-driven offset resolution. The fix is to compute offsets from class metadata (or pass them explicitly from frontend lowering) and remove magic constants.

### 256. `include/circt/Dialect/SV/SVTypes.td:18`
This TODO is a low-priority structural note, not an immediate capability blocker: it suggests moving definitions to `SVTypesImpl.td` once SV-specific types exist. What is missing is file-organization cleanup only when/if types are added. The fix is to keep it as deferred refactor and execute during first real SV type introduction.

### 257. `lib/Dialect/LLHD/Transforms/HoistSignals.cpp:652`
Using ad-hoc constants where a semantic `llhd.dontcare` would be more precise is a real IR expressiveness gap. What is missing is first-class don't-care materialization in this transform path. The fix is to introduce/route through an LLHD don't-care representation and update downstream consumers/tests accordingly.

### 258. `lib/Bindings/Python/support.py:281`
This TODO marks exception taxonomy debt: unconnected-backedge diagnostics still use generic text instead of `UnconnectedSignalError`. What is missing is consistent typed error signaling for Python users. The fix is to emit `UnconnectedSignalError` and update tests/documentation to match the richer exception type.

### 259. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:62`
Supported logic-op simulation is intentionally narrow (`aig::AndInverterOp` only), limiting cut rewriting opportunities. What is missing is broader combinational op support (`comb.and/xor/or` noted in TODO). The fix is to extend `isSupportedLogicOp` and simulation semantics for these ops with regression coverage.

### 260. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:174`
Assuming `numOutputs == 1` is a functional limitation in truth-table extraction and blocks multi-output cuts. What is missing is generalized multi-output table construction/return. The fix is to refactor `simulateCut` to produce all outputs and thread multi-output support through callers/tests.

### 261. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:343`
This TODO is a real performance gap in cut merging: the code re-derives topological order instead of merging already-sorted operation lists. What is missing is linear-time merge of pre-sorted cut operation vectors. The fix is to implement merge-sort-by-index for `operations`/`other.operations` and keep duplicate elimination cheap.

### 262. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:402`
Merged cut inputs are not explicitly sorted by defining operation, which can make ordering unstable and affect downstream reproducibility/caching. What is missing is deterministic canonical input ordering. The fix is to sort merged inputs by stable op index and lock with deterministic-output tests.

### 263. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:403`
Area and delay are not recomputed on merged cuts, so cost modeling is incomplete after merge. What is missing is cost recomputation with merged-structure awareness. The fix is to update area/delay bookkeeping during merge and validate with cut-ranking regression tests.

### 264. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:552`
Using full sorting where a priority queue would suffice is a known efficiency gap in cut selection. What is missing is incremental best-cut retrieval. The fix is to replace repeated sort-heavy selection with a priority queue and benchmark compile-time wins on large netlists.

### 265. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:684`
Cut enumeration currently rejects variadic ops and non-single-bit results, constraining applicability. What is missing is generalized operation/result-shape support. The fix is to add variadic handling and bit-slicing/multi-bit result support in enumeration and simulation.

### 266. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:887`
Patterns with multiple outputs are still hard-rejected, which blocks richer pattern libraries. What is missing is multi-output matching/rewrite support. The fix is to remove this guard once multi-output truth-table/extraction support is implemented end-to-end.

### 267. `lib/Dialect/Synth/Transforms/CutRewriter.cpp:957`
Primary-input delays ignore global arrival-time context, so delay estimation is incomplete. What is missing is explicit arrival-time modeling on IR values. The fix is to propagate/capture arrival-time metadata and include it in input delay computation.

### 268. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:122`
This FIXME indicates conservative SOP cut limits are used because CutRewriter performance is currently weak. What is missing is efficient enough cut rewriting to use stronger defaults comparable to ABC/mockturtle. The fix is to optimize CutRewriter and then retune `maxCutInputSize` defaults with benchmark-backed thresholds.

### 269. `lib/Dialect/Synth/Transforms/SynthesisPipeline.cpp:139`
Pipeline TODO notes missing major synthesis stages (balancing/rewriting/FRAIG, etc.). What is missing is fuller logic-optimization flow parity. The fix is to add these passes with configurable ordering and regression/quality-of-result benchmarks.

### 270. `lib/Dialect/Synth/Transforms/TechMapper.cpp:135`
Mapped instance naming is placeholder-quality, which hurts readability/debugging and can complicate downstream tooling. What is missing is stable, meaningful naming policy. The fix is to derive names from source op/module/cut identifiers with collision-safe suffixing.

### 271. `lib/Dialect/Synth/Transforms/TechMapper.cpp:174`
Technology-library metadata is currently represented as an ad-hoc attribute dictionary. What is missing is structured IR representation for techlib semantics. The fix is to introduce dedicated techlib ops/types/attrs and migrate mapper logic to them.

### 272. `lib/Dialect/Synth/Transforms/TechMapper.cpp:183`
Mapping currently runs broadly rather than being constrained to target hierarchy scopes. What is missing is hierarchy-scoped mapping control. The fix is to add scope selection (attribute/option driven) and gate mapping to explicit module subtrees.

### 273. `lib/Dialect/Synth/Transforms/TechMapper.cpp:201`
This line is ordinary numeric conversion (`convertToDouble`) and not a TODO/fixme/unsupported marker. This is a scanner false positive. The fix is to tighten marker matching to explicit debt tokens instead of incidental symbols/identifiers.

### 274. `lib/Dialect/Synth/Transforms/TechMapper.cpp:207`
Delay parsing assumes integer attributes, which is a temporary limitation against richer timing models/units. What is missing is typed timing attributes with unit semantics. The fix is to migrate to structured cell timing attributes and update parsing/validation accordingly.

### 275. `lib/Dialect/Synth/Transforms/LowerWordToBits.cpp:189`
Known-bits fallback uses a depth-limited, uncached path, which is both potentially imprecise and slower. What is missing is cache-aware, depth-robust known-bits computation integration. The fix is to thread cached known-bits analysis and avoid repeated recomputation.

### 276. `lib/Dialect/Synth/Transforms/LowerVariadic.cpp:134`
Only top-level ops are lowered due missing topological handling across nested regions, leaving nested-region variadics untouched. What is missing is region-aware traversal/scheduling. The fix is to implement nested topological rewrite order and extend coverage with nested-region tests.

### 277. `lib/Tools/circt-lec/ConstructLEC.cpp:54`
Fetched LLVM globals are not fully sanity-checked before reuse, which risks attribute/type mismatches going unnoticed. What is missing is strict global-shape validation. The fix is to verify linkage/type/attrs against expected schema and emit actionable diagnostics on mismatch.

### 278. `lib/Tools/circt-lec/ConstructLEC.cpp:219`
The TODO calls out avoidable LLVM-specific construction in result plumbing. What is missing is a cleaner dialect-level reporting path. The fix is to replace LLVM-constant/return scaffolding with higher-level ops or a dedicated reporting abstraction.

### 279. `lib/Tools/circt-lec/ConstructLEC.cpp:231`
Result reporting is currently implemented by injecting LLVM-print style plumbing, which is acknowledged as inelegant. What is missing is first-class LEC result-reporting mechanism. The fix is a dedicated report op/API and streamlined lowering to the final presentation layer.

### 280. `lib/Analysis/FIRRTLInstanceInfo.cpp:229`
`anyInstanceUnderDut` is a normal API function name, not a debt marker. This is a scanner false positive likely triggered by broad token matching. The fix is to match only explicit TODO/FIXME/TBD/unsupported markers in comments/diagnostics.

### 281. `lib/Analysis/FIRRTLInstanceInfo.cpp:239`
Same as entry 280: this is routine API code (`anyInstanceUnderEffectiveDut`), not unresolved work. The fix is scanner narrowing to actionable marker patterns.

### 282. `lib/Analysis/FIRRTLInstanceInfo.cpp:240`
Again this is ordinary implementation code returning computed status, not a TODO/unsupported gap. The fix is improved audit heuristics to avoid generic identifier false positives.

### 283. `lib/Analysis/FIRRTLInstanceInfo.cpp:247`
`anyInstanceUnderLayer` is part of analysis API surface, not a debt comment. This is non-actionable and should be filtered from gap scans.

### 284. `lib/Analysis/FIRRTLInstanceInfo.cpp:257`
`anyInstanceInDesign` is also normal function code with no TODO semantics. The missing item is scan precision, not implementation work.

### 285. `lib/Analysis/FIRRTLInstanceInfo.cpp:267`
Same false-positive class as entries 280-284: ordinary API naming captured as debt. The fix is to require explicit marker comments/messages for audit inclusion.

### 286. `lib/Bindings/Python/dialects/synth.py:57`
This TODO marks a real Python binding gap: wrapper objects are not yet linked back to MLIR value/op identities. What is missing is stable association to underlying IR objects for diagnostics/introspection. The fix is to store MLIR handles in the wrapper and add lifecycle-safe accessors/tests.

### 287. `lib/Dialect/Synth/Transforms/AIGERRunner.cpp:207`
Repeated `comb.extract` creation in bit expansion is uncached, leaving avoidable overhead in large multi-bit mappings. What is missing is extract-op reuse. The fix is local caching keyed by `(value, bitPosition)` to reuse existing extracts and reduce IR bloat.

### 288. `include/circt/Dialect/HW/HWStructure.td:247`
This TODO acknowledges a modeling hack around `verilogName` due missing proper parameterized-type representation in HW dialect. What is missing is first-class parameterization/type abstraction. The fix is to introduce parameterized type modeling and retire this naming workaround.

### 289. `lib/Analysis/CMakeLists.txt:67`
Linting subdirectory is disabled behind a TODO due Slang header-path issues, leaving analysis linting unavailable in normal builds. What is missing is robust include-path integration for CIRCTLinting. The fix is to resolve header path configuration and re-enable linting target with CI coverage.

### 290. `lib/Dialect/Synth/SynthOps.cpp:68`
Majority folding is incomplete for constant patterns (`maj(x,1,1)=1`, `maj(x,0,0)=0`), missing easy canonicalizations. What is missing is these constant-folding rules in `MajorityInverterOp::fold`. The fix is to implement these cases and add fold regression tests.

### 291. `lib/Dialect/Synth/SynthOps.cpp:298`
The current variadic-and/inverter lowering uses a balanced binary tree regardless of signal timing or fanout, which is correct but leaves QoR on the table. What is missing is a cost-aware tree-construction strategy that can optimize for critical-path delay and/or area. The fix is to add a heuristic or analysis-guided decomposition policy and validate it with timing/size regressions on representative netlists.

### 292. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:97`
This TODO captures a real dialect-conversion limitation: direct RAUW is used as a workaround because `replaceAllUsesWith` is not safely supported in this conversion context. What is missing is a first-class replacement path compatible with `ConversionPatternRewriter` semantics. The fix is to migrate to the canonical conversion API once available (or encapsulate the workaround centrally) and add regression coverage to prevent mutate-after-erase assertions.

### 293. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:319`
Same underlying gap as entry 292: this path still depends on direct RAUW during dialect conversion due framework limitations. What is missing is a robust conversion-safe value replacement mechanism for this forwarding case. The fix is shared migration to supported conversion rewrite utilities and removal of ad hoc RAUW once upstream support lands.

### 294. `lib/Dialect/Kanagawa/Transforms/KanagawaPortrefLowering.cpp:359`
Again this is the same conversion-framework workaround pattern, indicating repeated technical debt in the portref lowering pass. What is missing is a unified, safe replacement strategy instead of duplicated direct RAUW sites. The fix is to refactor these replacement points behind one helper and switch to native `DialectConversion` support when available.

### 295. `lib/Runtime/uvm-core/src/base/uvm_port_base.svh:528`
`// TBD tidy` here marks cleanup debt around duplicated phase-state checks for late connection handling, not a direct functional correctness hole. What is missing is a normalized helper for this phase gating logic and clearer phase-state intent in diagnostics. The fix is to factor the condition into a shared predicate/API and keep one authoritative check path.

### 296. `lib/Runtime/uvm-core/src/base/uvm_port_base.svh:638`
Same cleanup debt as entry 295 in a second call path (`debug_connected_to`): equivalent phase-state logic is open-coded again. What is missing is consistency and maintainability for this guard behavior. The fix is to reuse the same helper predicate and remove duplicated state-check fragments.

### 297. `include/circt/Dialect/HW/HWOps.h:42`
This TODO is an architectural gap: module helper functions are free functions instead of being surfaced through a `hw::ModuleLike` interface. What is missing is a uniform interface abstraction that makes module-like operations interchangeable across passes. The fix is to move these helpers into interface methods, migrate call sites, and keep compatibility shims only during transition.

### 298. `lib/Tools/circt-bmc/LowerToBMC.cpp:3312`
The pass currently injects LLVM dialect ops just to return `0` from generated `main`, which mixes abstraction levels and complicates lowering boundaries. What is missing is a cleaner dialect-level return construction in this stage. The fix is to construct `main` purely with non-LLVM core ops (or dedicated helper API) and leave LLVM materialization to later lowering passes.

### 299. `lib/Dialect/Kanagawa/Transforms/KanagawaPassPipelines.cpp:62`
This TODO points to a real verification gap: there is no pass ensuring unexpected `memref.alloca` values are gone after `mem2reg`. What is missing is an explicit pipeline invariant check before downstream SSA assumptions. The fix is to add a verifier/cleanup pass that rejects illegal residual allocas (except approved member-variable cases) with actionable diagnostics.

### 300. `lib/Analysis/TestPasses.cpp:238`
`anyInstanceUnderDut` here is expected diagnostic-print text in a test analysis pass, not unresolved implementation work. This is a scanner false positive from broad token matching. The fix is to ignore ordinary API identifier text in pass debug output when building debt lists.

### 301. `lib/Analysis/TestPasses.cpp:242`
This is continuation of expected analysis output formatting (`anyInstanceUnderEffectiveDut`) and is non-actionable from a gap perspective. The missing issue is scanner precision, not code functionality. The fix is marker filtering to comments/diagnostics that explicitly encode TODO/FIXME/TBD/unsupported debt.

### 302. `lib/Analysis/TestPasses.cpp:243`
Same false-positive class as entry 301: this line simply prints a computed analysis value. No implementation gap is implied. The fix is to exclude ordinary stream-output lines from debt scans.

### 303. `lib/Analysis/TestPasses.cpp:246`
`anyInstanceUnderLayer` label output in a test pass is intentional and not a TODO or unsupported marker. This is non-actionable audit noise. The fix is tighter pattern heuristics that avoid API-name collisions.

### 304. `lib/Analysis/TestPasses.cpp:247`
Same as entry 303: expected test-pass printout, not deferred work. The missing capability is scanner disambiguation between identifiers and debt markers. The fix is filtering for comment/diagnostic contexts only.

### 305. `lib/Analysis/TestPasses.cpp:250`
`anyInstanceInDesign` appears in normal debug output for analysis tests, which should not be classified as a project gap. This is another false positive from token overmatching. The fix is to treat this class of output-string lines as non-actionable by default.

### 306. `lib/Analysis/TestPasses.cpp:254`
This line is also expected label output (`anyInstanceInEffectiveDesign`) in a test pass and does not indicate missing implementation. The true gap is scan quality. The fix is debt-marker parsing that requires explicit TODO-like syntax.

### 307. `lib/Analysis/TestPasses.cpp:255`
Same false-positive pattern as entries 300-306: a printed value expression, not a gap marker. No code change is needed in this file for feature completeness. The fix is scanner refinement.

### 308. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:211`
Name extraction for block arguments only handles `hw::HWModuleOp` and falls back to `<unknown-argument>` otherwise, reducing analysis/debug fidelity on other operation kinds. What is missing is generalized argument naming across additional module-like ops. The fix is to extend `getNameImpl` dispatch to other relevant parent ops and add tests that verify stable naming in those contexts.

### 309. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:729`
Debug-point collection is always instantiated, which adds memory/runtime overhead even when path-debug data is not needed. What is missing is a mode to disable this bookkeeping for performance-focused runs. The fix is to gate debug-point factory creation and propagation behind an option, keeping output parity only when debug mode is enabled.

### 310. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1307`
Memory write endpoints are modeled without incorporating the write address, which collapses distinct memory locations and can blur path precision. What is missing is address-sensitive endpoint tracking for `seq::FirMemWriteOp`. The fix is to include address information in endpoint keys/path state and add regressions demonstrating distinct per-address behavior.

### 311. `lib/Dialect/Synth/Analysis/LongestPathAnalysis.cpp:1312`
Same precision gap as entry 310 for `seq::FirMemReadWriteOp`: endpoint modeling ignores address. What is missing is consistent address-aware handling across both write-capable memory op forms. The fix is to apply the same address-aware representation and tests to read-write ports.

### 312. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:293`
Argument/result name collection is open-coded with a TODO noting it belongs in `ModulePortInfo`, so port metadata ownership is currently fragmented. What is missing is a single structured source of port naming data. The fix is to extend `ModulePortInfo` to carry canonical arg/result names and remove local reconstruction logic.

### 313. `lib/Dialect/Kanagawa/Transforms/KanagawaContainersToHW.cpp:314`
This is the same dialect-conversion RAUW workaround pattern as entries 292-294, repeated in container-to-HW lowering. What is missing is conversion-safe replacement support that avoids direct RAUW in these patterns. The fix is to centralize and eventually eliminate this workaround once the conversion rewriter supports the needed API.

### 314. `lib/Analysis/DependenceAnalysis.cpp:169`
`replaceOp` currently performs a full scan of all dependence entries to rewrite source references, which is simple but can become expensive at scale. What is missing is reverse indexing from source op to dependent edges for efficient updates. The fix is to maintain an inverted index alongside `results` and update both maps on insertion/replacement.

### 315. `lib/Analysis/DebugInfo.cpp:165`
Fallback debug-info construction for instances does not track port assignments, leaving incomplete connectivity context in the resulting debug model. What is missing is representation of instance port-to-value bindings in fallback DI paths. The fix is to add a port-assignment structure to debug-info instances and populate it when visiting `hw.instance`.

### 316. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:537`
`TBD add more useful debug` marks observability debt in phase execution tracing, not a direct semantic bug. What is missing is richer debug output for phase control transitions/termination behavior. The fix is to define a concise debug schema and emit consistent trace points guarded by the existing phase-trace switch.

### 317. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:553`
`m_aa2string` is flagged `TBD tidy`, indicating code-hygiene debt in formatting predecessor/successor maps. What is missing is cleaner, reusable stringification for phase edge sets. The fix is to refactor this helper for readability/reuse and align formatting with other UVM diagnostics.

### 318. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:762`
The add-schedule API lacks validation that referenced phase nodes belong to the current schedule/domain, which can allow invalid graph edits before later failure. What is missing is early structural error checking for schedule membership. The fix is to add explicit membership checks for all relationship parameters and emit immediate fatal diagnostics on mismatch.

### 319. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1218`
State-change callback payload fields are manually poked with a comment noting no official setter path, which indicates API design debt around phase-state transitions. What is missing is a sanctioned constructor/setter interface for `m_state_chg` updates. The fix is to introduce a formal update helper/API and route all state transitions through it.

### 320. `lib/Runtime/uvm-core/src/base/uvm_phase.svh:1383`
`find` is explicitly marked as not doing a full search, so phase lookup coverage is intentionally incomplete in some graph/scope patterns. What is missing is comprehensive traversal semantics (with proper scope controls) for phase search. The fix is to implement full search behavior with cycle-safe traversal and regression tests for in-scope and cross-scope queries.
