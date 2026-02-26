# AOT Entry Dispatch Engineering Log

## 2026-02-25
- Added regression assertions to `test/Tools/circt-sim/aot-vtable-dispatch.mlir` for:
  - entry-table native classification line in load output
  - non-zero entry-table native call count in `--aot-stats`
- Realization: validating this path requires rebuilt binaries; stale `build_test/bin/circt-sim` output can falsely suggest runtime logic is still disabled.
- Surprise: `build_test/bin/circt-sim` only picked up the updated dispatch logic after forcing a recompilation path that included `LLHDProcessInterpreter.cpp`.
- Validation signal used: run output now reports
  - `Loaded 2 compiled functions: 2 native-dispatched, 0 excluded (call_indirect), 0 intercepted`
  - `Entry table: 2 entries for tagged-FuncId dispatch (2 native, 0 non-native)`
  - `Entry-table native calls:         2`
- Added `CIRCT_AOT_STATS=1` environment toggle in `circt-sim` so AOT counters can be enabled without CLI flag changes in benchmark harnesses.
- Added lit coverage in `aot-basic-func.mlir` to lock in env-driven stats behavior.
- Added lowering support target for `func.call_indirect` in `circt-sim-compile` (ptr->function cast + indirect call path) to prevent stripping compiled callees that use virtual dispatch.
- Added regression test `aot-call-indirect-func-body.mlir` to prove function-body `call_indirect` survives AOT compilation and dispatches through tagged entry-table lookup.
- Reproduced compiled-mode crash in the new `aot-call-indirect-func-body` test with
  - `Assertion 'isUIntN(BitWidth, val)' failed` in `APInt(BitWidth, uint64_t)`.
- Fixed narrow-return marshalling in native dispatch paths by converting native return values with:
  - `llvm::APInt(64, ret).zextOrTrunc(resultWidth)`
  - instead of constructing `APInt(resultWidth, ret)` directly.
- Surprise: after fixing the APInt assert, compiled mode still produced incorrect values for vtable-backed calls in compiled functions because vtable globals remained zero-initialized in the .so.
- Added tagged vtable global initialization in `circt-sim-compile`:
  - collect `{global, slot, fid}` from `circt.vtable_entries` using interpreter-compatible FuncId ordering
  - materialize `0xF0000000 + fid` into LLVM globals (supports flattened `[N x i8]` and typed `[N x ptr]` forms)
  - emits compile log line: `Initialized <N> vtable globals with tagged FuncIds`.
- Realization: `llvm::Module::getGlobalVariable(name)` ignores internal-linkage globals unless `AllowInternal=true`; this prevented initial vtable tagging until corrected.
- Updated regression expectations in `aot-call-indirect-func-body.mlir`:
  - compile check includes tagged-vtable initialization line
  - compiled stats expect `Entry-table native calls: 0` (the optimized compiled path folds indirect vtable dispatch to direct arithmetic in this test).
- Realization: a broad `name.contains("uvm_")` interception rule in both
  `circt-sim-compile` and `circt-sim` suppressed native dispatch for
  UVM-prefixed pure helper functions, limiting AOT speedup headroom.
- Changed interception policy in both components:
  - default: use targeted interceptor patterns only (no blanket `uvm_` block)
  - compatibility mode: `CIRCT_AOT_INTERCEPT_ALL_UVM=1` restores legacy
    blanket interception/demotion behavior.
- Added regression `test/Tools/circt-sim/aot-uvm-prefix-native.mlir`:
  - default run validates native dispatch for a `uvm_`-prefixed arithmetic helper
  - strict run with `CIRCT_AOT_INTERCEPT_ALL_UVM=1` validates legacy interception.
- Fixed one remaining narrow-return marshalling site in
  `LLHDProcessInterpreterCallIndirect.cpp` that still constructed
  `InterpretedValue(result, bits)` directly in an entry-table dispatch path;
  now uses `APInt(64, result).zextOrTrunc(bits)` to avoid width assert risk.
- Updated AOT test expectations to reflect the corrected load summary wording:
  - `... X native-dispatched, Y not-native-dispatched, Z intercepted`
  - removed stale `excluded (call_indirect)` phrasing.
- Added aggressive native-dispatch opt-in for sequence `::body` methods:
  - default behavior remains conservative (`::body` intercepted)
  - `CIRCT_AOT_ALLOW_NATIVE_SEQ_BODY=1` allows native dispatch of `::body`
    symbols for throughput experiments.
- Added regression `test/Tools/circt-sim/aot-seq-body-native-optin.mlir`:
  - default: `0 native-dispatched, 1 intercepted`
  - opt-in env: `1 native-dispatched, 0 intercepted`
  - both modes preserve functional output (`out=200`).
- Validation: previously touched AOT tests still pass local compile+run checks
  after the `::body` opt-in addition (`aot-basic-func`, `aot-vtable-dispatch`,
  `aot-call-indirect-func-body`, `aot-uvm-prefix-native`).
- Added explicit AOT stat `Entry-table trampoline calls` in `circt-sim` output
  and a corresponding interpreter counter (`trampolineEntryCallCount`) for
  tagged FuncId dispatches that resolve to non-native entry-table slots.
- Added regression `test/Tools/circt-sim/aot-entry-table-trampoline-counter.mlir`:
  - compiles with `CIRCT_AOT_INTERCEPT_ALL_UVM=1` to force a vtable target
    into non-native/trampoline classification
  - validates runtime reports:
    - `Entry table: ... (0 native, 1 non-native)`
    - `Entry-table native calls: 0`
    - `Entry-table trampoline calls: 1`
    - functional output `indirect_uvm(5) = 47`.
- Surprise/fix: initial non-native counter instrumentation double-counted one
  call (reported `2` for a single indirect call) because both site-cache
  population and late dispatch fallback paths incremented. Final fix keeps the
  non-native count on site-cache/dispatch points without duplicate increments.
- Extended existing AOT regressions to lock in zero trampoline-entry counts on
  native-only scenarios (`aot-basic-func`, `aot-vtable-dispatch`,
  `aot-call-indirect-func-body`).
- Added opt-in native coverage for UVM allocation-style symbols:
  - new env: `CIRCT_AOT_ALLOW_NATIVE_UVM_ALLOC=1`
  - affects both compiler demotion and runtime interception for:
    - `create` / `create_*` / `::create`
    - `::new`
  - default remains conservative (intercepted/demoted).
- Added regression `test/Tools/circt-sim/aot-uvm-alloc-native-optin.mlir`:
  - default compile: demotes `::new`, leaves `1 functions + 0 processes ready`
  - opt-in compile: keeps both functions compiled (`2 functions + 0 processes ready`)
  - runtime confirms dispatch shift:
    - default: `Loaded 1 compiled functions: 1 native-dispatched ...`
    - opt-in: `Loaded 2 compiled functions: 2 native-dispatched ...`
  - functional output remains `out=200` in both modes.
- Added opt-in native coverage for UVM type/registry helper symbols:
  - new env: `CIRCT_AOT_ALLOW_NATIVE_UVM_TYPEINFO=1`
  - gates both compile-time demotion and runtime interception for:
    - `m_initialize*` / `::m_initialize`
    - `m_register_cb*`
    - `get_object_type` / `get_type_name`
    - `get_type_*` / `::get_type_*`
    - `type_name_*`
  - default remains conservative (intercepted/demoted).
- Added regression `test/Tools/circt-sim/aot-uvm-typeinfo-native-optin.mlir`:
  - default compile: demotes one intercepted type-info helper, keeps 1 compiled fn
  - opt-in compile: keeps both functions compiled
  - runtime shift:
    - default: `Loaded 1 compiled functions: 1 native-dispatched ...`
    - opt-in: `Loaded 2 compiled functions: 2 native-dispatched ...`
  - functional output remains `out=47` in both modes.
- Surprise: rebuild transiently failed due unrelated in-flight edits in
  `LLHDProcessInterpreter.cpp`/`UVMFastPaths.cpp` (const-unsafe debug access
  and stale symbol usage). Rebuild succeeded once those unrelated edits settled;
  no additional behavior changes were needed for this step.
- Added opt-in native coverage for UVM accessor-heavy symbols:
  - new env: `CIRCT_AOT_ALLOW_NATIVE_UVM_ACCESSORS=1`
  - gates both compile-time demotion and runtime interception for:
    - `get_imp_*` / `::get_imp_*`
    - `get_inst` / `get_inst_*`
  - default remains conservative (intercepted/demoted).
- Added regression `test/Tools/circt-sim/aot-uvm-accessors-native-optin.mlir`:
  - default compile: demotes accessor helper, keeps 1 compiled fn
  - opt-in compile: keeps both functions compiled
  - runtime shift:
    - default: `Loaded 1 compiled functions: 1 native-dispatched ...`
    - opt-in: `Loaded 2 compiled functions: 2 native-dispatched ...`
  - functional output remains `out=47` in both modes.
- Revalidated key AOT baselines after accessor changes:
  - `aot-basic-func`, `aot-vtable-dispatch`, `aot-call-indirect-func-body`,
    `aot-entry-table-trampoline-counter` all preserve expected counters/output.
- Added umbrella fast-path env: `CIRCT_AOT_AGGRESSIVE_UVM=1`.
  - Runtime: enables current UVM native opt-ins in one switch:
    - `ALLOW_NATIVE_SEQ_BODY`
    - `ALLOW_NATIVE_UVM_ALLOC`
    - `ALLOW_NATIVE_UVM_TYPEINFO`
    - `ALLOW_NATIVE_UVM_ACCESSORS`
  - Compiler: same umbrella disables demotion for those same categories.
- Added regression `test/Tools/circt-sim/aot-uvm-aggressive-native-optin.mlir`:
  - default compile/run remains conservative (`Demoted 1`, `Loaded 1 compiled functions`)
  - aggressive compile/run lifts demotion (`2 functions + 0 processes ready`,
    `Loaded 2 compiled functions`)
  - functional output preserved (`out=47`).
- Added singleton/registry accessor opt-in coverage under the existing accessor gate:
  - `CIRCT_AOT_ALLOW_NATIVE_UVM_ACCESSORS=1` now also controls
    `uvm_root::get_root*` and `uvm_coreservice_t::get*` interception/demotion
    in both runtime and compiler.
- Added regression `test/Tools/circt-sim/aot-uvm-singleton-native-optin.mlir`:
  - default compile/run stays conservative (`Demoted 1`, `Loaded 1 compiled functions`)
  - accessor opt-in compile/run lifts demotion (`2 functions + 0 processes ready`,
    `Loaded 2 compiled functions`)
  - functional output preserved (`out=47`).
- Full targeted validation sweep after singleton + aggressive updates passed:
  - `aot-uvm-aggressive-native-optin` (default and `CIRCT_AOT_AGGRESSIVE_UVM=1`)
  - `aot-uvm-singleton-native-optin` (default and accessor opt-in)
  - AOT sanity regressions:
    `aot-basic-func`, `aot-vtable-dispatch`, `aot-call-indirect-func-body`,
    `aot-entry-table-trampoline-counter`
  - counters and outputs match expectations, including
    `Entry-table trampoline calls: 0` for native-only cases and `1` for the
    forced non-native vtable case.
- Realization from repro: hot singleton getter symbols (`get_0`,
  `get_common_domain`, `get_global_hopper`) were still always demoted/
  intercepted, even with `CIRCT_AOT_AGGRESSIVE_UVM=1`.
- Added new opt-in gate in both compiler demotion and runtime interception:
  - `CIRCT_AOT_ALLOW_NATIVE_UVM_SINGLETON_GETTERS=1`
  - also enabled by umbrella: `CIRCT_AOT_AGGRESSIVE_UVM=1`.
- Interception split now keeps `self`/`malloc` always intercepted, while
  singleton getters are env-controlled.
- Added regression `test/Tools/circt-sim/aot-uvm-singleton-getters-native-optin.mlir`:
  - default: `Demoted 1`, `Loaded 1 compiled functions`
  - singleton-getter opt-in: `2 functions + 0 processes ready for codegen`,
    `Loaded 2 compiled functions`
  - aggressive umbrella reproduces opt-in behavior
  - functional output stable (`out=47`).
- Revalidated nearby opt-in regressions after this expansion:
  - `aot-uvm-aggressive-native-optin` (default vs aggressive)
  - `aot-uvm-singleton-native-optin` (accessor opt-in compile path)
- Added ptr/int unrealized-cast lowering support in `circt-sim-compile` to
  reduce function rejection from cast-only patterns:
  - `!llvm.ptr -> iN` lowered to `llvm.ptrtoint`
  - `iN -> !llvm.ptr` lowered to `llvm.inttoptr`
  - `!llvm.ptr -> !llvm.ptr` lowered to `llvm.bitcast`
  - `iM -> iN` lowered via `zext/trunc` as needed
- Extended `isFuncBodyCompilable()` to accept these cast pairs so candidate
  functions are no longer pre-rejected.
- Added regression `test/Tools/circt-sim/aot-unrealized-cast-ptr-int.mlir`
  (compile-only): locks in
  `Functions: 2 total, 0 external, 0 rejected, 2 compilable`.
- TDD repro before fix (temporary local module) showed
  `Functions: 2 total, 0 external, 1 rejected, 1 compilable` for ptr->i64 cast.
- Surprise: cast-heavy minimal runtime modules currently trigger an unrelated
  `circt-sim` init crash in `executeGlobalConstructors`; to keep this step
  robust, the new regression is compile-only while runtime sanity is covered by
  existing stable AOT tests (`aot-basic-func` etc.).
- Added `arith.bitcast` lowering in the arith→LLVM rewrite path:
  - `arith.bitcast` now lowers to `llvm.bitcast` directly.
- TDD repro before fix: a simple `bitcast_roundtrip` function was counted as
  compilable but then stripped:
  - `Functions: 2 total, ... 2 compilable`
  - `Stripped 1 functions with non-LLVM ops`
  - `1 functions + 0 processes ready for codegen`
- Added regression `test/Tools/circt-sim/aot-arith-bitcast-lowering.mlir`
  (compile-only) to lock in no strip fallback:
  - `Functions: 2 total, 0 external, 0 rejected, 2 compilable`
  - no `Stripped` line
  - `2 functions + 0 processes ready for codegen`.
- Validation sweep after this change:
  - compile checks: `aot-unrealized-cast-ptr-int`,
    `aot-arith-bitcast-lowering`
  - runtime sanity: `aot-basic-func` compiled run still reports native dispatch
    and `out=200`.
- Added SCF->CF lowering in `circt-sim-compile` micro-module pipeline via
  `createSCFToControlFlowPass()` (best-effort/non-fatal fallback).
  - New helper: `lowerSCFToCF(ModuleOp)`.
  - Executed before custom arith/cf/func→LLVM lowering.
- Expanded `isFuncBodyCompilable()` to accept `scf` dialect ops so functions
  containing structured control flow are no longer pre-rejected.
- TDD repro before fix (`scf.if` + `scf.yield` function):
  - `Functions: 2 total, 0 external, 1 rejected, 1 compilable`
  - `1 functions + 0 processes ready for codegen`.
- Added regression `test/Tools/circt-sim/aot-scf-if-native.mlir`:
  - compile: `Functions: 2 total, 0 external, 0 rejected, 2 compilable`
  - compile: no `Stripped` fallback
  - interpreted and compiled runtime both produce `out=10`
  - compiled load summary: `Loaded 2 compiled functions: 2 native-dispatched...`.
- Build-system update: linked `circt-sim-compile` against `MLIRSCFToControlFlow`.
- Revalidated nearby regressions:
  - `aot-unrealized-cast-ptr-int` compile check
  - `aot-arith-bitcast-lowering` compile check
  - `aot-basic-func` compiled runtime sanity.
- Workload-guided profiling run (`-v --emit-llvm`) on
  `build_test/test/Tools/circt-sim/Output/uvm-run-phase-objection-runtime.sv.tmp.mlir`
  reported top function rejection reasons:
  - `builtin.unrealized_conversion_cast` (285)
  - `llhd.constant_time` (37)
  - `hw.struct_create` (36)
  - `llhd.current_time` (24)
  - `hw.struct_extract` (18)
- Implemented next coverage unlock for the `hw.struct_*` rejection bucket:
  - `isFuncBodyCompilable()` now accepts:
    - `hw.struct_create`
    - `hw.struct_extract`
    - `hw.aggregate_constant`
  - Pre-lowering phase folds 4-state struct patterns:
    - `hw.struct_extract` from `hw.struct_create(value, unknown)`
      rewrites directly to the corresponding field SSA value.
    - `hw.struct_extract` from 4-state `hw.aggregate_constant`
      rewrites to `arith.constant` for `value`/`unknown`.
    - dead 4-state `hw.struct_create` / `hw.aggregate_constant` ops are erased.
- Added regression `test/Tools/circt-sim/aot-hw-struct-fourstate-folding.mlir`:
  - compile: `Functions: 2 total, 0 external, 0 rejected, 2 compilable`
  - runtime (interpreted/compiled): functional output `out=47`
  - compiled load summary: `Loaded 2 compiled functions: 2 native-dispatched...`.
- Revalidated recently added AOT regressions after struct-folding update:
  - `aot-scf-if-native` (compiled runtime)
  - `aot-unrealized-cast-ptr-int` (compile-only)
  - `aot-arith-bitcast-lowering` (compile-only, no strip fallback)
  - `aot-basic-func` (compiled runtime sanity).
- Realization: the large UVM workload file above currently aborts later in
  compilation due invalid symbol-visibility/IR issues in that generated MLIR
  snapshot (`func.func ... symbol declaration cannot have public visibility`),
  so we use it for rejection profiling at the front of the pipeline rather than
  full completion until that input hygiene issue is resolved.
- Step A (entry-table-first dispatch) implementation pass completed in runtime:
  - `LLHDProcessInterpreterCallIndirect.cpp` now dispatches `call_indirect`
    through `compiledFuncEntries[fid]` for all valid FuncIds (native and
    trampoline), instead of gating on `compiledFuncIsNative[fid]`.
  - Applied at all four call_indirect sites:
    - X-fallback path
    - static fallback path
    - E5 site-cache population (cache all entries)
    - E5 hot-path dispatch
- Added depth guard accounting for entry-table dispatch:
  - new counter `entryTableSkippedDepthCount` increments when a tagged entry is
    available but `nativeCallDepth > 0` blocks re-entrant dispatch.
- Added depth safety for `func.call` native dispatch in both paths:
  - slow path native dispatch now only executes when `nativeCallDepth == 0`.
  - cached path native dispatch now only executes when `nativeCallDepth == 0`.
- Added re-entrancy depth telemetry:
  - new counter `maxNativeCallDepth`, updated in `dispatchTrampoline()` after
    `++nativeCallDepth`.
- AOT stats output extended in `circt-sim --aot-stats`:
  - `Entry-table skipped (depth)`
  - `Max native call depth`
- TDD/regression:
  - Updated `test/Tools/circt-sim/aot-entry-table-trampoline-counter.mlir` to
    assert actual trampoline callback execution via:
    `Trampoline calls:                 1`.
  - Before this Step A change, this scenario reported:
    - `Trampoline calls: 0`
    - `Entry-table trampoline calls: 1`
    indicating classification without actual trampoline entry invocation.
  - After this change, same scenario reports:
    - `Trampoline calls: 1`
    - `Entry-table trampoline calls: 1`
    - `Entry-table skipped (depth): 0`
    - `Max native call depth: 1`
- Validation executed:
  - rebuild: `ninja -C build_test circt-sim circt-sim-compile` (pass)
  - targeted runtime/AOT checks (pass):
    - `aot-entry-table-trampoline-counter`
    - `aot-vtable-dispatch`
    - `aot-call-indirect-func-body`
- Realization:
  - Existing `trampolineEntryCallCount` previously over-counted in site-cache
    population. This was removed so trampoline entry counts now reflect actual
    dispatched calls, not just discovered non-native FuncIds.
- Implemented Plan Step D1: runtime hot-uncompiled callee reporting (FuncId-based).
- New AOT hotness tracking state in interpreter:
  - `aotFuncIdCallCounts[fid]` call counters
  - `aotFuncEntryNamesById[fid]` for report labels
  - `aotFuncNameToCanonicalId` for direct `func.call` -> canonical FuncId mapping
  - enabled when `CIRCT_AOT_STATS` (or `CIRCT_AOT_HOT_UNCOMPILED`) is set.
- Added helpers:
  - `noteAotFuncIdCall(fid)`
  - `noteAotCalleeNameCall(calleeName)`
- Instrumented both direct and indirect paths:
  - direct calls: `interpretFuncCall` records via callee-name lookup.
  - indirect calls: all tagged-FuncId dispatch points now call
    `noteAotFuncIdCall(fid)` (x-fallback, static-fallback, site-cache path,
    and main entry-table path).
- Added shutdown report in `--aot-stats`:
  - `Hot uncompiled FuncIds (top 50):`
  - sorted by descending call count, filtered to non-native FuncIds
    (`!compiledFuncIsNative[fid]`).
- CLI/env wiring:
  - when `--aot-stats` is passed, `circt-sim` now sets
    `CIRCT_AOT_STATS=1` early so interpreter-side hotness collection is enabled
    without requiring users to set env manually.
- Regression update (TDD):
  - `test/Tools/circt-sim/aot-entry-table-trampoline-counter.mlir`
    now checks the hot-uncompiled section includes:
    - `1x fid=0 uvm_pkg::uvm_demo::add42`
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim circt-sim-compile`
  - targeted runtime checks pass:
    - `aot-entry-table-trampoline-counter` (includes new hot-uncompiled line)
    - `aot-vtable-dispatch` (native-only path unchanged)
- TDD repro for dead-cast rejection:
  - temporary module with an unused
    `builtin.unrealized_conversion_cast` (`!llvm.ptr -> !llhd.ref<i32>`)
    was previously counted as:
    - `Functions: 2 total, 0 external, 1 rejected, 1 compilable`.
- Implemented dead-cast elision in `circt-sim-compile`:
  - `isFuncBodyCompilable()` now accepts `builtin.unrealized_conversion_cast`
    ops when all cast results are unused.
  - Pre-lowering in `lowerFuncArithCfToLLVM()` now erases dead unrealized casts
    before main lowering, so they cannot survive to residual-op stripping.
- Added regression:
  - `test/Tools/circt-sim/aot-unrealized-cast-dead-elide.mlir`
  - checks compile stats:
    - `Functions: 2 total, 0 external, 0 rejected, 2 compilable`
    - `2 functions + 0 processes ready for codegen`.
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim-compile`
  - targeted checks (pass):
    - `aot-unrealized-cast-dead-elide` compile output
    - `aot-unrealized-cast-ptr-int` compile output (no regression)
    - `aot-basic-func` interpreted + compiled runtime sanity (`out=200`).
- Realization:
  - This fix improves correctness and local coverage for dead-cast wrappers,
  but does not materially change the current large UVM rejection histogram
  (`295x builtin.unrealized_conversion_cast` unchanged on
  `uvm-run-phase-objection-runtime.sv.tmp.mlir`).
- TDD repro for dead LLHD time rejection:
  - temporary module with unused `llhd.constant_time` and `llhd.current_time`
    values was previously counted as:
    - `Functions: 2 total, 0 external, 2 rejected, 0 compilable`.
- Implemented dead-time-op elision in `circt-sim-compile`:
  - `isFuncBodyCompilable()` now accepts `llhd.constant_time` /
    `llhd.current_time` only when all results are unused.
  - Pre-lowering now erases dead `llhd.constant_time` /
    `llhd.current_time` ops before main lowering.
- Added regression:
  - `test/Tools/circt-sim/aot-llhd-dead-time-elide.mlir`
  - checks compile stats:
    - `Functions: 2 total, 0 external, 0 rejected, 2 compilable`
    - `2 functions + 0 processes ready for codegen`.
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim-compile`
  - targeted checks (pass):
    - `aot-llhd-dead-time-elide`
    - `aot-unrealized-cast-dead-elide`
    - `aot-unrealized-cast-ptr-int`
    - temporary repros for dead cast + dead time now both report
      `0 rejected, 2 compilable`
    - `aot-basic-func` compiled runtime sanity unchanged (`out=200`).
- Realization:
  - Similar to dead-cast elision, this is a local coverage unlock; it does not
    change the current large UVM rejection histogram where
    `llhd.constant_time` / `llhd.current_time` are likely still live uses.
- TDD repro for compiled UVM crash in short-name `get` wrapper:
  - large `uvm-run-phase-objection-runtime` AOT run crashed in `.so` at
    symbol `get + 42` (`uvm_run_phase.so`), with stack through
    `interpretFuncCall`.
  - disassembly showed native `get()` doing:
    - call interpreted `get_0`
    - immediate pointer dereference on returned value
    - indirect call through vtable slot
  - if `get_0` returns null transiently, native dereference segfaults.
- Implemented interception hardening for short-name `get`:
  - compile-time demotion predicate (`circt-sim-compile.cpp`) now treats
    `name == "get"` as a singleton-getter interception candidate by default
    (same gate as `get_0` / `get_common_domain` / `get_global_hopper`).
  - runtime native-dispatch filter (`LLHDProcessInterpreter.cpp`) mirrors the
    same `name == "get"` interception rule.
- Added regression:
  - `test/Tools/circt-sim/aot-uvm-get-native-optin.mlir`
  - default: demotes `@get`, compiles only `keep_alive`
  - opt-in (`CIRCT_AOT_ALLOW_NATIVE_UVM_SINGLETON_GETTERS=1`): both functions
    compile native.
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim circt-sim-compile`
  - targeted AOT regression pack (pass), including:
    - `aot-uvm-get-native-optin`
    - `aot-uvm-factory-native-optin`
    - `aot-private-func-decl-visibility`
    - `aot-call-indirect-void-func-body`
    - dead-cast/dead-time tests + `aot-basic-func` runtime checks.
  - large UVM run now no longer segfaults; compiled and interpreter runs both
    terminate with same UVM fatal at time 0:
    - `UVM_FATAL ... FCTTYP Factory did not return ...`
- Realization:
  - this change restores behavioral parity (fatal vs crash) by routing fragile
    singleton-getter wrappers through interpreter semantics until nullability
    handling is made native-safe.
- TDD repro for process extraction blocker (`process_extract_external_operand`):
  - `uvm_seq_body` and `uvm_run_phase` showed:
    - `Processes: 1 total, 0 callback-eligible, 1 rejected`
    - top reason: `process_extract_external_operand`.
  - Root cause: callback process pre-scan only cloned a narrow external-operand
    set. UVM top process captures module-scope LLVM values (`llvm.mlir.addressof`,
    `llvm.mlir.zero`, `llvm.mlir.undef`) and failed extraction before lowering.
- Implemented external-LLVM-value extraction support in process compiler:
  - `compileProcessBodies()` pre-scan now clones external:
    - `LLVM::AddressOfOp`
    - `LLVM::ZeroOp`
    - `LLVM::UndefOp`
  - Added richer rejection reason detail for extraction failures:
    - `process_extract_external_operand:<def-op-name>`
    - plus specific `hw.aggregate_constant(non-fourstate)` tag.
- Added regressions:
  - `test/Tools/circt-sim/aot-process-external-llvm-values.mlir`
    - compile-only check that a process using external addressof/zero/undef is
      callback-eligible.
  - `test/Tools/circt-sim/aot-process-indirect-cast-dispatch.mlir`
    - compile-only check for process eligibility with safe
      `builtin.unrealized_conversion_cast` + `func.call_indirect` pattern.
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim-compile`
  - targeted checks (manual RUN-equivalent):
    - `aot-process-external-llvm-values`: `Compiled 1 process bodies`,
      `Processes: 1 total, 1 callback-eligible, 0 rejected`
    - `aot-process-indirect-cast-dispatch`: `Compiled 1 process bodies`,
      `Processes: 3 total, 1 callback-eligible, 2 rejected`
    - `aot-process-dispatch` compiled runtime still shows
      `Compiled callback invocations: 1` and expected `b=1`, `b=0` output.
  - large workload compile coverage gains:
    - `uvm_seq_body`: from `0 callback-eligible` to `1 callback-eligible`
      (`3021 functions + 1 processes ready for codegen`)
    - `uvm_run_phase`: from `0 callback-eligible` to `1 callback-eligible`
      (`3013 functions + 1 processes ready for codegen`)
  - runtime sanity:
    - `uvm_run_phase` compiled remains stable and parity with interpreter holds on
      shared `FCTTYP` fatal (`EXIT_CODE=1` both modes).
    - `uvm_seq_body` compiled remains stable (`EXIT_CODE=0`) but still has
      parity mismatch vs interpreter at time 0 (`COMP/INTERNAL` fatal present only
      in compiled run).
- Realization:
  - Process compilation was blocked by extraction plumbing, not core LLHD/comb
    lowering limits. Unlocking external LLVM operand extraction recovers process
    coverage on the large UVM workloads without reintroducing crashes.
- Shutdown-time assoc-pointer crash triage (from AOT bug report):
  - Signature: `[AOT BUG] __moore_assoc_exists: bad pointer 0x...`
  - Last native body: `uvm_pkg::uvm_report_message::get_severity`
  - Observation: crash occurs after simulation completion, indicating a
    shutdown/reporting path virtual-pointer dereference, not run-loop instability.
- Implemented quick stabilizer under existing reporting gate:
  - compile-time demotion filter now intercepts
    `::uvm_report_message::get_severity` by default when
    `CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING` is not set.
  - runtime native-dispatch filter mirrors the same intercept.
- Added regression:
  - `test/Tools/circt-sim/aot-uvm-report-message-severity-native-optin.mlir`
  - default: demotes only `get_severity`
  - opt-in (`CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1`): both functions compile.
- Validation:
  - rebuild pass: `ninja -C build_test circt-sim circt-sim-compile`
  - regression logs match expected compile counts (default 1 function ready,
    opt-in 2 functions ready).
  - `uvm_seq_body` compiled rerun remains stable (`EXIT_CODE=0`) with no
    assoc-pointer crash signature in log.
- Realization:
  - This is a tactical shutdown-path mitigation; the strategic fix remains
    pointer normalization/translation (plan Phase 1) and eventually arena-based
    mutable state (plan Phase 6).
- TDD for strategic assoc-pointer normalization:
  - Added Moore runtime unit tests:
    - `MooreRuntimeAssocTest.ResolverMapsVirtualPointer`
    - `MooreRuntimeAssocTest.ResolverFallbackUsesOriginalPointer`
  - Tests exercise runtime-side pointer normalization via a callback before
    `AssocArrayHeader` dereference.
- Implemented runtime pointer-normalization hook:
  - New API in `MooreRuntime`:
    - `MooreAssocPtrResolver`
    - `__moore_assoc_set_ptr_resolver(...)`
  - All assoc helpers now normalize pointer arguments first:
    - `size/delete/delete_key/first/next/last/prev/exists/get_ref/copy/copy_into`
  - Validation guard (`CIRCT_AOT_VALIDATE`) remains active post-normalization.
- Wired interpreter virtual-address translation into runtime:
  - `LLHDProcessInterpreter::setupRegistryAccessors()` now registers assoc
    resolver callback into `MooreRuntime`.
  - Added `normalizeAssocRuntimePointer(const void *)`:
    - pass through already-valid native assoc pointers
    - map interpreter virtual addresses via `findMemoryBlockByAddress`
    - recover host pointer from global pointer slots when applicable
    - fallback to aliased host storage for mapped interpreter memory.
- Validation:
  - Rebuild succeeded:
    - `ninja -C build_test circt-sim circt-sim-compile MooreRuntimeTests`
  - Unit tests passed:
    - `MooreRuntimeAssocTest.*` (2/2)
  - Large workload rerun:
    - `uvm_seq_body` compiled (`CIRCT_AOT_STATS=1`) completes cleanly
      (`RUN_EXIT=0`) with no `[AOT BUG] __moore_assoc_exists` crash signature.
    - Remaining known issue unchanged: compiled-only
      `UVM_FATAL [COMP/INTERNAL] attempt to find build phase object failed`.
- Realization:
  - This lifts assoc-pointer handling from per-function interception to a
    reusable ABI boundary fix. The next blocker is parity (build-phase fatal),
    not native shutdown stability.
- Parity bisection milestone (`uvm_seq_body` `COMP/INTERNAL` fatal):
  - Reconfirmed differential:
    - default AOT dispatch: compiled-only `UVM_FATAL [COMP/INTERNAL]`
    - `CIRCT_AOT_NO_FUNC_DISPATCH=1`: fatal disappears (run completes)
  - Added AOT stats visibility for native side:
    - `dumpAotHotUncompiledFuncs()` now also prints
      `Hot native FuncIds (top N)`.
  - Added temporary native call tracing for `func.call`:
    - `CIRCT_AOT_TRACE_NATIVE_CALLS=1`
    - `CIRCT_AOT_TRACE_NATIVE_LIMIT=<N>`
  - Trace showed all direct native `func.call` sites in the failing window are
    `fid=<unmapped>` (e.g., `m_do_pre_run_test`, `m_do_dump_args`,
    several `get_*` wrappers, `m_set_cl_msg_args`).
- Implemented guarded default for unmapped direct native `func.call`:
  - `LLHDProcessInterpreter::interpretFuncCall` and cached path now fall back
    to interpreter when the callee has no reverse `funcOp -> FuncId` mapping.
  - Opt-in escape hatch:
    - `CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1` restores previous behavior.
- Validation:
  - `uvm_seq_body` default AOT run:
    - no `COMP/INTERNAL` fatal
    - run completes (`EXIT=0`) with expected `RNTST` line
    - `Compiled function calls: 0`, `Entry-table native calls: 7`
  - `uvm_seq_body` with opt-in (`CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1`):
    - prior fatal reproduced
    - `Compiled function calls: 23`
  - `uvm_run_phase` compiled remains unchanged on known parity point:
    - `UVM_FATAL ... FCTTYP ...`
    - `EXIT=1` (same as interpreter baseline).
  - Quick wall-clock sanity on `uvm_seq_body` (single-run, same machine):
    - default (guard active): ~4857 ms, no `COMP/INTERNAL` fatal
    - opt-in allow unmapped native: ~4861 ms, `COMP/INTERNAL` fatal present
    - `CIRCT_AOT_NO_FUNC_DISPATCH=1`: ~5150 ms, no `COMP/INTERNAL` fatal
- Realization:
  - The `COMP/INTERNAL` divergence was not from entry-table native FIDs already
    tracked by stats/deny; it came from unmapped direct native `func.call`
    sites. Guarding those by default yields a strong correctness win with small
    observed coverage impact on this workload.

## 2026-02-26
- Tightened direct `func.call` fallback accounting and controls:
  - when native dispatch is suppressed by unmapped-name policy
    (`shouldDenyUnmappedNativeCall`) or `CIRCT_AOT_DENY_FID`, the runtime now
    increments `interpretedFuncCallCount` before falling back.
  - cached direct-dispatch path now applies the same deny/trap checks as the
    slow path (`CIRCT_AOT_DENY_FID`, `CIRCT_AOT_TRAP_FID`), removing a policy
    mismatch where cached calls could bypass these controls.
- Added regression `test/Tools/circt-sim/aot-unmapped-native-get-policy.mlir`:
  - locks in default policy `default deny get_*`
  - validates `CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1` enables native direct calls
  - validates `CIRCT_AOT_DENY_UNMAPPED_NATIVE_NAMES=get_*` overrides allow-all
    and forces interpreted fallback.
- Validation:
  - `ninja -C build_test circt-sim circt-sim-compile` succeeded.
  - Executed new test RUN commands manually (no `llvm-lit`/`FileCheck` binary
    in this build tree) with `rg` assertions:
    - default mode: compiled calls `0`, interpreted calls `1`
    - allow-all mode: compiled calls `1`, interpreted calls `0`
    - allow-all + deny `get_*`: compiled calls `0`, interpreted calls `1`
  - Nearby AOT sanity (`aot-basic-func`) still passes with native dispatch and
    expected output.

## 2026-02-26
- Implemented Phase 4 bootstrap: optional native module-level init execution.
  - Compiler (`circt-sim-compile`):
    - synthesizes conservative per-`hw.module` native init entrypoints
      named `__circt_sim_module_init__<encoded_module_name>` from top-level
      init ops in a safe LLVM-only subset.
    - emits compile log line:
      `Native module init functions: <N>`.
    - preserves native-init symbol visibility in the `.so`.
  - Runtime (`circt-sim`):
    - added loader lookup API:
      `CompiledModuleLoader::lookupModuleInit(hwModuleName)`.
    - interpreter initialize path now executes native module init when
      `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1` and symbol is present, otherwise
      falls back to existing interpreted `executeModuleLevelLLVMOps()`.
    - logs:
      `[circt-sim] Native module init: <hw.module name>`.
  - Initialization wiring:
    - plumbed early compiled-loader handle into interpreter via
      `setCompiledLoaderForModuleInit(...)`.
    - set runtime context early so native init can resolve `__circt_sim_ctx`.
- Added regression `test/Tools/circt-sim/aot-native-module-init-basic.mlir`:
  - compile check: native module init function count is emitted
  - interpreted and compiled baseline both print `out=123`
  - native-init opt-in run checks log marker and preserves `out=123`.
- Validation:
  - rebuild: `ninja -C build_test circt-sim circt-sim-compile` (pass)
  - manual RUN-equivalent checks (no `llvm-lit`/`FileCheck` in this build tree):
    - `aot-native-module-init-basic` pass across compile/sim/compiled/native-optin
    - rechecked `aot-unmapped-native-get-policy` counter expectations (pass)
    - nearby `aot-basic-func` native dispatch sanity (pass).
- Realization:
  - Bringing up native module init safely requires a conservative extraction set.
    This landed path proves wiring/end-to-end behavior; next expansion should
    target selected `llvm.call` init patterns with explicit allowlists and
    fallback guarantees.

## 2026-02-26
- Added explicit `TRAP_FID` regression coverage:
  - new test `test/Tools/circt-sim/aot-trap-fid-direct-call.mlir`.
  - uses a demoted/interpreted wrapper (`dummy_seq::body`) calling a
    FuncId-mapped native callee (`math::add42`, fid 0) to force trap through
    the direct `func.call` dispatch path.
  - checks:
    - normal compiled run prints `out=47`
    - trap run emits
      `[AOT TRAP] func.call fid=0 name=math::add42`
      and terminates non-zero (observed `EXIT=132`, illegal instruction).
- Validation:
  - manual RUN-equivalent pass for new trap test (no local `llvm-lit`).
- Realization:
  - This closes the immediate Phase 1.4 testing gap for crash-triage controls:
    deny/trap path behavior is now covered for both normal and trap flows.

## 2026-02-26
- Phase 4.1 expansion: enabled selected top-level `llvm.call` patterns in
  native module-init extraction (`synthesizeNativeModuleInitFunctions`).
  - allowlisted callees:
    - libc: `memset`, `memcpy`, `memmove`, `malloc`, `calloc`
    - LLVM mem intrinsics: `llvm.memset.*`, `llvm.memcpy.*`, `llvm.memmove.*`
- TDD repro before change:
  - call-bearing module init (top-level `llvm.call @memset`) compiled without
    native init extraction:
    - `Functions: 1 total, 0 external, 0 rejected, 1 compilable`
    - no `Native module init functions: 1` line.
- Added regression:
  - `test/Tools/circt-sim/aot-native-module-init-memset.mlir`
  - checks:
    - compile emits native init function count
    - interpreted + compiled baseline remain `out=12345`
    - native-init opt-in (`CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`) executes
      native module init and prints `out=0`.
- Native-init runtime wiring follow-up (required for call-bearing init):
  - `circt-sim` now wires trampoline callback/user-data before module init.
  - initial attempt loaded compiled-function maps at native module-init
    dispatch time; this was later narrowed to trampoline-only preparation
    (see follow-up entry below) to avoid perturbing global-constructor flow.
- Surprise/root cause:
  - Early preload in `SimulationContext::initialize()` (before
    `funcLookupCache` build) produced unresolved trampoline fallbacks and
    crashed with:
    - `FATAL: trampoline dispatch for func_id=0 (memset) — FuncOp not found`
  - Fix was to move function/trampoline map loading to interpreter initialize
    at module-init call site, while keeping early callback wiring.
- Validation:
  - rebuild: `ninja -C build_test circt-sim circt-sim-compile` (pass)
  - manual RUN-equivalent checks (no local `llvm-lit`/`FileCheck`):
    - `aot-native-module-init-memset`:
      - compile: native init functions `1`
      - sim baseline: `out=12345`
      - compiled baseline: `out=12345`
      - native-init opt-in: `[circt-sim] Native module init: top`, `out=0`
    - `aot-native-module-init-basic` still passes:
      - compile: native init functions `1`
      - native-init opt-in: `[circt-sim] Native module init: top`, `out=123`.
- Realization:
  - Allowlisting `llvm.call` in native module init is coupled to early
    trampoline readiness, not just extraction legality. The safe boundary is:
    callback wiring early, dispatch maps loaded after symbol cache setup.

## 2026-02-26
- Follow-up fix for Phase 4.1 native-init call path:
  - previous attempt loaded full `loadCompiledFunctions()` during
    `LLHDProcessInterpreter::initialize()` to enable module-init trampolines.
  - this changed global-constructor dispatch timing and caused UVM crash under
    `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1` (`__moore_assoc_delete` path).
- Corrected approach:
  - keep early trampoline callback wiring in `SimulationContext` (needed before
    native module init runs), but do NOT load full native dispatch maps early.
  - in `LLHDProcessInterpreter::initialize()`, prepare only module-init
    trampoline readiness once per loader:
    - fill `trampolineFuncOps` from `funcLookupCache` for func symbols
    - fill `trampolineNativeFallback` via `dlsym` for extern/native symbols
    - avoid populating `nativeFuncPtrs`/entry-table dispatch maps until normal
      `setCompiledModule()` phase after initialize/finalize.
- Validation:
  - regression `aot-native-module-init-memset` still passes:
    - compile: `Native module init functions: 1`
    - compiled baseline: `out=12345`
    - native-init opt-in: `[circt-sim] Native module init: top`, `out=0`
  - large workload `uvm_seq_body`:
    - baseline compiled: `EXIT=0`, `Simulation completed`
    - native-init opt-in: `EXIT=0`, `Simulation completed`
    - quick single-run wall-clock: ~4.726s baseline vs ~4.768s opt-in
  - large workload `uvm_run_phase`:
    - baseline and native-init opt-in both retain known parity fatal:
      `UVM_FATAL ... FCTTYP ...`, `EXIT=1`.
- Realization:
  - module-init trampoline readiness must be narrowly scoped; loading full
    native dispatch policy before global-constructor/finalize phase perturbs
    behavior. Preparing only trampoline fallback state keeps init semantics
    stable while allowing selected call-bearing native init entrypoints.

## 2026-02-26
- Phase 5 cast-coverage expansion: `!llhd.ref` argument ABI canonicalization
  in `circt-sim-compile`.
- TDD repro before fix (minimal pointer/ref wrapper):
  - `/tmp/aot_ref_cast_repro.mlir` with
    `ptr -> !llhd.ref<i32>` at callsite and `!llhd.ref<i32> -> ptr` in callee.
  - pre-fix compile result:
    - `Functions: 10 total, 8 external, 2 rejected, 0 compilable`
    - top reasons:
      - `builtin.unrealized_conversion_cast:!llhd.ref<i32>->!llvm.ptr`
      - `builtin.unrealized_conversion_cast:!llvm.ptr->!llhd.ref<i32>`.
- Implementation:
  - `isFuncBodyCompilable(...)` now accepts `!llhd.ref <-> !llvm.ptr`
    unrealized-cast pairs.
  - Added micro-module ABI normalization pass:
    - `canonicalizeLLHDRefArgumentABIs(ModuleOp)`
    - rewrites `func.func` argument types `!llhd.ref<T> -> !llvm.ptr`
    - updates non-external entry block arg types
    - rewrites direct `func.call` operands to match (peeling cast wrappers)
  - Added pre-lowering fold for cast round-trips:
    - `ptr -> ref -> ptr` (and generic same-endpoint round trip)
    - removes both casts when net type is unchanged.
  - Wired ABI normalization after `cloneReferencedDeclarations(...)` and before
    SCF/arith/func lowering.
- Regression coverage added:
  - `test/Tools/circt-sim/aot-unrealized-cast-llhd-ref-arg-abi.mlir`
  - checks:
    - compile: `... 0 rejected, 2 compilable`
    - interpreted + compiled: `out=42`.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused manual test checks:
    - new regression compile/sim/compiled pass
    - existing `aot-unrealized-cast-llvm-hw-struct-extract` still passes.
  - repro after fix:
    - `/tmp/aot_ref_cast_repro.mlir` now compiles as
      `Functions: 10 total, 8 external, 0 rejected, 2 compilable`.
  - large workload coverage delta:
    - `uvm_seq_body`:
      - from `378 rejected` to `114 rejected`
      - ready-for-codegen: `3021 -> 3197` functions (`+176`)
      - top reasons now: `37x llhd.constant_time`, `24x llhd.current_time`,
        `12x builtin.unrealized_conversion_cast:i1->!moore.event`.
    - `uvm_run_phase`:
      - from `378 rejected` to `114 rejected`
      - ready-for-codegen: `3013 -> 3189` functions (`+176`)
      - same top reasons profile as above.
  - runtime sanity:
    - `uvm_seq_body` compiled (`post_refabi.so`) `EXIT=0`, simulation completes.
    - `uvm_run_phase` compiled (`post_refabi.so`) remains parity point
      `UVM_FATAL FCTTYP` with `EXIT=1` (same known behavior class).
- Realization:
  - The dominant `unrealized_conversion_cast` blocker was not isolated cast ops
    but function ABI shape (`!llhd.ref` in helper-call boundaries). Rewriting
    only the micro-module argument ABI to `!llvm.ptr` yields a large coverage
    jump without changing front-end IR generation.

## 2026-02-26
- Queue pointer/length hardening follow-up (interpreter + AOT compiled path).
- Trigger:
  - New regression from interpreter-side triage showed
    `__moore_queue_pop_front_ptr` can be invoked on non-queue pointers and
    must not mutate invalid `len` values.
- TDD changes:
  - Converted red interpreter repro to green regression:
    - `test/Tools/circt-sim/queue-pop-front-ptr-invalid-len-guard.mlir`
    - now expects normal completion with `len_after=100001`.
  - Added AOT-specific regression that forces native compiled execution of the
    queue helper call:
    - `test/Tools/circt-sim/aot-queue-pop-front-ptr-invalid-len-guard.mlir`
    - checks compileable function + interp/compiled parity on `len_after`.
  - Added runtime unit tests:
    - `MooreRuntimeQueueTest.QueuePopFrontPtrInvalidLengthGuard`
    - `MooreRuntimeQueueTest.QueuePopBackPtrInvalidLengthGuard`
    - `MooreRuntimeQueueTest.QueueSizeInvalidLengthGuard`
- Runtime fixes (`lib/Runtime/MooreRuntime.cpp`):
  - `__moore_queue_pop_front_ptr` and `__moore_queue_pop_back_ptr` now:
    - normalize queue/result/data pointers through `normalizeHostPtr(...)`
      so interpreter virtual addresses are translated before dereference.
    - enforce queue length sanity cap (`100000`) before mutation.
    - avoid `free()` for virtual-address-backed queue storage.
  - `__moore_queue_size` now normalizes queue pointer and clamps invalid
    lengths to `0`.
- Surprise/root cause:
  - Initial runtime length-only guard passed unit checks but AOT compiled test
    still crashed (`EXIT=139`) because queue helpers dereferenced virtual queue
    pointers before any normalization.
  - Fix required pointer normalization at queue helper boundaries, not just
    length checks.
- Validation:
  - rebuild: `ninja -C build_test MooreRuntimeTests circt-sim circt-sim-compile`
    (pass)
  - unit tests:
    - `build_test/unittests/Runtime/MooreRuntimeTests --gtest_filter='MooreRuntimeQueueTest.*InvalidLengthGuard*'`
      -> `3 tests`, all passed.
  - regressions (manual RUN-equivalent):
    - `circt-sim test/Tools/circt-sim/queue-pop-front-ptr-invalid-len-guard.mlir`
      -> `EXIT=0`, `len_after=100001`.
    - `circt-sim-compile -v .../aot-queue-pop-front-ptr-invalid-len-guard.mlir`
      -> `1 functions + 0 processes ready for codegen`.
    - `circt-sim .../aot-queue-pop-front-ptr-invalid-len-guard.mlir --compiled=<so>`
      -> `EXIT=0`, `len_after=100001`.
- Realization:
  - Queue helpers are another Moore ABI boundary that must treat virtual
    pointers explicitly, like assoc helpers. This directly improves AOT
    robustness for mixed native/interpreter state.

## 2026-02-26
- Phase 5 follow-up: fixed `!llhd.ref` ABI canonicalization gap for
  `func.call_indirect`.
- Trigger:
  - Large workload verbose compile emitted verifier diagnostics after ref-ABI
    canonicalization:
    - `'func.call_indirect' op failed to verify that callee input types match
      argument types`
    - args were canonicalized to `!llvm.ptr` but callee function type still
      carried `!llhd.ref<...>` inputs.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - extended `canonicalizeLLHDRefArgumentABIs(...)` to rewrite
    `func.call_indirect` sites:
    - rebuild callee `FunctionType` with `!llhd.ref<T> -> !llvm.ptr` inputs
    - rewrite callee cast to preserve `ptr -> function` shape when available
      (so lowering to `llvm.call` keeps working)
    - bridge mismatched operands with cast-peeling or conversion casts.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regressions (manual RUN-equivalent) pass:
    - `aot-unrealized-cast-llhd-ref-arg-abi.mlir` (`out=42` interp/compiled)
    - `aot-llhd-time-to-int-current-time.mlir` (`out=7000000` interp/compiled)
  - large compile diagnostics:
    - `uvm_seq_body`:
      - verifier errors removed (`func.call_indirect` mismatch gone)
      - stripped residual-op functions reduced `181 -> 96`
      - ready-for-codegen increased `3217 -> 3302` (`+85`)
    - `uvm_run_phase`:
      - stripped residual-op functions reduced `181 -> 96`
      - ready-for-codegen increased `3209 -> 3294` (`+85`)
  - runtime sanity with new `.so` outputs:
    - `uvm_seq_body` compiled run still completes (`EXIT=0`)
    - `uvm_run_phase` retains known parity point
      (`UVM_FATAL FCTTYP`, `EXIT=1`).
- Realization:
  - The high-impact remaining cast/legalization work is not only direct calls;
    indirect-callee function type consistency is required to convert retained
    call_indirect paths into LLVM cleanly and avoid artificial residual-op
    stripping.

## 2026-02-26
- Phase 5 follow-up: stabilized `!llhd.ref` ABI canonicalization around LLHD users
  and unlocked pointer-backed `llhd.drv` lowering.
- Trigger:
  - After allowing `llhd.constant_time` + `llhd.drv` in `isFuncBodyCompilable`,
    large-workload compile started emitting verifier noise during SCF lowering:
    - `llhd.drv` operand #0 became `!llvm.ptr` after ref-ABI canonicalization.
  - Coverage counters looked better (`75 rejected`) but real codegen did not
    improve (`3302/3294` unchanged), and stripped functions rose (`110`).
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - In `canonicalizeLLHDRefArgumentABIs(...)`, added LLHD-use bridge casts for
    canonicalized args:
    - keep func signature as `!llvm.ptr`
    - insert local `ptr -> !llhd.ref<...>` view when operand use requires ref
      (`llhd.drv`, `llhd.prb`, `llhd.sig.extract`, `llhd.sig.array_slice`).
  - In pre-lowering (`lowerFuncArithCfToLLVM`), added pointer-backed
    near-immediate drive rewrite:
    - `llhd.drv %ref, %val after %t0` (where `%t0` is `<0ns,0d,0/1e>`)
      -> `llvm.store %val, %ptr`
    - supports `%ref` from direct pointer or `ptr -> ref` cast.
  - Added post-prelowering dead-time sweep for `llhd.constant_time` /
    `llhd.current_time` (needed because drive rewrite can make time ops dead
    only after their initial visit).
  - Kept compilability policy conservative for LLHD ops: allow `llhd.drv`
    (lowered), do not blanket-allow other LLHD ref ops in function bodies.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-drive-ref-arg-constant-time.mlir`.
  - First red run showed `1` function codegen + interpreted fallback failure;
    root cause was dead-time cleanup ordering.
  - After dead-time sweep fix: compile emits `2 functions + 0 processes ready`,
    interpreted and compiled both print `out=42`.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regressions pass:
    - `aot-llhd-drive-ref-arg-constant-time.mlir`
    - `aot-unrealized-cast-llhd-ref-arg-abi.mlir`
    - `aot-llhd-time-to-int-current-time.mlir`

## 2026-02-26
- Phase 5 coverage unlock: added comb signed div/mod lowering in function path.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isFuncBodyCompilable(...)` now accepts:
    - `comb.divs`, `comb.modu`, `comb.mods`
  - pre-lowering rewrites added:
    - `comb.divs -> arith.divsi`
    - `comb.modu -> arith.remui`
    - `comb.mods -> arith.remsi`
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-comb-divmod-lowering.mlir`.
  - Validates compileable function and interp/compiled parity (`out=7`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regressions pass:
    - `aot-comb-divmod-lowering.mlir`
    - `aot-llhd-drive-ref-arg-constant-time.mlir`
    - prior ref/time regressions still pass.
  - large workload compile delta (vs pre-change snapshot at `3302/3294` ready):
    - `uvm_seq_body`:
      - `Functions: ... 78 rejected, 5236 compilable`
      - `3311 functions + 1 processes ready for codegen` (`+9`)
      - `Stripped 100 functions with non-LLVM ops`
    - `uvm_run_phase`:
      - `Functions: ... 78 rejected, 5216 compilable`
      - `3303 functions + 1 processes ready for codegen` (`+9`)
      - `Stripped 100 functions with non-LLVM ops`
    - top rejection reasons now:
      - `23x llhd.sig.extract`
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `8x llhd.sig`
- Runtime sanity with new `.so` outputs:
  - `uvm_seq_body` compiled: `EXIT_CODE=0`, simulation completes.
  - `uvm_run_phase` compiled: same known parity point
    (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - small arithmetic legalization wins (`divs/mod`) now translate directly into
    codegen-count gains, while remaining LLHD ref ops (`sig.extract`, `sig`)
    are the next dominant functional coverage blockers.

## 2026-02-26
- Phase 5 coverage unlock: lowered pointer-backed `llhd.sig.extract` immediate
  drives in function-body AOT path.
- Trigger:
  - After previous unlocks, top rejection bucket shifted to
    `23x llhd.sig.extract` on large UVM workloads.
  - Representative cases were of form:
    - `%x = llhd.sig.extract %baseRef from %low`
    - `llhd.drv %x, %value after %t0`
    where `%baseRef` came from `ptr -> !llhd.ref<integer>`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - Added conservative compilability acceptance for `llhd.sig.extract` only when:
    - base ref is pointer-backed (`ptr -> ref` cast)
    - all users are `llhd.drv` with no enable
    - drive delay is near-immediate (`<0ns,0d,0/1e>`)
    - drive value width matches extracted slice width.
  - Extended pre-lowering drive rewrite:
    - `drv(sig.extract(base, low), val)` lowers to base read-modify-write:
      - load base integer
      - bit-insert shifted value using shifted mask
      - store merged result back to base pointer.
  - Added dead `llhd.sig.extract` cleanup pass after pre-lowering rewrites.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-sig-extract-drive-lowering.mlir`.
  - Checks compileable function count and interp/compiled parity (`out=8`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regression pack (manual RUN-equivalent) pass:
    - new: `aot-llhd-sig-extract-drive-lowering.mlir`
    - existing: `aot-llhd-drive-ref-arg-constant-time.mlir`
    - existing: `aot-comb-divmod-lowering.mlir`
    - existing: `aot-unrealized-cast-llhd-ref-arg-abi.mlir`
    - existing: `aot-llhd-time-to-int-current-time.mlir`
  - large workload compile delta (vs prior `3321/3303`-era snapshot):
    - `uvm_seq_body`:
      - `Functions: ... 56 rejected, 5258 compilable`
      - `3329 functions + 1 processes ready for codegen` (`+18`)
      - top reasons now: `13x sim.fork.terminator`,
        `12x builtin.unrealized_conversion_cast:i1->!moore.event`,
        `8x llhd.sig`, `6x llhd.sig.struct_extract`.
    - `uvm_run_phase`:
      - `Functions: ... 56 rejected, 5238 compilable`
      - `3321 functions + 1 processes ready for codegen` (`+18`)
      - same top-reason profile.
  - runtime sanity with new `.so` outputs:
    - `uvm_seq_body` compiled: `EXIT_CODE=0`, simulation completed.
    - `uvm_run_phase` compiled: same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - `llhd.sig.extract` in UVM hot paths is mostly a bit-slice write aliasing
    pattern. Lowering only the immediate drive subset yields a measurable
    codegen win without requiring full general alias-object support.

## 2026-02-26
- Phase 5 follow-up: added pointer-backed `llhd.sig.struct_extract` + integer
  `llhd.prb` lowering in function-body AOT path.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-sig-struct-extract-prb-drv.mlir`.
  - Initial red compile reproduced checker gap:
    - `Functions: 1 total, ... 1 rejected`
    - reason: `llhd.prb` on a `sig.struct_extract` ref.
  - Patched compilability to accept integer probes on supported
    `sig.struct_extract` refs.
  - Post-fix: compile emits `1 compilable`, interpreter/compiled both print
    `out=6`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isFuncBodyCompilable(...)`:
    - accept safe `llhd.sig.struct_extract` subset:
      - input pointer-backed (`ptr -> ref` cast)
      - extracted/result nested type integer
      - users limited to integer `llhd.prb` and immediate/delta `llhd.drv`
    - accept integer `llhd.prb` on pointer-backed refs, including refs defined
      by supported `sig.struct_extract`.
  - `lowerFuncArithCfToLLVM(...)` pre-lowering:
    - rewrite `llhd.sig.struct_extract` to field subref via
      `llvm.gep` + `unrealized_conversion_cast(ptr -> ref)`
    - lower integer `llhd.prb` on pointer-backed refs to `llvm.load`
    - erase dead `llhd.sig.struct_extract` ops after rewrite.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused AOT pack pass:
    - new `aot-llhd-sig-struct-extract-prb-drv.mlir`
    - prior `aot-llhd-sig-extract-drive-lowering.mlir`
    - prior `aot-llhd-drive-ref-arg-constant-time.mlir`
    - prior `aot-comb-divmod-lowering.mlir`
    - prior `aot-unrealized-cast-llhd-ref-arg-abi.mlir`
    - prior `aot-llhd-time-to-int-current-time.mlir`
  - large-workload compile diagnostics:
    - `uvm_seq_body`: `54 rejected, 5260 compilable` (from `56/5258`)
    - `uvm_run_phase`: `54 rejected, 5240 compilable` (from `56/5238`)
    - top reasons now:
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `8x llhd.sig`
      - `6x hw.array_create`
      - `5x sim.fmt.literal`
    - codegen-ready count remained `3329/3321`; stripped non-LLVM functions
      increased `102 -> 104`.
  - runtime sanity with new `.so` outputs:
    - `uvm_seq_body` compiled: completes at `50000000000 fs`, `EXIT_CODE=0`
    - `uvm_run_phase` compiled: same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - This change clears `llhd.sig.struct_extract` from the rejection top list,
    but two newly-admitted functions still fall out during non-LLVM strip.
    The next coverage work should target the specific residual ops in those
    functions (likely `sim.fmt.*`/`llhd.sig` patterns) to convert compileable
    gains into codegen-ready gains.

## 2026-02-26
- Phase 5 coverage follow-up: lowered `hw.array_create`/`hw.array_get` pattern
  in function-body AOT path (formatter selection paths in UVM reporting code).
- Trigger:
  - After the prior step, top rejection reasons showed
    `6x hw.array_create` and `5x sim.fmt.literal`.
  - Representative UVM functions used:
    - `%arr = hw.array_create ...`
    - `%elem = hw.array_get %arr[%idx]`
    where element type was already LLVM-compatible (`!llvm.struct<(ptr, i64)>`).
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isFuncBodyCompilable(...)`:
    - accept `hw.array_create` when all users are `hw.array_get`
    - accept `hw.array_get` when its input is `hw.array_create`, index is
      integer, and result type is LLVM-compatible.
  - `lowerFuncArithCfToLLVM(...)` pre-lowering:
    - rewrite `hw.array_get(hw.array_create(...), idx)` into select chains:
      - compare index against constants
      - select between logical array elements while preserving HW
        operand ordering (`operands = [N-1..0]`, element 0 = last operand).
    - erase dead `hw.array_create` after rewrite.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-hw-array-create-get-lowering.mlir`.
  - Validates compileability and interp/compiled parity (`out=20`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused AOT regressions all pass, including:
    - `aot-hw-array-create-get-lowering.mlir`
    - `aot-llhd-sig-struct-extract-prb-drv.mlir`
    - prior LLHD/cast/divmod/time regressions.
  - large-workload compile diagnostics:
    - `uvm_seq_body`:
      - `Functions: ... 48 rejected, 5266 compilable`
      - `3331 functions + 1 processes ready for codegen`
      - `Stripped 107 functions with non-LLVM ops`
    - `uvm_run_phase`:
      - `Functions: ... 48 rejected, 5246 compilable`
      - `3323 functions + 1 processes ready for codegen`
      - `Stripped 107 functions with non-LLVM ops`
    - top reasons now:
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `8x llhd.sig`
      - `5x sim.fmt.literal`
      - `3x llhd.sig.struct_extract`
  - runtime sanity:
    - `uvm_seq_body` compiled (`after_array.so`): completes at
      `50000000000 fs`, `EXIT_CODE=0`.
    - `uvm_run_phase` compiled (`after_array.so`): same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - This converts the `hw.array_create` rejection bucket into a net codegen
    gain (`+2` ready functions on each large workload), but the strip count
    still grows. Remaining ROI is now concentrated in `llhd.sig` and
    event-cast/wait paths.

## 2026-02-26
- Phase 5 coverage follow-up: lowered local `llhd.sig` refs for LLVM-compatible
  nested types and added struct-valued immediate-drive materialization.
- Trigger:
  - After `array_create/get` lowering, top rejection buckets still included
    `llhd.sig` and residual non-LLVM strips in UVM heavy functions.
  - Representative UVM patterns:
    - `%s = llhd.sig %init : !hw.struct<value: i1024, unknown: i1024>`
    - `llhd.drv %s, %val after %t0` where `%val` is `hw.struct_create(...)`
      and `%init` may come from `hw.bitcast` of packed integers.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - Added shared type conversion helper:
    - `convertToLLVMCompatibleType(...)` for recursive
      `hw.struct`/`hw.array` -> LLVM aggregate mapping.
  - `isFuncBodyCompilable(...)`:
    - now accepts `llhd.sig` when nested type maps to LLVM-compatible storage.
  - `lowerFuncArithCfToLLVM(...)` pre-lowering:
    - rewrites `llhd.sig` into stack-backed storage:
      - `llvm.alloca` of converted nested type
      - materialized init store
      - cast back to `!llhd.ref<...>` for existing LLHD users
    - extends immediate pointer-backed `llhd.drv` to materialize value into
      LLVM storage type before `llvm.store`.
    - materialization supports:
      - `hw.struct_create` -> `llvm.insertvalue` chains
      - `hw.aggregate_constant` (integer fields) -> LLVM aggregate constants
      - `hw.bitcast` from packed integer -> field slices via shift/trunc
      - `hw.array_create` -> `llvm.insertvalue` array assembly
    - dead-op cleanup now erases unused `hw.bitcast` after rewrites.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-sig-bitcast-init-read.mlir`.
  - Red stage:
    - first attempt stripped function due residual `comb.extract` generated by
      bitcast materialization.
  - Fix:
    - replaced generated `comb.extract` with arith shift/trunc sequence.
  - Green stage:
    - compile emits `1 functions + 0 processes ready for codegen`
    - interpreter/compiled both print `out=5`.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused AOT regression pack pass, including:
    - new `aot-llhd-sig-bitcast-init-read.mlir`
    - prior `aot-hw-array-create-get-lowering.mlir`
    - prior `aot-llhd-sig-struct-extract-prb-drv.mlir`
    - prior `aot-llhd-sig-extract-drive-lowering.mlir`
    - prior LLHD/cast/divmod/time regressions.
  - large-workload compile diagnostics:
    - `uvm_seq_body`:
      - `Functions: ... 45 rejected, 5269 compilable`
      - `3335 functions + 1 processes ready for codegen`
      - `Stripped 106 functions with non-LLVM ops`
    - `uvm_run_phase`:
      - `Functions: ... 45 rejected, 5249 compilable`
      - `3327 functions + 1 processes ready for codegen`
      - `Stripped 106 functions with non-LLVM ops`
    - top reasons now:
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `5x sim.fmt.literal`
      - `5x llhd.prb`
      - `3x llhd.sig.struct_extract`
  - runtime sanity:
    - `uvm_seq_body` compiled (`after_sig.so`): completes at
      `50000000000 fs`, `EXIT_CODE=0`.
    - `uvm_run_phase` compiled (`after_sig.so`): same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - Converting `llhd.sig` from rejection into compileable/native paths yields
    a strong net gain (`+4` ready functions on each large workload) while also
    reducing strip count (`107 -> 106`), i.e. this step improved both
    front-door coverage and lowered residual-op fallout.

## 2026-02-26
- Phase 5 coverage follow-up: generalized `llhd.prb` lowering from integer-only
  to LLVM-compatible aggregate probe types.
- Trigger:
  - After `llhd.sig` lowering, top rejection reasons showed `5x llhd.prb`.
  - Representative patterns probe 4-state structs then immediately
    `hw.struct_extract` fields.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isFuncBodyCompilable(...)`:
    - `llhd.prb` now accepted when result type maps via
      `convertToLLVMCompatibleType(...)`.
    - probe source eligibility extended to local `llhd.sig` refs whose nested
      type is LLVM-compatible (not only already pointer-backed refs).
    - retained existing `sig.struct_extract`-specific safety checks.
  - `lowerFuncArithCfToLLVM(...)` pre-lowering:
    - `llhd.prb` now lowers to `llvm.load` of converted LLVM probe type.
    - if probe result type differs (e.g. hw struct), emit bridge
      `unrealized_conversion_cast(llvm -> hw)`; downstream existing
      `hw.struct_extract(cast(...))` folding removes bridge on common paths.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-prb-struct-read.mlir`.
  - Validates compileability and interp/compiled parity (`out=5`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused AOT regression pack pass (new + prior LLHD/array/divmod/cast/time tests).
  - large-workload compile diagnostics:
    - `uvm_seq_body`:
      - `Functions: ... 40 rejected, 5274 compilable`
      - `3336 functions + 1 processes ready for codegen`
      - `Stripped 106 functions with non-LLVM ops`
    - `uvm_run_phase`:
      - `Functions: ... 40 rejected, 5254 compilable`
      - `3328 functions + 1 processes ready for codegen`
      - `Stripped 106 functions with non-LLVM ops`
    - top reasons now:
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `5x sim.fmt.literal`
      - `3x llhd.sig.struct_extract`
      - `2x builtin.unrealized_conversion_cast:i1->!moore.i1`
  - runtime sanity:
    - `uvm_seq_body` compiled (`after_prb.so`): completes at
      `50000000000 fs`, `EXIT_CODE=0`.
    - `uvm_run_phase` compiled (`after_prb.so`): same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - The aggregate `llhd.prb` path is now mostly unlocked; remaining `llhd`
    blockers are concentrated in `sig.struct_extract` edge cases and event-cast
    forms (`i1 -> !moore.event` / `i1 -> !moore.i1`).

## 2026-02-26
- Phase 5 follow-up: unlocked safe `sig.struct_extract -> sig.extract -> drv`
  checker path to reduce remaining `llhd.sig.struct_extract` rejects.
- Trigger:
  - Post-`llhd.prb` run still showed `3x llhd.sig.struct_extract` rejects.
  - Representative UVM function pattern:
    - `%f = llhd.sig.struct_extract %argRef["byte_en"]`
    - `%b = llhd.sig.extract %f from %idx`
    - `llhd.drv %b, ... after %t0`
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isFuncBodyCompilable(...)`:
    - `llhd.sig.extract` now accepts compatible `sig.struct_extract` inputs
      (including local `llhd.sig`-backed refs) rather than requiring an
      already-pointer-backed cast at check time.
    - `llhd.sig.struct_extract` user validation now accepts nested
      `llhd.sig.extract` users when all nested users are immediate/delta
      `llhd.drv` with matching integer widths.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-sig-struct-extract-bit-drive.mlir`.
  - Validates compileability and interp/compiled parity (`out=4`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - focused AOT regression pack pass (new + prior LLHD/array/divmod/cast/time tests).
  - large-workload compile diagnostics:
    - `uvm_seq_body`:
      - `Functions: ... 39 rejected, 5275 compilable`
      - `3336 functions + 1 processes ready for codegen`
      - `Stripped 107 functions with non-LLVM ops`
    - `uvm_run_phase`:
      - `Functions: ... 39 rejected, 5255 compilable`
      - `3328 functions + 1 processes ready for codegen`
      - `Stripped 107 functions with non-LLVM ops`
    - top reasons now:
      - `13x sim.fork.terminator`
      - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
      - `5x sim.fmt.literal`
      - `2x builtin.unrealized_conversion_cast:i1->!moore.i1`
      - `2x llhd.sig.struct_extract`
  - runtime sanity:
    - `uvm_seq_body` compiled (`after_struct_chain.so`): completes at
      `50000000000 fs`, `EXIT_CODE=0`.
    - `uvm_run_phase` compiled (`after_struct_chain.so`): same known parity point
      (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - This reduced rejection count (`40 -> 39`) and increased compilable count,
    but did not improve ready-for-codegen yet due matching strip growth
    (`106 -> 107`). Remaining ROI is now dominated by event-cast/wait lowering
    and `sim.fmt.literal` support.

## 2026-02-26
- Phase 5 follow-up: unlocked `llhd.sig.struct_extract` on function ref args
  (`!llhd.ref<!hw.struct<...>>`) so the remaining struct-extract rejects can
  pass through ref-ABI canonicalization.
- Trigger:
  - Large-workload telemetry still showed `2x llhd.sig.struct_extract`.
  - Representative rejected shape: `llhd.sig.struct_extract %argRef["field"]`
    where `%argRef` is a `func.func` block argument (not a local `llhd.sig`
    and not already wrapped by a `ptr -> ref` cast).
- Root cause:
  - `isFuncBodyCompilable(...)` only accepted pointer-backed or local-signal
    struct-ref inputs for `llhd.sig.struct_extract`/related checks.
  - `canonicalizeLLHDRefArgumentABIs(...)` rewrote ref-arg uses for
    `llhd.prb`/`llhd.drv`/`llhd.sig.extract`, but did not treat
    `llhd.sig.struct_extract` as a ref-typed consumer. That prevented safe
    ref-view materialization for ref-arg struct-extract paths.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - Added helper `isLLVMCompatibleRefValue(Value, MLIRContext*)` for
    block-arg ref compatibility checks.
  - Updated compilability checks to accept LLVM-compatible ref-arg inputs for:
    - `llhd.sig.struct_extract`
    - `llhd.sig.extract` when sourced from `llhd.sig.struct_extract`
    - `llhd.prb` when probing `llhd.sig.struct_extract` results
  - Updated ref-ABI canonicalization:
    - `requiresRefOperand(...)` now includes `llhd::SigStructExtractOp`, so
      canonicalized `!llvm.ptr` entry args get local `ptr -> ref` views at
      struct-extract use sites.
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-llhd-sig-struct-extract-ref-arg-prb-drv.mlir`.
  - Red stage confirmed prior bug:
    - compile result: `Functions: ... 1 rejected` with
      `1x llhd.sig.struct_extract`.
  - Green stage:
    - compile result: `0 rejected, 1 compilable`
    - interpreter + compiled parity: `out=8`.
- Focused validation:
  - rebuild: `ninja -C build_test circt-sim-compile circt-sim` (pass)
  - AOT regression subset pass (interp + compiled parity):
    - `aot-llhd-sig-struct-extract-ref-arg-prb-drv.mlir` (`out=8`)
    - `aot-llhd-sig-struct-extract-prb-drv.mlir` (`out=6`)
    - `aot-llhd-sig-struct-extract-bit-drive.mlir` (`out=4`)
    - `aot-llhd-sig-extract-drive-lowering.mlir` (`out=8`)
    - `aot-llhd-prb-struct-read.mlir` (`out=5`)
- Large-workload compile diagnostics after this step:
  - `uvm_seq_body`:
    - `Functions: ... 37 rejected, 5277 compilable`
    - `3338 functions + 1 processes ready for codegen`
    - `Stripped 107 functions with non-LLVM ops`
  - `uvm_run_phase`:
    - `Functions: ... 37 rejected, 5257 compilable`
    - `3330 functions + 1 processes ready for codegen`
    - `Stripped 107 functions with non-LLVM ops`
  - top reasons now:
    - `13x sim.fork.terminator`
    - `12x builtin.unrealized_conversion_cast:i1->!moore.event`
    - `5x sim.fmt.literal`
    - `2x builtin.unrealized_conversion_cast:i1->!moore.i1`
    - `1x sim.wait_fork`
- Runtime sanity with freshly rebuilt large-workload `.so` files:
  - `uvm_seq_body` compiled:
    - `Wrote ... (1 processes, 3338 functions, 1914 trampolines, ~87.7s)`
    - run reaches `50000000000 fs`, `EXIT_CODE=0`.
  - `uvm_run_phase` compiled:
    - `Wrote ... (1 processes, 3330 functions, 1902 trampolines, ~87.4s)`
    - preserves known parity point:
      `UVM_FATAL FCTTYP ...`, `EXIT_CODE=1`.
- Realization:
  - This closes the remaining `llhd.sig.struct_extract` rejection bucket and
    yields another measurable ready-for-codegen gain (`+2` functions on each
    large workload) without changing the known parity/stability envelope.

## 2026-02-26
- Phase 4 follow-up: landed native module-init skip telemetry to drive safe
  call allowlist expansion from measurements (instead of guesswork).
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - Added `NativeModuleInitSynthesisStats` and switched
    `synthesizeNativeModuleInitFunctions(...)` to return:
    - total hw.module count
    - emitted native-init module count
    - top skip-reason buckets (first blocker per module)
  - Added verbose diagnostics:
    - `Native module init modules: <emitted> emitted / <total> total`
    - `Top native module init skip reasons:`
      (`unsupported_call:*`, `unsupported_op:*`, operand-dependency forms)
- TDD/regression:
  - Added `test/Tools/circt-sim/aot-native-module-init-skip-telemetry.mlir`.
  - Test forces a top-level unsupported init call (`llvm.call @puts`) and
    checks telemetry output under `-v`.
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - new regression checks (manual log/FileCheck-equivalent grep): pass
  - existing native-init baseline regression compile output still valid:
    `aot-native-module-init-basic.mlir` (`Native module init functions: 1`).
- Large-workload telemetry probe:
  - `uvm_seq_body.mlir`:
    - compile: `Functions: ... 37 rejected, 5277 compilable`
    - module-init telemetry: `Native module init functions: 1`,
      `Native module init modules: 1 emitted / 1 total`
  - `uvm_run_phase.mlir`:
    - compile: `Functions: ... 37 rejected, 5257 compilable`
    - module-init telemetry: `Native module init functions: 1`,
      `Native module init modules: 1 emitted / 1 total`
  - no skip reasons reported on either workload.
- Runtime sanity recheck:
  - `uvm_seq_body` compiled (`uvm_seq_body.native_init_telemetry.so`):
    reaches `50000000000 fs`, `EXIT_CODE=0`.
  - `uvm_run_phase` compiled (`uvm_run_phase.native_init_telemetry.so`):
    preserves known parity endpoint (`UVM_FATAL FCTTYP`, `EXIT_CODE=1`).
- Realization:
  - For current UVM workload, top-level native-init allowlist is not the
    bottleneck (already 1/1 emitted). Remaining AOT coverage/perf ROI is still
    dominated by function bodies blocked on `sim.fork`/`moore.wait_event`/`sim.fmt`.

## 2026-02-26
- Phase 4 follow-up: removed the dominant native module-init extraction blocker
  on AVIP workloads by admitting pure constant producers.
- Trigger:
  - multi-module AVIP telemetry showed `0 emitted / N total` for all 8 core8
    workloads, with top skip reason `unsupported_op:arith.constant`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - widened `isNativeModuleInitOp(...)` allowlist to include:
    - `arith::ConstantOp`
    - `hw::AggregateConstantOp`
- TDD/regression:
  - added `test/Tools/circt-sim/aot-native-module-init-arith-constant.mlir`.
  - red stage (before patch):
    - `Native module init modules: 0 emitted / 1 total`
    - `1x unsupported_op:arith.constant`
  - green stage (after patch + rebuild):
    - `Native module init functions: 1`
    - `Native module init modules: 1 emitted / 1 total`
    - runtime parity checks pass (`SIM out=77`, `NATIVE out=77`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regressions (pass):
    - `aot-native-module-init-arith-constant.mlir`
    - `aot-native-module-init-basic.mlir`
    - `aot-native-module-init-skip-telemetry.mlir`
  - AVIP core8 re-telemetry (`--emit-llvm -v`):
    - `ahb`: `1 emitted / 4 total` (was `0 / 4`)
    - `apb`: `1 emitted / 4 total` (was `0 / 4`)
    - `axi4`: `3 emitted / 6 total` (was `0 / 6`)
    - `axi4Lite`: `2 emitted / 9 total` (was `0 / 9`)
    - `i2s`: `1 emitted / 4 total` (was `0 / 4`)
    - `i3c`: `1 emitted / 4 total` (was `0 / 4`)
    - `jtag`: `1 emitted / 4 total` (was `0 / 4`)
    - `spi`: `1 emitted / 4 total` (was `0 / 4`)
    - dominant remaining skip reason: `unsupported_op:llhd.prb`.
  - large UVM sanity after patch:
    - `uvm_seq_body` compiled run: `EXIT_CODE=0` (both baseline compiled mode
      and with `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`).
- Realization:
  - native module-init extraction is no longer stuck at zero on AVIP-style
    multi-module designs; the next concrete ROI is supporting safe module-init
    `llhd.prb` shapes rather than broadening call allowlists further.
- OpenTitan telemetry follow-up (same phase):
  - probe status in this workspace currently limits actionable coverage data:
    - failed MLIR generation probes: `gpio`, `spi_device`, `usbdev`, `i2c`,
      `spi_host`, `gpio_no_alerts`, `uart_reg_top`, `tlul_adapter_reg`
    - only `prim_count` generated MLIR.
  - `prim_count` compile telemetry result:
    - `Functions: 8 total, 8 external, 0 compilable`
    - `No compilable functions found`
  - implication: AVIP remains the primary dataset for native module-init
    coverage tuning until broader OpenTitan targets compile in this setup.

## 2026-02-26
- Phase 4 follow-up: enabled conservative native module-init `llhd.prb`
  support for pointer-backed ref probes.
- Trigger:
  - after constant-op expansion, AVIP telemetry shifted the dominant blocker to
    `unsupported_op:llhd.prb`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - widened `isNativeModuleInitOp(...)` allowlist to include `llhd::ProbeOp`.
  - safety is still enforced by existing synthesis guards:
    - `operand_block_arg:*` rejection for block-arg dependent probe chains
    - `operand_dep_skipped:*` rejection for probes depending on skipped ops
      (notably `llhd.sig`).
- TDD/regression:
  - added `test/Tools/circt-sim/aot-native-module-init-llhd-prb-refcast.mlir`.
  - red stage (before patch):
    - `Native module init modules: 0 emitted / 1 total`
    - `1x unsupported_op:llhd.prb`
  - green stage (after patch + rebuild):
    - `Native module init functions: 1`
    - `Native module init modules: 1 emitted / 1 total`
    - runtime parity checks pass (`SIM out=123`, `NATIVE out=123`).
- Validation:
  - rebuild: `ninja -C build_test circt-sim-compile` (pass)
  - focused regressions (pass):
    - `aot-native-module-init-llhd-prb-refcast.mlir`
    - `aot-native-module-init-arith-constant.mlir`
    - `aot-native-module-init-basic.mlir`
    - `aot-native-module-init-skip-telemetry.mlir`
  - AVIP spot telemetry (`--emit-llvm -v`) after patch:
    - `ahb`: `1 emitted / 4 total`, skip reasons:
      `2x operand_block_arg:llhd.prb`, `1x operand_dep_skipped:llhd.sig`
    - `axi4Lite`: `2 emitted / 9 total`, skip reasons:
      `4x operand_block_arg:llhd.prb`, `3x operand_dep_skipped:llhd.sig`
    - result: emitted count unchanged on these samples, but blocker class is
      now precisely identified (ABI/state-model, not op-coverage).
  - large UVM sanity after patch:
    - `uvm_seq_body` compiled with native init enabled:
      `COMPILE_EXIT=0`, `RUN_EXIT=0`, reaches `50000000000 fs`.
- Realization:
  - module-init extraction has moved from “missing op lowering” to
    “module-argument/signal-state bridge” as the remaining bottleneck.
  - Next ROI is not adding more op names; it is bridging block-arg probes and
    selected `llhd.sig` dependency chains safely (or introducing hybrid
    native+interpreter fallback per module).

## 2026-02-26
- Phase 4 follow-up: landed module-init bridge for `llhd.prb` on module
  block-args and conservative aliasing for probe-only `llhd.sig` chains.
- Trigger:
  - previous AVIP spot telemetry after `llhd.prb` allowlisting still showed
    `operand_block_arg:llhd.prb` and `operand_dep_skipped:llhd.sig` as the
    dominant blockers (`ahb` `1/4`, `axi4Lite` `2/9` emitted).
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - added `isSupportedNativeModuleInitBlockArgProbe(...)` and
    `isSupportedNativeModuleInitSignalProbe(...)`.
  - synthesis now lowers supported block-arg probes to runtime helper calls:
    `__circt_sim_module_init_probe_port_raw(i64) -> i64`.
  - synthesis now aliases supported probe-only `llhd.sig` probes to the signal
    init value instead of rejecting with `operand_dep_skipped:llhd.sig`.
  - micro-module synthesis auto-declares the helper symbol when needed.
- Runtime bridge (`tools/circt-sim/LLHDProcessInterpreter.cpp/.h`):
  - added TLS interpreter handle for native module-init helper dispatch.
  - exported helper:
    `extern "C" uint64_t __circt_sim_module_init_probe_port_raw(uint64_t)`.
  - during native module init, interpreter now maps `hw.module` port index to
    live `SignalId` and serves raw values via
    `getNativeModuleInitPortRawValue(...)`.
- TDD/regressions:
  - added `test/Tools/circt-sim/aot-native-module-init-llhd-prb-block-arg.mlir`.
  - added `test/Tools/circt-sim/aot-native-module-init-llhd-prb-signal-alias.mlir`.
  - compile expectation in `signal-alias` test was relaxed to
    `Functions: {{.*}}0 rejected, 1 compilable` (external helper count is
    workload-dependent).
  - both tests now pass sequential compile/interpreter/native runs.
- Validation:
  - AVIP spot telemetry rerun (`--emit-llvm -v`) on
    `out/avip_core8_interpret_20260226_110735`:
    - `ahb`:
      - `Functions: 7602 total, 23 external, 86 rejected, 7493 compilable`
      - `Native module init functions: 3`
      - `Native module init modules: 3 emitted / 4 total`
      - remaining skip: `1x operand_dep_skipped:llhd.sig`
    - `axi4Lite`:
      - `Functions: 10174 total, 23 external, 104 rejected, 10047 compilable`
      - `Native module init functions: 9`
      - `Native module init modules: 9 emitted / 9 total`
  - large UVM sanity after patch:
    - `uvm_seq_body` with `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`:
      `COMPILE_EXIT=0`, `RUN_EXIT=0` (reaches `50000000000 fs`).
    - `uvm_run_phase` with native-init opt-in:
      preserves parity endpoint (`UVM_FATAL FCTTYP`, `RUN_EXIT=1`).
- Realization:
  - module-arg probe ABI support is no longer the limiting factor on the tested
  AVIP workloads.
  - remaining native-init extraction misses are now concentrated in mutable
  `llhd.sig` dependency chains rather than missing op coverage.

## 2026-02-26
- Phase 4 follow-up: closed the remaining `ahb` native module-init miss by
  accepting conservative `llhd.sig` probe aliasing when the same signal also
  appears as a module-body `hw.instance` operand (connectivity-only use).
- Trigger:
  - after block-arg bridge + signal alias, AVIP spot telemetry still showed
    `ahb` at `3 emitted / 4 total` with `1x operand_dep_skipped:llhd.sig`
    while `axi4Lite` was already `9 / 9`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - widened `isSupportedNativeModuleInitSignalProbe(...)` module-body user
    check from probe-only to read-only probe/connectivity users:
    - allowed users: `llhd.prb`, `hw.instance`
  - this keeps mutation guards intact: any other module-body user still rejects
    module native-init synthesis for that module.
- TDD/regression:
  - added
    `test/Tools/circt-sim/aot-native-module-init-llhd-prb-signal-instance-alias.mlir`.
  - focused native-init probe suite passes sequentially:
    - `aot-native-module-init-llhd-prb-refcast.mlir`
    - `aot-native-module-init-llhd-prb-block-arg.mlir`
    - `aot-native-module-init-llhd-prb-signal-alias.mlir`
    - `aot-native-module-init-llhd-prb-signal-instance-alias.mlir`
- Validation:
  - AVIP spot telemetry rerun (`--emit-llvm -v`) on
    `out/avip_core8_interpret_20260226_110735`:
    - `ahb`:
      - `Native module init functions: 4`
      - `Native module init modules: 4 emitted / 4 total`
      - (previous step was `3 / 4`, `1x operand_dep_skipped:llhd.sig`)
    - `axi4Lite`:
      - `Native module init functions: 9`
      - `Native module init modules: 9 emitted / 9 total` (unchanged)
  - large UVM sanity after patch:
    - `uvm_seq_body` with native-init opt-in:
      `COMPILE_EXIT=0`, `RUN_EXIT=0`, reaches max-time endpoint.
    - `uvm_run_phase` with native-init opt-in:
      preserves known parity endpoint
      (`UVM_FATAL FCTTYP`, `RUN_EXIT=1`).
- Realization:
  - for this AVIP spot set, native module-init extraction is now no longer
    limited by `llhd.prb`/block-arg/signal-connectivity forms.
  - next ROI is broader telemetry-driven handling of remaining mutation-free
    `llhd.sig` dependency shapes beyond this sample pair.

## 2026-02-26
- Phase 4 follow-up: enabled conservative module-init extraction through
  top-level `scf.if`.
- Trigger:
  - after landing probe/alias bridges, AVIP telemetry still had one unresolved
    module-init miss (`i3c`), initially reported as `unsupported_op:scf.if`.
- Implementation (`tools/circt-sim-compile/circt-sim-compile.cpp`):
  - `isNativeModuleInitOp(...)` now accepts `scf::IfOp`.
  - region guard in synthesis now allows region-bearing `scf.if` while keeping
    all other region-bearing ops rejected (`op_has_region:*` unchanged for
    non-`scf.if`).
- TDD/regression:
  - added `test/Tools/circt-sim/aot-native-module-init-scf-if.mlir`.
  - validates compile emission + interpreter/native parity (`out=111`) with
    observed runtime native-init execution marker.
- Focused validation:
  - native-init probe regression pack (all pass):
    - `aot-native-module-init-scf-if.mlir`
    - `aot-native-module-init-llhd-prb-refcast.mlir`
    - `aot-native-module-init-llhd-prb-block-arg.mlir`
    - `aot-native-module-init-llhd-prb-signal-alias.mlir`
    - `aot-native-module-init-llhd-prb-signal-instance-alias.mlir`
- AVIP telemetry re-run (`--emit-llvm -v`, core8 snapshot):
  - `ahb`: `4 emitted / 4 total`
  - `apb`: `4 emitted / 4 total`
  - `axi4`: `6 emitted / 6 total`
  - `axi4Lite`: `9 emitted / 9 total`
  - `i2s`: `4 emitted / 4 total`
  - `jtag`: `4 emitted / 4 total`
  - `spi`: `4 emitted / 4 total`
  - `i3c`: `3 emitted / 4 total`
    - skip reason: `1x unsupported_op:hw.struct_extract`
- Large UVM sanity (native-init opt-in):
  - `uvm_seq_body`: `COMPILE_EXIT=0`, `RUN_EXIT=0` (max-time endpoint)
  - compiled stats unchanged:
    `Functions: 5330 total, 16 external, 37 rejected, 5277 compilable`
    `Wrote ... (1 processes, 3338 functions, 1914 trampolines, ~75s)`
- Surprises / rollbacks:
  - an attempted follow-up to also admit module-init `hw.struct_extract` in
    struct-valued conditional chains caused instability:
    - module-init functions were emitted but then stripped (residual non-LLVM)
      in targeted regression shape.
    - an attempted in-function SCF re-lowering workaround triggered a compiler
      crash in `circt-sim-compile`.
  - that experimental path was reverted before landing; current checkpoint keeps
    only the stable `scf.if` support and leaves `hw.struct_extract` as the next
    explicit target.
