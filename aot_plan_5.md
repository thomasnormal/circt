# AOT Plan 5 (Updated)

This plan assumes the core architecture is correct:

- tagged FuncIds
- unified entry table in AOT `.so`
- runtime `dlopen` + descriptor loading
- pre-alias/global patching

The remaining gap to Xcelium-level end-to-end throughput is now mostly:

1. native dispatch correctness/stability
2. compiled/interpreter parity
3. init-time reduction
4. hot-path coverage expansion
5. long-term mutable-state simplification

---

## Current status snapshot

- Entry-table dispatch and tagged indirect lowering are already implemented.
- Large UVM segfaults were reduced via targeted interception/demotion.
- Process extraction/lowering coverage improved recently:
  - `uvm_seq_body`: `Processes: 1 total, 1 callback-eligible, 0 rejected`
  - `uvm_run_phase`: `Processes: 1 total, 1 callback-eligible, 0 rejected`
- A shutdown/reporting-path crash signature in `__moore_assoc_exists` was
  traced to `uvm_pkg::uvm_report_message::get_severity` reading a virtual
  pointer.
  - follow-up in progress: remove the specific `get_severity` default demotion
    (keep broader reporting demotions), relying on assoc-pointer normalization.
  - regression updated to require native-default codegen + compiled run:
    `aot-uvm-report-message-severity-native-optin.mlir`.
  - pending: re-run large workload validation on rebuilt binaries.
- Strategic assoc-pointer fix is now partially landed:
  - MooreRuntime has assoc-pointer normalization callback plumbing.
  - LLHD interpreter registers a resolver that maps virtual/global addresses
    to host-dereferenceable storage (including pointer-slot recovery).
  - `MooreRuntimeAssocTest` coverage added for resolver path.
  - `uvm_seq_body` compiled run now completes (`RUN_EXIT=0`) without the
    prior `__moore_assoc_exists` shutdown crash signature.
- Queue helper pointer/length hardening now covers compiled AOT dispatch too:
  - runtime `__moore_queue_pop_front_ptr` / `__moore_queue_pop_back_ptr` now:
    - normalize queue/result/data pointers through TLS virtual->host mapper
    - reject invalid queue lengths (`<0` or `>100000`) before mutation
    - avoid freeing virtual-address-backed queue storage
  - `__moore_queue_size` now applies the same invalid-length guard.
  - regressions:
    - interpreter: `queue-pop-front-ptr-invalid-len-guard.mlir` (green)
    - AOT: `aot-queue-pop-front-ptr-invalid-len-guard.mlir` (compiled path)
  - unit tests:
    - `MooreRuntimeQueueTest.QueuePopFrontPtrInvalidLengthGuard`
    - `MooreRuntimeQueueTest.QueuePopBackPtrInvalidLengthGuard`
    - `MooreRuntimeQueueTest.QueueSizeInvalidLengthGuard`
- `uvm_seq_body` parity mismatch (`COMP/INTERNAL`) is now mitigated with a
  root-cause-specific demotion:
  - culprit class was generated short-name build-phase singleton getters
    (for example `get_2160`) touching
    `@"uvm_pkg::uvm_pkg::uvm_build_phase::m_inst"` under unmapped direct
    native `func.call`.
  - compile-time demotion now strips functions that access that global to
    trampolines by default.
  - unmapped direct native `func.call` policy is back to default allow-all;
    targeted deny remains available via
    `CIRCT_AOT_DENY_UNMAPPED_NATIVE_NAMES=...`.
- Direct-call policy consistency hardening landed:
  - cached `func.call` path now applies the same `CIRCT_AOT_DENY_FID` and
    `CIRCT_AOT_TRAP_FID` checks as the slow path.
  - interpreted-call accounting now includes policy-forced native fallbacks
    (explicit unmapped-name deny + `DENY_FID`) for accurate AOT stats.
- Regression coverage added:
  - `aot-unmapped-native-get-policy.mlir` locks in default/allow/allow+deny
    behavior for unmapped direct native `func.call`.
  - `aot-uvm-build-phase-singleton-getter-demote.mlir` locks in targeted
    demotion of build-phase singleton getter wrappers.
- Phase 4 bootstrap landed (opt-in):
  - `circt-sim-compile` now emits conservative per-`hw.module` native init
    entrypoints (`__circt_sim_module_init__*`) when top-level init ops are in
    a safe LLVM-only subset.
  - runtime can execute these during initialize() with
    `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`; otherwise existing interpreted
    module-init path remains unchanged.
  - new regression `aot-native-module-init-basic.mlir` validates native init
    execution and parity on a module-level global store/read path.
  - selected top-level `llvm.call` init patterns are now supported via
    allowlist (`memset`/`memcpy`/`memmove`/`malloc`/`calloc` + `llvm.mem*`).
  - trampoline callback wiring is now available before native module init, and
    module-init trampoline fallback tables are prepared at dispatch time (after
    function lookup cache population) to avoid unresolved fallback calls.
  - new regression `aot-native-module-init-memset.mlir` validates call-bearing
    native init (`out=12345` baseline, `out=0` with native-init opt-in).
  - large-workload sanity with native init opt-in now runs cleanly:
    - `uvm_seq_body`: `EXIT_CODE=0` both baseline and native-init opt-in
      (single-run wall-clock ~4.73s baseline vs ~4.77s opt-in).
    - `uvm_run_phase`: preserves known parity point (`FCTTYP` fatal, `EXIT=1`)
      in both baseline and native-init opt-in modes.
- `uvm_run_phase` compiled/interpreter still match on `FCTTYP` fatal (`EXIT_CODE=1` both).
- Indirect aggregate-return ABI parity fix landed for native vtable dispatch:
  - root cause for `uvm_phase::add -> uvm_object::get_name` crash was
    mixed call ABI on indirect calls (caller register-return vs callee sret).
  - `flattenAggregateFunctionABIs` now rewrites indirect `llvm.call` sites
    when callee function types carry aggregate args/returns.
- Trampoline generation is now scoped to actual interpreter-fallback symbols:
  - host extern declarations (for example lowered `printf(... )`) are no
    longer forced through interpreter trampoline synthesis.
  - AOT regression added:
    `aot-host-vararg-extern-no-trampoline.mlir`.
- Phase 5 cast-coverage expansion landed for `!llhd.ref` argument ABIs:
  - micro-module `func.func` arg canonicalization: `!llhd.ref<T> -> !llvm.ptr`
  - direct callsite rewrites to peel wrapper casts
  - indirect callsite rewrites: `func.call_indirect` callee function type +
    operands are now canonicalized when `!llhd.ref` inputs are present
  - pre-lowering round-trip cast fold (`ptr -> ref -> ptr`)
  - new regression: `aot-unrealized-cast-llhd-ref-arg-abi.mlir`
- Phase 5 follow-up landed for pointer-backed ref writes:
  - canonicalized `!llhd.ref` arguments now keep LLHD operand typing valid via
    local ref-view casts at LLHD use sites (`llhd.drv`/`llhd.prb`/`llhd.sig.extract`).
  - pre-lowering now lowers near-immediate pointer-backed `llhd.drv` to
    `llvm.store` (`<0ns,0d,0/1e>` delay), followed by dead time-op cleanup.
  - new regression: `aot-llhd-drive-ref-arg-constant-time.mlir`.
- Added comb arithmetic coverage unlocks:
  - `comb.divs`, `comb.modu`, `comb.mods` are now accepted in compilability and
  lowered to `arith.divsi`, `arith.remui`, `arith.remsi`.
  - new regression: `aot-comb-divmod-lowering.mlir`.
- Added `llhd.sig.extract` + immediate-drive coverage unlock:
  - compilability now accepts pointer-backed `llhd.sig.extract` when all users
    are immediate/delta `llhd.drv` writes.
  - pre-lowering now rewrites those drives to base-signal read-modify-write
    stores and erases dead `llhd.sig.extract`.
  - new regression: `aot-llhd-sig-extract-drive-lowering.mlir`.
- Coverage impact on large workloads (post-change):
  - `uvm_seq_body`:
    - `Functions: ... 56 rejected, 5258 compilable`
    - `3329 functions + 1 processes ready for codegen`
  - `uvm_run_phase`:
    - `Functions: ... 56 rejected, 5238 compilable`
    - `3321 functions + 1 processes ready for codegen`
  - prior baseline was `378 rejected` with ~`3021/3013` ready-for-codegen
    functions (`uvm_seq_body` / `uvm_run_phase` respectively).
- Added pointer-backed `llhd.sig.struct_extract` + integer `llhd.prb` lowering:
  - compilability now accepts the safe subset:
    - `sig.struct_extract` input is pointer-backed (`ptr -> ref` cast)
    - extracted field/result are integer-typed
    - users are integer `llhd.prb` and/or immediate/delta `llhd.drv`
  - pre-lowering now rewrites:
    - `llhd.sig.struct_extract` to field-pointer subref via `llvm.gep` + cast
    - integer `llhd.prb` on pointer-backed refs to `llvm.load`
  - new regression:
    - `aot-llhd-sig-struct-extract-prb-drv.mlir` (`out=6` interp/compiled)
  - large-workload metrics after this step:
    - `uvm_seq_body`: `54 rejected, 5260 compilable`
    - `uvm_run_phase`: `54 rejected, 5240 compilable`
    - codegen-ready remained `3329/3321`; stripped non-LLVM functions rose
      `102 -> 104` (the newly-admitted functions still contain residual ops).
- Added `hw.array_create` + `hw.array_get` lowering for formatter-style
  selection patterns:
  - compilability now accepts `hw.array_create` when all users are
    `hw.array_get`, and accepts `hw.array_get` when fed by `hw.array_create`.
  - pre-lowering rewrites `array_get(array_create(...), idx)` into
    LLVM-compatible select chains (preserving HW operand ordering).
  - new regression:
    - `aot-hw-array-create-get-lowering.mlir` (`out=20` interp/compiled)
  - large-workload metrics after this step:
    - `uvm_seq_body`: `48 rejected, 5266 compilable`, `3331 + 1 ready`
    - `uvm_run_phase`: `48 rejected, 5246 compilable`, `3323 + 1 ready`
    - stripped non-LLVM functions: `104 -> 107` (net ready gain `+2` vs prior).
- Added `llhd.sig` local-ref lowering for LLVM-compatible nested types:
  - compilability now accepts `llhd.sig` when nested type can be lowered to
    LLVM-compatible storage (`int` / `hw.struct` / `hw.array` recursive subset).
  - pre-lowering now rewrites:
    - `llhd.sig` to stack storage (`llvm.alloca` + init `llvm.store`) with
      ref-view cast back to `!llhd.ref<...>`.
    - pointer-backed immediate `llhd.drv` stores with value materialization to
      LLVM storage type (including `hw.struct_create` and `hw.bitcast`-sourced
      packed 4-state struct values).
  - new regression:
    - `aot-llhd-sig-bitcast-init-read.mlir` (`out=5` interp/compiled)
  - large-workload metrics after this step:
    - `uvm_seq_body`: `45 rejected, 5269 compilable`, `3335 + 1 ready`
    - `uvm_run_phase`: `45 rejected, 5249 compilable`, `3327 + 1 ready`
    - stripped non-LLVM functions: `107 -> 106` (net ready gain `+4` vs prior).
- Added non-integer `llhd.prb` lowering for LLVM-compatible aggregate types:
  - compilability now accepts `llhd.prb` when probe result type maps to an
    LLVM-compatible type (not just integers), including local `llhd.sig` refs.
  - pre-lowering now lowers such probes to `llvm.load` plus
    `llvm -> hw` bridge cast when needed; existing struct-extract folds then
    consume the bridge on hot paths.
  - new regression:
    - `aot-llhd-prb-struct-read.mlir` (`out=5` interp/compiled)
  - large-workload metrics after this step:
    - `uvm_seq_body`: `40 rejected, 5274 compilable`, `3336 + 1 ready`
    - `uvm_run_phase`: `40 rejected, 5254 compilable`, `3328 + 1 ready`
    - stripped non-LLVM functions unchanged at `106`.
- Added checker support for `sig.struct_extract -> sig.extract -> drv` chains:
  - compilability now accepts `llhd.sig.extract` when its input is a compatible
    `llhd.sig.struct_extract` path (including local `llhd.sig`-backed refs).
  - `llhd.sig.struct_extract` user checks now include safe nested
    `llhd.sig.extract` immediate-drive users.
  - new regression:
    - `aot-llhd-sig-struct-extract-bit-drive.mlir` (`out=4` interp/compiled)
  - large-workload metrics after this step:
    - `uvm_seq_body`: `39 rejected, 5275 compilable`, `3336 + 1 ready`
    - `uvm_run_phase`: `39 rejected, 5255 compilable`, `3328 + 1 ready`
    - stripped non-LLVM functions: `106 -> 107` (coverage moved forward,
      but no additional ready-for-codegen net gain yet).
- Added block-argument `!llhd.ref<hw.struct<...>>` support for
  `llhd.sig.struct_extract` paths:
  - fixed checker gap where ref-arg struct-extracts were rejected before
    ref-ABI canonicalization.
  - fixed ref-ABI canonicalization gap by treating `llhd.sig.struct_extract`
    as a ref-typed operand user (inserts local `ptr -> ref` view cast).
  - new regression:
    - `aot-llhd-sig-struct-extract-ref-arg-prb-drv.mlir` (`out=8`
      interp/compiled).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `37 rejected, 5277 compilable`, `3426 + 1 ready`
    - `uvm_run_phase`: `37 rejected, 5257 compilable`, `3418 + 1 ready`
    - stripped non-LLVM functions: `1` on both workloads.
- Added `sim.proc.print` + `sim.fmt.*` lowering for function bodies:
  - lowered to direct `llvm.call @printf` with recursive format-fragment
    expansion from `sim.fmt.*` producer trees.
  - format ops are now treated as compile-time metadata and dead-erased after
    print lowering.
  - new regression:
    - `aot-sim-fmt-literal-print-lowering.mlir`
      (literal + dec + concat in compiled func body; interp/compiled parity).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `35 rejected, 5279 compilable`, `3427 + 1 ready`
    - `uvm_run_phase`: `35 rejected, 5259 compilable`, `3419 + 1 ready`
    - stripped non-LLVM functions: `1` on both workloads.
- Suspensive Moore call hardening landed for func-body AOT:
  - uncovered failure mode: functions with `moore.wait_event` could be lowered
    to `__moore_wait_event` and native-dispatched, causing runtime loader
    failure (`undefined symbol: __moore_wait_event`) in compiled mode.
  - fix: default-demote to trampolines any function that:
    - contains `moore.wait_event`, or
    - calls suspending Moore runtime helpers
      (`__moore_wait_event`, `__moore_wait_condition`,
      `__moore_process_await`).
  - this keeps suspension semantics on the interpreter path until a proper
    resumable/native ABI is implemented for these helpers.
  - regression:
    - `aot-moore-wait-event-func-lowering.mlir` now locks in demotion +
      compiled/interpreter parity (no symbol-lookup crash).
  - workload status after hardening:
    - `uvm_seq_body`: `21 rejected, 5293 compilable`, `3416 + 1 ready`,
      `RUN_EXIT=0` (known parity endpoint preserved at time 0).
    - `uvm_run_phase`: `21 rejected, 5273 compilable`, `3408 + 1 ready`,
      `RUN_EXIT=1` (`FCTTYP` parity endpoint unchanged).
- Added `hw.array_get` coverage for `hw.aggregate_constant` array literals:
  - compilability now accepts integer `hw.array_get` when input is an
    integer-element `hw.aggregate_constant` array (not just `hw.array_create`).
  - pre-lowering now rewrites
    `array_get(aggregate_constant[...], idx)` to LLVM-compatible select chains
    with constant element materialization (same logical index ordering as the
    existing `array_create` lowering).
  - regression:
    - `aot-hw-array-get-aggregate-constant-lowering.mlir`
      (`out=-1` interpreter/compiled parity).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `20 rejected, 5294 compilable`, `3417 + 1 ready`
    - `uvm_run_phase`: `20 rejected, 5274 compilable`, `3409 + 1 ready`
    - runtime parity endpoints preserved (`RUN_EXIT=0` / `RUN_EXIT=1`).
- Added sim-control op compileability with explicit trampoline demotion:
  - `isFuncBodyCompilable` now accepts:
    - `sim.pause`
    - `sim.terminate`
    - `sim.clocked_pause`
    - `sim.clocked_terminate`
  - interception demotion now conservatively trampolines any function body
    containing the above ops (`hasSimControlOp(...)`).
  - this removes these ops from rejection buckets while preserving interpreter
    control-flow semantics for termination/pause behavior.
  - regression:
    - `aot-sim-control-func-demote.mlir`
      (`Functions: 3 ... 0 rejected`, `Demoted 2 ...`, `out=7` parity).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `18 rejected, 5296 compilable`, `3417 + 1 ready`
    - `uvm_run_phase`: `18 rejected, 5276 compilable`, `3409 + 1 ready`
    - parity endpoints preserved:
      - `uvm_seq_body`: `RUN_EXIT=0` (`UVM_FATAL :0`, simulation completes)
      - `uvm_run_phase`: `RUN_EXIT=1` (`FCTTYP` at time 0).
- Closed wait-event-local cast rejection for `iN <-> !moore.iN` bridge casts:
  - `isFuncBodyCompilable` now accepts same-width integer/moore-int
    `builtin.unrealized_conversion_cast` when it appears under
    `moore.wait_event` (where function bodies are already demoted).
  - this targets the remaining UVM pattern in
    `uvm_pkg::uvm_wait_for_nba_region`:
    `i32 -> !moore.i32` before `moore.detect_event any`.
  - regression:
    - `aot-moore-wait-event-int-cast-demote.mlir`
      (`Functions: 2 ... 0 rejected`, `Demoted 1 ...`, `out=2` parity).
  - existing wait-event regression preserved:
    - `aot-moore-wait-event-func-lowering.mlir` (`out=1` parity).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `17 rejected, 5297 compilable`, `3417 + 1 ready`
    - `uvm_run_phase`: `17 rejected, 5277 compilable`, `3409 + 1 ready`.
- Added fork-family compileability with explicit trampoline demotion:
  - `isFuncBodyCompilable` now accepts:
    - `sim.fork`
    - `sim.fork.terminator`
    - `sim.wait_fork`
    - `sim.disable_fork`
  - demotion policy now also trampolines any function containing fork-family
    ops (`hasSimForkFamilyOp(...)`), preserving interpreter semantics while
    removing hard rejections.
  - regression:
    - `aot-sim-fork-family-func-demote.mlir`
      (`Functions: 2 ... 0 rejected`, `Demoted 1 ...`, `out=9`).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `1 rejected, 5313 compilable`, `3417 + 1 ready`
    - `uvm_run_phase`: `1 rejected, 5293 compilable`, `3409 + 1 ready`
    - only remaining top rejection reason:
      - `comb.parity` (1).
  - runtime sanity preserved on full compiled runs:
    - `uvm_seq_body`: `RUN_EXIT=0`, `UVM_FATAL :0`, simulation completed.
    - `uvm_run_phase`: `RUN_EXIT=1`, known `FCTTYP` parity endpoint.
- Added comb-parity safe demotion closure (robustness-first):
  - attempted native `comb.parity` lowering uncovered a backend crash in LLVM
    translation (`mlir::PtrLikeTypeInterface::getMemorySpace`) on the large
    UVM workloads after lowering/codegen start.
  - to keep AOT robust while removing the final rejection bucket, parity ops
    are now accepted for compilability and parity-containing functions are
    demoted to trampolines (`hasCombParityOp(...)`).
  - regression:
    - `aot-comb-parity-lowering.mlir`
      (`Functions: 2 ... 0 rejected`, `Demoted 1 ...`, `out=1` parity).
  - large-workload metrics after this step:
    - `uvm_seq_body`: `0 rejected, 5314 compilable`, `3417 + 1 ready`
      (`Demoted 1896`).
    - `uvm_run_phase`: `0 rejected, 5294 compilable`, `3409 + 1 ready`
      (`Demoted 1884`).
  - full compiled runtime sanity preserved:
    - `uvm_seq_body`: `RUN_STATUS=0`, `UVM_FATAL :0`, simulation completed.
    - `uvm_run_phase`: `RUN_STATUS=1`, known `FCTTYP` parity endpoint.
- MLIR->LLVM translation crash class was root-caused and fixed:
  - culprit pattern: `llvm.getelementptr` carrying non-LLVM `elem_type`
    (for example `!hw.struct<...>`) reaching LLVM translation.
  - fix:
    - `isFuncBodyCompilable` now validates GEP `elem_type` convertibility
      (`llvm.getelementptr:elem_type` rejection on non-convertible cases).
    - pre-translation canonicalization rewrites non-LLVM GEP element types to
      LLVM-compatible types via `convertToLLVMCompatibleType`.
  - regression:
    - `test/Tools/circt-sim/aot-llvm-gep-non-llvm-elemtype-lowering.mlir`.
  - large parity-admission compile repros now succeed (no
    `PtrLikeTypeInterface::getMemorySpace` signatures):
    - `uvm_seq_body`: `0 rejected, 5314 compilable`, `.so` emitted.
    - `uvm_run_phase`: `0 rejected, 5294 compilable`, `.so` emitted.
- parity native codegen is now re-enabled by default:
  - removed parity-specific demotion gating in function stripping.
  - updated regression `aot-comb-parity-lowering.mlir` to expect native codegen
    (`2 functions + 0 processes ready for codegen`).
  - default large-workload compile metrics after re-enable:
    - `uvm_seq_body`: `Demoted 1895`, `3418 + 1 ready`.
    - `uvm_run_phase`: `Demoted 1883`, `3410 + 1 ready`.
  - compiled runtime endpoints remain stable:
    - `uvm_seq_body`: `EXIT=0`, known `UVM_FATAL :0` endpoint.
    - `uvm_run_phase`: `EXIT=1`, known `FCTTYP` endpoint.
- Current top function rejection reasons are now:
  - none (current compile telemetry shows `0 rejected` on both tracked UVM
    workloads).
- Phase-graph opt-in stabilization follow-up landed:
  - enabling `CIRCT_AOT_ALLOW_NATIVE_UVM_PHASE_GRAPH=1` was shown to recover
    large compile coverage (`uvm_seq_body` `3428+1 -> 3472+1`) but crashed in
    `uvm_phase` state/query paths.
  - symbolized crash culprits:
    - `uvm_phase::set_state`
    - `uvm_phase::get_domain`
    - `uvm_phase::get_full_name`
  - added dedicated gate:
    - `CIRCT_AOT_ALLOW_NATIVE_UVM_PHASE_STATE`
    - default keeps the above phase-state methods intercepted even with
      phase-graph opt-in.
  - regression:
    - `aot-uvm-phase-state-mutator-native-optin.mlir`.
  - post-fix phase-graph opt-in workload status:
    - `uvm_seq_body`: `3468 + 1 ready` (still +40 vs default) and `RUN_EXIT=0`
    - `uvm_run_phase`: `3460 + 1 ready`, parity endpoint preserved (`EXIT=1`).

---

## Phase 1: Fix `__moore_assoc_exists` class of crashes correctly

### Goal

Prevent native code from dereferencing non-host pointers by classifying and normalizing pointers at the Moore runtime ABI boundary.

### 1.1 Runtime resolver bridge (implemented)

Land callback bridge at Moore runtime boundary:

- `MooreAssocPtrResolver`
- `__moore_assoc_set_ptr_resolver(...)`
- interpreter registers resolver during runtime accessor setup

### 1.2 Pointer normalization policy (implemented)

Normalize assoc-array pointers before dereference:

- pass through known host pointers
- map interpreter virtual addresses to aliased host storage
- recover host pointer from global pointer slots when the virtual address
  points at a pointer field
- pointer-class diagnostics with optional trace stream (`CIRCT_AOT_TRACE_PTR`):
  - `class=virtual ... -> host=... (tls-normalize)`
  - `class=virtual ... unresolved (tls-miss|no-tls-callback)`
  - `class=tagged_funcid ...` trap.
- interpreter resolver provenance tracing now classifies:
  - `assoc-host`, `virtual-direct`, `virtual-pointer-slot`,
    `virtual-unmapped`.
- unit coverage:
  - `MooreRuntimeAssocTest.TlsNormalizeMapsVirtualPointer`
  - `MooreRuntimeAssocTest.TracePtrLogsVirtualTranslation`.

### 1.3 Assoc helper integration (implemented)

All assoc helper entry points now normalize pointer operands before
`AssocArrayHeader` dereference (`size/delete/delete_key/first/next/last/prev/exists/get_ref/copy/copy_into`).

Acceptance criteria:

- no crash at `header->type` dereference in assoc helper
- failures become deterministic traps with pointer provenance (met for
  pointer-class tracing in runtime + resolver path)
- tactical intercept exceptions (like `get_severity`) can be removed once
  normalization + translation is verified.

### 1.4 Fast crash triage controls

Ensure these are implemented and used in both direct and indirect dispatch:

- `CIRCT_AOT_DENY_FID=...`
- `CIRCT_AOT_ALLOW_ONLY_FID=...`
- `CIRCT_AOT_TRAP_FID=...`

Optional: `CIRCT_AOT_TRACE_PTR=1` for pointer normalization tracing.

Status:

- `CIRCT_AOT_DENY_FID` / `CIRCT_AOT_TRAP_FID` checks are now aligned between
  slow and cached direct `func.call` paths.
- Added deterministic `TRAP_FID` regression:
  - `aot-trap-fid-direct-call.mlir` validates direct `func.call` trap behavior
    on mapped FuncId (`fid=0`) with expected fatal marker.

---

## Phase 2: Full entry-table dispatch, no `compiledFuncIsNative` gate

### Goal

Dispatch indirect calls through `all_func_entries[fid]` for all FuncIds (native or trampoline) without recursion instability.

### 2.1 Re-entry guard contract

Use `aotDepth` guard in interpreter trampoline entry path:

- increment on `__circt_sim_call_interpreted` entry
- while `aotDepth > 0`, disable re-dispatch back to native
- decrement on return

This allows uniform entry-table dispatch while preventing AOT<->interp recursive storms.

### 2.2 Remove gating reliance

After guard is validated, remove `compiledFuncIsNative` as a dispatch gate (keep only for stats if needed).

Acceptance criteria:

- no crash when dispatching trampoline entries via entry table
- stable behavior with full indirect dispatch enabled

---

## Phase 3: Resolve compiled vs interpreter parity divergence

### Goal

Eliminate semantic mismatches (e.g. `uvm_seq_body` time-0 `COMP/INTERNAL` fatal only in compiled mode).

### 3.1 Add parity bisection workflow

For failing test(s), run:

- interpreter baseline
- compiled run

Compare compact invariants:

- phase progression markers
- factory/registration events
- first fatal/error signatures

Use FID bisection via deny-lists to isolate culprit path quickly.

Status update:

- FID-only deny did not resolve `uvm_seq_body` mismatch.
- native call tracing showed culprit class was direct native `func.call` with
  no reverse `funcOp -> FuncId` mapping (`fid=<unmapped>`).
- narrower bisection isolated generated build-phase singleton getter wrappers
  (for example `get_2160`) reading `uvm_build_phase::m_inst`.
- compile-time interception now demotes functions touching
  `uvm_build_phase::m_inst` to trampolines.
- default unmapped direct native `func.call` policy is restored to allow-all;
  targeted deny remains opt-in (`CIRCT_AOT_DENY_UNMAPPED_NATIVE_NAMES`).

### 3.2 Culprit handling policy

For each culprit function/fid:

- if simulator-integrated semantics are required -> intercept (Tier 0)
- otherwise fix native semantics and keep native-enabled (Tier 1)

Acceptance criteria:

- `uvm_seq_body` parity restored (fatal mismatch removed) - met with default
  build-phase singleton demotion
- no broad fallback that undoes coverage gains

---

## Phase 4: Init-time collapse (highest end-to-end ROI)

### Goal

Remove interpreter-heavy module-level initialization cost.

### 4.1 Compile module-level LLVM init ops

Compile current `executeModuleLevelLLVMOps()` work into AOT entry:

- synthesize `void circt_sim_module_init(CirctSimCtx *ctx)` in `.so`
- call it at runtime instead of interpreting module-level LLVM ops

Status update (incremental):

- Landed conservative extraction + runtime wiring:
  - symbol form: `__circt_sim_module_init__<encoded_hw_module_name>`
  - runtime lookup/dispatch via `CompiledModuleLoader::lookupModuleInit(...)`
  - execution guarded by env: `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`
  - fallback remains interpreter module-init when symbol unavailable or env off.
- Expanded native-init subset now includes selected `llvm.call` patterns:
  - libc: `memset`, `memcpy`, `memmove`, `malloc`, `calloc`
  - LLVM mem intrinsics: `llvm.memset.*`, `llvm.memcpy.*`, `llvm.memmove.*`
- Runtime module-init dispatch now ensures trampoline fallback readiness:
  - trampoline callback wired before module-init execution.
  - module-init trampoline fallback tables prepared at module-init call site,
    after function lookup cache setup (without enabling full native dispatch
    policy before finalize).
- Native-init telemetry for unsupported top-level ops is now landed:
  - verbose summary: emitted/total module count
  - top skip reasons bucketed by first blocker
    (`unsupported_call:*`, `unsupported_op:*`, operand dependency forms)
  - regression: `aot-native-module-init-skip-telemetry.mlir`
- Workload readout with telemetry:
  - `uvm_seq_body`: `Native module init modules: 1 emitted / 1 total`
  - `uvm_run_phase`: `Native module init modules: 1 emitted / 1 total`
  - no skip reasons reported on this workload, so current init bottleneck is
    not from top-level allowlist misses.
- Native module-init allowlist now includes pure constant producers:
  - `arith.constant`
  - `hw.aggregate_constant`
  - regression: `aot-native-module-init-arith-constant.mlir`
- Native module-init allowlist now also includes conservative `llhd.prb`:
  - supported shape is pointer-backed ref probes (e.g. probe of
    `builtin.unrealized_conversion_cast(!llvm.ptr -> !llhd.ref<...>)`)
  - regression: `aot-native-module-init-llhd-prb-refcast.mlir`
- AVIP core8 telemetry after this expansion (`--emit-llvm -v`):
  - `ahb`: `1 emitted / 4 total` (was `0 / 4`)
  - `apb`: `1 emitted / 4 total` (was `0 / 4`)
  - `axi4`: `3 emitted / 6 total` (was `0 / 6`)
  - `axi4Lite`: `2 emitted / 9 total` (was `0 / 9`)
  - `i2s`: `1 emitted / 4 total` (was `0 / 4`)
  - `i3c`: `1 emitted / 4 total` (was `0 / 4`)
  - `jtag`: `1 emitted / 4 total` (was `0 / 4`)
  - `spi`: `1 emitted / 4 total` (was `0 / 4`)
  - top remaining blocker is now `unsupported_op:llhd.prb`.
- AVIP spot telemetry after `llhd.prb` enablement:
  - `ahb`: still `1 emitted / 4 total`, skip reasons now
    `operand_block_arg:llhd.prb` and `operand_dep_skipped:llhd.sig`
  - `axi4Lite`: still `2 emitted / 9 total`, skip reasons now
    `operand_block_arg:llhd.prb` and `operand_dep_skipped:llhd.sig`
  - interpretation: blocker moved from unsupported op coverage to module-init
    ABI/state-model limits (module block-arg probes + skipped `llhd.sig` deps).
- Module-init block-arg probe bridge is now landed:
  - compile-time synthesis rewrites supported `llhd.prb` on `hw.module` block
    arguments into runtime helper calls:
    `__circt_sim_module_init_probe_port_raw(i64) -> i64`
  - runtime now exposes current module-port raw values during native init via
    TLS interpreter context and port-index to signal-id mapping.
  - regression: `aot-native-module-init-llhd-prb-block-arg.mlir`
- Conservative `llhd.sig` probe aliasing is now landed for module init:
  - supported shape: `llhd.prb` of a module-body `llhd.sig` where module-body
    users of that signal are probes only (no mutations).
  - synthesis aliases probe result to signal init value instead of rejecting
    due `operand_dep_skipped:llhd.sig`.
  - regression: `aot-native-module-init-llhd-prb-signal-alias.mlir`
- AVIP spot telemetry after block-arg bridge + signal alias (`--emit-llvm -v`):
  - `ahb`: `3 emitted / 4 total` (was `1 / 4`), remaining skip:
    `1x operand_dep_skipped:llhd.sig`
  - `axi4Lite`: `9 emitted / 9 total` (was `2 / 9`)
  - interpretation: module-arg probe ABI gap is resolved on these samples;
    remaining work is selective handling of mutable/complex `llhd.sig`
    dependency chains.
- Follow-up landed: allow conservative module-init signal-probe aliasing when
  the same `llhd.sig` is also connected via module-body `hw.instance` users
  (non-mutating connectivity).
  - implementation keeps the same safety intent: module-body users must remain
    read-only forms (`llhd.prb` and/or `hw.instance`), with no module-body
    mutation users.
  - new regression:
    `aot-native-module-init-llhd-prb-signal-instance-alias.mlir`
- AVIP spot telemetry after this follow-up (`--emit-llvm -v`):
  - `ahb`: `4 emitted / 4 total` (was `3 / 4`)
  - `axi4Lite`: unchanged at `9 emitted / 9 total`
  - interpretation: the prior residual `ahb` skip due
    `operand_dep_skipped:llhd.sig` is now closed for this sample.
- Follow-up landed: conservative top-level `scf.if` support in native module
  init synthesis.
  - `isNativeModuleInitOp(...)` now admits `scf.if`.
  - region guard now permits `scf.if` regions during native-init extraction
    (other region-bearing ops remain rejected).
  - regression: `aot-native-module-init-scf-if.mlir`.
- AVIP core8 telemetry after re-run (`--emit-llvm -v`):
  - fully emitted:
    - `ahb` `4/4`
    - `apb` `4/4`
    - `axi4` `6/6`
    - `axi4Lite` `9/9`
    - `i2s` `4/4`
    - `i3c` `4/4`
    - `jtag` `4/4`
    - `spi` `4/4`
- UVM sanity after this expansion remains stable:
  - `uvm_seq_body` with `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1`:
    `COMPILE_EXIT=0`, `RUN_EXIT=0` (max-time reached)
  - `uvm_run_phase` with native-init opt-in:
    preserves known parity endpoint `UVM_FATAL FCTTYP`, `RUN_EXIT=1`.
- Follow-up landed for the former `i3c` gap:
  - added safe native module-init handling for:
    - `hw.struct_extract` from 4-state `scf.if`/`hw.struct_create`/
      `hw.aggregate_constant` producers.
    - top-level 4-state `hw.struct_create` (`value`/`unknown`) in module init.
  - module-init-only rewrite lowers `hw.struct_extract(scf.if(...))` to
    field-typed `scf.if` before SCF->CF to avoid residual non-LLVM forms.
  - regressions:
    - `aot-native-module-init-scf-if-struct-extract.mlir`
    - `aot-native-module-init-hw-struct-create-fourstate.mlir`
- Next expansion target: move from op-coverage closure to init wall-time
  reduction (Phase 4.2 snapshot/native-init integration) and broader telemetry
  beyond the current AVIP core8 set.
- Phase 4.2 snapshot/native-init integration is now landed:
  - `.csnap` loading now auto-enables native module init by default when a
    compiled module is present (`native.so` auto-load or explicit `--compiled`),
    preserving explicit env overrides.
  - explicit opt-out is supported via
    `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=0`.
  - runtime native-init gate now uses truthy semantics (only
    `1/true/yes/on` enables), replacing prior env-presence behavior.
  - regression added:
    - `aot-snapshot-native-module-init-auto.mlir`
      (snapshot default-on + explicit-off behavior).
- Current OpenTitan telemetry constraint in this workspace:
  - probe set mostly fails during MLIR generation (`gpio`, `spi_device`,
    `usbdev`, `i2c`, `spi_host`, `uart_*`, `tlul_adapter_reg`).
  - only `prim_count` generated MLIR, but compile telemetry reports
    `Functions: 8 total, 8 external, 0 compilable`, so it is not yet useful
    for module-init coverage measurement.

### 4.2 Snapshot integration

Reuse preprocessed IR + compiled `.so`; extend snapshot path to include native init flow.

Status update:

- Snapshot flow now sets `CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1` by default
  when loading `.csnap` with an available compiled module and no explicit env
  override.
- Explicit disable is preserved (`CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=0`).
- `LLHDProcessInterpreter` native-init execution now obeys truthy parsing
  rather than raw env-presence checks, closing the prior `=0 still enabled`
  behavior gap.
- Regression coverage:
  - `aot-snapshot-native-module-init-auto.mlir`
    (compile snapshot, run default-on, run explicit-off).

Acceptance criteria:

- init no longer dominates wall time
- end-to-end AVIP time approaches parse+load+run envelope

---

## Phase 5: Coverage expansion from hotness and rejection telemetry

### Goal

Increase native hot-path share with targeted, measurable wins.

### 5.1 Keep high-value counters

Track and report at shutdown:

- `indirect_calls_total`
- `indirect_calls_native`
- `indirect_calls_trampoline`
- `direct_calls_native`
- `direct_calls_interpreted`
- `aotDepth_max`

Status:

- Landed in `circt-sim` AOT stats output (`CIRCT_AOT_STATS=1`) as canonical
  shutdown telemetry lines, alongside legacy AOT counters.
- Regressions updated/validated:
  - `aot-basic-func.mlir`
  - `aot-unmapped-native-get-policy.mlir`

### 5.2 Address highest-impact rejection patterns first

Current priority:

1. live time-op lowering gaps (`llhd.constant_time` / `llhd.current_time`)
2. remaining cast legalization (`i1 -> !moore.event` family)
3. hot `sim.*`/`llhd.*` paths identified from runtime hotness

### 5.3 Process compilation telemetry

Retain detailed process rejection reasons (`process_extract_external_operand:<op>`, `process_lowering_failed:<op>`, etc.) and drive fixes from measured counts.

Status update:

- landed residual-strip telemetry in `circt-sim-compile -v`:
  - after residual-op stripping, now emits top strip reasons
    (`body_nonllvm_op:*`, `sig_nonllvm_arg:*`, `global_nonllvm_type:*`, etc.)
  - regression added: `aot-strip-non-llvm-telemetry.mlir`
- current large-workload readout (`uvm_seq_body`) after this telemetry:
  - `Stripped 107 functions with non-LLVM ops`
  - top reasons:
    - `34x body_nonllvm_op:hw.struct_create`
    - `33x sig_nonllvm_arg:!hw.struct<value: i4096, unknown: i4096>`
    - `9x body_nonllvm_op:hw.bitcast`
    - `9x sig_nonllvm_arg:!hw.struct<value: i64, unknown: i64>`
    - `4x body_nonllvm_op:builtin.unrealized_conversion_cast`
- implication:
  - next high-ROI coverage step is now clear and measured: eliminate residual
    4-state HW struct ABI/body op shapes (`hw.struct_create` / `hw.bitcast` +
    non-LLVM hw-struct signatures) in function lowering.
- follow-up landed: `hw.struct_extract(hw.bitcast(iN -> hw.struct<...>), field)`
  now folds to integer slicing when all struct fields are integers.
  - regression added:
    `aot-hw-bitcast-struct-extract-lowering.mlir`
  - large-workload impact (`uvm_seq_body`):
    - stripped residual functions: `107 -> 100`
    - codegen-ready functions: `3338 -> 3345` (`+7`)
  - prior `body_nonllvm_op:hw.bitcast` top reason was removed; dominant
      residual blocker is now concentrated on `hw.struct_create` + 4-state
      hw-struct ABI signatures.
- follow-up landed: canonical fold for non-LLVM aggregate casts
  (`builtin.unrealized_conversion_cast`) when target is LLVM-compatible.
  - specifically, pre-lowering now materializes LLVM values for cast sites
    like `!hw.struct -> !llvm.struct` from `hw.struct_create` producers.
  - regression added:
    `aot-hw-struct-create-cast-llvm-lowering.mlir`
  - large-workload impact (`uvm_seq_body`):
    - stripped residual functions: `100 -> 98`
    - codegen-ready functions: `3345 -> 3347` (`+2`)
    - top reason trend:
      - `body_nonllvm_op:hw.struct_create` now `34` (from `36` before
        bitcast/extract + cast-materialize follow-ups).
- robustness follow-up landed: tagged-indirect rewriting now requires a real
  in-module `@__circt_sim_func_entries` definition and is skipped otherwise.
  - avoids link failures in function-only/no-FuncId-table compiles where
    lowering previously synthesized unresolved hidden references.
  - regression added:
    `aot-call-indirect-no-funcid-table.mlir`
  - existing strip telemetry regression updated to remain deterministic under
    the new cast-lowering behavior:
    `aot-strip-non-llvm-telemetry.mlir` now checks
    `sig_nonllvm_arg:!hw.struct<f: i8>`.
- major coverage follow-up landed: canonicalized four-state HW struct
  function boundary ABIs in the micro-module:
  - rewrites `!hw.struct<value: iN, unknown: iN>` func args/results to
    LLVM-compatible boundary types.
  - rewrites direct and indirect call sites accordingly with boundary bridge
    casts.
  - leverages existing pre-lowering folds to eliminate bridge casts in-body
    (e.g. `extract(cast(llvm.struct -> hw.struct), field)`).
  - regression added:
    `aot-fourstate-abi-canonicalization.mlir`.
  - large-workload impact (`uvm_seq_body`):
    - stripped residual functions: `74 -> 2`
    - codegen-ready functions: `3371 -> 3425` (`+54`)
    - remaining strip reasons:
      - `1x body_nonllvm_op:hw.struct_create`
      - `1x body_nonllvm_op:arith.cmpf`.
- float-op follow-up landed: lowered `arith.cmpf`, `arith.fptosi`,
  and `arith.fptoui` in the arith->LLVM rewrite phase.
  - regressions added:
    - `aot-arith-cmpf-lowering.mlir`
    - `aot-arith-fptosi-lowering.mlir`
  - large-workload impact (`uvm_seq_body`):
    - stripped residual functions: `2 -> 1`
    - codegen-ready functions: `3425 -> 3426` (`+1`)
    - only remaining residual strip reason:
      - `1x body_nonllvm_op:hw.struct_create`.
- type-compat follow-up landed: treat `FloatType` as LLVM-compatible in
  boundary/aggregate conversion (`convertToLLVMCompatibleType`).
  - regression added:
    - `aot-hw-struct-float-field-cast-lowering.mlir`
  - status on `uvm_seq_body` unchanged after this patch:
    - still `1x body_nonllvm_op:hw.struct_create` residual.
- residual-strip closure follow-up landed: canonicalized 4-state `cf` block
  arguments after SCF->CF to LLVM boundary types, so merged CFG values no
  longer keep `hw.struct_create` alive.
  - implementation:
    - new `canonicalizeFourStateCFBlockArguments(...)`
    - runs between `lowerSCFToCF(...)` and `lowerFuncArithCfToLLVM(...)`.
  - regression added:
    - `aot-hw-struct-create-cf-blockarg-lowering.mlir`.
  - large-workload impact:
    - `uvm_seq_body`: ready-for-codegen `3427 -> 3428`, no `Stripped` output.
    - `uvm_run_phase`: ready-for-codegen `3419 -> 3420`, no `Stripped` output.
  - net status:
    - prior tail `1x body_nonllvm_op:hw.struct_create` is now closed.
- arena symbol visibility follow-up landed:
  - root cause: runtime patch path uses `dlsym("__circt_sim_arena_base")`, but
    compile-side hidden-visibility pass was hiding the symbol, leaving arena
    base null in native code.
  - fix: keep `__circt_sim_arena_base` default-visible (same treatment as
    `__circt_sim_ctx`) in `setDefaultHiddenVisibility`.
  - effect:
    - `.so` now exports `__circt_sim_arena_base` (verified by `nm -D`)
    - removes the loader warning (`missing __circt_sim_arena_base`) and the
      immediate `m_uvm_get_root` native segfault class.
  - next parity/stability target after this fix:
    - native crash moved forward to `uvm_root::m_do_dump_args` path; continue
      root-cause/fix (do not reintroduce broad gating).
- call_indirect pointer-arg normalization follow-up landed:
  - root cause: native `func.call_indirect` entry-table dispatch packed pointer
    args as raw `uint64_t` without `normalizeNativeCallPointerArg(...)`, unlike
    direct `func.call`.
  - fix: all 4 native call_indirect dispatch paths now normalize pointer-typed
    operands before native invocation.
  - regression added:
    - `aot-call-indirect-native-pointer-tag-normalize.mlir`.
  - status:
    - targeted pointer-tag behavior is fixed (`out=0` with
      `Entry-table native calls: 1`), but large `uvm_seq_body` still crashes in
      `uvm_root::m_do_dump_args` with null `arg0`, indicating a separate issue
      in the compiled-process/run_test path remains.
- arena-global state coherency fix landed (Phase 6 integration bug):
  - root cause:
    - `CompiledModuleLoader::setupArenaGlobals(...)` was implemented but not
      called from `circt-sim`; arena globals were skipped by
      `preAliasGlobals/aliasGlobals`, so interpreter/native mutable state
      diverged.
  - fixes:
    - `circt-sim.cpp` now calls `setupArenaGlobals(...)` before init
      (`compiledLoaderForPreAlias` path) and again in `setCompiledModule(...)`.
    - `setupArenaGlobals(...)` now preserves preexisting global bytes +
      initialized flags when rebasing onto arena storage.
  - validation:
    - `aot-native-module-init-basic.mlir` compiled output restored from `out=0`
      back to `out=123`.
    - `uvm_seq_body` compiled run moved from `EXIT=139` crash to clean exit;
      remaining endpoint is `UVM_FATAL :0` parity at time 0 (no segfault).

---

## Phase 6: Long-term architecture simplification

### 6.1 Arena-based mutable state (`ctx + offset`)

Move mutable globals from `.so` storage to runtime-owned arena:

- lower mutable `llvm.mlir.addressof @G` to arena base + offset
- keep constants/vtable metadata in `.rodata`

### 6.2 Arena-backed vtable FuncIds

Represent vtable slots directly as FuncIds in mutable arena state; keep indirect dispatch as pure table lookup.

Expected result:

- less global patch complexity
- fewer stale-pointer/coherency bugs
- simpler runtime load path

---

## Phase 7 (after 1-5 are stable): coroutine lowering for process-heavy TB paths

Incremental order:

1. `moore.wait_event` yield lowering
2. `sim.fork` join_none
3. join/join_any
4. nested fork/cancel paths as needed

Keep mixed model:

- stackless callbacks for simple RTL processes
- coroutines for true suspension-heavy TB processes

---

## Immediate execution sequence (do now)

1. Implement pointer normalization + classification around assoc/runtime pointer dereference boundaries.
2. Re-enable full all-entry dispatch with `aotDepth` guard semantics.
3. Use FID deny/trap bisection to isolate and fix parity-divergent call paths.
4. Compile module-level init path (`circt_sim_module_init`) and wire runtime call.
5. Continue hotness-driven compile coverage expansion.

---

## Practical fork for assoc-pointer failures

When `array` pointer is bad at assoc boundary:

- **Virtual and mappable** -> translation/normalization path is missing or incomplete.
- **Unmappable** -> likely layout/offset mismatch between writer and native reader.

The classifier + normalize attempt must decide this in one run, so fixes are surgical instead of heuristic.
