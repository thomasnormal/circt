# Block-Level JIT Design (Option C) — Technical Design Document

## 1. Goal

Compile "hot" basic blocks in LLHD process bodies to native x86 code via
LLVM ORC, replacing the interpreter's per-op dispatch with direct native
execution. This targets the **~50-100x interpreter overhead** that dominates
circt-sim's performance gap to Xcelium.

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Process Scheduler                   │
│  (ProcessScheduler: delta cycles, sensitivity, time)  │
└──────────────┬──────────────────────┬────────────────┘
               │ executeProcess()     │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────┐
│   Thunk Dispatch     │  │   Interpreter Fallback     │
│  (fast-path check)   │  │  (executeStep loop)        │
│                      │  │                             │
│  1. PeriodicToggle   │  │  interpretOperation():     │
│  2. ResumableWait    │  │    → dyn_cast chain         │
│  3. ★ NativeBlock ★  │  │    → handler per op type   │
│  4. SingleBlock      │  │    → getValue/setValue      │
│  5. MultiBlock       │  │    → APInt arithmetic       │
└──────┬───────────────┘  └───────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              JIT-Compiled Native Code                 │
│                                                       │
│  void compiled_block(RuntimeCtx *ctx) {               │
│    int32_t clk = ctx->read_signal(sig_clk);           │
│    int32_t out = ~clk;                                │
│    ctx->drive_signal(sig_out, out, delay);             │
│  }                                                    │
│                                                       │
│  Produced by: MLIR block → BehavioralLowering →       │
│               LLVM IR → ORC JIT → native fn ptr       │
└──────────────────────────────────────────────────────┘
```

## 3. Hot Block Identification

### 3.1 What Makes a Block "Hot"

A basic block within an LLHD process body is "hot" if it:
1. Executes repeatedly (once per clock cycle or sensitivity trigger)
2. Contains only JIT-compatible operations (no interpreter-only ops)
3. Is the steady-state body — the block reached after process initialization

### 3.2 Identification Strategy

**Static analysis at thunk installation time** (no profiling needed):

```
For each llhd.process:
  1. Walk CFG to find the "steady-state loop":
     - Entry block → branch to loop header
     - Loop header → wait → resume block → (ops) → branch back to header
  2. The "resume block" (wait destination) is the hot block candidate
  3. Validate all ops in the resume block are JIT-compatible
```

This is exactly the pattern already matched by `tryBuildPeriodicToggleClockThunkSpec`
(line 1661 of NativeThunkExec.cpp) and `isResumableWaitSelfLoopNativeThunkCandidate`,
but generalized to handle arbitrary op sequences instead of only probe→toggle→drive.

### 3.3 JIT-Compatible Operations

Operations that BehavioralLowering already handles (or trivially could):

| Category | Ops | BehavioralLowering Status |
|----------|-----|--------------------------|
| Signal I/O | `llhd.prb`, `llhd.drv` | ✅ Lowers to `__arc_sched_{read,drive}_signal` |
| Combinational | `comb.and`, `comb.or`, `comb.xor`, `comb.icmp`, `comb.mux`, etc. | ✅ Via `populateCombToArithConversionPatterns` |
| Arithmetic | `arith.addi`, `arith.subi`, `arith.trunci`, `arith.extui`, etc. | ✅ Via `arith::populateArithToLLVMConversionPatterns` |
| Constants | `hw.constant`, `arith.constant` | ✅ Via HW/Arith to LLVM |
| LLVM memory | `llvm.load`, `llvm.store`, `llvm.getelementptr`, `llvm.alloca` | ✅ Already LLVM dialect |
| Control flow | `cf.br`, `cf.cond_br` | ✅ Via `cf::populateControlFlowToLLVMConversionPatterns` |

Operations that CANNOT be JIT-compiled (require interpreter):

| Category | Ops | Reason |
|----------|-----|--------|
| Function calls | `func.call`, `func.call_indirect` | Interceptor system (UVM, coverage, etc.) |
| Forking | `sim.fork`, `sim.join*` | Scheduler integration |
| Events | `moore.wait_event`, `moore.detect_event` | Sensitivity list management |
| Print/format | `sim.proc.print`, `sim.fmt.*` | VPI/output system |

**Key insight**: The hot block in a typical clock process (probe→compute→drive→wait)
contains ONLY JIT-compatible ops. The wait/resume handling stays in the thunk dispatch.

## 4. Block Extraction to Standalone MLIR Function

### 4.1 Extraction Procedure

Given a hot block `B` in an LLHD process, extract it as:

```mlir
// Original process:
llhd.process {
  ^entry:
    cf.br ^loop
  ^loop:
    %t = llhd.int_to_time %delay
    llhd.wait %t, ^body  // observed: [%sig_clk]
  ^body:                  // ← HOT BLOCK
    %v = llhd.prb %sig_clk
    %neg = comb.xor %v, %ones
    llhd.drv %sig_out, %neg, %t
    cf.br ^loop
}

// Extracted function:
func.func @__jit_proc_3_block_body(%ctx: !llvm.ptr) {
  %sig_clk_handle = llvm.load %ctx[offset_sig_clk] : !llvm.ptr
  %sig_out_handle = llvm.load %ctx[offset_sig_out] : !llvm.ptr
  %v = call @__arc_sched_read_signal(%sig_clk_handle) → load i1
  %neg = arith.xori %v, %ones
  call @__arc_sched_drive_signal(%sig_out_handle, %neg_ptr, %delay, %true)
  return
}
```

### 4.2 Value Binding

The extracted function receives a `RuntimeCtx*` pointer that contains:
- Signal handles (opaque pointers to scheduler signal storage)
- Local variable addresses (alloca-backed refs)
- Constants (delay values, bit widths)

These are populated at thunk installation time by the interpreter, which
already knows the signal→SignalId mapping (`valueToSignal` map) and
memory block addresses.

### 4.3 Implementation Detail

```cpp
// In LLHDProcessInterpreterNativeThunkPolicy.cpp:
struct JITBlockSpec {
  Block *hotBlock;                    // MLIR block to JIT-compile
  llhd::WaitOp waitOp;               // The wait that resumes into this block
  SmallVector<SignalId> signalReads;  // Signals probed in the block
  SmallVector<SignalId> signalDrives; // Signals driven in the block
  SmallVector<Value> localRefs;       // Alloca-backed ref accesses
  void (*nativeFunc)(void *ctx);      // JIT-compiled function pointer
};
```

## 5. BehavioralLowering Integration

### 5.1 Per-Block Lowering Pipeline

Rather than lowering the entire module (which BehavioralLowering currently does),
we create a **micro-module** containing just the extracted function:

```cpp
// Create a temporary MLIR module with just the hot block function
auto microModule = ModuleOp::create(loc);
// Clone the extracted function into it
microModule.push_back(extractedFunc.clone());
// Add __arc_sched_* function declarations
declareSchedulerRuntimeFunctions(microModule);

// Run the lowering pipeline
PassManager pm(&context);
pm.addPass(createCanonicalizerPass());
pm.addPass(createLowerBehavioralToLLVMPass());
pm.run(microModule);
```

### 5.2 Required BehavioralLowering Enhancements

Current BehavioralLowering handles the ops we need, but needs small additions:

1. **Remove ProcessOpLowering from the extracted function path** — we're lowering
   individual blocks, not whole processes
2. **Keep llhd.int_to_time as identity** — already handled (IntToTimeLowering)
3. **Signal ref handling** — `llhd.prb` and `llhd.drv` already lower to runtime
   calls; we just need the runtime functions to exist

No major changes needed — the existing patterns cover the hot block ops.

## 6. LLVM ORC JIT Compilation

### 6.1 JIT Engine Setup

Use `mlir::ExecutionEngine` (same as arcilator):

```cpp
// One-time setup in LLHDProcessInterpreter constructor or first JIT compile
mlir::ExecutionEngineOptions engineOpts;
engineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default; // O2, not O3
engineOpts.transformer = mlir::makeOptimizingTransformer(2, 0, nullptr);

auto engine = mlir::ExecutionEngine::create(microModule, engineOpts);
auto funcPtr = engine->lookupPacked("__jit_proc_3_block_body");
// Store funcPtr in JITBlockSpec for the thunk to call
```

### 6.2 Compilation Cost

- **One-time cost**: ~1-10ms per block (LLVM O2 on a small function)
- **When**: At thunk installation time (first process activation), same as
  current thunk analysis
- **Memory**: ~100KB per compiled block (LLVM module + JIT code)
- For a typical AVIP with ~10 hot clock processes, total: ~1MB JIT overhead

### 6.3 Lazy vs Eager

Start with **eager** compilation at thunk installation. The blocks are tiny
(3-20 ops), so compilation is fast. Lazy compilation can be added later if
needed for designs with many processes.

## 7. Runtime Bridge: `__arc_sched_*` Implementation

### 7.1 Function Signatures

```cpp
// In a new file: tools/circt-sim/JITSchedulerRuntime.cpp

struct JITRuntimeContext {
  ProcessScheduler *scheduler;
  ProcessId processId;
  // Signal handles: indexed array for fast access
  SignalId *signalHandles;
  size_t numSignals;
  // Local memory: base pointer for alloca-backed refs
  uint8_t *localMemBase;
};

// Thread-local context set before calling JIT-compiled code
static thread_local JITRuntimeContext *g_jitCtx = nullptr;

extern "C" {

int64_t __arc_sched_current_time() {
  return g_jitCtx->scheduler->getCurrentTime().toFemtoseconds();
}

void *__arc_sched_read_signal(void *signalHandle) {
  SignalId sigId = reinterpret_cast<SignalId>(signalHandle);
  const SignalValue &val = g_jitCtx->scheduler->getSignalValue(sigId);
  return const_cast<void *>(static_cast<const void *>(val.getRawData()));
}

void __arc_sched_drive_signal(void *signalHandle, void *valuePtr,
                               int64_t delayFs, bool enable) {
  if (!enable) return;
  SignalId sigId = reinterpret_cast<SignalId>(signalHandle);
  // Read value from pointer, create SignalValue, schedule update
  const auto &currentVal = g_jitCtx->scheduler->getSignalValue(sigId);
  unsigned width = currentVal.getBitWidth();
  APInt newVal(width, 0);
  std::memcpy(newVal.getRawData(), valuePtr, (width + 7) / 8);

  if (delayFs == 0) {
    // NBA: schedule for next delta
    g_jitCtx->scheduler->updateSignal(sigId, SignalValue(newVal));
  } else {
    // Timed: schedule via event queue
    g_jitCtx->scheduler->scheduleTimedUpdate(sigId, SignalValue(newVal),
                                              SimTime::fromFemtoseconds(delayFs));
  }
}

void *__arc_sched_create_signal(void *initPtr, int64_t sizeBytes) {
  // Not needed for hot blocks (signals are pre-registered)
  // Include for completeness
  return nullptr;
}

} // extern "C"
```

### 7.2 Signal Handle Encoding

The `void *signalHandle` passed to JIT code is the `SignalId` cast to pointer:
```cpp
void *handle = reinterpret_cast<void *>(static_cast<uintptr_t>(sigId));
```

This avoids heap allocation for handles and allows O(1) lookup in the
scheduler's signal map.

### 7.3 Binding to LLVM ORC

Register the runtime functions as symbols in the execution engine:

```cpp
engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
  llvm::orc::SymbolMap map;
  map[interner("__arc_sched_current_time")] = {
    llvm::orc::ExecutorAddr::fromPtr(&__arc_sched_current_time),
    llvm::JITSymbolFlags::Exported};
  map[interner("__arc_sched_read_signal")] = {
    llvm::orc::ExecutorAddr::fromPtr(&__arc_sched_read_signal),
    llvm::JITSymbolFlags::Exported};
  map[interner("__arc_sched_drive_signal")] = {
    llvm::orc::ExecutorAddr::fromPtr(&__arc_sched_drive_signal),
    llvm::JITSymbolFlags::Exported};
  return map;
});
```

## 8. Thunk Dispatch Integration

### 8.1 New Thunk Executor

Add `executeJITCompiledBlockNativeThunk` to the thunk dispatch chain in
`executeTrivialNativeThunk` (NativeThunkExec.cpp line 189), inserted
**before** the generic `executeSingleBlockTerminatingNativeThunk`:

```cpp
// In executeTrivialNativeThunk(), after ResumableMultiblockWait:

if (executeJITCompiledBlockNativeThunk(procId, it->second, thunkState))
  return;

// Then fall through to existing SingleBlock/MultiBlock thunks
```

### 8.2 Execution Flow

```cpp
bool LLHDProcessInterpreter::executeJITCompiledBlockNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {

  auto specIt = jitBlockSpecs.find(procId);
  if (specIt == jitBlockSpecs.end())
    return false;  // Not a JIT candidate

  auto &spec = specIt->second;

  // Guard: process must be resuming into the hot block after wait
  if (!state.waiting || state.destBlock != spec.hotBlock) {
    thunkState.deoptRequested = true;
    return true;
  }

  // 1. Set up runtime context
  JITRuntimeContext ctx;
  ctx.scheduler = &scheduler;
  ctx.processId = procId;
  ctx.signalHandles = spec.signalHandleArray.data();
  ctx.numSignals = spec.signalHandleArray.size();
  g_jitCtx = &ctx;

  // 2. Call the JIT-compiled native function
  spec.nativeFunc(&ctx);

  // 3. Clean up and schedule next wait
  g_jitCtx = nullptr;

  // Re-enter wait state (same as periodic toggle clock thunk)
  state.waiting = false;
  if (failed(interpretWait(procId, spec.waitOp))) {
    thunkState.deoptRequested = true;
    return true;
  }

  thunkState.halted = state.halted;
  thunkState.waiting = state.waiting;
  return true;
}
```

### 8.3 Deopt Fallback

If the JIT block encounters an unexpected state (e.g., new signal
registered, control flow changed), it deopts to the interpreter:

```cpp
if (spec.nativeFunc == nullptr) {
  // JIT compilation failed — permanent deopt to interpreter
  thunkState.deoptRequested = true;
  thunkState.deoptDetail = "jit_compiled_block:compile_failed";
  return true;
}
```

## 9. End-to-End Example: Clock Toggle Process

### Before (interpreter, ~100 ops/execution):
```
executeStep: fetch llhd.prb → dyn_cast → interpretProbe → resolveSignalId →
  scheduler.getSignalValue → APInt copy → setValue
executeStep: fetch comb.xor → dyn_cast → interpretCombXor → getValue × 2 →
  APInt XOR → setValue
executeStep: fetch llhd.drv → dyn_cast → interpretDrive → resolveSignalId →
  getValue → scheduler.updateSignal → APInt copy
executeStep: fetch cf.br → dyn_cast → interpretBranch → set destBlock
executeStep: fetch llhd.wait → dyn_cast → interpretWait → set sensitivity →
  scheduler.suspendProcess
```

**Per-execution overhead**: ~5 executeStep calls × ~20 operations each
(map lookups, dyn_casts, APInt operations) = ~100 effective operations.

### After (JIT-compiled, ~10 native instructions):
```asm
; JIT-compiled clock toggle:
mov  rdi, [rsp+8]        ; load signal handle
call __arc_sched_read_signal
movzx eax, byte [rax]    ; load 1-bit value
xor  eax, 1              ; toggle
mov  [rsp-8], al         ; store to temp
lea  rsi, [rsp-8]        ; value pointer
mov  rdi, [rsp+16]       ; drive signal handle
mov  rdx, 5000           ; delay = 5ps
mov  ecx, 1              ; enable = true
call __arc_sched_drive_signal
ret
```

**Per-execution overhead**: ~10 native instructions + 2 function calls to
runtime bridge. The runtime bridge functions are simple (1-3 lines each).

**Expected speedup per block execution**: ~10x (100 interpreter ops → ~10 native ops + 2 function calls).

## 10. Effort Estimation

| Step | Description | Effort | Risk |
|------|-------------|--------|------|
| 1 | Create `JITSchedulerRuntime.cpp` with `__arc_sched_*` functions | 1-2 days | Low — straightforward bridge code |
| 2 | Add hot block identification (generalize `tryBuildPeriodicToggleClockThunkSpec`) | 1-2 days | Low — pattern matching exists |
| 3 | Implement block extraction (MLIR function creation) | 2-3 days | Medium — MLIR builder API |
| 4 | Wire `mlir::ExecutionEngine` into circt-sim (add LLVM target deps) | 1-2 days | Medium — CMake/build integration |
| 5 | Implement `executeJITCompiledBlockNativeThunk` | 1 day | Low — follows existing thunk pattern |
| 6 | End-to-end testing with clock toggle process | 1-2 days | Medium — debugging JIT miscompiles |
| 7 | Extend to multi-op blocks (BFM driver bodies) | 2-3 days | Medium — more ops to handle |

**Total**: ~10-15 days for a working prototype on clock processes.
**Risk areas**: Build integration (ExecutionEngine depends on LLVM target libs),
signal value encoding mismatch between interpreter APInt and JIT raw bytes.

## 11. What This Doesn't Solve

1. **UVM/class method overhead** (~10-20x): `func.call_indirect` dispatch through
   vtables with interceptor pattern matching. This requires a different approach
   (e.g., inline caching, devirtualization).

2. **Signal value encoding** (~2-5x): The interpreter uses `APInt` for all signal
   values, even 1-bit clocks. Native code would use native integer types. The
   runtime bridge functions hide this cost but don't eliminate it.

3. **Scheduler overhead** (~2-5x): Delta cycle management, sensitivity list
   scanning, process queue management. Orthogonal to JIT compilation.

4. **Complex process bodies**: Processes with `func.call_indirect`, `sim.fork`,
   `moore.wait_event`, etc. cannot be JIT-compiled with this approach. They
   remain in the interpreter.

## 12. Future Extensions

1. **Multi-block JIT**: Compile multiple consecutive blocks (e.g., the entire
   wait→compute→drive→wait loop) as a single native function with an internal
   state machine.

2. **Inline caching for call_indirect**: For monomorphic call sites (99% of UVM
   dispatch), cache the target function pointer and bypass vtable lookup.

3. **Native signal storage**: Replace APInt-based `SignalValue` with fixed-size
   native storage (uint8/16/32/64) for common signal widths. This eliminates
   the APInt overhead in both the runtime bridge and the scheduler.

4. **Profile-guided JIT**: Use the existing `opStats` infrastructure to identify
   hot functions (not just blocks) and JIT-compile entire function bodies.
