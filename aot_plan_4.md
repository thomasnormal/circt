# Plan: AOT Native Dispatch via Tagged FuncId Entry Table (Step 7A–7D)

## Context

AOT compilation works — globals are shared via pre-alias, .so loads cleanly, AVIP passes with `--compiled`. But **native execution is effectively off** because compiled code crashes on indirect calls through synthetic vtable addresses (`0xF0000000+N`). The fix is NOT to patch vtables, but to treat synthetic addresses as tagged FuncIds and route all indirect calls through a `func_entries[fid]` table.

This plan implements the user's Step 7A–7D to unlock both `call_indirect` and `func.call` native dispatch.

---

## Step 7A: Unified Function Entry Table in .so

### A1. Extend CirctSimCompiledModule ABI (v4)

**File: `include/circt/Runtime/CirctSimABI.h`**

Add to struct:
```c
uint32_t num_all_funcs;                    // total functions (compiled + trampolined)
const void *const *func_entries;           // func_entries[fid] → callable pointer (never null)
const char *const *func_entry_names;       // func_entry_names[fid] → symbol name
```

Bump `CIRCT_SIM_ABI_VERSION` to 4.

The key invariant: `func_entries[fid]` is **always callable** — either a native compiled function or a trampoline that calls `__circt_sim_call_interpreted`.

### A2. Build the entry table in circt-sim-compile

**File: `tools/circt-sim-compile/circt-sim-compile.cpp`**

Currently the compiler tracks two separate lists:
- `funcNames` / `funcEntries` — compiled functions (lines 2771–2807)
- `trampolineNames` — uncompiled externals (lines 714–748 in `generateTrampolines()`)

Change to build a **unified** list indexed by FuncId:

1. After `generateTrampolines()`, merge both lists into a single `allFuncEntries[]` array:
   - For each compiled function: `allFuncEntries[fid] = @compiled_func`
   - For each trampoline: `allFuncEntries[fid] = @trampoline_func`

2. Build a parallel `allFuncNames[fid]` array with symbol names.

3. The FuncId assignment must match the interpreter's `0xF0000000+N` scheme. Since the interpreter assigns FuncIds sequentially as it discovers vtable entries in `initializeGlobals()`, the compiler must assign the same IDs. **Approach**: Walk vtable globals with `circt.vtable_entries` in the same order as the interpreter, assign FuncIds in the same sequence.

4. Emit as LLVM globals:
   - `@__circt_sim_func_entries : [N x ptr]` (hidden, mutable — filled at link time)
   - `@__circt_sim_func_entry_names : [N x ptr]` (hidden, constant)

5. Update `synthesizeDescriptor()` (line 2193) to include these in the CirctSimCompiledModule struct.

### A3. Ensure trampolines cover ALL functions

Currently trampolines are only generated for external functions referenced by compiled code (line 714–748). For the entry table to be complete, we need trampolines for ALL functions that appear in vtable entries, even if no compiled code references them directly.

**Change**: After collecting vtable entries, generate trampolines for any vtable function not already compiled. This ensures `func_entries[fid]` is never null.

### A4. FuncId assignment synchronization

**Critical**: The interpreter assigns FuncIds in `initializeGlobals()` (LLHDProcessInterpreterGlobals.cpp:274–310) by walking globals with `circt.vtable_entries` and incrementing a counter. The compiler must produce the same assignment.

**Approach**: In circt-sim-compile, walk the module's globals with `circt.vtable_entries` in the same iteration order (module-level op walk), assign FuncIds sequentially. Store the mapping `funcName → fid` for use by the LLVM pass.

---

## Step 7B: LowerTaggedIndirectCalls LLVM Pass

### B1. New LLVM pass in circt-sim-compile

**New file: `tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp`**

An LLVM `FunctionPass` that transforms every indirect call:

```
Original:
  %fp = load ptr, ptr %vtable_slot
  %result = call %fp(%args...)

Transformed:
  %fp = load ptr, ptr %vtable_slot
  %fp_int = ptrtoint ptr %fp to i64
  %is_tagged = icmp uge i64 %fp_int, 0xF0000000
  %is_low = icmp ult i64 %fp_int, 0x100000000  ; < 4GB = not a real pointer
  %tagged = and i1 %is_tagged, %is_low
  br i1 %tagged, label %tagged_path, label %direct_path

tagged_path:
  %fid = trunc i64 (%fp_int - 0xF0000000) to i32
  %entry_ptr = getelementptr [0 x ptr], ptr @__circt_sim_func_entries, i32 0, i32 %fid
  %entry = load ptr, ptr %entry_ptr
  %result_tagged = call %entry(%args...)
  br label %merge

direct_path:
  %result_direct = call %fp(%args...)
  br label %merge

merge:
  %result = phi <result_type> [%result_tagged, %tagged_path], [%result_direct, %direct_path]
```

### B2. Pass registration and scheduling

**File: `tools/circt-sim-compile/circt-sim-compile.cpp`**

Insert the pass **after** MLIR→LLVM IR translation (line 3016) and **before** optimization passes (line 2622). This way:
1. MLIR→LLVM translation produces indirect calls with synthetic addresses
2. LowerTaggedIndirectCalls rewrites them to use entry table
3. LLVM optimizer can inline/simplify the tagged check

### B3. Reference `@__circt_sim_func_entries` in generated code

The pass references the global `@__circt_sim_func_entries` emitted in Step 7A. Since both are in the same LLVM module, this is a direct symbol reference — no dynamic lookup needed.

### B4. Handle Scheme B addresses (hash|0x80000000...)

The static GEP path creates addresses with high bit set: `hash_combine(globalName, index) | 0x8000000000000000ULL`. These are only created dynamically by the interpreter at runtime and never appear in compiled code vtable memory (which uses 0xF0000000+N from initializeGlobals). So the LLVM pass only needs to handle 0xF0000000+N.

---

## Step 7C: Re-enable Interpreter Native Dispatch

### C1. Re-enable call_indirect dispatch

**File: `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`**

Currently 5× `#if 0` blocks disable native dispatch. Change approach:

Instead of the old pattern (lookup in nativeFuncPtrs, type-switch on F0–F8), use the entry table:

```cpp
// When we resolve a tagged address to fid:
uint64_t funcAddr = ...;  // from vtable memory
if (funcAddr >= 0xF0000000 && funcAddr < 0x100000000) {
    uint32_t fid = funcAddr - 0xF0000000;
    if (fid < compiledModule->num_all_funcs) {
        void *entryPtr = compiledModule->func_entries[fid];
        // Marshal args, call entryPtr, unmarshal results
    }
}
```

This replaces the old `nativeFuncPtrs` lookup with a direct array index — O(1), no hash map.

### C2. Update E5 inline cache

**File: `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`**

Extend `callIndirectSiteCache` to store:
- `uint32_t cachedFid` — decoded FuncId
- `void *cachedEntryPtr` — `func_entries[fid]`

Fast path: key match → call cached entry pointer directly.

### C3. Re-enable func.call dispatch

**File: `tools/circt-sim/LLHDProcessInterpreter.cpp`**

Change `if (false && nativeFuncPtrs.count(...))` at lines ~22386/22446 back to enabled. Now safe because:
- Compiled callees that hit indirect calls will use the entry table (Step 7B)
- No more crashes on synthetic addresses

### C4. Remove call_indirect exclusion from loadCompiledFunctions

**File: `tools/circt-sim/LLHDProcessInterpreter.cpp`** (line ~37800)

Currently functions containing `call_indirect` are excluded from `nativeFuncPtrs`:
```cpp
if (isa<mlir::func::CallIndirectOp>(op))
    unsafeForNative = true;
```

Remove this check — with Step 7B, call_indirect inside compiled functions is safe.

---

## Step 7D: Validation

### D1. Debug counters

Add counters to CompiledModuleLoader or interpreter:
- `native_entry_calls` — calls dispatched through func_entries to native code
- `trampoline_entry_calls` — calls dispatched through func_entries to trampolines

Print on shutdown (or via env var `CIRCT_AOT_STATS=1`).

### D2. Test plan

1. **Existing lit tests**: All 677/678 must still pass
2. **AOT tests**: All 6 AOT tests must pass
3. **AVIP with --compiled**: Must complete without $cast failures
4. **Counter validation**: With AVIP, confirm `native_entry_calls > 0`
5. **Regression test**: Add a test that exercises vtable dispatch through compiled code

### D3. Expected outcome

With dispatch enabled, AVIP should show measurable speedup in run phase (currently ~6.3x baseline; with native dispatch potentially 50-500x for compiled hot paths).

---

## Files to Modify

| File | Change |
|------|--------|
| `include/circt/Runtime/CirctSimABI.h` | Add func_entries fields, bump ABI to v4 |
| `tools/circt-sim-compile/circt-sim-compile.cpp` | Build unified entry table, register LLVM pass, sync FuncId assignment |
| `tools/circt-sim-compile/LowerTaggedIndirectCalls.cpp` | **NEW** — LLVM pass to rewrite indirect calls |
| `tools/circt-sim-compile/CMakeLists.txt` | Add LowerTaggedIndirectCalls.cpp |
| `tools/circt-sim/CompiledModuleLoader.h` | Expose func_entries from loaded .so |
| `tools/circt-sim/CompiledModuleLoader.cpp` | Load func_entries, provide lookup-by-fid |
| `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp` | Re-enable dispatch via entry table |
| `tools/circt-sim/LLHDProcessInterpreter.cpp` | Re-enable func.call dispatch, remove call_indirect exclusion |

---

## Implementation Order

1. **7A**: Entry table generation in circt-sim-compile (ABI v4, unified func_entries)
2. **7B**: LowerTaggedIndirectCalls LLVM pass
3. **7C**: Re-enable interpreter dispatch (call_indirect + func.call)
4. **7D**: Validation (counters, tests, AVIP benchmark)

Each step is independently testable — 7A/7B can be verified by building a .so and inspecting it, before touching the interpreter.
