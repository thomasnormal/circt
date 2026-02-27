You’re in a great position now: Phases 1–5 mean the “AOT execution engine” is no longer the risky part. You have:

* **Correctness** (pointer normalization + aotDepth + parity)
* **Full entry-table dispatch** (tagged FuncId, native+trampoline)
* **Native module init** (you removed the biggest fixed bottleneck)
* **0 rejected / 0 stripped** (your compiler pipeline is closed)
* A very clear remaining frontier: **(6) state model cleanup** and **(7) suspension-heavy TB/process codegen**

To get to “Xcelium-level” performance, the next work should be structured around what actually still burns cycles in real runs:

1. **Trampoline-heavy suspension/control-flow paths** (Phase 7)
2. **Residual fixed overhead and complexity from `.so` globals + patching** (Phase 6)
3. **Finishing touches that matter at scale** (dlopen cost, snapshot restore, signal data layout, and eventually partition parallelism)

Below is a concrete plan with implementation details, exit criteria, and how to sequence it so you don’t destabilize what you’ve already made stable.

---

# North-star metrics to drive every decision

Before the next engineering push, lock in the metrics you’ll report for every benchmark run (these will tell you if Phase 6/7 is paying off):

## Run-time throughput

* **ns simulated / s wall** for at least one workload where UVM actually drives traffic and triggers waits/forks.
* Also report **events (delta activations) per second** if you can.

## “Native-ness” mix

From your existing AOT stats, add or standardize:

* `entry_calls_native`
* `entry_calls_trampoline`
* `entry_calls_total`
* `aotDepth_max`
* `yield_count_total` (once Phase 7 starts)
* `wait_event_count`, `fork_count`, `join_count`

## Fixed overhead

* `parse_ms` (mlirbc)
* `.so_load_ms` (dlopen+relocs)
* `init_ms` (native init should be close to constant and small)
* `snapshot_restore_ms` (once you add it)

Exit criteria for “Xcelium-level” isn’t “100% native”; it’s:
**(a) TB run dominates long runs, (b) TB run is mostly native, (c) fixed overhead is amortized or near-zero in snapshot mode.**

---

# Phase 6: Arena-based mutable state (ctx+offset) — do this next

Even though Phase 7 is the big ceiling raiser, **Phase 6 makes Phase 7 cheaper and safer**. It also makes your `.so` smaller, loading faster, and eliminates an entire class of pointer/alias bugs for good.

## 6.1 Design goal

**No mutable `.so` globals at all.** Compiled code accesses mutable state via:

* `base = ctx->valueArenaBase`
* `ptr = base + offset[globalId]`

Constant data (strings, tables) stays in `.rodata` in the `.so`.

This mirrors what you learned from Xcelium and will let you delete most of:

* preAlias/alias patch-table complexity
* “virtual pointer normalization for globals”
* a large fraction of `dlopen` relocation work

## 6.2 Minimal ABI surface

Bump ABI (v5) or add a v4-compatible “optional section”; simplest is a version bump.

Add to `CirctSimCompiledModule`:

* `uint32_t arena_size`
* `const uint32_t* global_offsets`  // indexed by GlobalId
* `const uint32_t* global_sizes`
* `uint32_t num_globals`

Optional (for debug/asserts):

* `const uint32_t* global_alignments`
* `const uint8_t* global_flags` (const/mutable/category)

## 6.3 Compiler changes (circt-sim-compile)

### A) Assign offsets deterministically

You already have `globalId` assignment. Now compute an arena layout:

* alignment policy: 16-byte aligned by default; honor stronger alignment if required
* pack globals tightly
* emit offsets

### B) Eliminate mutable llvm.globals

For each `llvm.mlir.addressof @G`:

* if `@G` is mutable → replace with:

  * load arena base (preferably via an ABI getter or known `ctx` field)
  * `gep i8, base, global_offsets[gid]`
  * bitcast to expected type
* if `@G` is constant → keep as `llvm.global constant` in `.rodata`

### C) Keep your existing “byte-array flattening” only where needed

Once the arena is byte-addressable, flattening is often simpler:

* store as `[N x i8]` in the arena
* rewrite all GEPs to byte offsets (you already do this for globals; reuse)

## 6.4 Runtime changes (circt-sim)

### A) Allocate arena once per sim instance

* `ctx->valueArenaBase = aligned_calloc(arena_size, 16)`
* set interpreter `globalMemoryBlocks[gid] = base + offsets[gid]`

### B) Delete patch table use for mutable globals

Patch table becomes:

* only for constants if you still need it (often you won’t)
* or removed entirely

### C) Make pointer normalization simpler

Your TLS pointer normalizer becomes:

* still useful as a safety net for old virtual handles
* but should rarely trigger for globals anymore

## 6.5 Tests and exit criteria

### Tests

* AOT test: interpreter writes a global, native reads it; native writes it, interpreter reads it.
* Include at least one “vtable global” and one “assoc-array reference stored in an object field/global”.

### Exit criteria

* Patch table shrinks dramatically (ideally to zero mutable entries).
* dlopen time improves measurably (fewer relocations + fewer data symbols).
* AOT stability improves (fewer pointer-normalization events in logs).

---

# Phase 7: Coroutine lowering for suspension-heavy TB paths

This is the big multiplier: those ~1895 “demoted to trampoline” are exactly the functions and processes that prevent you from becoming “native everywhere.”

Do this in a way that preserves your biggest advantage:

> **stackful coroutines** + `_setjmp/_longjmp` yield = you can suspend from deep call stacks naturally.

## 7.0 Rules of engagement (so you don’t destabilize dispatch)

### A) “Suspensive code runs only inside a coroutine context”

Any native function that may yield must only run when:

* `ctx->activeProcess != nullptr` (or equivalent)

If called outside (e.g., during init or via a direct call from non-process context):

* dispatch to trampoline instead

You can enforce this at *dispatch time* (best):

* mark each Fid with a `mayYield` bit at compile time
* if `mayYield && !ctx->activeProcess` → trampoline

This lets you compile suspensive functions without making the entire simulator “always in fiber mode.”

## 7.1 Lower `moore.wait_event` to a yield-capable runtime call

Right now you accept + demote. The next step is:

* accept + **lower to `__circt_sim_wait_event(ctx, evHandle, edgeMask)`**
* the runtime registers the waiter and then `__circt_sim_yield(kind=WAIT_EVENT, ...)`

On resume: the function continues after the call (stackful coroutine does the work).

### Runtime data you need to store on yield

* process id / pointer
* wait kind
* event handle(s) and edge kind
* optionally region to resume in (Active/Reactive/Observed), if you’re modeling that

## 7.2 Compile TB processes as native coroutines

This is where the hot loops live.

### A) Expand process compilation beyond callback-only

You already compile one callback process. Now add coroutine processes:

* allow multiple waits
* allow `moore.wait_event`
* allow `sim.fork` family (initially demote inside process until fork lowering is ready, then remove demotion)

### B) Fix the current immediate blocker you called out earlier

You mentioned `process_extract_external_operand` diagnostics in prior notes. The general solution is:

* build an explicit “process environment” object for each compiled process, containing:

  * baked signal IDs / aliases
  * pointers to arena slices (Phase 6 makes this trivial)
  * constant handles needed by the process
* pass a pointer to this env to the compiled process entry (or store in ctx indexed by ProcId)

That removes “external operand” fragility and makes compiled processes relocatable.

### C) Yield plumbing

At each yield:

* runtime captures wait metadata + updates process state
* `_longjmp` to scheduler
  On resume:
* scheduler `_longjmp` back

You already have this infrastructure; now it becomes the hot path.

## 7.3 Lower `sim.fork` incrementally

Don’t aim for “full SV fork semantics” first; aim for what UVM/AVIP actually uses most.

### Step 1: `fork ... join_none`

* runtime allocates a `ForkGroup` from a pool
* spawns child coroutines (each gets its own stack from pool)
* parent continues immediately
* group destroyed when all children complete

### Step 2: `fork ... join`

* parent yields until `remaining == 0`

### Step 3: `fork ... join_any`

* parent yields until any child finishes
* cancellation of others can come later (or you can let them run but ignore, depending on how your IR models it)

### Implementation detail that matters

Make the fork objects **allocation-free on the hot path**:

* fixed-size pool for `ForkGroup`
* intrusive lists for runnable children

## 7.4 Tests and exit criteria for Phase 7

### Unit tests (AOT lit)

* a coroutine process with:

  * wait_event
  * fork join_none
  * fork join
  * nested waits

### AVIP validation

Choose one AVIP core where you can run actual transactions (APB/AXI typically best once factory is healthy). Then:

Exit criteria:

* the demoted/trampoline count drops significantly for the run phase
* `entry_calls_trampoline / entry_calls_total` decreases
* run throughput increases on long windows (not just short 500ns runs)

---

# Phase 8: Make fixed overhead disappear in “snapshot mode”

You already have native init and mlirbc caching. Now finish the job:

## 8.1 Post-init snapshot/restore

Once Phase 6 arena exists, snapshot is straightforward:

* dump arena bytes
* dump any runtime tables required to start running immediately (or rebuild them cheaply)

Then:

* `circt-sim foo.csnap` restores arena and jumps directly to run
* init becomes ~0 for repeated runs (CI, regressions)

## 8.2 Reduce dlopen cost

You still have ~620ms `.so load`. That will matter in short runs and CI.

Concrete tactics:

* compile `.so` with `-fvisibility=hidden`
* export only the descriptor getter via a version script
* internalize everything else (you already do aggressive internalization; ensure it affects symbol visibility)
* use `-ffunction-sections -fdata-sections` + `--gc-sections`

Exit criteria:

* `.so load` becomes a smaller fraction of total in snapshot mode.

---

# Phase 9: Optional “match Xcelium on big SoCs” — partition parallelism

Once you’re close on single-core throughput, the remaining gap on huge designs is usually multicore scaling. Xcelium wins here with partition-level parallelism, not “thread per process.”

This is a separate milestone after Phase 7 is paying off:

* partition by clock domain / module clusters
* independent schedulers per partition
* lock-free rings for commId value changes
* deterministic barriers per time/delta window

Only start this when your single-core run is convincingly native-dominated; otherwise you’ll parallelize interpreter overhead.

---

# What to do next, concretely, starting now

Given your exact status, I’d sequence it like this:

1. **Start Phase 6 arena** with a strict “minimal invasive” approach:

   * bump ABI
   * implement arena allocation + offsets
   * migrate *only* mutable globals first
   * keep constants in `.rodata`
   * keep pointer-normalization as a debug safety net

2. In parallel, do Phase 7 groundwork:

   * add `mayYield` classification per fid
   * add dispatch gating: yield-capable fids run native only inside coroutine context

3. Then implement Phase 7 in this order:

   * `moore.wait_event` → yield runtime call
   * coroutine process compilation (multiple waits)
   * fork join_none → join → join_any

4. After that, implement post-init snapshot restore (Phase 8.1) to make end-to-end runs feel like a compiled simulator in CI.

---

## Why this plan gets you to Xcelium-level performance

* Xcelium-level throughput comes from “native + tight dispatch + minimal overhead.” You already have the dispatch and native init.
* The remaining gap is “the last mile of testbench suspension semantics” (Phase 7) and “clean state model” (Phase 6).
* Arena + coroutine lowering is also the clean path to a true compiled-simulator workflow: compile once, restore state, run long regressions fast and deterministically.

If you want, I can also sketch the exact data structures for the arena layout (alignment rules, how to represent `!llhd.ref` in arena, and how to handle vtables as pure FuncId arrays) and the minimal runtime ABI for `wait_event`/fork so you can implement Phase 7 without needing to revisit ABI again.

