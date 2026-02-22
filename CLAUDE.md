# CLAUDE.md — circt-sim Development Guide

## Session Management (CRITICAL — READ THIS FIRST)

**You are the team leader. You NEVER run builds, ninja, make, test suites, or
any large-output command directly.** Doing so causes context overflow that
crashes your session and kills your entire team.

**Don't write or run any code yourself. Always delegate to your agent team.**

### What You Must NEVER Do
```
# ALL OF THESE WILL CRASH YOU:
ninja circt-sim                          # build
ninja check-circt-tools-circt-sim        # test suite
llvm-lit test/Tools/circt-sim/           # test suite
bash utils/run_sv_tests_circt_sim.sh     # sv-tests
circt-sim some.mlir --top foo            # simulation run
git stash pop                            # can produce huge merge conflict output
```

### WARNING: Do NOT blindly revert files
**Other agents (e.g., the Codex agent working on circt-bmc) may have committed
or modified files in the same repo.** Before reverting any file with
`git checkout HEAD --`, check that you are not undoing another agent's work.
Only revert files YOU modified in your current session. When in doubt, use
`git diff HEAD -- <file>` to inspect what would be lost.

### What You Must ALWAYS Do Instead
```python
# 1. Create a team FIRST
TeamCreate(team_name="my-task")

# 2. Spawn team agents for ALL builds/tests
Task(
    name="builder",
    team_name="my-task",
    subagent_type="Bash",
    model="sonnet",         # MUST be explicit — "inherit" DOES NOT WORK
    mode="bypassPermissions",
    prompt="cd /home/thomas-ahle/circt/build-test && ninja circt-sim 2>&1 | tail -5 ..."
)

# 3. Wait for team messages — they arrive automatically
# 4. DON'T use run_in_background=true — use team members instead
```

### Model Selection
- `model: "sonnet"` — simple tasks (builds, commits, searches, grep)
- `model: "opus"` — hard tasks (debugging failures, architecture, code review)
- NEVER omit the model parameter; `"inherit"` silently fails

### Other Session Rules
- Team agents sometimes fail to auto-start — send a follow-up message to kick them
- Don't kill tmux panes without checking — pane %0 is often the main session
- Commit changes regularly — don't let the worktree accumulate >20 dirty files
- `git add`: use explicit file paths, not directories (avoids accidentally staging junk)

### Team Agent Best Practices
- Create team first (`TeamCreate`), then spawn agents with `team_name` — gives automatic message delivery
- Use `mode: "bypassPermissions"` for build/test agents so they don't block on permission prompts
- Don't use `run_in_background: true` — use team members instead (better messaging, no output file hassle)
- Don't run multiple agents against `build-test/` simultaneously — they fight over ninja locks
- Give agents **specific, command-oriented prompts** with exact commands to run and what to report:
  ```
  # GOOD:
  "1. Run: ninja circt-sim 2>&1 | tail -5
   2. Run: llvm-lit --threads=4 ... 2>&1 | grep '^FAIL:' | sort
   3. Report the sorted FAIL list and total pass/fail counts"

  # BAD:
  "Build and test and analyze the failures"
  ```
- Agents have NO context from your session — make prompts fully self-contained
- Parallelize independent work (e.g., build agent + code research agent)
- If an agent doesn't start, send a follow-up `SendMessage` to kick it

---

## Build & Test Commands (for team agents)

```bash
# Build
cd /home/thomas-ahle/circt/build-test && ninja circt-sim circt-verilog

# circt-sim lit tests
cd /home/thomas-ahle/circt/build-test && ninja check-circt-tools-circt-sim
# Or directly:
/home/thomas-ahle/circt/llvm/build/bin/llvm-lit --threads=4 build-test/test/Tools/circt-sim/

# sv-tests (simulation)
CIRCT_VERILOG=build-test/bin/circt-verilog CIRCT_SIM=build-test/bin/circt-sim \
  bash utils/run_sv_tests_circt_sim.sh

# AVIP simulation (always set wall-time guard)
CIRCT_MAX_WALL_MS=600000 CIRCT_UVM_ARGS="+UVM_TESTNAME=apb_8b_write_test" \
  build-test/bin/circt-sim build-test/apb_avip_dual_llhd.mlir \
  --top hvl_top --top hdl_top --max-time=500000000

# cocotb VPI tests
bash utils/run_cocotb_tests.sh
```

### AVIP Top Module Names
| Protocol | hvl_top |
|----------|---------|
| APB | `hvl_top` |
| AHB | `HvlTop` |
| SPI | `SpiHvlTop` |
| I2S | `hvlTop` |
| I3C | `hvl_top` |
| JTAG | `HvlTop` |
| AXI4 | `hvl_top` |
| AXI4Lite | `Axi4LiteHvlTop` |
| UART | `HvlTop` |

### Test Counts (Feb 19, 2026)
- circt-sim: 527/527 (100%)
- sv-tests: 907 total, 855 pass, 50 xfail, 0 fail (100%)
- ImportVerilog: 268/268 (100%)
- cocotb VPI: 48/50 (96%)

### Key Paths
- `llvm-lit`: `/home/thomas-ahle/circt/llvm/build/bin/llvm-lit`
- `FileCheck`: `/home/thomas-ahle/circt/llvm/build/bin/FileCheck`
- Build dir: `/home/thomas-ahle/circt/build-test/`
- AVIP MLIR: `build-test/apb_avip_dual_llhd.mlir` (NOT `build/`)
- Wall-time guard default: 300s (set `CIRCT_MAX_WALL_MS=600000` for AVIPs)

---

## Current State (Feb 22, 2026)

### Uncommitted Changes
- `lib/Dialect/Sim/ProcessScheduler.cpp`: advanceTime fix for minnow/delta interaction
  (14 lines — prevents minnow time-skip from skipping same-time delta events)

### Stash Contents (important ones)
- `stash@{0}`: E3 minnow + SVA thunk changes (EventQueue.h, ProcessScheduler.h/.cpp, NativeThunkExec)
- `stash@{1}`: **E4 edge fanout** — the main in-progress performance work
  (ProcessScheduler.h +8, ProcessScheduler.cpp +123/-52)
- `stash@{2}` and below: old, probably not needed

### E4 Status: IN PROGRESS — Has Test Regressions
E4 (batch clock-edge wake-up) adds `EdgeFanout` struct to partition
`signalToProcesses` by edge type (posedge/negedge/anyedge). Last test run
showed 507/535 passing (27 failures). Need to determine which failures are
E4-caused vs pre-existing by testing baseline (without E4). The advanceTime
fix in the worktree may also interact.

**To continue E4:**
1. Spawn a team agent to test baseline (stash E4, build, run tests)
2. Compare failing test lists to isolate E4-specific regressions
3. Fix or revert E4 as needed

---

## Project Architecture

### Key Source Files
| File | Purpose |
|------|---------|
| `include/circt/Dialect/Sim/ProcessScheduler.h` | Process scheduler: signals, sensitivity, edge fanout |
| `lib/Dialect/Sim/ProcessScheduler.cpp` | Scheduler: dispatch, time advance, triggers |
| `include/circt/Dialect/Sim/EventQueue.h` | TimeWheel event queue, MinnowInfo callbacks |
| `lib/Dialect/Sim/EventQueue.cpp` | Event scheduling, advanceTime, advanceTimeTo |
| `tools/circt-sim/circt-sim.cpp` | Main entry: parse → passes → init → run → _exit(0) |
| `tools/circt-sim/LLHDProcessInterpreter.cpp` | MLIR interpreter: 20K+ lines, op handlers, UVM interceptors |
| `tools/circt-sim/LLHDProcessInterpreterDrive.cpp` | Signal drive logic |
| `tools/circt-sim/LLHDProcessInterpreterMemory.cpp` | Memory block management |
| `tools/circt-sim/LLHDProcessInterpreterUvm.cpp` | UVM-specific interceptors |
| `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp` | Native thunk execution |
| `tools/circt-sim/AOTProcessCompiler.cpp` | AOT batch compilation (disabled, WIP) |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | SV→MLIR lowering (23K+ lines) |
| `lib/Dialect/Sim/VPIRuntime.cpp` | VPI signal/callback runtime |
| `lib/Runtime/MooreRuntime.cpp` | SV runtime: strings, arrays, randomize, coverage |

### Initialization Pipeline
`processInput()` in circt-sim.cpp:
1. **parse**: MLIR bytecode/text parsing
2. **passes**: Bottom-up simple canonicalizer
3. **init**: SimulationContext + interpreter init (4+ second bottleneck)
4. **run**: Simulation execution
5. **Exit**: `_exit(0)` before return — destructor takes MINUTES for UVM designs

### LLHDProcessInterpreter::initialize() Breakdown
1. `discoverOpsIteratively()` — ~2s, classifies all ops in single pass
2. `registerSignals()` — ~1s
3. `registerFirRegs()` — ~0.5s
4. `registerProcesses()` — ~1-2s
5. `initializeChildInstances()` — ~0.5s, recursive
6. `initializeGlobals()` — one-time, vtable setup
7. `executeModuleLevelLLVMOps()` — **up to 160s for AVIP APB** (main bottleneck)
8. `executeChildModuleLevelOps()` — ~1s, runs AFTER parent ops
9. `createInterfaceFieldShadowSignals()` — ~0.1s
10. `expandDeferredInterfaceSensitivityExpansions()` — ~0.5s

### MooreToCore Struct Layout (CRITICAL)
- **UNALIGNED** layout: `sizeof(struct) = sum of field sizes`, no padding
- Example: `struct<(i32, ptr)>` = 4+8 = 12 bytes (NOT 16)
- **NEVER use hardcoded byte offsets** — use `getLLVMStructFieldOffset()`
- LLVM: fields low-to-high bits; HW: fields high-to-low bits
- Phase ordering bug was caused by hardcoded aligned offsets (40, 104) vs correct unaligned (32, 96)

### Signal Resolution Chain
`resolveSignalId()`: valueToSignal → instanceOutputMap → block args → casts
- Alloca-backed refs: sigId=0 → `findMemoryBlockByAddress()` fallback
- Force sigId=0 for SigStructExtractOp/SigArrayGetOp (read-modify-write)
- Cast+probe tracing through `unrealized_conversion_cast(llhd.prb(sig))`
  Pattern: `%sig = llhd.sig %ptr` → `%val = llhd.prb %sig` → `%ref = cast %val to !llhd.ref`
  Without this, DUT `always @(posedge in_if.clk)` gets empty waitList → delta overflow

### !llhd.ref Patterns
- Local variables: `llvm.alloca` + `unrealized_conversion_cast` to `!llhd.ref`
- Class properties: `llvm.getelementptr` + cast
- Static properties: `llvm.mlir.addressof` + cast

### VIF Shadow Signals
Interface instances use `llhd.sig` holding `!llvm.ptr` (malloc'd struct).
`createInterfaceFieldShadowSignals()` creates per-field shadow signals.
- `interfaceFieldSignals`: address → SignalId
- `interfacePtrToFieldSignals`: parent signal → field signal list
- Bidirectional propagation: parent↔child via `interfaceFieldPropagation` + `childToParentFieldAddr`
- Parent and child have DIFFERENT struct types/layouts

### VTable Dispatch
- `!llvm.array<N x ptr>` globals with `circt.vtable_entries` attribute
- Synthetic addresses (0xF0000000 + N) → `addressToFunction`
- Three call_indirect paths: X-fallback (SSA trace), direct (addressToFunction), static (GEP)
- Runtime vtable override: check object's vtable at byte offset 4
- Object layout: `[i32 class_id][ptr vtable_ptr][...fields...]`

### Process Scheduler Internals
- Flat process vector with intrusive ready queue (E1)
- Edge fanout tables (E4, in progress): `signalEdgeFanout[signalId]` → posedge/negedge/anyedge process lists
- MinnowInfo (E3): 24-byte lightweight callbacks for CallbackTimeOnly processes
- `sensitivityTriggered()` checks edge match per process sensitivity entry
- `triggerSensitiveProcesses()` uses edge fanout for batch wake-up (with fallback)

### UVM Interceptors
Native interceptors fire inside `call_indirect` handler BEFORE `interpretFuncBody`.
Key interceptors: config_db, phase sequencing, sequencer interface, analysis ports,
objection system, factory, coverage, die() absorption.

- `return failure()` → error handler (WRONG). Must `return success()` with `waiting=true` for suspension.
- Phase hopper has OWN objection methods (different class) — separate interceptors needed.
- `executePhaseBlockingPhaseMap` must NOT propagate to master_phase_process children.
- config_db has TWO paths: func.call (~12277) and call_indirect (~6809)
- config_db class name: `config_db_DEFAULT_implementation_t`

### UVM Phase Sequencing
- execute_phase interceptor: sets currentExecutingPhaseAddr on entry
- master_phase_process: detected by fork name "master_phase_process" in interpretSimFork
- Per-process map: `executePhaseBlockingPhaseMap[procId]` → phase address
- 9 IMP phases in order: build→connect→EOE→SOS→run→extract→check→report→final
- Die() absorption: both sim.terminate and llvm.unreachable absorbed in phase context

### Sequencer Interface (Native)
- `start_item` → records item→sequencer in `itemToSequencer`
- `finish_item` → pushes to FIFO, BLOCKS until `item_done`
- `item_done` → marks done, DIRECTLY RESUMES finish_item waiter
- `seq_item_pull_port::get` → pops from FIFO or suspends with retry
- `sequencerGetRetryCallOp` has TWO code paths (callStack empty vs non-empty)

### Shutdown (CRITICAL)
- `_exit(0)` in processInput() BEFORE return, not in main()
- SimulationContext destructor takes MINUTES for large UVM designs
- Deferred sim.terminate: grace period (30s wall-clock) for forked children

---

## Performance Engineering Phases

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| E0 | classifyProcess | Done | 6-step algorithm, ExecModel enum, CallbackPlan struct |
| E1 | Flat process array | Done | Intrusive ready queue, no hash lookups |
| E2 | Callback dispatch | Done | Wire classifyProcess into execution loop |
| E3 | Minnow callbacks | Done | 24-byte MinnowInfo, bypass TimeWheel for time-only |
| E4 | Edge fanout | **In progress** | Pre-partition by edge type, batch wake-up (has regressions) |
| E5 | Inline cache | Planned | call_indirect dispatch caching |
| E6 | Sensitivity caching | Planned | Avoid re-scanning sensitivity lists |

### AVIP Benchmarks
- AHB baseline (Feb 21): init=4.4s, run=0.76s/100ns, total=9s for 100ns
- Performance: ~171 ns/s simulated time
- Always set `CIRCT_MAX_WALL_MS=600000` for AVIP runs
- Always delegate benchmarks to team agents

---

## Common Pitfalls
- **MooreToCore.cpp is 23K+ lines** — ripgrep may timeout, reads may be slow
- **Edit tool silently fails on large files** — always verify edits landed
- **AVIP `.mlir` files are pre-compiled** — recompile from .sv for new features
- **`timeout` EXIT_CODE=124** doesn't mean hang — sim just took too long
- **ccache stale objects** — clear with `ccache -C` if symbols mismatch
- **git stash pop** can produce enormous merge conflict output — be careful
- **Multiple agents sharing build-test/** causes contention — serialize builds
- **`resolveDrivers()` fix**: Old `getLSB()` broken for FourStateStruct; must group by full APInt
- **`__moore_randomize_basic`**: Must be no-op (only advance RNG) — filling object with random bytes corrupts vtable/class_id
- **Associative array assignment** must deep-copy (not shallow ptr copy) — root cause of UVM phase livelock

---

## VPI / Cocotb

### Key Constants (must match IEEE 1800)
- `vpiRealVar = 47` (NOT 29), `vpiStringVar = 616` (NOT 612)
- `vpiReg = 48`, `vpiNet = 36`

### 4-State Encoding
- FourStateStruct: `[value_bits | unknown_bits]`
- X: unknown=1, value=1; Z: unknown=1, value=0

### Key Files
- VPI runtime: `lib/Dialect/Sim/VPIRuntime.cpp`
- Runtime stubs: `lib/Runtime/MooreRuntime.cpp`
- cocotb VPI library: `libcocotbvpi_ius.so`

---

## Coverage System

### Feature Status
| Feature | Status |
|---------|--------|
| Basic covergroups | Works |
| Parametric sample() | Works (fixed: bind by name, not pointer) |
| Cross coverage | Works (basic + binsof/intersect) |
| Coverpoint iff guard | Missing |
| Auto sampling @(posedge) | Missing |
| Wildcard/transition bins | Missing/partial |

### Key Insight
slang's `copyArg()` creates DIFFERENT FormalArgumentSymbol pointers than what expressions reference.
Must match by NAME, not pointer identity. This caused 0% coverage in all 9 AVIPs.

---

## AOT Compilation (Disabled)
- Code in `AOTProcessCompiler.cpp/.h`
- Enabled via `CIRCT_SIM_AOT=1` env var
- Currently disabled pending debugging (tests hang)
- `__moore_coverage_control` and `__moore_coverage_get_max` stubbed (return 0 and 100)

---

## Key Documentation

### Project Files (in repo root)
- **`CHANGELOG.md`** — All changes, organized by date/feature
- **`perf_engineering_log.md`** — Performance measurements after each E-phase, AVIP timings
- **`avip_engineering_log.md`** — AVIP bring-up progress, dual-top debugging history
- **`~/.claude/plans/cached-zooming-fern.md`** — Full AOT/performance engineering project plan (phases E0–E6)

### Auto-Memory (persist across sessions)
- **`~/.claude/projects/-home-thomas-ahle-circt/memory/MEMORY.md`** — Master reference: architecture, workflow rules, all major fixes. First 200 lines auto-loaded.
- **`~/.claude/projects/-home-thomas-ahle-circt/memory/implementation-notes.md`** — config_db, UVM phase sequencing, event/wait, deferred sim.terminate.
- **`~/.claude/projects/-home-thomas-ahle-circt/memory/aot-status.md`** — AOT phase status, rejection ops, test pass rates.
