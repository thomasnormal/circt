# CIRCT Sim LLHDProcessInterpreter Churn Map

Date: February 20, 2026

## Scope and Method

This map covers `tools/circt-sim/LLHDProcessInterpreter*` files.

The churn snapshot below uses local git history for the last 12 months:

- `git log --since='2025-02-20' --no-merges --name-only -- ...`
- `git log --since='2025-02-20' --no-merges --numstat -- ...`

The goal is to identify the highest-change zones before subsystem extraction.

## File-Level Churn Snapshot (Last 12 Months)

| File | Touches | Added | Deleted |
| --- | ---: | ---: | ---: |
| `tools/circt-sim/LLHDProcessInterpreter.cpp` | 273 | 51,509 | 14,713 |
| `tools/circt-sim/LLHDProcessInterpreter.h` | 117 | 2,860 | 133 |
| `tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp` | 19 | 1,538 | 133 |
| `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp` | 15 | 2,332 | 168 |
| `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp` | 6 | 3,425 | 22 |
| `tools/circt-sim/LLHDProcessInterpreterWaitCondition.cpp` | 5 | 573 | 12 |
| `tools/circt-sim/LLHDProcessInterpreterModuleLevelInit.cpp` | 3 | 249 | 25 |
| `tools/circt-sim/LLHDProcessInterpreterGlobals.cpp` | 1 | 389 | 0 |
| `tools/circt-sim/LLHDProcessInterpreterStorePatterns.h` | 1 | 214 | 0 |

## High-Churn Zones

### 1. Core Dispatch and Operation Interpretation Monolith

Primary concentration remains in `tools/circt-sim/LLHDProcessInterpreter.cpp`:

- `interpretLLVMCall` at `tools/circt-sim/LLHDProcessInterpreter.cpp:27861` (largest block).
- `interpretOperation` at `tools/circt-sim/LLHDProcessInterpreter.cpp:10628`.
- `interpretFuncCall` at `tools/circt-sim/LLHDProcessInterpreter.cpp:18478`.
- `interpretLLVMStore` at `tools/circt-sim/LLHDProcessInterpreter.cpp:26741`.
- `interpretLLVMLoad` at `tools/circt-sim/LLHDProcessInterpreter.cpp:26103`.

This is the main conflict hotspot and should be shrunk first.

### 2. Drive/Store/Update Semantics

Drive execution and signal update logic is clustered in:

- `executeModuleDrives` at `tools/circt-sim/LLHDProcessInterpreter.cpp:5703`.
- `executeModuleDrivesForSignal` at `tools/circt-sim/LLHDProcessInterpreter.cpp:5935`.
- `executeContinuousAssignment` at `tools/circt-sim/LLHDProcessInterpreter.cpp:7299`.
- `interpretDrive` at `tools/circt-sim/LLHDProcessInterpreter.cpp:15070`.
- `interpretLLVMStore` at `tools/circt-sim/LLHDProcessInterpreter.cpp:26741`.

These directly map to the Phase 4 `drive/update` and `memory model` extraction items.

### 3. Wait/Fork/Scheduler Wakeup Paths

Suspension and wakeup flow remains split between large sections:

- `interpretWait` at `tools/circt-sim/LLHDProcessInterpreter.cpp:17102`.
- `interpretMooreWaitEvent` at `tools/circt-sim/LLHDProcessInterpreter.cpp:24782`.
- `resumeProcess` at `tools/circt-sim/LLHDProcessInterpreter.cpp:9914`.
- Queue/objection wake helpers (`wake*`) around `tools/circt-sim/LLHDProcessInterpreter.cpp:1190`-`1621`.
- `interpretMooreWaitConditionCall` at `tools/circt-sim/LLHDProcessInterpreterWaitCondition.cpp:64`.

This is a second conflict hotspot due to cross-cutting wait semantics.

### 4. Call/Call-Indirect and UVM/DPI Interception

Call handling spans multiple files and is still high-risk for merge conflicts:

- `interpretFuncCall` at `tools/circt-sim/LLHDProcessInterpreter.cpp:18478`.
- `interpretFuncCallIndirect` at `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp:209`.
- DPI interception blocks (`interceptDPIFunc`) around
  `tools/circt-sim/LLHDProcessInterpreter.cpp:36702` and
  `tools/circt-sim/LLHDProcessInterpreter.cpp:37122`.

This maps to the Phase 4 `call/call_indirect` and `UVM adapter/interceptor` extraction items.

### 5. Native Thunk Policy/Execution

Native-thunk code is already file-split, but remains active:

- `tryInstallProcessThunk` at
  `tools/circt-sim/LLHDProcessInterpreterNativeThunkPolicy.cpp:725`.
- `resumeSavedCallStackFrames` at
  `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp:1846`.
- Multi-mode thunk executors in
  `tools/circt-sim/LLHDProcessInterpreterNativeThunkExec.cpp:395`-`1660`.

This area is not the top churn source, but it is tightly coupled to dispatch.

## Extraction Priorities from This Map

1. `memory model` and `drive/update` extraction from `LLHDProcessInterpreter.cpp`.
2. `call/call_indirect` and UVM/DPI interception extraction.
3. `wait/wakeup` scheduler-path extraction.
4. `tracing/diagnostics` separation after the above boundaries are stable.

## Validation Gate for Phase 4 Follow-Up Slices

For each extraction slice, keep at least:

- focused `circt-sim` lit coverage for the touched subsystem,
- focused `LLHDProcessInterpreterTest` unit coverage for touched behavior,
- no public CLI behavior changes.
