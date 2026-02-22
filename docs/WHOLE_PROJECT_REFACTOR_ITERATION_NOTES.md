# Whole-Project Refactor Iteration Notes

Last updated: February 20, 2026

This file holds detailed iteration narratives referenced by concise entries in
`CHANGELOG.md`.

<a id="entry-78-native-real-harness-arg-quoting-hardening"></a>
## Entry 78: Native Real Harness Arg Quoting Hardening

Scope:
- `utils/run_mutation_mcy_examples.sh`

Changes:
- `validate_native_real_harness_args_spec` validates shell parseability (`shlex.split`) when python is available.
- `render_native_real_harness_args_suffix` canonicalizes args with `shlex.split` + `shlex.join`.

Coverage:
- Added:
  - `test/Tools/run-mutation-mcy-examples-native-real-harness-args-quoted-pass.test`
  - `test/Tools/run-mutation-mcy-examples-native-real-harness-args-quote-invalid.test`
- Updated:
  - `test/Tools/run-mutation-mcy-examples-native-real-harness-args-shell-escape-pass.test`

Validation highlights:
- Focused lit and suite lit passed.
- Native backend real-tests run on `~/mcy/examples` passed (`bitcnt`, `picorv32_primes`).

<a id="entry-79-native-backend-enforcement-gate"></a>
## Entry 79: Native Backend Enforcement Gate

Scope:
- `utils/run_mutation_mcy_examples.sh`

Changes:
- Added `--require-native-backend`.
- Added `CIRCT_MUT_REQUIRE_NATIVE_BACKEND=1` policy gate.
- Strict env parsing (`0|1`).

Coverage:
- Added:
  - `test/Tools/run-mutation-mcy-examples-require-native-backend-missing-native.test`
  - `test/Tools/run-mutation-mcy-examples-require-native-backend-pass.test`
  - `test/Tools/run-mutation-mcy-examples-require-native-backend-env-invalid.test`

Validation highlights:
- Focused lit and suite lit passed.
- Native backend enforcement run on `~/mcy/examples` passed.

<a id="entry-80-native-no-op-fallback-governance"></a>
## Entry 80: Native No-Op Fallback Governance

Scope:
- `utils/run_mutation_mcy_examples.sh`

Changes:
- Added `--fail-on-native-noop-fallback`.
- Added `CIRCT_MUT_FAIL_ON_NATIVE_NOOP_FALLBACK=1` policy gate.
- Integrated fallback marker checks (`CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER`).
- Fixed worker command status handling for timeout/retry/error classification.

Coverage:
- Added:
  - `test/Tools/run-mutation-mcy-examples-native-noop-fallback-pass.test`
  - `test/Tools/run-mutation-mcy-examples-native-noop-fallback-fail.test`

Validation highlights:
- Focused lit and suite lit passed.
- Native backend real harness run passed.

<a id="entry-81-orchestrator-engine-parity-contract"></a>
## Entry 81: Orchestrator Engine-Parity Contract

Scope:
- `utils/run_regression_unified.sh`
- `docs/UNIFIED_REGRESSION_ORCHESTRATOR.md`

Changes:
- Promoted `engine_parity.tsv` to first-class contract artifact.
- Parity states: `MATCH|DIFF|INCOMPLETE` with reason field.

Coverage:
- Added:
  - `test/Tools/run-regression-unified-parity-report.test`
  - `test/Tools/run-regression-unified-parity-incomplete.test`

<a id="entry-82-orchestrator-retryresumeshard-controls"></a>
## Entry 82: Orchestrator Retry/Resume/Shard Controls

Scope:
- `utils/run_regression_unified.sh`

Changes:
- Added `--lane-retries`, `--lane-retry-delay-ms`.
- Added `--resume` lane skipping for finalized lanes.
- Added `--shard-count`, `--shard-index`.
- Added `retry-summary.tsv`.
- Fixed lane failure exit-code propagation.

Coverage:
- Added:
  - `test/Tools/run-regression-unified-lane-retries-pass.test`
  - `test/Tools/run-regression-unified-resume-pass-only.test`
  - `test/Tools/run-regression-unified-shard-selection.test`

<a id="entry-83-bounded-parallel-lane-execution"></a>
## Entry 83: Bounded Parallel Lane Execution

Scope:
- `utils/run_regression_unified.sh`

Changes:
- Added `--jobs` for bounded concurrent lane execution.
- Preserved `--keep-going` semantics while draining in-flight work.
- Hardened command-exit capture under `set -e`.

Coverage:
- Added:
  - `test/Tools/run-regression-unified-jobs-invalid.test`
  - `test/Tools/run-regression-unified-jobs-pass.test`

<a id="entry-84-adapter-catalog-driven-lane-resolution"></a>
## Entry 84: Adapter-Catalog Driven Lane Resolution

Scope:
- `utils/run_regression_unified.sh`
- `docs/unified_regression_manifest.tsv`
- `docs/unified_regression_adapter_catalog.tsv`
- `docs/UNIFIED_REGRESSION_ORCHESTRATOR.md`

Changes:
- Added `--adapter-catalog` and adapter-driven command resolution.
- Added optional manifest extension columns:
  - `suite_root`, `circt_adapter`, `xcelium_adapter`, `adapter_args`
- `xcelium_adapter` defaults to `circt_adapter` when omitted.

Coverage:
- Added:
  - `test/Tools/run-regression-unified-adapter-catalog-pass.test`
  - `test/Tools/run-regression-unified-adapter-catalog-missing-entry.test`

<a id="entry-85-phase-7-ownership--maintenance-docs"></a>
## Entry 85: Phase 7 Ownership + Maintenance Docs

Scope:
- `docs/WHOLE_PROJECT_OWNERSHIP_MAP.md`
- `docs/WHOLE_PROJECT_MAINTENANCE_PLAYBOOK.md`
- `docs/WHOLE_PROJECT_REFACTOR_WORKFLOW.md`

Changes:
- Added explicit ownership map for runner/formal/mutation/simulation domains.
- Added maintenance playbook for baseline/schema/adapter updates.
- Updated workflow doc to require ownership/playbook consultation.

<a id="entry-87-phase-3-formal-declarative-manifests"></a>
## Entry 87: Phase 3 Declarative Formal Manifests

Scope:
- `docs/formal_manifests/README.md`
- `docs/formal_manifests/sv-tests.tsv`
- `docs/formal_manifests/verilator-verification.tsv`
- `docs/formal_manifests/yosys-tests.tsv`
- `docs/formal_manifests/opentitan-formal-targets.tsv`
- `docs/FormalRegression.md`
- `utils/validate_formal_manifests.py`

Changes:
- Added machine-readable declarative manifests for:
  - `sv-tests`
  - `verilator-verification`
  - `yosys/tests`
  - OpenTitan formal targets
- Added manifest schema documentation (`docs/formal_manifests/README.md`).
- Added validator utility (`utils/validate_formal_manifests.py`) with checks for:
  - header/schema correctness,
  - row arity,
  - duplicate `suite_id`,
  - valid profile tokens,
  - runner-path existence for non-`-` lanes.
- Updated `docs/FormalRegression.md` with manifest + validation usage.

Coverage:
- Added:
  - `test/Tools/validate-formal-manifests-pass.test`
  - `test/Tools/validate-formal-manifests-invalid-runner.test`

<a id="entry-88-phase-3-formal-shared-launch-retry-drift-helpers"></a>
## Entry 88: Phase 3 Shared Formal Launch/Retry/Drift Helpers

Scope:
- `utils/formal/lib/runner_common.py`
- `utils/run_opentitan_connectivity_circt_bmc.py`
- `utils/run_opentitan_connectivity_circt_lec.py`

Changes:
- Added shared formal helper module with reusable primitives for:
  - non-negative numeric parsing (`int`/`float`),
  - retryable exit-code parsing,
  - allowlist loading + matching (`exact|prefix|regex`),
  - logged command execution with bounded retry support,
  - status summary read/write helpers,
  - status drift TSV writer.
- Wired OpenTitan connectivity BMC wrapper to shared helpers for:
  - allowlist parse/match,
  - status summary read/write,
  - status drift writer.
- Wired OpenTitan connectivity LEC wrapper to shared helpers for:
  - allowlist parse/match,
  - status summary read/write,
  - status drift writer,
  - run/log command path with optional retry knobs:
    - `FORMAL_LAUNCH_RETRY_ATTEMPTS`
    - `FORMAL_LAUNCH_RETRY_BACKOFF_SECS`
    - `FORMAL_LAUNCH_RETRYABLE_EXIT_CODES`
    - `FORMAL_LAUNCH_RETRYABLE_PATTERNS`
- Preserved standalone compatibility for copied script fixtures by keeping local
  fallbacks when shared helper module is unavailable.

Coverage:
- Added:
  - `test/Tools/Inputs/formal_runner_common_retry.py`
  - `test/Tools/formal-runner-common-retry.test`

Validation highlights:
- Python compile checks: PASS
  - `python3 -m py_compile utils/formal/lib/runner_common.py utils/run_opentitan_connectivity_circt_bmc.py utils/run_opentitan_connectivity_circt_lec.py`
- Focused lit slice: PASS
  - `build-ot/bin/llvm-lit -sv test/Tools --filter='formal-runner-common-retry|run-opentitan-connectivity-circt-(bmc|lec)-status-(summary|drift-fail|drift-allowlist)|run-formal-all-opentitan-connectivity-(bmc|lec)-status-(drift-forwarding|baseline-update)'`

<a id="entry-89-phase-3-formal-wrapper-migration-complete"></a>
## Entry 89: Phase 3 Formal Wrapper Migration Completion

Scope:
- `utils/run_opentitan_circt_bmc.py`
- `utils/run_opentitan_circt_lec.py`
- `utils/run_opentitan_fpv_circt_bmc.py`
- `utils/run_opentitan_fpv_circt_lec.py`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Completed shared-helper migration across remaining OpenTitan formal wrappers:
  - OpenTitan BMC wrapper now consumes shared non-negative parsing helper.
  - OpenTitan LEC wrapper now consumes shared logged launch helper with
    optional retry knobs:
    - `FORMAL_LAUNCH_RETRY_ATTEMPTS`
    - `FORMAL_LAUNCH_RETRY_BACKOFF_SECS`
    - `FORMAL_LAUNCH_RETRYABLE_EXIT_CODES`
    - `FORMAL_LAUNCH_RETRYABLE_PATTERNS`
  - OpenTitan FPV LEC wrapper now consumes shared parse + logged launch/retry
    helpers using the same retry knob contract.
  - OpenTitan FPV BMC wrapper keeps shared parse/allowlist adoption from the
    previous slice; this completes wrapper coverage for the Phase 3 migration
    goal.
- Preserved standalone fixture compatibility by retaining local implementations
  and applying shared-helper overrides only when `utils/formal/lib` is present.
- Restored FPV LEC objective-reason contract to `projected_case_*` tokens for
  assertion/cover outputs.
- Marked Phase 3 TODO items as complete for both BMC and LEC wrapper migration.

Validation highlights:
- Python compile checks: PASS
  - `python3 -m py_compile utils/run_opentitan_fpv_circt_lec.py utils/run_opentitan_circt_lec.py utils/run_opentitan_circt_bmc.py`
- Focused OpenTitan wrapper lit slice: PASS
  - `build-ot/bin/llvm-lit -sv -j 1 test/Tools --filter='run-opentitan-fpv-circt-lec-(basic|failing-status)|run-opentitan-lec-(default-x-optimistic|x-optimistic|no-assume-known|diagnose-xprop|xprop-summary|xprop-fail-detail|error-diag|diag-fallback|mode-label|resolved-contracts-file|dump-unknown-sources|strict-auto-assume-known)|run-opentitan-bmc-(mode-label|case-policy-file|case-policy-invalid|case-policy-regex|case-policy-ambiguous-pattern|case-policy-extra-args|case-policy-extra-args-invalid|case-policy-provenance)'`
- Shared-helper regression slice: PASS
  - `build-ot/bin/llvm-lit -sv -j 1 test/Tools --filter='formal-runner-common-retry|run-opentitan-connectivity-circt-(bmc|lec)-status-(summary|drift-fail|drift-allowlist)|run-formal-all-opentitan-connectivity-(bmc|lec)-status-(drift-forwarding|baseline-update)'`

<a id="entry-90-phase-4-llhdprocessinterpreter-churn-map"></a>
## Entry 90: Phase 4 LLHDProcessInterpreter Churn Map

Scope:
- `docs/CIRCT_SIM_INTERPRETER_CHURN_MAP.md`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added a dedicated churn map for `tools/circt-sim/LLHDProcessInterpreter*`
  files with:
  - reproducible history method (`git log --since='2025-02-20'`),
  - per-file touch and line-churn snapshot,
  - high-churn zone mapping with file/line anchors for:
    - core dispatch/operation interpretation,
    - drive/store/update semantics,
    - wait/fork/wakeup scheduling paths,
    - call/call_indirect + UVM/DPI interception,
    - native thunk policy/execution coupling zones.
- Added extraction priority ordering derived from the churn map to guide the
  next Phase 4 subsystem splits.
- Marked Phase 4 TODO item complete for high-churn mapping.

Validation highlights:
- Churn snapshot commands executed successfully:
  - `git log --since='2025-02-20' --no-merges --name-only -- tools/circt-sim/LLHDProcessInterpreter*`
  - `git log --since='2025-02-20' --no-merges --numstat -- tools/circt-sim/LLHDProcessInterpreter*`

<a id="entry-91-phase-4-memory-model-subsystem-extraction-slice-1"></a>
## Entry 91: Phase 4 Memory Model Subsystem Extraction (Slice 1)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterMemory.cpp`
- `tools/circt-sim/CMakeLists.txt`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added new source file `LLHDProcessInterpreterMemory.cpp` and moved core LLVM
  memory handlers out of the monolithic interpreter file:
  - `interpretLLVMAlloca`
  - `interpretLLVMLoad`
  - `interpretLLVMStore`
  - `interpretLLVMGEP`
- Registered `LLHDProcessInterpreterMemory.cpp` in the `circt-sim` tool target
  CMake source list.
- Preserved behavior by keeping helper semantics unchanged, including:
  - `safeInsertBits` clamping behavior for four-state layout conversions,
  - native block registration from `{ptr, i64}` loaded aggregates.
- Marked Phase 4 memory-model extraction TODO as in progress (`[~]`) because
  this slice extracts primary handlers while follow-up cleanup can still move
  additional memory-support helpers.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused `circt-sim` memory-op lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='llvm-(assoc-native-ref-load-store|struct-load-store|load-unknown-native|alloca-ref-probe-drive)|store-load-address-fallback|native-store-oob-fallback|jit-process-thunk-(alloca-gep-load-store-halt|fork-branch-alloca-gep-load-store-terminator)|llhd-sig-extract-gep-backed'`

<a id="entry-92-phase-4-drive-update-subsystem-extraction-slice-1"></a>
## Entry 92: Phase 4 Drive/Update Subsystem Extraction (Slice 1)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterDrive.cpp`
- `tools/circt-sim/CMakeLists.txt`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added new source file `LLHDProcessInterpreterDrive.cpp` and moved core
  drive/update handlers out of the monolithic interpreter file:
  - `normalizeImplicitZDriveStrength`
  - `resolveTriStateDriveSourceFieldSignal`
  - `tryEvaluateTriStateDestDriveValue`
  - `executeContinuousAssignment`
  - `detectArrayElementDrive`
- Registered `LLHDProcessInterpreterDrive.cpp` in the `circt-sim` CMake source
  list.
- Kept behavior unchanged by preserving:
  - tri-state source-field cache keying,
  - disabled-driver high-Z/X release behavior,
  - strength-aware distinct continuous-driver IDs.
- Removed now-unused tri-state cache-key helper from
  `LLHDProcessInterpreter.cpp` after extraction.
- Marked Phase 4 drive/update extraction TODO as in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused drive/tri-state lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='interface-tristate-(passive-observe-vif|signalcopy-redirect|suppression-cond-false)|interface-inout-(shared-wire-bidirectional|tristate-propagation)|interface-intra-tristate-propagation|llhd-drv-(array-get-struct-field-offset|memory-backed-struct-array-func-arg|sig-extract-oob-noop)|llhd-prb-subfield-pending-epsilon|sig-extract-struct-array-bit-memory-layout|llhd-ref-cast-(array-subfield-store-func-arg|subfield-store-func-arg)'`

<a id="entry-93-phase-4-call-indirect-subsystem-extraction-slice-1"></a>
## Entry 93: Phase 4 Call/Call-Indirect Subsystem Extraction (Slice 1)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Moved shared call-interception helper methods from the monolithic interpreter
  file into `LLHDProcessInterpreterCallIndirect.cpp`:
  - `tryReadStringKey`
  - `readMooreStringStruct(ProcessId, InterpretedValue)`
  - `readMooreStringStruct(ProcessId, Value)`
  - `tryInterceptConfigDbCallIndirect`
- Kept call/call_indirect behavior unchanged by preserving:
  - dynamic string + memory/native lookup order for string decoding,
  - config_db set/get wildcard/fuzzy key matching,
  - config_db write-back behavior for both ref-backed and pointer-backed
    outputs.
- Marked Phase 4 call/call_indirect extraction TODO as in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + config_db/vtable lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-94-phase-4-call-indirect-subsystem-extraction-slice-2"></a>
## Entry 94: Phase 4 Call/Call-Indirect Subsystem Extraction (Slice 2)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`

Changes:
- Added a dedicated call-site helper declaration:
  - `tryInterceptUvmPortCall(ProcessId, StringRef, LLVM::CallOp)`
- Implemented the helper in `LLHDProcessInterpreterCallIndirect.cpp` and moved
  the `llvm.call` UVM-port interception block into it:
  - `uvm_port_base::connect`
  - `uvm_port_base::size` (native connection graph path)
  - `uvm_port_base::resolve_bindings` no-op interception
- Replaced duplicated inline logic in
  `LLHDProcessInterpreter::interpretLLVMCall` with a single helper invocation.
- Preserved behavior by keeping the same address canonicalization/cache
  invalidation and result encoding semantics.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + config_db/vtable/uvm-port lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-95-phase-4-uvm-adapter-interceptor-extraction-slice-1"></a>
## Entry 95: Phase 4 UVM Adapter/Interceptor Extraction (Slice 1)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterUvm.cpp`
- `tools/circt-sim/CMakeLists.txt`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added `LLHDProcessInterpreterUvm.cpp` and moved a bounded UVM
  adapter/cache helper set out of the monolithic interpreter file:
  - `lookupUvmSequencerQueueCache`
  - `cacheUvmSequencerQueueAddress`
  - `invalidateUvmSequencerQueueCache`
  - `canonicalizeUvmObjectAddress`
  - `seedAnalysisPortConnectionWorklist`
- Registered `LLHDProcessInterpreterUvm.cpp` in the `circt-sim` CMake source
  list.
- Marked Phase 4 UVM adapter/interceptor extraction TODO as in progress (`[~]`).
- Kept behavior unchanged by preserving cache hit/miss accounting, canonical
  address mapping, and analysis-port worklist seeding logic.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused UVM sequencer/port + config_db lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|uvm-printer-fast-path-call-indirect|config-db-(dual-top|native-call-indirect-writeback|native-call-indirect-writeback-offset)'`

<a id="entry-96-phase-4-call-indirect-subsystem-extraction-slice-3"></a>
## Entry 96: Phase 4 Call/Call-Indirect Subsystem Extraction (Slice 3)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Moved additional call-interception-adjacent helper methods from the monolithic
  interpreter file into `LLHDProcessInterpreterCallIndirect.cpp`:
  - `readObjectVTableAddress`
  - `isAhbMonitorSampleFunctionForTrace`
  - `readRawMemoryBytesForTrace`
  - `traceAhbTxnPayload`
- Kept behavior unchanged by preserving:
  - vtable pointer decoding from interpreter and native memory,
  - AHB trace env gating (`CIRCT_SIM_TRACE_AHB_TXN_*`),
  - trace payload byte formatting and bounded print behavior.
- Kept the Phase 4 call/call_indirect TODO item in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-102-phase-4-tracing-diagnostics-subsystem-extraction-slice-5"></a>
## Entry 102: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 5)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared tracing helpers in `LLHDProcessInterpreterTrace.cpp` and wired
  them from fork/join handlers:
  - `maybeTraceJoinNoneCheck`
  - `maybeTraceJoinNoneWait`
  - `maybeTraceJoinNoneResume`
  - `maybeTraceForkTerminator`
  - `maybeTraceJoinAnyImmediate`
- Replaced inlined fork/join diagnostic print blocks in
  `LLHDProcessInterpreter.cpp` with the shared helpers.
- Kept behavior unchanged by preserving:
  - `traceForkJoinEnabled` gating and poll-threshold limits,
  - fork terminator and join-any immediate diagnostic limits,
  - existing diagnostic message formats/fields.
- Simplified `traceI3CForkRuntimeEvent` state-name formatting by using
  `getProcessStateName(...)` directly (no behavior change).
- Recorded the slice in TODO by checking:
  - `Extract fork/join diagnostic emission helpers into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-103-phase-4-tracing-diagnostics-subsystem-extraction-slice-6"></a>
## Entry 103: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 6)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared disable-fork diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceDisableForkBegin`
  - `maybeTraceDisableForkDeferredPoll`
  - `maybeTraceDisableForkDeferredArm`
  - `maybeTraceDisableForkChild`
- Replaced inlined diagnostic formatting in:
  - `fireDeferredDisableFork`
  - `interpretSimDisableFork`
  with the shared tracing helpers.
- Kept behavior unchanged by preserving:
  - `CIRCT_SIM_TRACE_DISABLE_FORK` env-gated emission,
  - deferred/immediate mode labels and diagnostic field formats,
  - existing `traceI3CForkRuntimeEvent` side-traces.
- Recorded the slice in TODO by checking:
  - `Extract disable_fork diagnostic emission helpers into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-104-phase-4-tracing-diagnostics-subsystem-extraction-slice-7"></a>
## Entry 104: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 7)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared fork-trace formatting helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceForkIntercept`
  - `maybeTraceForkInterceptObjectionWait`
  - `maybeTraceForkCreate`
  - `maybeTraceJoinNoneYield`
- Replaced inlined diagnostics in `interpretSimFork` with shared helpers for:
  - `FORK-INTERCEPT` shape/intercept trace lines,
  - `FORK-INTERCEPT` objection-zero wait trace lines,
  - `FORK-CREATE` launch trace lines,
  - `JOIN-NONE-YIELD` defer trace lines.
- Kept behavior unchanged by preserving:
  - `traceForkJoinEnabled` gating,
  - existing field names/formatting (including `phase=0x...` formatting),
  - existing join type/function/name attribution values.
- Recorded the slice in TODO by checking:
  - `Extract fork intercept/create diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-105-phase-4-tracing-diagnostics-subsystem-extraction-slice-8"></a>
## Entry 105: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 8)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared process-finalize diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceProcessFinalize`
- Replaced inline `[PROC-FINALIZE]` tracing block in `finalizeProcess` with
  the shared helper call.
- Kept behavior unchanged by preserving:
  - `CIRCT_SIM_TRACE_FINALIZE` env-gated emission,
  - output fields/format (`proc`, `name`, `killed`, `sched_state`, `func`),
  - scheduler/process-state lookup semantics.
- Recorded the slice in TODO by checking:
  - `Extract process-finalize diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-106-phase-4-tracing-diagnostics-subsystem-extraction-slice-9"></a>
## Entry 106: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 9)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared wait-sensitivity diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceWaitSensitivityList`
- Replaced inlined `CIRCT_SIM_TRACE_WAIT_SENS`/`[WAIT-SENS]` lambda
  diagnostics in `interpretWait` with shared helper calls for:
  - observed wait lists,
  - inferred wait lists.
- Kept behavior unchanged by preserving:
  - `CIRCT_SIM_TRACE_WAIT_SENS` env-gated emission,
  - per-entry signal/edge formatting and process name lookup,
  - call sites/tags (`observed`, `inferred`) and scheduling behavior.
- Recorded the slice in TODO by checking:
  - `Extract wait-sensitivity diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-107-phase-4-tracing-diagnostics-subsystem-extraction-slice-10"></a>
## Entry 107: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 10)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared wait-event tracing helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceWaitEventCache`
  - `maybeTraceWaitEventNoop`
- Replaced inlined `CIRCT_SIM_TRACE_WAIT_EVENT_CACHE` and
  `CIRCT_SIM_TRACE_WAIT_EVENT_NOOP` diagnostic emission in
  `interpretMooreWaitEvent` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_WAIT_EVENT_CACHE`,
    `CIRCT_SIM_TRACE_WAIT_EVENT_NOOP`),
  - trace formats/tags (`[WAIT-EVENT-CACHE] hit/store`,
    `[WAIT-EVENT-NOOP]`),
  - timestamp/op-pointer/list formatting and scheduler interactions.
- Recorded the slice in TODO by checking:
  - `Extract wait-event cache/no-op diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body'`

<a id="entry-108-phase-4-tracing-diagnostics-subsystem-extraction-slice-11"></a>
## Entry 108: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 11)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared call-indirect site-cache tracing helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceCallIndirectSiteCacheHit`
  - `maybeTraceCallIndirectSiteCacheStore`
- Replaced inlined `[CI-SITE-CACHE]` emission in
  `getCachedCallIndirectStaticMethodIndex` with shared helper calls.
- Kept behavior unchanged by preserving:
  - `traceCallIndirectSiteCacheEnabled` gating,
  - hit/store diagnostics format and static/dynamic method-index labeling,
  - call-site cache state transitions and method-index resolution flow.
- Recorded the slice in TODO by checking:
  - `Extract call-indirect site-cache diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call-indirect/vtable lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-printer-fast-path-call-indirect|config-db-native-call-indirect-writeback(-offset)?'`

<a id="entry-109-phase-4-tracing-diagnostics-subsystem-extraction-slice-12"></a>
## Entry 109: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 12)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface-sensitivity tracing helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceSensitivityBegin`
  - `maybeTraceInterfaceSensitivitySource`
  - `maybeTraceInterfaceSensitivityAddedField`
- Replaced inlined `CIRCT_SIM_TRACE_IFACE_SENS` diagnostics in
  `expandDeferredInterfaceSensitivityExpansions` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_IFACE_SENS`),
  - diagnostics format (`[IFACE-SENS]` proc/source/field lines),
  - sensitivity expansion and scheduling behavior.
- Recorded the slice in TODO by checking:
  - `Extract interface-sensitivity diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-110-phase-4-tracing-diagnostics-subsystem-extraction-slice-13"></a>
## Entry 110: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 13)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface-overwrite diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `traceInterfaceSignalOverwrite`
- Replaced inlined `[IFACE-OVERWRITE]` emission in interface-field registration
  logic with the shared helper call.
- Kept behavior unchanged by preserving:
  - unconditional overwrite diagnostic emission,
  - output format/fields (`addr`, `old sig`, `new sig`, field name),
  - interface field signal map update flow.
- Recorded the slice in TODO by checking:
  - `Extract interface-overwrite diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-111-phase-4-tracing-diagnostics-subsystem-extraction-slice-14"></a>
## Entry 111: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 14)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared multi-driver post-NBA diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceMultiDriverPostNbaConditionalSignals`
  - `maybeTraceMultiDriverPostNbaApply`
  - `maybeTraceMultiDriverPostNbaSkipFirReg`
- Replaced inlined `CIRCT_SIM_TRACE_MULTI_DRV`/`[MULTI-DRV-POST-NBA]`
  emission in the post-NBA callback with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_MULTI_DRV`),
  - diagnostics format (`conditional signals`, `applying`, `SKIPPED`),
  - post-NBA update ordering and firreg suppression behavior.
- Recorded the slice in TODO by checking:
  - `Extract multi-driver post-NBA diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-112-phase-4-tracing-diagnostics-subsystem-extraction-slice-15"></a>
## Entry 112: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 15)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared module-drive diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceExecuteModuleDrives`
  - `maybeTraceModuleDriveTrigger`
- Replaced inlined `CIRCT_SIM_TRACE_MOD_DRV` diagnostics in:
  - `executeModuleDrives` (`[EXEC-MOD-DRV]`)
  - `executeModuleDrivesForSignal` (`[MOD-DRV]` trigger tracing)
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_MOD_DRV`),
  - output formats/fields (`t`, `d`, `proc`, trigger/destination ids + names),
  - existing destination-signal filter (`dstSigId == 9 || dstSigId == 10`).
- Recorded the slice in TODO by checking:
  - `Extract module-drive execution/trigger diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-113-phase-4-tracing-diagnostics-subsystem-extraction-slice-16"></a>
## Entry 113: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 16)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared module-drive multi-driver diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceMultiDriverSuppressedUnconditional`
  - `maybeTraceMultiDriverConditionalEnable`
  - `maybeTraceMultiDriverStoredConditional`
- Replaced inlined `CIRCT_SIM_TRACE_MULTI_DRV`/`[MULTI-DRV]` emission in
  `executeModuleDrives` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_MULTI_DRV`),
  - existing signal filter for detailed traces (`sigId == 7`),
  - output fields/format (`enable`, time, width, isX, nonzero, hex payload).
- Recorded the slice in TODO by checking:
  - `Extract module-drive multi-driver diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-114-phase-4-tracing-diagnostics-subsystem-extraction-slice-17"></a>
## Entry 114: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 17)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface-propagation diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfacePropagationSource`
  - `maybeTraceInterfacePropagationChild`
- Replaced inlined `CIRCT_SIM_TRACE_IFACE_PROP`/`[IFACE-PROP]` emission in
  `forwardPropagateOnSignalChange` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_IFACE_PROP`),
  - source/child diagnostics fields (signal id/name, value text, fanout, time),
  - propagation/update ordering and memory-shadow synchronization behavior.
- Recorded the slice in TODO by checking:
  - `Extract interface-propagation diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-115-phase-4-tracing-diagnostics-subsystem-extraction-slice-18"></a>
## Entry 115: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 18)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface tri-state rule diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceTriStateTrigger`
  - `maybeTraceInterfaceTriStateRule`
- Replaced inlined `CIRCT_SIM_TRACE_INTERFACE_TRISTATE`/`[TRI-RULE]`
  emission in `applyInterfaceTriStateRules` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_INTERFACE_TRISTATE`),
  - trigger/rule diagnostics fields and formatting,
  - rule evaluation and propagation behavior.
- Recorded the slice in TODO by checking:
  - `Extract interface tri-state rule diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-116-phase-4-tracing-diagnostics-subsystem-extraction-slice-19"></a>
## Entry 116: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 19)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared conditional-branch diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceCondBranch`
- Replaced inlined `CIRCT_SIM_TRACE_CONDBR`/`[CONDBR]` emission in
  `mlir::cf::CondBranchOp` handling with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_CONDBR`),
  - true/false/X condition labeling and branch destination fields,
  - defining-op + operand trace formatting.
- Recorded the slice in TODO by checking:
  - `Extract conditional-branch diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-117-phase-4-tracing-diagnostics-subsystem-extraction-slice-20"></a>
## Entry 117: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 20)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared firreg update diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceFirRegUpdate`
- Replaced inlined `CIRCT_SIM_TRACE_FIRREG`/`[FIRREG]` emission in firreg
  update handling with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_FIRREG`),
  - signal id/name/value/posedge/time fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract firreg update diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-118-phase-4-tracing-diagnostics-subsystem-extraction-slice-21"></a>
## Entry 118: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 21)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared instance-output diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInstanceOutput`
  - `maybeTraceInstanceOutputUpdate`
- Replaced inlined `CIRCT_SIM_TRACE_INST_OUT`/`[INST-OUT]` and
  `CIRCT_SIM_TRACE_INST_OUT_UPDATE`/`[INST-OUT-UPD]` emission in
  `LLHDProcessInterpreter.cpp` with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_INST_OUT`,
    `CIRCT_SIM_TRACE_INST_OUT_UPDATE`),
  - signal/source/process count fields and formatting,
  - update value/time diagnostics formatting.
- Recorded the slice in TODO by checking:
  - `Extract instance-output diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-119-phase-4-tracing-diagnostics-subsystem-extraction-slice-22"></a>
## Entry 119: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 22)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared combinational trace-through diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceCombTraceThroughHit`
  - `maybeTraceCombTraceThroughMiss`
- Replaced inlined `CIRCT_SIM_TRACE_COMB_TT`/`[COMB-TT]` and
  `[COMB-TT-MISS]` emission in `evaluateCombinationalOp` with shared helper
  calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_COMB_TT`),
  - signal name/value/time fields and formatting on hit,
  - `inCombMap`/`visited`/time fields and formatting on miss.
- Recorded the slice in TODO by checking:
  - `Extract combinational trace-through diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-120-phase-4-tracing-diagnostics-subsystem-extraction-slice-23"></a>
## Entry 120: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 23)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared continuous-fallback diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceContinuousFallback`
- Replaced inlined `CIRCT_SIM_TRACE_CONT_FALLBACK`/`[CONT-FALLBACK]` emission
  in `evaluateContinuousValue` fallback evaluation with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_CONT_FALLBACK`),
  - value-kind and invalidated-count fields/formatting.
- Recorded the slice in TODO by checking:
  - `Extract continuous fallback diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-121-phase-4-tracing-diagnostics-subsystem-extraction-slice-24"></a>
## Entry 121: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 24)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared drive-schedule diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceDriveSchedule`
- Replaced inlined `CIRCT_SIM_TRACE_DRIVE_SCHEDULE`/`[DRV-SCHED]` emission
  in `interpretDrive` with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_DRIVE_SCHEDULE`),
  - signal/value/now/target/delay fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract drive schedule diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-122-phase-4-tracing-diagnostics-subsystem-extraction-slice-25"></a>
## Entry 122: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 25)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared array-drive remap diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceArrayDriveRemap`
- Replaced inlined `CIRCT_SIM_TRACE_ARRAY_DRIVE`/`[ARRAY-DRV]` emission
  in `interpretDrive` with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_ARRAY_DRIVE`),
  - remap-kind/signal-id/parent-id/index fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract array-drive remap diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-123-phase-4-tracing-diagnostics-subsystem-extraction-slice-26"></a>
## Entry 123: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 26)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared drive-failure diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceDriveFailure`
- Replaced local `emitDriveFailure` lambda usage in `interpretDrive` with
  shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_DRIVE_FAILURE`),
  - proc/name/function/reason/signal/location fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract drive failure diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-124-phase-4-tracing-diagnostics-subsystem-extraction-slice-27"></a>
## Entry 124: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 27)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared array-drive scheduling diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceArrayDriveSchedule`
- Replaced inlined `CIRCT_SIM_TRACE_ARRAY_DRIVE`/`[ARRAY-DRV-SCHED]` emission
  in `interpretDrive` with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_ARRAY_DRIVE`),
  - signal/index/offset/width/value/delay fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract array-drive scheduling diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-125-phase-4-tracing-diagnostics-subsystem-extraction-slice-28"></a>
## Entry 125: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 28)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C address-bit diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CAddressBitDrive`
- Replaced inlined `CIRCT_SIM_TRACE_I3C_ADDR_BITS`/`[I3C-ADDR-BIT]` emission
  in `interpretDrive` with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_I3C_ADDR_BITS`),
  - function-name filter (`i3c_target_driver_bfm::sample_target_address`),
  - bit/value/time/delta fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C address-bit diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-126-phase-4-tracing-diagnostics-subsystem-extraction-slice-29"></a>
## Entry 126: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 29)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C ref-cast diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CRefCast`
- Replaced inlined `CIRCT_SIM_TRACE_I3C_REF_CASTS`/`[I3C-REF-CAST]` emission
  in `interpretCall` with shared helper calls for both resolved and unresolved
  cast paths.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_I3C_REF_CASTS`),
  - process/name/function/resolution/address/bitOffset/in/out fields and
    formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C ref-cast diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-127-phase-4-tracing-diagnostics-subsystem-extraction-slice-30"></a>
## Entry 127: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 30)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C cast-layout diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CCastLayout`
- Replaced inlined `CIRCT_SIM_TRACE_I3C_CAST_LAYOUT`/`[I3C-CAST]` emission
  in `interpretCall` layout-conversion handling with a shared helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_I3C_CAST_LAYOUT`),
  - I3C transfer-struct type guard,
  - input/output field extraction and diagnostic formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C cast-layout diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-128-phase-4-tracing-diagnostics-subsystem-extraction-slice-31"></a>
## Entry 128: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 31)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared ref-arg resolve diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceRefArgResolveFailure`
- Replaced inlined `CIRCT_SIM_TRACE_REF_ARG_RESOLVE`/`[REF-RESOLVE]` emission
  in `interpretDrive` unresolved parent-signal handling with a shared helper
  call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_REF_ARG_RESOLVE`),
  - unresolved value and optional block-arg source diagnostics formatting.
- Recorded the slice in TODO by checking:
  - `Extract ref-arg resolve diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-129-phase-4-tracing-diagnostics-subsystem-extraction-slice-32"></a>
## Entry 129: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 32)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C field-drive signal-struct diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CFieldDriveSignalStruct`
- Replaced inlined `CIRCT_SIM_TRACE_I3C_FIELD_DRIVES`/
  `[I3C-FIELD-DRV-SIGNAL-STRUCT]` emission in `interpretDrive` with a shared
  helper call.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_I3C_FIELD_DRIVES`),
  - field-name filter for tracked I3C transfer fields,
  - process/function/field/offset/width/value/signal/time formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C field-drive signal-struct diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-130-phase-4-tracing-diagnostics-subsystem-extraction-slice-33"></a>
## Entry 130: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 33)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C field-drive array/mem diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CFieldDriveMem`
  - `maybeTraceI3CFieldDrive`
- Replaced inlined `CIRCT_SIM_TRACE_I3C_FIELD_DRIVES` emissions in
  `interpretDrive` for:
  - `[I3C-FIELD-DRV-MEM]`
  - `[I3C-FIELD-DRV]`
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - env-gated emission (`CIRCT_SIM_TRACE_I3C_FIELD_DRIVES`),
  - tracked field-name filter for the `[I3C-FIELD-DRV]` path
    (`writeData|readData|writeDataStatus|readDataStatus`),
  - process/function/index/offset/value/time/delta formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C field-drive array/mem diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-131-phase-4-tracing-diagnostics-subsystem-extraction-slice-34"></a>
## Entry 131: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 34)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C config-handle diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CConfigHandleGet`
  - `maybeTraceI3CConfigHandleSet`
- Replaced inlined `[I3C-CFG]` emission in `interpretFuncCall` for both:
  - `get` path (`config_db` read/writeback intercept),
  - `set` path (`config_db` store intercept),
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `traceI3CConfigHandles` env-gated call-site guard,
  - existing `fieldName.contains(\"i3c_\")` filter,
  - `callee/key/value_ptr/out_ref/field` formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C config-handle diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-132-phase-4-tracing-diagnostics-subsystem-extraction-slice-35"></a>
## Entry 132: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 35)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C handle/call-stack diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CHandleCall`
  - `maybeTraceI3CCallStackSave`
- Replaced inlined diagnostics in `LLHDProcessInterpreter.cpp` for:
  - `[I3C-HANDLE]` emission in `interpretFuncCall`,
  - `[I3C-CS-SAVE]` emission when pushing saved call frames.
- Kept behavior unchanged by preserving:
  - existing call-site env/target filters
    (`traceI3CConfigHandles`, `traceI3CCallStackSave`, callee/function-name
    match predicates),
  - existing `[I3C-HANDLE]` print cap (`512`),
  - process/callee/arg/time and saved-frame formatting.
- Recorded the slice in TODO by checking:
  - `Extract I3C handle-call diagnostics into tracing source`.
  - `Extract I3C call-stack-save diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-133-phase-4-tracing-diagnostics-subsystem-extraction-slice-36"></a>
## Entry 133: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 36)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared I3C to-class diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceI3CToClassArgs`
- Replaced inlined `[I3C-TO-CLASS]` emission in `interpretFuncCall` with a
  shared helper call.
- Kept behavior unchanged by preserving:
  - existing env-gated call-site guard (`CIRCT_SIM_TRACE_I3C_TO_CLASS_ARGS`),
  - existing call-target filter (`to_class` / `to_class_*`),
  - existing argument decode and output formatting
    (caller func/block info, low/hw field decodes, optional out_ref).
- Recorded the slice in TODO by checking:
  - `Extract I3C to-class argument diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-134-phase-4-tracing-diagnostics-subsystem-extraction-slice-37"></a>
## Entry 134: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 37)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared config-db func.call diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceConfigDbFuncCallGetBegin`
  - `maybeTraceConfigDbFuncCallGetHit`
  - `maybeTraceConfigDbFuncCallGetMiss`
  - `maybeTraceConfigDbFuncCallSet`
- Replaced inlined diagnostics in `interpretFuncCall` for:
  - `[CFG-FC-GET]` begin/hit/miss emissions,
  - `[CFG-FC-SET]` emissions,
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing call-site guard (`traceConfigDbEnabled`),
  - existing wildcard/fuzzy key resolution and intercept flow,
  - existing output field formatting.
- Recorded the slice in TODO by checking:
  - `Extract config-db func.call diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-135-phase-4-tracing-diagnostics-subsystem-extraction-slice-38"></a>
## Entry 135: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 38)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared sequencer func.call diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceSequencerFuncCallSize`
  - `maybeTraceSequencerFuncCallGet`
  - `maybeTraceSequencerFuncCallPop`
  - `maybeTraceSequencerFuncCallTryMiss`
  - `maybeTraceSequencerFuncCallWait`
  - `maybeTraceSequencerFuncCallItemDoneMiss`
- Replaced inlined diagnostics in `interpretFuncCall` for:
  - `[SEQ-SIZE]` and `[SEQ-FC]` get/pop/try-miss/wait/item_done-miss emissions,
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing call-site guard (`traceSeq`),
  - existing sequencer queue/port resolution control flow,
  - existing field formatting (hex widths and fallback flag).
- Recorded the slice in TODO by checking:
  - `Extract sequencer func.call diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-136-phase-4-tracing-diagnostics-subsystem-extraction-slice-39"></a>
## Entry 136: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 39)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared analysis-write func.call diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceAnalysisWriteFuncCallBegin`
  - `maybeTraceAnalysisWriteFuncCallTerminals`
  - `maybeTraceAnalysisWriteFuncCallMissingVtableHeader`
  - `maybeTraceAnalysisWriteFuncCallMissingAddressToGlobal`
  - `maybeTraceAnalysisWriteFuncCallMissingAddressToFunction`
  - `maybeTraceAnalysisWriteFuncCallMissingModuleFunction`
  - `maybeTraceAnalysisWriteFuncCallDispatch`
- Replaced inlined `[ANALYSIS-WRITE-FC]` emissions in `interpretFuncCall` with
  shared helper calls.
- Kept behavior unchanged by preserving:
  - existing call-site guard (`traceAnalysisEnabled`),
  - existing analysis-port fanout traversal and dispatch flow,
  - existing field formatting.
- Recorded the slice in TODO by checking:
  - `Extract analysis write func.call diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-137-phase-4-tracing-diagnostics-subsystem-extraction-slice-40"></a>
## Entry 137: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 40)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared UVM run-test entry diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceUvmRunTestEntry`
- Replaced inlined `[UVM-RUN-TEST]` emission in
  `checkUvmRunTestEntry` with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing call-site guard (`traceUvmRunTestEnabled`),
  - existing process/state fields and formatting,
  - existing run-test enforcement/error behavior.
- Recorded the slice in TODO by checking:
  - `Extract UVM run_test entry diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-138-phase-4-tracing-diagnostics-subsystem-extraction-slice-41"></a>
## Entry 138: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 41)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared get-name loop diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceGetNameLoop`
- Replaced inlined `[GETNAME-LOOP]` emission in `interpretFuncCall` with a
  shared helper call.
- Kept behavior unchanged by preserving:
  - existing call-site guard (`traceGetNameLoop`) and emit throttling logic,
  - existing fields and formatting,
  - existing local streak/sequence accounting semantics.
- Recorded the slice in TODO by checking:
  - `Extract get_name loop diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-139-phase-4-tracing-diagnostics-subsystem-extraction-slice-42"></a>
## Entry 139: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 42)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared baud fast-path diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceBaudFastPathReject`
  - `maybeTraceBaudFastPathNullSelfStall`
  - `maybeTraceBaudFastPathHit`
- Replaced inlined `[BAUD-FP]` reject/null-self-stall/hit emissions in
  `handleBaudClkGeneratorFastPath` with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing fast-path guard and rejection flow,
  - existing null-self-stall gating behavior,
  - existing hit payload fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract baud fast-path reject/hit/null-self-stall diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-140-phase-4-tracing-diagnostics-subsystem-extraction-slice-43"></a>
## Entry 140: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 43)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared baud fast-path diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceBaudFastPathGep`
  - `maybeTraceBaudFastPathMissingFields`
- Replaced inlined `[BAUD-FP]` gep/missing-fields emissions in
  `handleBaudClkGeneratorFastPath` with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `traceBaudFastPath` + `gepSeen <= 20` gating,
  - existing raw GEP index formatting and field names,
  - existing missing-field rejection flow.
- Recorded the slice in TODO by checking:
  - `Extract baud fast-path gep/missing-fields diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-141-phase-4-tracing-diagnostics-subsystem-extraction-slice-44"></a>
## Entry 141: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 44)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared baud delay-batching diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceBaudFastPathBatchParityAdjust`
  - `maybeTraceBaudFastPathBatchMismatch`
  - `maybeTraceBaudFastPathBatchSchedule`
- Replaced inlined `[BAUD-FP]` batch-parity-adjust, batch-mismatch, and
  batch-schedule emissions in `handleBaudClkGeneratorFastPath` with shared
  helper calls.
- Kept behavior unchanged by preserving:
  - existing `traceBaudFastPath` guard semantics,
  - existing per-event emission caps (`< 50`) via helper-local static counters,
  - existing payload fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract baud fast-path batch-parity/batch-mismatch/batch-schedule diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-142-phase-4-tracing-diagnostics-subsystem-extraction-slice-45"></a>
## Entry 142: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 45)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared LLVM get-name loop diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceGetNameLoopLLVM`
- Replaced inlined `[GETNAME-LLVM]` emission in `interpretLLVMCall` with a
  shared helper call.
- Kept behavior unchanged by preserving:
  - existing `traceGetNameLoopLLVM` guard and emit throttling conditions,
  - existing process-name lookup behavior,
  - existing payload fields and formatting.
- Recorded the slice in TODO by checking:
  - `Extract get_name LLVM loop diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-143-phase-4-tracing-diagnostics-subsystem-extraction-slice-46"></a>
## Entry 143: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 46)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared function-cache diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceFuncCacheSharedHit`
  - `maybeTraceFuncCacheSharedStore`
- Replaced inlined `[FUNC-CACHE]` shared hit/store emissions in:
  - `interpretFuncCall`
  - `interpretFuncBody`
  with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `CIRCT_SIM_TRACE_FUNC_CACHE` enable semantics,
  - existing emitted fields and formatting (`func`, `arg_hash`),
  - existing cache hit/store control flow.
- Recorded the slice in TODO by checking:
  - `Extract function-cache shared hit/store diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-144-phase-4-tracing-diagnostics-subsystem-extraction-slice-47"></a>
## Entry 144: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 47)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared phase-order diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTracePhaseOrderProcessPhase`
  - `maybeTracePhaseOrderProcessPhaseWaitPred`
  - `maybeTracePhaseOrderProcessPhaseWaitUnknownImp`
  - `maybeTracePhaseOrderFinishPhase`
  - `maybeTracePhaseOrderWakeWaiter`
  - `maybeTracePhaseOrderExecutePhase`
- Replaced inlined `[PHASE-ORDER]` emissions in:
  - `uvm_phase_hopper::process_phase` handling,
  - `uvm_phase_hopper::finish_phase` handling,
  - `uvm_phase_hopper::execute_phase` handling.
- Kept behavior unchanged by preserving:
  - existing `tracePhaseOrderEnabled()` gate conditions,
  - existing field formatting and values,
  - existing wake/scheduling control flow.
- Recorded the slice in TODO by checking:
  - `Extract phase-order diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-145-phase-4-tracing-diagnostics-subsystem-extraction-slice-48"></a>
## Entry 145: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 48)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared mailbox diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceMailboxTryPut`
  - `maybeTraceMailboxWakeGetByTryPut`
  - `maybeTraceMailboxGet`
- Replaced inlined mailbox diagnostics in `interpretFuncCall` with shared
  helper calls for:
  - `[MAILBOX-TRYPUT]`
  - `[MAILBOX-WAKE-GET]` (tryput wake path)
  - `[MAILBOX-GET]` (immediate and block modes)
- Kept behavior unchanged by preserving:
  - existing `traceMailbox` gating at callsites,
  - existing field formatting and mode labels,
  - existing mailbox wake/scheduling behavior.
- Recorded the slice in TODO by checking:
  - `Extract mailbox tryput/get diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-146-phase-4-tracing-diagnostics-subsystem-extraction-slice-49"></a>
## Entry 146: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 49)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared randomize diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceRandClassSrandom`
  - `maybeTraceRandBasic`
  - `maybeTraceRandBytes`
  - `maybeTraceRandRange`
- Replaced inlined `[RAND]` emissions in `interpretFuncCall` with shared
  helper calls for:
  - class RNG seeding (`class_srandom`),
  - basic randomize summary,
  - byte-range randomize summary,
  - ranged randomize summary.
- Kept behavior unchanged by preserving:
  - existing `traceRandomize` gating at callsites,
  - existing field formatting and values,
  - existing randomization control flow and result writes.
- Recorded the slice in TODO by checking:
  - `Extract randomize diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused wait-event/interface/fork lit slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`

<a id="entry-147-phase-4-tracing-diagnostics-subsystem-extraction-slice-50"></a>
## Entry 147: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 50)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared tail-wrapper fast-path diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceTailWrapperSuspendElide`
- Replaced inlined suspend-elide diagnostics in `interpretFuncBody` with a
  shared helper call for:
  - `[MON-DESER-FP]`
  - `[DRV-SAMPLE-FP]`
  - `[TAIL-WRAP-FP]`
- Kept behavior unchanged by preserving:
  - existing env-gating semantics for all three fast-path trace knobs,
  - existing global emission cap (`< 100`),
  - existing payload fields (`proc`, `wrapper`, `callee`).
- Recorded the slice in TODO by checking:
  - `Extract tail-wrapper fast-path diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Existing `.sv`-heavy focused slice: FAIL (unrelated parser/front-end breakage in current dirty tree)
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='moore-wait-event|wait-event-class-member|fork-wait-event|moore-wait-event-sensitivity-cache|disable-fork-halt|fork-disable-defer-poll|i3c-samplewrite-disable-fork-ordering|i3c-samplewrite-joinnone-disable-fork-ordering|fork-join-wait|fork-join-none-nested-join-body|interface-tristate-passive-observe-vif|interface-tristate-signalcopy-redirect|interface-inout-shared-wire-bidirectional|interface-tristate-suppression-cond-false|interface-inout-tristate-propagation|interface-intra-tristate-propagation'`
  - failing tests:
    - `CIRCT :: Tools/circt-sim/fork-disable-defer-poll.sv`
    - `CIRCT :: Tools/circt-sim/i3c-samplewrite-disable-fork-ordering.sv`
    - `CIRCT :: Tools/circt-sim/i3c-samplewrite-joinnone-disable-fork-ordering.sv`
  - observed failure mode: parse-stage errors from generated MLIR in `circt-sim` input (`custom op 'b123' is unknown`, parse failures), occurring before runtime execution.
- Alternate runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-148-phase-4-tracing-diagnostics-subsystem-extraction-slice-51"></a>
## Entry 148: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 51)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared on-demand-load diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceOnDemandLoadSignal`
  - `maybeTraceOnDemandLoadNoSignal`
- Replaced inlined `[ONDEMAND-LOAD]` emissions in `getValue` with shared
  helper calls for:
  - interface-field signal hit tracing,
  - interface-field miss tracing.
- Kept behavior unchanged by preserving:
  - existing `traceOnDemandLoad` callsite gate,
  - existing payload fields/formatting (`addr`, `sig`, `name`),
  - existing load-resolution control flow.
- Recorded the slice in TODO by checking:
  - `Extract on-demand-load diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-149-phase-4-tracing-diagnostics-subsystem-extraction-slice-52"></a>
## Entry 149: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 52)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared struct-inject X diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceStructInjectX`
- Replaced the inlined `[STRUCT-INJECT-X]` emission in `evaluateExpression`
  (`EvalKind::StructInject`) with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing trigger condition (`structVal.isX() || newVal.isX()`),
  - existing payload fields (`field`, `structX`, `newValX`, `totalWidth`),
  - existing X-result fallback behavior.
- Recorded the slice in TODO by checking:
  - `Extract struct-inject X diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-150-phase-4-tracing-diagnostics-subsystem-extraction-slice-53"></a>
## Entry 150: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 53)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared AHB transaction field diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceAhbTxnFieldWrite`
- Replaced three inlined `[AHB-TXN-FIELD]` emission blocks in
  `interpretDrive` with shared helper calls:
  - struct field drive (alloca-backed path),
  - struct field drive (memory-backed ref path),
  - struct+array element drive (memory-backed ref path, indexed variant).
- Kept behavior unchanged by preserving:
  - existing call-site env gates (`CIRCT_SIM_TRACE_AHB_TXN_FIELD_WRITES`,
    `CIRCT_SIM_TRACE_AHB_TXN_MIN_TIME_FS`),
  - existing monitor-sample function filter logic,
  - existing payload field formatting (including optional `idx=`).
- Recorded the slice in TODO by checking:
  - `Extract AHB transaction field diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-151-phase-4-tracing-diagnostics-subsystem-extraction-slice-54"></a>
## Entry 151: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 54)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared fread diagnostic helpers in `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceFreadSignalPath`
  - `maybeTraceFreadSignalWidth`
  - `maybeTraceFreadPointerPath`
- Replaced inlined `[fread]` emissions in `interpretFuncCall` for:
  - signal-path metadata trace,
  - signal-width/raw-bytes result trace,
  - pointer-path metadata trace.
- Kept behavior unchanged by preserving:
  - existing `traceFRead` gating logic at callsites,
  - existing field names and formatting,
  - existing fread execution and writeback flow.
- Recorded the slice in TODO by checking:
  - `Extract fread diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-152-phase-4-tracing-diagnostics-subsystem-extraction-slice-55"></a>
## Entry 152: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 55)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared function-progress diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceFuncProgress`
- Replaced two inlined `[circt-sim] func progress` emission blocks with shared
  helper calls in:
  - `interpretFunc` path,
  - `interpretLLVMFuncBody` path.
- Kept behavior unchanged by preserving:
  - existing progress trigger cadence (`funcBodySteps & 0xFFFFFF`),
  - existing payload fields/formatting (`process`, `funcBodySteps`,
    `totalSteps`, `func`, `callDepth`, `op`),
  - existing step accounting and overflow/abort checks around the trace point.
- Recorded the slice in TODO by checking:
  - `Extract function-progress diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-153-phase-4-tracing-diagnostics-subsystem-extraction-slice-56"></a>
## Entry 153: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 56)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared function-body step-overflow diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceProcessStepOverflowInFunc`
- Replaced two inlined `ERROR(PROCESS_STEP_OVERFLOW in func)` emission blocks
  with shared helper calls in:
  - `interpretFunc` path,
  - `interpretLLVMFuncBody` path.
- Kept behavior unchanged by preserving:
  - existing overflow trigger condition (`totalSteps > effectiveMaxProcessSteps`)
    in both paths,
  - existing message contract including `process`, step budget, function name, and
    `totalSteps`,
  - existing halted/cleanup/failure control flow after overflow detection.
- Recorded the slice in TODO by checking:
  - `Extract function-step-overflow diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-154-phase-4-tracing-diagnostics-subsystem-extraction-slice-57"></a>
## Entry 154: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 57)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared assertion-failure diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceSvaAssertionFailed`
  - `maybeTraceImmediateAssertionFailed`
- Replaced two inlined assertion-failure emissions with shared helper calls:
  - clocked SVA failure emission in `executeClockedAssertion`,
  - immediate `verif.assert` failure emission in `executeStep`.
- Kept behavior unchanged by preserving:
  - existing failure trigger conditions for both assertion kinds,
  - existing payload formatting (`label`, time/location fields),
  - existing failure accounting (`clockedAssertionFailures`) and control flow.
- Recorded the slice in TODO by checking:
  - `Extract assertion-failure diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-155-phase-4-tracing-diagnostics-subsystem-extraction-slice-58"></a>
## Entry 155: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 58)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared UVM run_test re-entry diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceUvmRunTestReentryError`
- Replaced the inlined `UVM run_test entered more than once` error emission in
  `checkUvmRunTestEntry` with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing re-entry trigger condition (`uvmRunTestEntryCount > 1` under
    `enforceSingleUvmRunTestEntry`),
  - existing payload fields and formatting (`count`, `callee`, optional process
    name),
  - existing failure return behavior after diagnostic emission.
- Recorded the slice in TODO by checking:
  - `Extract UVM run_test re-entry diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-156-phase-4-tracing-diagnostics-subsystem-extraction-slice-59"></a>
## Entry 156: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 59)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared bounded func.call internal-failure warning helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceFuncCallInternalFailureWarning`
- Replaced the inlined warning emission/counter block in `interpretFuncCall`
  (`func.call ... failed internally (absorbing)`) with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing abort guard (`!isAbortRequested()`),
  - existing bounded emission behavior (`<= 5` warnings),
  - existing warning payload formatting and fallback-zero control flow.
- Recorded the slice in TODO by checking:
  - `Extract func.call internal-failure warning diagnostics into tracing source`.

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Runtime-focused `.mlir` slice: PASS
  - `python3 llvm/llvm/utils/lit/lit.py -sv build-test/test/Tools/circt-sim --filter='func-(tail-wrapper-generic-resume-fast-path|drive-to-bfm-resume-fast-path|start-monitoring-resume-fast-path|baud-clk-generator-fast-path(-delay-batch|-null-self|-count-visible)?|generate-baud-clk-resume-fast-path)|randomize-(bytes|basic|with-ranges)|jit-process-thunk-(llvm-call-randomize-(basic|with-range)-halt|multiblock-scf-if-randomize-range-halt)|uvm-phase-hopper-wait-for-waiters-backoff|phase-hopper-objection|mailbox-(dpi-(blocking|blocking-bounded|nonblocking)|hopper-pattern)'`

<a id="entry-157-phase-4-tracing-diagnostics-subsystem-extraction-slice-60"></a>
## Entry 157: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 60)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared execute-loop diagnostics helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceProcessActivationStepLimitExceeded`
  - `maybeTraceProcessStepOverflow`
- Replaced two inlined execute-loop diagnostics in `executeProcess` with shared
  helper calls:
  - per-activation step-limit warning (`kMaxStepsPerActivation`),
  - global process-step overflow error (`effectiveMaxProcessSteps`).
- Kept behavior unchanged by preserving:
  - existing trigger conditions and finalize/exit control flow,
  - existing payload fields/formatting (process name, step counters, optional
    current function and last-op context),
  - existing step accounting around the checks.
- Recorded the slice in TODO by checking:
  - `Extract execute-loop step-limit/overflow diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-158-phase-4-tracing-diagnostics-subsystem-extraction-slice-61"></a>
## Entry 158: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 61)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared sim.terminate trigger diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceSimTerminateTriggered`
- Replaced inlined verbose `sim.terminate triggered in process ID ...` emission
  in `interpretTerminate` with a shared helper call.
- Added `#include "JITBlockCompiler.h"` to `LLHDProcessInterpreter.cpp` to keep
  JIT types complete in this TU under the concurrent JIT integration.
- Kept behavior unchanged by preserving:
  - existing `if (verbose)` gate for the diagnostic emission,
  - existing payload formatting (process id + location),
  - existing terminate control flow semantics.
- Recorded the slice in TODO by checking:
  - `Extract sim.terminate trigger diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-159-phase-4-tracing-diagnostics-subsystem-extraction-slice-62"></a>
## Entry 159: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 62)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared UVM JIT promotion-candidate diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceUvmJitPromotionCandidate`
- Replaced the inlined `UVM JIT promotion candidate` emission in
  `noteUvmFastPathActionHit` with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing `uvmJitTracePromotions` gate and promotion threshold/budget logic,
  - existing payload formatting (`stableKey`, `hits`, `threshold`,
    `budget_remaining`),
  - existing promotion-side effects and early-return conditions.
- Recorded the slice in TODO by checking:
  - `Extract UVM JIT promotion-candidate diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-160-phase-4-tracing-diagnostics-subsystem-extraction-slice-63"></a>
## Entry 160: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 63)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface propagation-map diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceFieldPropagationMap`
- Replaced the inlined `interfaceFieldPropagation map (...)` diagnostic dump
  block in `initialize()` with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing `traceInterfacePropagation && !interfaceFieldPropagation.empty()`
    gate at the callsite,
  - existing per-parent and per-child payload formatting (`sig`, names, widths),
  - existing control flow around initialization.
- Recorded the slice in TODO by checking:
  - `Extract interface propagation-map diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-161-phase-4-tracing-diagnostics-subsystem-extraction-slice-64"></a>
## Entry 161: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 64)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface auto-link diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceAutoLinkMatch`
  - `maybeTraceInterfaceAutoLinkTotal`
- Replaced two inlined interface auto-link emissions in `initialize()` with
  shared helper calls:
  - `Auto-linked ... fields from child interface ...`,
  - `Total auto-linked ... BFM interface fields ...`.
- Kept behavior unchanged by preserving:
  - existing `traceInterfacePropagation` gates and `autoLinked > 0` condition,
  - existing payload formatting (`bestMatchCount`, child/parent interface ids,
    total linked count),
  - existing auto-link algorithm and side effects.
- Recorded the slice in TODO by checking:
  - `Extract interface auto-link diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-162-phase-4-tracing-diagnostics-subsystem-extraction-slice-65"></a>
## Entry 162: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 65)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface intra-link diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceIntraLinkDetection`
  - `maybeTraceInterfaceIntraLinkReverseTarget`
  - `maybeTraceInterfaceIntraLinkBlock`
  - `maybeTraceInterfaceIntraLinkChildBlocks`
  - `maybeTraceInterfaceIntraLinkMatch`
  - `maybeTraceInterfaceIntraLinkTotal`
- Replaced inlined intra-link diagnostic emissions in `initialize()` with shared
  helper calls:
  - detection summary + per-reverse-target lines,
  - per-interface-block and child-block-count lines,
  - per-link match line,
  - total added intra-link summary line.
- Kept behavior unchanged by preserving:
  - existing `traceInterfacePropagation` gates and `intraLinks > 0` condition,
  - existing payload fields/formatting (signal ids/names, counts),
  - existing intra-link selection and propagation logic.
- Recorded the slice in TODO by checking:
  - `Extract interface intra-link diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-163-phase-4-tracing-diagnostics-subsystem-extraction-slice-66"></a>
## Entry 163: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 66)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface tri-state candidate diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceTriStateCandidateInstall`
  - `maybeTraceInterfaceTriStateCandidateSummary`
- Replaced inlined tri-state candidate diagnostics in `initialize()` with
  shared helper calls:
  - per-installed-rule line (`TriState rule: ...`),
  - per-pass summary line (`TriState candidates: ...`).
- Kept behavior unchanged by preserving:
  - existing `traceInterfacePropagation` gating at callsites,
  - existing payload formatting (`cond/src/dest`, `condBit`, `else`,
    candidate/installed/unresolved/total counts),
  - existing candidate resolution and rule-install control flow.
- Recorded the slice in TODO by checking:
  - `Extract interface tri-state candidate diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-164-phase-4-tracing-diagnostics-subsystem-extraction-slice-67"></a>
## Entry 164: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 67)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface copy/signal-copy diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceCopyPairLink`
  - `maybeTraceInterfaceDeferredSameInterfaceLink`
  - `maybeTraceInterfaceCopyPairSummary`
  - `maybeTraceInterfaceSignalCopyLink`
  - `maybeTraceInterfaceSignalCopySummary`
- Replaced inlined interface propagation diagnostics in `initialize()` with
  shared helper calls:
  - `CopyPair link ...`,
  - `Deferred same-interface link ...`,
  - `childModuleCopyPairs ...` summary block,
  - `SignalCopy link ...`,
  - `interfaceSignalCopyPairs ...` summary line.
- Kept behavior unchanged by preserving:
  - existing `traceInterfacePropagation` gates and summary conditions,
  - existing payload formatting (signal ids, addresses, widths, counters),
  - existing propagation and reverse-map control flow.
- Recorded the slice in TODO by checking:
  - `Extract interface copy/signal-copy diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-165-phase-4-tracing-diagnostics-subsystem-extraction-slice-68"></a>
## Entry 165: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 68)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface auto-link discovery diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceAutoLinkSignalDumpHeader`
  - `maybeTraceInterfaceAutoLinkSignalDumpEntry`
  - `maybeTraceInterfaceParentSignals`
  - `maybeTraceInterfaceInstanceAwareLink`
- Replaced inlined discovery diagnostics in `initialize()` with shared helper
  calls:
  - interface signal dump header and per-interface field-width dump entries,
  - parent-interface signal-id list,
  - instance-aware child-to-parent mapping lines.
- Kept behavior unchanged by preserving:
  - existing `enableHeuristicAutoLink`/size gating at the callsite,
  - existing payload formatting (signal names/ids, field widths, mappings),
  - existing auto-link discovery and mapping control flow.
- Recorded the slice in TODO by checking:
  - `Extract interface auto-link discovery diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-166-phase-4-tracing-diagnostics-subsystem-extraction-slice-69"></a>
## Entry 166: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 69)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface field-signal dump diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceFieldSignalDumpHeader`
  - `maybeTraceInterfaceFieldSignalDumpEntry`
- Replaced the remaining inlined interface field-signal `LLVM_DEBUG` emission
  in `initialize()` with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at the callsite,
  - existing payload formatting (address, signal id, optional signal name),
  - existing emission order over `interfaceFieldSignals`.
- Recorded the slice in TODO by checking:
  - `Extract interface field-signal dump diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-167-phase-4-tracing-diagnostics-subsystem-extraction-slice-70"></a>
## Entry 167: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 70)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared instance-output dependency diagnostic helper in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInstanceOutputDependencySignals`
- Replaced inlined `LLVM_DEBUG` instance-output source dependency emission in
  child instance output setup with a shared helper call.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at the callsite,
  - existing payload formatting (`signal id`, source signal count, ids),
  - existing dependency discovery and scheduling control flow.
- Recorded the slice in TODO by checking:
  - `Extract instance-output dependency diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-168-phase-4-tracing-diagnostics-subsystem-extraction-slice-71"></a>
## Entry 168: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 71)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared interface field-shadow scan diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceInterfaceFieldShadowScanSummary`
  - `maybeTraceInterfaceParentGepStructName`
  - `maybeTraceInterfaceParentScanResult`
- Replaced inlined `LLVM_DEBUG` diagnostics in
  `createInterfaceFieldShadowSignals()` with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at each callsite,
  - existing payload formatting (counts, struct names, scan stats, address),
  - existing interface field-shadow discovery and layout scan control flow.
- Recorded the slice in TODO by checking:
  - `Extract interface field-shadow scan diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-169-phase-4-tracing-diagnostics-subsystem-extraction-slice-72"></a>
## Entry 169: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 72)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared child-instance discovery/registration diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceChildInstanceFound`
  - `maybeTraceChildInstanceMissingRootModule`
  - `maybeTraceChildInstanceMissingModule`
  - `maybeTraceRegisteredChildSignal`
  - `maybeTraceRegisteredChildProcess`
  - `maybeTraceRegisteredChildCombinational`
  - `maybeTraceRegisteredChildInitialBlock`
- Replaced inlined `LLVM_DEBUG` child-instance diagnostics in
  `initializeChildInstances()` with shared helper calls.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at each callsite,
  - existing payload formatting (instance/module names, IDs, sensitivity count),
  - existing child-instance initialization and registration control flow.
- Recorded the slice in TODO by checking:
  - `Extract child-instance discovery/registration diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-170-phase-4-tracing-diagnostics-subsystem-extraction-slice-73"></a>
## Entry 170: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 73)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared child-instance input mapping/mismatch helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceChildInputMapped`
  - `maybeTraceChildInstanceOutputCountMismatch`
  - `maybeTraceInitializationRegistrationSummary`
- Replaced inlined `LLVM_DEBUG` diagnostics in initialization/child-instance
  setup with shared helper calls:
  - post-init signal/process registration summary,
  - child input block-arg mapping lines,
  - instance result/output count mismatch warning.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at each callsite,
  - existing payload formatting (names, ids, counts),
  - existing initialization and child-instance wiring control flow.
- Recorded the slice in TODO by checking:
  - `Extract child-instance input mapping/mismatch diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-171-phase-4-tracing-diagnostics-subsystem-extraction-slice-74"></a>
## Entry 171: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 74)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared signal discovery/registration diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceIterativeDiscoverySummary`
  - `maybeTraceRegisteredPortSignal`
  - `maybeTraceMappedExternalPortSignal`
  - `maybeTraceRegisteredOutputSignal`
- Replaced inlined `LLVM_DEBUG` diagnostics in discovery and signal
  registration paths with shared helper calls:
  - iterative operation discovery summary,
  - ref-port signal registration lines,
  - external-port block-arg mapping lines,
  - llhd.output implicit signal registration lines.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at each callsite,
  - existing payload formatting (counts, names, IDs, widths),
  - existing discovery/registration control flow and data mappings.
- Recorded the slice in TODO by checking:
  - `Extract signal discovery/registration diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-172-phase-4-tracing-diagnostics-subsystem-extraction-slice-75"></a>
## Entry 172: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 75)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added shared per-signal registration diagnostic helpers in
  `LLHDProcessInterpreterTrace.cpp`:
  - `maybeTraceRegisteredSignalInitialValue`
  - `maybeTraceRegisteredSignal`
- Replaced remaining inlined `LLVM_DEBUG` diagnostics in `registerSignal()`
  with shared helper calls:
  - explicit initial-value set line,
  - final per-signal registration summary line.
- Kept behavior unchanged by preserving:
  - existing `LLVM_DEBUG`-only gating at each callsite,
  - existing payload formatting (`APInt` text, name/ID/width fields),
  - existing signal registration and initialization control flow.
- Recorded the slice in TODO by checking:
  - `Extract per-signal init/registration diagnostics into tracing source`.

Validation highlights:
- Compile-only TU checks: PASS
  - `LLHDProcessInterpreter.cpp` via `build-test/compile_commands.json`
  - `LLHDProcessInterpreterTrace.cpp` via `build-test/compile_commands.json`
- Full build/lit status:
  - blocked by concurrent JIT integration errors in `tools/circt-sim/JITBlockCompiler.cpp`
    during `ninja -C build-test circt-sim`.

<a id="entry-97-phase-4-uvm-adapter-interceptor-extraction-slice-2"></a>
## Entry 97: Phase 4 UVM Adapter/Interceptor Extraction (Slice 2)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterUvm.cpp`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Moved sequencer queue resolution helpers from the monolithic interpreter file
  into `LLHDProcessInterpreterUvm.cpp`:
  - `canonicalizeUvmSequencerQueueAddress`
  - `resolveUvmSequencerQueueAddress`
- Added a local terminal-collection helper in `LLHDProcessInterpreterUvm.cpp`
  to preserve existing queue graph traversal behavior.
- Kept behavior unchanged by preserving:
  - queue candidate promotion/canonicalization logic,
  - owner/vtable fallback resolution semantics,
  - trace logging contract under `traceSeqEnabled`.
- Kept the Phase 4 UVM adapter/interceptor TODO item in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused UVM sequencer + call_indirect lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-98-phase-4-tracing-diagnostics-subsystem-extraction-slice-1"></a>
## Entry 98: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 1)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`
- `tools/circt-sim/CMakeLists.txt`
- `docs/WHOLE_PROJECT_REFACTOR_TODO.md`

Changes:
- Added `LLHDProcessInterpreterTrace.cpp` and moved a bounded diagnostics/profile
  helper set out of the monolithic interpreter file:
  - `getOrCreateJitRuntimeIndirectSiteData`
  - `noteJitRuntimeIndirectResolvedTarget`
  - `noteJitRuntimeIndirectUnresolved`
  - `lookupJitRuntimeIndirectSiteProfile`
  - `getJitRuntimeIndirectSiteProfiles`
  - `getJitDeoptProcessName`
  - `getJitCompileHotThreshold`
  - `dumpOpStats`
  - `dumpProcessStats`
- Moved local diagnostics utility `countRegionOps` into the new tracing source
  to keep per-process op-count reporting behavior unchanged.
- Registered `LLHDProcessInterpreterTrace.cpp` in the `circt-sim` CMake source
  list.
- Marked Phase 4 tracing/diagnostics extraction TODO as in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-99-phase-4-tracing-diagnostics-subsystem-extraction-slice-2"></a>
## Entry 99: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 2)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`

Changes:
- Moved `dumpProcessStates` from `LLHDProcessInterpreter.cpp` into
  `LLHDProcessInterpreterTrace.cpp`.
- Kept behavior unchanged by preserving:
  - process state dump fields and formatting,
  - profiling summaries (function/fast-path/cache),
  - sequencer/cache/native-state summaries,
  - memory summary/peak/delta/process-top sections.
- Kept the Phase 4 tracing/diagnostics TODO item in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`

<a id="entry-100-phase-4-tracing-diagnostics-subsystem-extraction-slice-3"></a>
## Entry 100: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 3)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`

Changes:
- Moved `traceI3CForkRuntimeEvent` from `LLHDProcessInterpreter.cpp` into
  `LLHDProcessInterpreterTrace.cpp`.
- Kept behavior unchanged by preserving:
  - `traceI3CForkRuntimeEnabled` gating,
  - per-process scheduler/runtime state formatting,
  - timestamp/fork-tag diagnostic emission contract.
- Kept the Phase 4 tracing/diagnostics TODO item in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`
<a id="entry-101-phase-4-tracing-diagnostics-subsystem-extraction-slice-4"></a>
## Entry 101: Phase 4 Tracing/Diagnostics Subsystem Extraction (Slice 4)

Scope:
- `tools/circt-sim/LLHDProcessInterpreter.h`
- `tools/circt-sim/LLHDProcessInterpreter.cpp`
- `tools/circt-sim/LLHDProcessInterpreterCallIndirect.cpp`
- `tools/circt-sim/LLHDProcessInterpreterTrace.cpp`

Changes:
- Added shared class helper:
  - `maybeTraceFilteredCall(ProcessId, StringRef, StringRef, int64_t, uint64_t)`
- Moved the `CIRCT_SIM_TRACE_CALL_FILTER` parsing/trace emission logic into
  `LLHDProcessInterpreterTrace.cpp`.
- Removed duplicated file-local `maybeTraceFilteredCall` implementations from:
  - `LLHDProcessInterpreter.cpp`
  - `LLHDProcessInterpreterCallIndirect.cpp`
- Preserved behavior by keeping:
  - trace enable/all/filter token semantics,
  - call trace output format and timestamp/delta emission.
- Kept the Phase 4 tracing/diagnostics TODO item in progress (`[~]`).

Validation highlights:
- Build: PASS
  - `ninja -C build-test circt-sim`
- Focused call/call_indirect + UVM lit slice: PASS
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-sim --filter='uvm-port-size-fallback|finish-item-(blocks-until-item-done|port-specific-item-done-order|multiple-outstanding-item-done)|seq-get-next-item-(empty-waiter-event-isolation|event-wakeup|empty-fallback-backoff)|config-db-native-(call-indirect-writeback|call-indirect-writeback-offset|impl-direct-writeback|impl-direct-writeback-offset|wrapper-writeback|wrapper-writeback-offset)|call-indirect-runtime-(vtable-slot-cache|override-site-cache)|vtable-(indirect-call|fallback-dispatch|fallback-corrupt-ptr|dispatch|dispatch-global-store|dispatch-cross-func|dispatch-internal-failure)|uvm-(printer-fast-path-call-indirect|report-getters-fast-path|report-handler-set-severity-action-fast-path)|timeout-no-spurious-vtable-warning'`
