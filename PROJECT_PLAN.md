# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/mbit/*_avip` testbenches using only CIRCT tools and the library ~/uvm-core.
Secondary goal: Get to 100% in the ~/sv-tests/ and ~/verilator-verification/ test suites.

---

## Simulation Workstream (circt-sim) — February 12, 2026

### Native JIT Roadmap (February 17, 2026)
- Comprehensive execution roadmap added: `docs/CIRCT_SIM_FULL_NATIVE_JIT_PLAN.md`
- Scope locked to big-bang rollout with strict-native as end-state.
- Correctness gate locked to bit-exact parity before default-on.
- Phase A scaffolding now in-tree (Feb 17):
  - `circt-sim` `--jit-report` JSON artifacts with stable keys.
  - deterministic interpret-vs-compile AVIP parity harness and checker scripts.
- Compile-governor scaffold now in-tree (Feb 17):
  - `JITCompileManager` counters/deopt-reason accounting.
  - initial process-thunk cache seam (`install/has/execute/invalidate`) for ORC.
  - `--jit-hot-threshold`, `--jit-compile-budget`, `--jit-cache-policy`,
    `--jit-fail-on-deopt` (+ env equivalents).
  - strict deopt policy path for compile-mode gating on per-process
    `missing_thunk` deopts.
  - initial native thunk path now executes for trivial terminating process
    shapes, including one-op `llhd.halt` (with yielded process results) and
    `sim.proc.print` + `llhd.halt`, plus trivial `seq.initial` print/yield
    shapes; compile mode can produce non-zero `jit_compiles_total` and
    `jit_exec_hits_total` on supported paths.
  - initial deopt-bridge scaffold now captures/restores process state when a
    native thunk requests deopt, and reports `guard_failed` in compile-mode
    telemetry.
  - first resumable native thunk path is in-tree for delay-wait processes
    (`llhd.wait delay -> llhd.halt`), exercising native suspend/resume flow.
  - resumable native thunk coverage now also includes
    `llhd.wait delay -> sim.proc.print -> llhd.halt`.
  - per-process native resume tokens are now wired through thunk dispatch and
    deopt snapshot/restore state.
  - periodic toggle clock native thunk dispatch now uses explicit token-guarded
    activation phases with deopt fallback on mismatched token/state transitions.
  - resumable wait thunks now include event-sensitive waits with a pre-wait
    probe (`llhd.wait (observed ...)`) for print/halt terminal bodies.
  - event-sensitive resumable wait thunks now also support multi-observed wait
    lists with matching pre-wait probe sequences.
  - resumable wait thunks now also support process-result terminal shapes with
    `llhd.wait yield (...)` + destination block operands and terminal
    `llhd.halt` yield operands.
  - event-sensitive resumable wait thunks are now covered for process-result
    terminal shapes as well (single- and multi-observed wait lists), with
    dedicated strict/deopt regressions.
  - deopt reasons are now split between `missing_thunk` and
    `unsupported_operation` when compile is attempted on unsupported bodies.
  - bounded AVIP mode-parity smoke remains green on `jtag`/seed `1`
    (`rows_interpret=1`, `rows_compile=1`; both bounded-timeout under 120s cap).
  - bounded AVIP compile-mode profiling smokes (jtag/seed=1) at 90s and 180s
    bounds both timed out before graceful profile-summary emission.
  - refreshed bounded AVIP compile-mode smoke (jtag/seed=1, 90s) remains:
    compile `OK`, bounded sim `TIMEOUT`.
  - parallel-runtime hardening gap identified:
    minimal LLHD process tests under `--parallel=4` currently hang or abort in
    both interpret and compile modes (allocator corruption crashes observed),
    so multi-threaded parity remains blocked pending scheduler/runtime fixes.
  - parallel safety gate landed:
    `--parallel` now defaults to a stable sequential fallback with warning;
    force-enable path for continuing hardening:
    `CIRCT_SIM_EXPERIMENTAL_PARALLEL=1`.

### Current Status
- **sv-tests simulation**: 1172 pass + 321 xfail = 1493/1493 (100%), 0 fail (Feb 17)
- **circt-sim unit tests**: 307/318 pass (11 UVM fast-path caching test failures)
- **ImportVerilog tests**: 268/268 (100%)
- **AVIP parity (v16 baseline, Feb 17)**: All 7 AVIPs complete simulation. 3 at 100% coverage (JTAG, SPI, AXI4-cg1). AHB 90%/100%, APB 88%/84%. I2S/I3C at 0%.
- **Performance**: JTAG completes in <60s, AXI4 in ~120s

### Recently Completed (v16 Baseline, Feb 17, 2026)

1. **Memory migration fix for parent process halt** (critical bug fix):
   When a parent process halts, `valueMap.clear()` and `memoryBlocks.clear()` destroy
   memory that child processes still reference through the parentProcessId chain.
   This broke UVM's phase hopper: `run_phases` creates an alloca for the phase output
   variable, then forks a loop child that calls `get()` storing to that alloca via a
   ref parameter. After the parent halts, stores are silently skipped.
   Fix: migrate parent memory blocks/valueMap to first active child before clearing.

2. **v16 AVIP baseline results** (all 7 AVIPs, 120s timeout):

   | AVIP | Sim Time | Errors | Coverage |
   |------|----------|--------|----------|
   | JTAG | 255 ns | 0 | 100% / 100% |
   | SPI  | 5.43 us | 1 | 100% / 100% |
   | AXI4 | 63.14 us | 0 | 100% / 96.49% |
   | AHB  | 3.25 us | 4 | 90% / 100% |
   | APB  | 21.35 us | 5 | 87.89% / 83.85% |
   | I2S  | 45.46 us | 2 | 0% / 0% |
   | I3C  | 169.91 us | 0 | 0% / 0% |

3. **sv-tests 100% pass rate**: 1172 pass, 321 xfail, 0 fail (up from 952/76).
   Auto-fast-skip for unlisted UVM tests, black-parrot xfail, gclk/power xfail.

4. **Regression tests**: sig-extract-x-drive-as-zero.mlir, iface-field-reverse-propagation.mlir.

5. **OpenTitan**: 30/39 tests pass. Fixed stuck-at-time-0 for spi_device, alert_handler.
6. **SVA importer diagnostics preservation** (Feb 21, 2026):
   Concurrent assertion action blocks with simple severity calls now preserve
   the diagnostic message as assertion labels during import
   (e.g. `else $error("fail")` / `$display("fail")` ->
   `verif.assert ... label "fail"`).
7. **SVA event-port clocking diagnostic cleanup** (Feb 21, 2026):
   Fixed false-positive importer diagnostic for nested event-typed assertion
   port clocking in `$past(..., @(event_port))` scenarios.
8. **SVA module-level labeled assertion lowering fix** (Feb 21, 2026):
   Fixed importer block-placement for module-scope labeled concurrent
   assertions (`label: assert property ...`) to avoid invalid multi-block
   `moore.module` IR (`cf.br` split around terminator). This unblocked
   yosys SVA `basic00` and broader `basic0[0-3]` smoke in CIRCT BMC.
9. **SVA compound match-item local-var assignment support** (Feb 21, 2026):
   Importer now lowers compound sequence match-item assignments for local
   assertion variables (e.g. `z += 1`, `s <<= 1`) instead of rejecting them
   as unsupported assignment kinds.
10. **SVA bounded property eventually support** (Feb 22, 2026):
   ImportVerilog now lowers bounded `eventually [m:n]` and
   `s_eventually [m:n]` when their operand is property-typed by constructing
   a delay-shifted property disjunction, rather than rejecting the form.
11. **SVA property nexttime support** (Feb 22, 2026):
   ImportVerilog now lowers property-typed `nexttime` and `s_nexttime`
   (including explicit single-cycle counts like `[N]`) via delayed-true
   implication, instead of erroring out.
12. **SVA bounded property always support** (Feb 22, 2026):
   ImportVerilog now lowers bounded property wrappers `always [m:n]` and
   `s_always [m:n]` by constructing conjunctions over delay-shifted property
   instances.
13. **SVA unbounded property always support** (Feb 22, 2026):
   ImportVerilog now lowers unbounded `always p` on property operands via
   duality (`not(eventually(not p))`) and adds explicit diagnostics to block
   unsound open upper-bound property-range lowering in unary wrappers.
14. **SVA open-range property unary support** (Feb 22, 2026):
   ImportVerilog now lowers parser-accepted open-range property forms
   `s_eventually [m:$]` and `always [m:$]` using shifted-property unbounded
   eventual/always encodings.
15. **SVA explicit-clock unpacked-array sampled support** (Feb 22, 2026):
   ImportVerilog now lowers explicit-clocked `$stable/$changed` for fixed-size
   unpacked arrays via helper-procedure sampled state and `moore.uarray_cmp`.
   Added dedicated positive/negative regressions for explicit clocking and
   canonical diagnostics on unsupported `$rose/$fell` unpacked-array operands.
16. **SVA explicit-clock unpacked-array `$past` support** (Feb 22, 2026):
   ImportVerilog now lowers explicit-clocked `$past` on fixed-size unpacked
   arrays using typed helper history storage, closing the previous
   `unsupported $past value type with explicit clocking` gap.
17. **SVA implicit-clock `$past` enable support with default clocking**
   (Feb 22, 2026):
   ImportVerilog now lowers `$past(expr, ticks, enable)` in procedural/default
   clocking contexts by resolving implicit clocking and routing through
   helper-based clocked lowering when needed.
18. **SVA procedural `_gclk` sampled-value support** (Feb 22, 2026):
   ImportVerilog now resolves global clocking for non-assertion sampled
   `_gclk` variants (`$changed_gclk/$stable_gclk/$rose_gclk/$fell_gclk`) and
   lowers them with helper-based clocked sampled state.
19. **SVA procedural default-clocking sampled-value support** (Feb 22, 2026):
   ImportVerilog now infers in-scope `default clocking` for non-assertion
   `$rose/$fell/$stable/$changed` (without explicit clock argument) and lowers
   them with helper-based sampled clocked state.
20. **SVA interface-contained concurrent assertion lowering** (Feb 22, 2026):
   Interface instance elaboration now lowers assertion-generated procedural
   members from interface bodies at instance sites, closing a silent-drop gap
   where interface assertions were previously omitted from imported IR.
21. **SVA unpacked-struct sampled-value support** (Feb 22, 2026):
   ImportVerilog now lowers `$stable/$changed` on unpacked struct operands
   (regular and explicit sampled clock helper paths), and extends explicit
   clock `$past` helper storage support to unpacked struct values.
22. **Unpacked-struct logical equality lowering (`==` / `!=`)** (Feb 22, 2026):
   ImportVerilog now lowers unpacked-struct equality/inequality expressions via
   recursive fieldwise comparisons, enabling direct full-struct SVA forms like
   `($past(s, ...) == s)` instead of requiring field-by-field workarounds.
23. **Unpacked-struct case equality lowering (`===` / `!==`)** (Feb 22, 2026):
   ImportVerilog now lowers unpacked-struct case equality/inequality via
   recursive fieldwise `case_eq` reduction, enabling SVA case-comparison
   assertions on unpacked structs.
24. **Unpacked-array case equality lowering (`===` / `!==`)** (Feb 22, 2026):
   ImportVerilog now lowers unpacked-array case equality/inequality directly
   via `moore.uarray_cmp` predicates (`eq`/`ne`), enabling procedural and SVA
   case-comparison assertions on fixed-size unpacked arrays.
25. **Unpacked-union sampled + explicit-clock `$past` support** (Feb 22, 2026):
   ImportVerilog now lowers `$stable/$changed` on unpacked union operands and
   supports explicit-clock `$past` for unpacked unions in assertions, closing
   previous bit-vector-cast and helper type-rejection gaps.
26. **Unpacked-union equality/case-equality support** (Feb 22, 2026):
   ImportVerilog now lowers unpacked-union `==/!=/===/!==` expressions via
   aggregate recursive comparison helpers (union member extraction + reduction),
   enabling direct union-compare SVA assertions without field workarounds.
27. **Nested aggregate case-equality regression hardening** (Feb 22, 2026):
   Added dedicated importer regression for unpacked-struct case equality with
   nested unpacked-array members to lock helper-recursion behavior.
28. **Unpacked aggregate `$rose/$fell` sampled support** (Feb 22, 2026):
   ImportVerilog now lowers `$rose/$fell` for fixed unpacked aggregates
   (arrays/structs/unions) in both direct assertion-clock and explicit sampled
   clock helper paths, using recursive aggregate-to-bool sampling.
29. **Dynamic/open-array sampled-value support** (Feb 22, 2026):
   ImportVerilog now supports dynamic-array/open-unpacked-array operands for
   `$stable/$changed/$rose/$fell` in assertions (direct and explicit-clock
   helper lowering), using size-aware elementwise comparison and boolean
   reduction through array locator primitives.
30. **Queue sampled-value semantic parity** (Feb 22, 2026):
   ImportVerilog now lowers queue operands for `$stable/$changed/$rose/$fell`
   with size-aware/element-aware queue semantics, replacing prior queue-to-zero
   fallback behavior.
31. **Dynamic-array/queue logical equality parity** (Feb 22, 2026):
   ImportVerilog now lowers `==`/`!=` on open unpacked arrays and queues with
   size-aware elementwise semantics (including SVA expression contexts), rather
   than constant fallback results.
32. **Dynamic-array/queue case-equality parity** (Feb 22, 2026):
   ImportVerilog now lowers `===`/`!==` on open unpacked arrays and queues with
   size-aware elementwise case-equality semantics (including nested aggregate
   members), replacing unsupported/incorrect dynamic case-compare behavior in
   procedural and SVA expression contexts.
33. **Explicit-clock `$past` for dynamic arrays/queues** (Feb 22, 2026):
   ImportVerilog now supports explicit sampled-clock `$past(..., @(event))` for
   open unpacked arrays and queues in assertions, using typed helper history
   state instead of rejecting those operand types.
15. **SVA packed sampled-value explicit-clocking support** (Feb 22, 2026):
   ImportVerilog now lowers explicit-clocking sampled-value calls on packed
   operands (`$changed/$stable/$rose/$fell`), by normalizing packed types to
   simple bit vectors in sampled-value helper paths.
16. **SVA packed `$past` explicit-clocking support** (Feb 22, 2026):
   ImportVerilog now lowers explicit-clocked `$past` on packed operands by
   sampling in simple-bit-vector form and converting results back to packed
   type, instead of rejecting non-int operands.
17. **SVA packed sampled regression hardening** (Feb 22, 2026):
   Added dedicated importer regression coverage for packed sampled-value
   operators in regular assertion-clocked usage (`$changed/$stable`).
18. **SVA string sampled explicit-clocking support** (Feb 22, 2026):
   ImportVerilog now lowers explicit-clocked sampled-value operators on string
   operands by sampling the existing 32-bit string-to-int bit-vector form.
19. **SVA string `$past` explicit-clocking regression hardening** (Feb 22, 2026):
   Added dedicated importer regression coverage for explicit-clocked `$past` on
   string operands (string-to-int sampled state plus int-to-string reification).
20. **SVA sampled explicit-clocking crash hardening** (Feb 22, 2026):
   Fixed null-deref crash paths when unsupported sampled operands fail
   bit-vector normalization under explicit clocking; importer now emits
   diagnostics reliably and is covered by verify-diagnostics regression.
21. **SVA unpacked-array sampled support (assertion clocking)** (Feb 22, 2026):
   ImportVerilog now lowers regular assertion-clocked `$changed/$stable` on
   fixed-size unpacked arrays using `moore.uarray_cmp` sampled comparisons.
10. **SVA `$future_gclk` temporal semantics fix** (Feb 21, 2026):
   Importer now lowers `$future_gclk(expr)` through forward temporal
   `ltl.delay(expr, 1, 0)` semantics instead of approximating with
   `moore.past(expr, 1)`, and regression checks now lock this behavior.
11. **SVA unclocked `_gclk` global-clocking semantics fix** (Feb 21, 2026):
   Unclocked properties using sampled `_gclk` functions now correctly lower
   with the scope's global clocking event instead of producing unclocked
   assertions.
12. **SVA `@($global_clock)` timing-control support + assertion-drop hardening** (Feb 21, 2026):
   ImportVerilog now lowers `$global_clock` assertion timing controls through
   the scope's global clocking block and no longer silently drops assertions
   when assertion-expression lowering fails (except true dead-generate
   `InvalidAssertionExpr` branches).
13. **SVA sampled explicit `@($global_clock)` support** (Feb 21, 2026):
   Explicit sampled-value clocking-argument forms (for example,
   `$rose(a, @($global_clock))`) now lower through resolved global clocking
   events instead of failing import.
14. **SVA assertion clock event-list support** (Feb 21, 2026):
   Property clocking with event lists (for example,
   `@(posedge clk or negedge clk)`) now lowers by clocking per-event and OR
   combining the resulting clocked properties.
15. **SVA `$global_clock iff ...` guard preservation** (Feb 21, 2026):
   ImportVerilog now preserves outer `iff` guards for `$global_clock` in both
   property clocking and explicit sampled-value clocking arguments.
16. **SVA yosys `counter` known-profile baseline cleanup** (Feb 21, 2026):
   Removed stale expected-XFAIL baseline entries for `counter` fail-mode in
   the known-input profile now that this case passes consistently.

### Previously Completed (Iteration 1401, Feb 14, 2026)
1. **sv-tests 100% coverage**: 952 PASS + 76 XFAIL = 1028/1028. Zero silent skips. Key additions: SVA LTLToCore pipeline, CompRegOp support, AnalysisManager integration, tagged union checker, runner compile-only mode for preprocessing tests.
2. **Build infrastructure**: lld linker (was GNU ld), ccache, RelWithDebInfo, parallel link job limiting.

### Previously Completed (Iteration 1171, Feb 12, 2026)
1. **randomize() vtable corruption fix**: `__moore_randomize_basic` was filling entire class object memory with random bytes, corrupting class_id (bytes 0-3) and vtable pointer (bytes 4-11). Now a no-op — individual rand fields set by `_with_range`/`_with_dist`. This fixed sub-sequence body dispatch (factory-created sequences now call derived body()).
2. **Blocking finish_item / item_done handshake** (iter 1165): Direct-wake mechanism, process-level retry fix.
3. **Phase hopper objection fix**: Broadened interceptors for `uvm_phase_hopper::` variants.
4. **`wait_for` wasEverRaised tracking**: Prevents premature phase completion.
5. **Dual-top sim.terminate handling**: Graceful shutdown when HVL finishes before HDL.
6. **Stack size increase**: 64MB stack for deep UVM sequence nesting.
7. **Diagnostic cleanup**: Removed all 21 DIAG-* debug logging blocks.
8. **config_db native write fix for dynamic arrays**: `uvm_config_db::get` writeback now reliably updates heap-backed dynamic-array slots (`new[]`) across all interpreter interception paths; added `test/Tools/circt-sim/config-db-dynamic-array-native.sv`.

### Feature Gap Table (Simulation)

| Feature | Status | Notes |
|---------|--------|-------|
| UVM phase sequencing | DONE | All 9 function phase IMPs complete in order |
| Phase hopper objections | DONE | get/raise/drop/wait_for for uvm_phase_hopper |
| VIF shadow signals | DONE | Interface field propagation + store interception |
| Sequencer interface | DONE | start_item/finish_item(blocking)/get/item_done; direct-wake handshake |
| Analysis port write | DONE | Chain-following BFS dispatch via vtable |
| Associative array deep copy | DONE | Prevents UVM phase livelock |
| Runtime vtable override | DONE | All 3 call_indirect paths check runtime vtable |
| Per-process RNG | DONE | IEEE 1800-2017 §18.13 random stability |
| Coverage collection | DONE | Covergroups + coverpoints + iff guards |
| Constraint solver | DONE | Soft/hard, inheritance, inline, dynamic bounds |
| Stack size (64MB) | DONE | Handles deep UVM sequence nesting |
| Sub-sequence body dispatch | DONE | Fixed by randomize_basic no-op (vtable preservation) |
| BFM/driver transactions | DONE | finish_item blocks until item_done; direct-wake handshake |
| SVA concurrent assertions | IN PROGRESS | LTLToCore pipeline + CompRegOp + module-level assert eval; 26 compile-only SVA tests now compile+link; full assertion failure detection pending |
| BFM APB state machine | DONE | Full APB transaction cycle works (88% coverage) |
| Multi-AVIP coverage | DONE | 7/7 AVIPs complete simulation; 3 at 100% coverage |
| Parent memory migration | DONE | Fixed silent store-skip when parent process halts before child |

### Next Steps (Simulation)
1. **Fix I2S/I3C 0% coverage**: Both AVIPs complete simulation but report 0% coverage. I2S has RX channel issue; I3C likely a coverage collection wiring problem.
2. **Fix APB/AHB scoreboard errors**: 5 APB and 4 AHB UVM_ERROR from scoreboard comparisons (master vs slave data mismatches). May be signal propagation timing issues.
3. **Convert 321 sv-tests xfail to pass**: black-parrot (6), remaining $*_gclk sampled value functions (~8 pending after `$future_gclk` + unclocked-global-clocking fixes), power operator (2), easyUVM tests. Need to fix underlying parser/lowering bugs.
4. **Performance optimization**: Target >500 ns/s for practical AVIP runs.
5. **SVA assertion runtime**: Genuine SVA runtime for concurrent assertion checking during simulation (task #38).

### Known Limitations (Simulation)
- I2S/I3C: 0% coverage despite completing simulation (coverage collection not triggered)
- APB: 5 UVM_ERROR (scoreboard comparison), AHB: 4 UVM_ERROR (scoreboard comparison)
- SPI: 1 UVM_ERROR (MOSI comparison, 0 data comparisons passed)
- 321 sv-tests marked xfail (target: convert all to pass)
- 11 circt-sim unit test failures (UVM fast-path caching)

---

## Formal Workstream (circt-mut) — February 12, 2026

Historical note: prior `### Formal Closure Snapshot Update (...)` entries were
migrated to `CHANGELOG.md` under `Historical Migration - February 14, 2026`.

### Milestone Progress Update (February 14, 2026)

1. Launch-telemetry strict governance completed for formal lanes:
   - absolute launch event limits (`--fail-on-any-{bmc,lec}-launch-events`)
   - bounded launch event limits (`--max-{bmc,lec}-launch-event-rows`)
   - launch reason-key drift checks
     (`--fail-on-new-{bmc,lec}-launch-reason-keys`)
2. Launch reason-key allowlist support is now first-class and strict-gate aware:
   - `--{bmc,lec}-launch-reason-key-allowlist-file`
   - allowlisted reason keys no longer cause strict-gate prefix drift on
     dynamic `*_launch_reason_*_events` counters.
3. OpenTitan FPV target-selection governance completed:
   - added baseline/update/fail/allowlist controls for
     `opentitan-fpv-target-manifest.tsv` drift.
   - strict-gate now auto-enforces target-manifest drift when a target-manifest
     baseline is configured.
4. Compile-contract task-policy governance completed:
   - compile-contract drift now covers task-policy metadata drift keys:
     `stopat_mode`, `blackbox_policy`, `task_policy_fingerprint`.
   - strict compile-contract drift gates now catch sec_cm task-policy mapping
     drift even when filelist-level contract fingerprints are unchanged.
5. Per-assertion drift governance completed for OpenTitan FPV BMC:
   - added baseline/update/fail/allowlist controls for
     per-assertion evidence drift (`opentitan-fpv-bmc-assertion-results.tsv`).
   - strict-gate now auto-enforces per-assertion drift when an assertion
     baseline is configured.
6. Per-assertion row-level drift allowlisting completed for OpenTitan FPV BMC:
   - added row-granularity allowlist controls for per-assertion drift
     suppressions using `<case_id>::<assertion_id>::<kind>` tokens.
   - strict-gate/fail-mode semantics are enforced for row-level allowlists to
     prevent silent drift suppression outside governed runs.
7. FPV-summary row-level drift allowlisting completed for OpenTitan FPV BMC:
   - added row-granularity allowlist controls for assertion-summary drift
     suppressions using `<target_name>::<kind>` tokens.
   - enables precise vacuous/unreachable drift rollouts without target-wide
     drift suppression.
8. Per-target assertion status-policy governance completed for OpenTitan FPV BMC:
   - added first-class required/forbidden assertion status policy controls:
     `--opentitan-fpv-bmc-assertion-status-policy-file`
   - strict-gate auto-enables status-policy fail mode when a policy file is
     configured.
9. Task-profile-aware status-policy presets and grouped diagnostics completed:
   - added task-profile preset policy controls:
     `--opentitan-fpv-bmc-assertion-status-policy-task-profile-presets-file`
   - added grouped violation diagnostics by task_profile/status class:
     `--opentitan-fpv-bmc-assertion-status-policy-grouped-violations-file`
10. OpenTitan missing-results diagnostics hardened for FPV BMC + LEC lanes:
   - missing-results classification now recognizes unresolved module failures as
     first-class reason `runner_command_unknown_module`.
   - `opentitan-missing-results-diagnostics.tsv` now carries extracted
     `unknown_modules` context for triage.
   - `opentitan-unresolved-modules.tsv` now receives deterministic per-module
     rows from both FPV BMC and LEC missing-results paths.
11. Regression coverage added for missing-results unknown-module paths:
   - `test/Tools/run-formal-all-opentitan-fpv-bmc-missing-results-unknown-module.test`
   - `test/Tools/run-formal-all-opentitan-lec-missing-results-unknown-module.test`
   - updated:
     `test/Tools/run-formal-all-opentitan-lec-missing-results-reason.test`
12. Grouped assertion-status-policy drift governance completed for OpenTitan FPV
    BMC:
   - added baseline/update/fail/allowlist controls for grouped status-policy
     diagnostics:
     - `--opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline-file`
     - `--opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift-file`
     - `--opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift-allowlist-file`
     - `--opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift-row-allowlist-file`
     - `--update-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-baseline`
     - `--fail-on-opentitan-fpv-bmc-assertion-status-policy-grouped-violations-drift`
   - strict-gate now auto-enables grouped status-policy drift fail mode when a
     grouped baseline is configured.
13. Canonical OpenTitan FPV BMC policy packaging completed:
   - shipped repo-managed task-profile status-policy presets:
     `utils/opentitan_fpv_policy/task_profile_status_presets.tsv`.
   - added first-class CLI toggle in `run_formal_all.sh`:
     `--opentitan-fpv-bmc-use-canonical-task-profile-presets`.
   - added checked-in baseline workflow wrapper:
     `utils/run_opentitan_fpv_bmc_policy_workflow.sh` (`update` / `check`).
14. Remaining limitation vs Jasper/VCF-style mature flows:
   - canonical policy inputs/workflow are now available, but reviewed baseline
     snapshots for broad OpenTitan target cohorts (`ip`/`prim`/`sec_cm`) are
     not yet materialized and governed as committed artifacts.
   - next practical gap is scaling the new workflow from harness-only policy
     controls to routinely updated, target-cohort baseline packs.
15. Cohort baseline-pack orchestration completed:
   - added checked-in profile-pack definitions:
     `utils/opentitan_fpv_policy/profile_packs.tsv`
     - `prim_all`, `ip_all`, `sec_cm_all`
   - added profile-pack orchestration runner:
     `utils/run_opentitan_fpv_bmc_policy_profiles.sh`
     - executes `update` / `check` workflow per pack
     - supports profile filtering and forwards lane args
   - workflow wrapper now supports per-pack baseline names via
     `--baseline-prefix`, enabling independent baseline artifacts per cohort.
   - added checked-in drift triage playbook:
     `utils/opentitan_fpv_policy/DRIFT_TRIAGE.md`.
16. Next formal milestone (OpenTitan-aligned, backend-generic):
   - materialize reviewed baseline snapshots for the three canonical profile
     packs from real OpenTitan runs and add strict check cadence:
     - commit initial reviewed baseline artifacts for `prim_all`, `ip_all`,
       and `sec_cm_all`;
     - wire recurring profile-pack `check` invocation into formal cadence
       workflows for early drift detection.
17. OpenTitan FPV profile-pack execution hardening completed:
   - fixed compile-contract lifetime for FPV BMC runs by defaulting
     `--opentitan-fpv-compile-contracts-workdir` to a persistent path under
     `OUT_DIR` when FPV cfg selection is active.
   - fixed FPV BMC baseline update behavior to always materialize
     assertion-results baselines (including empty baselines), preventing
     follow-on `check` mode failures due missing baseline files.
   - added focused regressions:
     - `test/Tools/run-formal-all-opentitan-fpv-compile-contracts-default-workdir.test`
     - `test/Tools/run-formal-all-opentitan-fpv-bmc-empty-assertion-baseline-update.test`
   - materialized canary baseline pack artifacts for:
     - `prim_all`, `ip_all`, `sec_cm_all`
     under `utils/opentitan_fpv_policy/baselines/`.
18. Remaining high-priority limitations (OpenTitan FPV cadence):
   - `check` mode with workflow default `--strict-gate` still depends on
     `utils/formal-baselines.tsv` containing matching OpenTitan FPV rows;
     otherwise strict gate reports `missing baseline row`.
   - smoke canary (`prim_all`) now passes end-to-end for BMC, but broad
     OpenTitan parity still requires scaling from single-target smoke to
     reviewed multi-target strict-gated cohorts (`prim_all`, `ip_all`,
     `sec_cm_all`) with stable baseline/update cadence.
19. OpenTitan FPV frontend ingestion hardening completed:
   - compile-contract resolver now excludes `is_include_file` entries from the
     compile-unit `files` list while still forwarding include directories.
   - `run_pairwise_circt_bmc.py` now supports generic single-unit policy mode
     via `BMC_VERILOG_SINGLE_UNIT_MODE` (`auto|on|off`, default `auto`).
   - in `auto`, verilog frontend now retries once without `--single-unit` when
     diagnostics match known macro-preprocessor single-unit failures:
     - `macro operators may only be used within a macro definition`
     - `unexpected conditional directive`
   - retry provenance is preserved:
     - initial failure log saved to `circt-verilog.single-unit.log`
     - launch telemetry emits
       `single_unit_preprocessor_failure`.
20. LLHD signal-array reference closure completed for OpenTitan FPV BMC:
   - `StripLLHDInterfaceSignals` now supports `llhd.sig.array_get` in LEC/BMC
     stripping by carrying array-element ref steps and materializing updates
     through `hw.array_get` / `hw.array_inject`.
   - added regression:
     - `test/Tools/circt-lec/lec-strip-llhd-signal-array-get.mlir`
   - real canary uplift:
     - `utils/run_opentitan_fpv_bmc_policy_profiles.sh --profile prim_all`
       (`BMC_SMOKE_ONLY=1`, explicit `build-test` toolchain) now passes:
       `total=1 pass=1 error=0`.
20a. OpenTitan FPV objective-parity projected-evidence governance completed:
   - `utils/check_opentitan_fpv_objective_parity.py` now preserves lane
     evidence/reason metadata in parity TSV rows:
     - `bmc_evidence`, `lec_evidence`, `bmc_reason`, `lec_reason`.
   - `run_formal_all.sh` now reports projected-reason parity counters:
     - `fpv_objective_parity_projected_rows`
     - `fpv_objective_parity_projected_non_allowlisted_rows`
     - `fpv_objective_parity_projected_assertion_rows`
     - `fpv_objective_parity_projected_cover_rows`.
   - this closes a visibility gap for native FPV LEC auto-produced evidence
     (`projected_case_*`) and makes strict-gate triage auditable at objective
     row granularity.
20b. Remaining high-priority limitation (objective parity depth):
   - objective parity status governance is now projected-reason aware, but
   parity still compares normalized status classes only and does not yet
   enforce reason-class contracts (e.g. `projected_case_eq` vs
   `projected_case_unknown`) as a fail condition.
20c. Objective reason-drift enforcement completed for OpenTitan FPV parity:
   - `check_opentitan_fpv_objective_parity.py` now supports
     `--reason-policy ignore|projected|all` and emits first-class
     reason drift rows (`assertion_reason` / `cover_reason`).
   - `run_formal_all.sh` now exposes/forwards:
     `--opentitan-fpv-objective-parity-reason-policy`.
   - strict-gate default now upgrades FPV objective parity reason policy to
     `projected` when FPV LEC assertion evidence is present.
   - lane summaries now expose reason-specific parity counters for drift triage.
20d. Remaining high-priority limitation (OpenTitan FPV objective parity):
   - objective reason drift is enforced per run, but there is still no
     baseline-driven reason-drift artifact governance layer (baseline/update/
     allowlist drift reports) for FPV objective parity rows analogous to FPV
     BMC summary/assertion baseline flows.
20e. Objective-parity baseline drift governance completed:
   - added new checker:
     - `utils/check_opentitan_fpv_objective_parity_drift.py`
   - `run_formal_all.sh` now supports baseline lifecycle and drift controls
     for FPV objective parity:
     - `--opentitan-fpv-objective-parity-baseline-file`
     - `--update-opentitan-fpv-objective-parity-baseline`
     - `--opentitan-fpv-objective-parity-drift-file`
     - `--opentitan-fpv-objective-parity-drift-allowlist-file`
     - `--opentitan-fpv-objective-parity-drift-row-allowlist-file`
     - `--fail-on-opentitan-fpv-objective-parity-drift`
   - lane summaries now include objective-parity drift counters (total,
     allowlisted/non-allowlisted, and drift-kind breakdown).
20f. Remaining high-priority limitation (OpenTitan FPV objective parity depth):
   - objective parity drift governance is now baseline-driven, but there is no
     committed broad-target baseline pack yet for FPV objective parity
     (multi-target `prim/ip/sec_cm` cohort snapshots).
   - next step is to materialize reviewed cohort baselines and run strict-gated
     cadence checks so objective-level drift is governed at OpenTitan scale.
20g. Objective-parity governance integrated into policy wrappers/profile packs:
   - `utils/run_opentitan_fpv_bmc_policy_workflow.sh` now supports
     objective-parity baseline management:
     - `--enable-objective-parity`
     - `--objective-parity-reason-policy`
     - managed baseline artifact:
       `${baseline_prefix}-objective-parity-baseline.tsv`
     - mode wiring:
       - update:
         `--update-opentitan-fpv-objective-parity-baseline`
       - check:
         `--fail-on-opentitan-fpv-objective-parity-drift`
   - `utils/run_opentitan_fpv_bmc_policy_profiles.sh` now supports both
     workflow-level and per-profile objective-parity controls:
     - workflow:
       `--workflow-enable-objective-parity`,
       `--workflow-objective-parity-reason-policy`
     - profile TSV optional columns:
       `objective_parity`, `objective_parity_reason_policy`
     - lane routing now expands to include `opentitan/FPV_OBJECTIVE_PARITY`
       when objective parity is enabled for a profile.
20h. Remaining high-priority limitation (objective-parity cohort rollout):
   - wrappers/profile packs can now govern objective-parity baselines, but
     reviewed committed objective-parity baselines for default canary packs are
     not yet materialized (`prim_all`, `ip_all`, `sec_cm_all`).
   - next step is to run profile-pack update/check in real OpenTitan cohorts
     and commit reviewed objective-parity baseline snapshots.
21. Frontend macro-compat retry hardening completed for FPV BMC ingestion:
   - added external preprocessor retry controls in
     `run_pairwise_circt_bmc.py`:
     - `BMC_VERILOG_EXTERNAL_PREPROCESS_MODE=auto|on|off`
     - `BMC_VERILOG_EXTERNAL_PREPROCESS_CMD`
   - fixed default verilator include-flag forwarding (`-I<dir>`), removing
     preprocessing command false-negatives.
   - added focused regression:
     - `test/Tools/run-pairwise-circt-bmc-external-preprocess-auto-retry.test`
22. Prim-assert include-shim retry completed for OpenTitan-style macro stacks:
   - added auto-retry path in `run_pairwise_circt_bmc.py` for parser failures
     rooted in `prim_assert*` headers; retry emits and applies deterministic
     case-local include overrides:
     - `prim_assert.sv`
     - `prim_flop_macros.sv`
     - `prim_assert_sec_cm.svh`
     - `circt-verilog.assert-macro-shim.sv`
   - launch provenance now records:
     - `prim_assert_include_shim_macro_compat`
   - added focused regression:
     - `test/Tools/run-pairwise-circt-bmc-prim-assert-include-shim-auto-retry.test`
23. Remaining high-priority limitation (OpenTitan parity path):
   - full `pinmux_fpv` remains a heavy compile-unit stress case; retry
     hardening is now in place, but broad OpenTitan FPV closure still needs:
     - additional parser-compat closure beyond current shim scope, and
     - compile-unit scale/performance improvements for large FPV targets.
24. Frontend retry ordering hardened for OpenTitan-scale compile units:
   - `run_pairwise_circt_bmc.py` now prioritizes `prim_assert` include-shim
     retry over external preprocessing when diagnostics indicate
     `prim_assert*` macro-compatibility failures.
   - external preprocessing fallback remains available for non-`prim_assert`
     macro/preprocessor failures.
   - added regression:
     - `test/Tools/run-pairwise-circt-bmc-prim-assert-retry-precedes-external-preprocess.test`
   - impact:
     - reduces avoidable heavyweight preprocessing retries in the dominant
       OpenTitan prim-assert failure family, improving scale behavior while
       preserving generic fallback coverage.
25. Frontend compile caching milestone completed (generic, backend-agnostic):
   - `run_pairwise_circt_bmc.py` now supports deterministic frontend artifact
     caching for verilog import outputs:
     - `BMC_VERILOG_CACHE_MODE=off|read|readwrite`
     - `BMC_VERILOG_CACHE_DIR`
   - cache key includes tool fingerprint + frontend option surface + source
     file fingerprints, enabling safe cross-run reuse under stable contracts.
   - cache operation is fail-open to preserve run reliability under cache I/O
     faults.
   - added regression:
     - `test/Tools/run-pairwise-circt-bmc-verilog-cache-basic.test`
   - impact:
     - reduces repeated `circt-verilog` cost for large OpenTitan FPV reruns and
     sharded re-execution paths, moving toward Jasper/VCF-scale throughput
     expectations without sacrificing determinism.
26. OpenTitan compile-contract scoped cache forwarding completed:
   - `run_opentitan_fpv_circt_bmc.py` now supports OpenTitan-runner-level
     cache controls:
     - `--verilog-cache-mode` (`off|read|readwrite|auto`)
     - `--verilog-cache-dir`
     - env defaults:
       - `BMC_OPENTITAN_VERILOG_CACHE_MODE`
       - `BMC_OPENTITAN_VERILOG_CACHE_DIR`
   - compile contracts now ingest `contract_fingerprint` into per-target
     metadata and use it to derive deterministic cache namespaces per policy
     group (`stopat`/`blackbox` partition).
   - added regression:
     - `test/Tools/run-opentitan-fpv-circt-bmc-verilog-cache-forwarding.test`
   - impact:
     - enables predictable frontend reuse from OpenTitan orchestration without
       hard-coding OpenTitan-only frontend behavior into the generic pairwise
       runner.
27. OpenTitan FPV cache policy forwarding at orchestration layer completed:
   - `run_formal_all.sh` now exposes and forwards OpenTitan FPV BMC cache
     controls:
     - `--opentitan-fpv-bmc-verilog-cache-mode`
     - `--opentitan-fpv-bmc-verilog-cache-dir`
   - added fail-closed mode validation and lane-dependency checks
     (`--with-opentitan-fpv-bmc` required).
   - added regressions:
     - `test/Tools/run-formal-all-opentitan-fpv-bmc-verilog-cache-forwarding.test`
     - `test/Tools/run-formal-all-opentitan-fpv-bmc-verilog-cache-mode-invalid.test`
   - impact:
     - lets profile-pack and strict-gate workflows enable deterministic cache
       behavior from the top-level formal entrypoint, not only runner-local
       invocations.
28. OpenTitan FPV policy-wrapper cache governance completed:
   - `utils/run_opentitan_fpv_bmc_policy_workflow.sh` now supports and
     forwards:
     - `--opentitan-fpv-bmc-verilog-cache-mode`
     - `--opentitan-fpv-bmc-verilog-cache-dir`
   - `utils/run_opentitan_fpv_bmc_policy_profiles.sh` now supports:
     - workflow defaults:
       - `--workflow-verilog-cache-mode`
       - `--workflow-verilog-cache-dir`
     - optional per-profile overrides in TSV:
       - `verilog_cache_mode`
       - `verilog_cache_dir`
   - both wrappers enforce fail-closed validation:
     - invalid cache mode rejected.
     - cache dir requires cache mode.
   - added/updated regressions:
     - `test/Tools/run-opentitan-fpv-bmc-policy-workflow.test`
     - `test/Tools/run-opentitan-fpv-bmc-policy-profiles.test`
   - impact:
     - OpenTitan policy pack runs can now deterministically control frontend
       cache reuse per cohort/profile, enabling stable large-target cadence
       without introducing OpenTitan-specific behavior into generic runners.
29. Launch reason-event budget governance completed for BMC/LEC lanes:
   - added first-class strict-gate controls in `run_formal_all.sh`:
     - `--fail-on-any-bmc-launch-reason-events`
     - `--fail-on-any-lec-launch-reason-events`
     - `--max-bmc-launch-reason-event-rows`
     - `--max-lec-launch-reason-event-rows`
   - policy semantics are allowlist-aware via existing reason-key files:
     - non-allowlisted reason-event rows are enforced by nonzero/max gates.
     - allowlisted reason keys are excluded from reason-event budget
       violations.
   - allowlist gate preconditions now accept reason-event budget policies
     (not only new-reason-key drift mode), keeping fail-closed behavior while
     enabling controlled rollout of transient launch reason keys.
   - added focused regressions:
     - `test/Tools/run-formal-all-strict-gate-bmc-launch-reason-events-budget.test`
     - `test/Tools/run-formal-all-strict-gate-lec-launch-reason-events-budget.test`
     - updated:
       `test/Tools/run-formal-all-launch-reason-key-allowlists-require-gate.test`
30. OpenTitan FPV policy-pack rollout for launch reason-event budgets completed:
   - `utils/run_opentitan_fpv_bmc_policy_workflow.sh` now supports check-only
     launch reason-event policy forwarding:
     - `--check-bmc-launch-reason-key-allowlist-file`
     - `--check-lec-launch-reason-key-allowlist-file`
     - `--check-max-bmc-launch-reason-event-rows`
     - `--check-max-lec-launch-reason-event-rows`
     - `--check-fail-on-any-bmc-launch-reason-events`
     - `--check-fail-on-any-lec-launch-reason-events`
   - `utils/run_opentitan_fpv_bmc_policy_profiles.sh` now supports workflow
     defaults and per-profile overrides for the same check-only launch policy
     surface, including relative-path resolution for profile-local allowlists.
   - canonical canary-pack policy inputs updated:
     - `utils/opentitan_fpv_policy/profile_packs.tsv`
     - `utils/opentitan_fpv_policy/bmc_launch_reason_key_allowlist.txt`
   - focused wrapper regressions updated:
     - `test/Tools/run-opentitan-fpv-bmc-policy-workflow.test`
     - `test/Tools/run-opentitan-fpv-bmc-policy-profiles.test`
31. Launch retry reason taxonomy expanded across BMC lanes:
   - runner-level launch retry classification now emits explicit reason keys
     beyond `etxtbsy`:
     - `permission_denied`
     - `posix_spawn_failed`
     - `resource_temporarily_unavailable`
   - implemented in:
     - `utils/run_pairwise_circt_bmc.py`
     - `utils/run_sv_tests_circt_bmc.sh`
     - `utils/run_verilator_verification_circt_bmc.sh`
     - `utils/run_yosys_sva_circt_bmc.sh`
     - `utils/run_sv_tests_circt_lec.sh`
   - `run_formal_all.sh` launch-event summaries now expose dedicated counters:
     - `*_launch_permission_denied_events`
     - `*_launch_posix_spawn_failed_events`
     - `*_launch_resource_temporarily_unavailable_events`
   - focused regressions added/updated:
     - `test/Tools/run-sv-tests-bmc-launch-retry-posix-spawn-failed.test`
     - `test/Tools/run-formal-all-sv-tests-launch-reason-classification-summary.test`
     - updated launch-reason expectation tests across pairwise/sv-tests/verilator/yosys BMC runners.
32. Per-reason launch event budget governance completed for BMC/LEC:
   - `run_formal_all.sh` now supports per-reason max budgets for non-allowlisted
     launch-reason counters:
     - `--bmc-launch-reason-event-budget-file`
     - `--lec-launch-reason-event-budget-file`
   - budget files are selector-driven (`exact|prefix|regex|*`) and enforced in
     strict-gate evaluation with dedicated diagnostics when a reason exceeds its
     per-key max.
   - allowlist semantics remain authoritative:
     allowlisted reason keys are excluded from per-reason budget enforcement.
   - OpenTitan FPV policy wrappers now forward check-mode budget files:
     - `utils/run_opentitan_fpv_bmc_policy_workflow.sh`:
       - `--check-bmc-launch-reason-event-budget-file`
       - `--check-lec-launch-reason-event-budget-file`
     - `utils/run_opentitan_fpv_bmc_policy_profiles.sh`:
       workflow defaults + profile TSV columns:
       - `check_bmc_launch_reason_event_budget_file`
       - `check_lec_launch_reason_event_budget_file`
   - canonical canary pack now carries a checked-in default:
     - `utils/opentitan_fpv_policy/bmc_launch_reason_event_budget.tsv`
     - wired in `utils/opentitan_fpv_policy/profile_packs.tsv`.
   - added/updated regressions:
     - `test/Tools/run-formal-all-strict-gate-bmc-launch-reason-event-budget-file.test`
     - `test/Tools/run-formal-all-strict-gate-lec-launch-reason-event-budget-file.test`
     - `test/Tools/run-opentitan-fpv-bmc-policy-workflow.test`
     - `test/Tools/run-opentitan-fpv-bmc-policy-profiles.test`
     - `test/Tools/run-formal-all-launch-reason-key-allowlists-require-gate.test`
33. Launch retry taxonomy extended with stale-handle transient classification:
   - added `stale_file_handle` retry reason classification across formal launchers:
     - `utils/run_pairwise_circt_bmc.py`
     - `utils/run_sv_tests_circt_bmc.sh`
     - `utils/run_sv_tests_circt_lec.sh`
     - `utils/run_verilator_verification_circt_bmc.sh`
     - `utils/run_yosys_sva_circt_bmc.sh`
   - `run_sv_tests_circt_lec.sh` retry trigger detection is now taxonomy-aligned:
     - recognizes `Text file busy`/`ETXTBSY`, `Permission denied`,
       `posix_spawn failed`, `resource temporarily unavailable`,
       and `stale file handle`/`ESTALE` as retryable launch failures.
   - pairwise runner now maps `errno.ESTALE` as retryable `stale_file_handle`.
   - canonical OpenTitan canary per-reason budget policy now carries explicit
     `exact:stale_file_handle` governance row:
     - `utils/opentitan_fpv_policy/bmc_launch_reason_event_budget.tsv`
   - added regressions:
     - `test/Tools/run-pairwise-circt-bmc-launch-retry-stale-file-handle.test`
     - `test/Tools/run-sv-tests-bmc-launch-retry-stale-file-handle.test`
     - `test/Tools/run-sv-tests-lec-verilog-stale-file-handle-retry.test`
     - `test/Tools/run-verilator-verification-circt-bmc-launch-retry-stale-file-handle.test`
     - `test/Tools/run-yosys-sva-bmc-launch-retry-stale-file-handle.test`
     - updated:
       `test/Tools/run-formal-all-sv-tests-launch-reason-classification-summary.test`
34. Launch retry taxonomy extended with file-descriptor exhaustion classification:
   - added `too_many_open_files` retry reason classification across formal
     launchers:
     - `utils/run_pairwise_circt_bmc.py`
     - `utils/run_sv_tests_circt_bmc.sh`
     - `utils/run_sv_tests_circt_lec.sh`
     - `utils/run_verilator_verification_circt_bmc.sh`
     - `utils/run_yosys_sva_circt_bmc.sh`
   - pairwise runner now maps transient launcher `OSError` classes:
     - `errno.EMFILE`
     - `errno.ENFILE` (when present on host libc)
     to retryable reason `too_many_open_files`.
   - launch-summary counters in `run_formal_all.sh` now expose dedicated metrics:
     - `bmc_launch_too_many_open_files_events`
     - `lec_launch_too_many_open_files_events`
   - canonical OpenTitan canary per-reason budget policy now carries explicit
     `exact:too_many_open_files` governance row:
     - `utils/opentitan_fpv_policy/bmc_launch_reason_event_budget.tsv`
   - added regressions:
     - `test/Tools/run-pairwise-circt-bmc-launch-retry-too-many-open-files.test`
     - `test/Tools/run-sv-tests-bmc-launch-retry-too-many-open-files.test`
     - `test/Tools/run-sv-tests-lec-verilog-too-many-open-files-retry.test`
     - `test/Tools/run-verilator-verification-circt-bmc-launch-retry-too-many-open-files.test`
     - `test/Tools/run-yosys-sva-bmc-launch-retry-too-many-open-files.test`
     - updated:
       `test/Tools/run-formal-all-sv-tests-launch-reason-classification-summary.test`
35. Launch retry taxonomy extended with memory-pressure classification:
   - added `cannot_allocate_memory` retry reason classification across formal
     launchers:
     - `utils/run_pairwise_circt_bmc.py`
     - `utils/run_sv_tests_circt_bmc.sh`
     - `utils/run_sv_tests_circt_lec.sh`
     - `utils/run_verilator_verification_circt_bmc.sh`
     - `utils/run_yosys_sva_circt_bmc.sh`
   - pairwise runner now maps transient launcher `OSError`:
     - `errno.ENOMEM` -> `cannot_allocate_memory`.
   - launch-summary counters in `run_formal_all.sh` now expose dedicated metrics:
     - `bmc_launch_cannot_allocate_memory_events`
     - `lec_launch_cannot_allocate_memory_events`
   - canonical OpenTitan canary per-reason budget policy now carries explicit
     `exact:cannot_allocate_memory` governance row:
     - `utils/opentitan_fpv_policy/bmc_launch_reason_event_budget.tsv`
   - added regressions:
     - `test/Tools/run-pairwise-circt-bmc-launch-retry-cannot-allocate-memory.test`
     - `test/Tools/run-sv-tests-bmc-launch-retry-cannot-allocate-memory.test`
     - `test/Tools/run-sv-tests-lec-verilog-cannot-allocate-memory-retry.test`
     - `test/Tools/run-verilator-verification-circt-bmc-launch-retry-cannot-allocate-memory.test`
     - `test/Tools/run-yosys-sva-bmc-launch-retry-cannot-allocate-memory.test`
     - updated:
       `test/Tools/run-formal-all-sv-tests-launch-reason-classification-summary.test`
36. Canonical OpenTitan profile-pack launch budget parity completed (BMC + LEC):
   - added canonical LEC per-reason launch budget defaults:
     - `utils/opentitan_fpv_policy/lec_launch_reason_event_budget.tsv`
       (explicit `stale_file_handle` / `too_many_open_files` /
       `cannot_allocate_memory` + wildcard default).
   - wired checked-in profile packs to carry both BMC and LEC check-mode launch
     budget controls:
     - `utils/opentitan_fpv_policy/profile_packs.tsv` now includes:
       - `check_max_lec_launch_reason_event_rows`
       - `check_lec_launch_reason_event_budget_file`
       - `check_fail_on_any_lec_launch_reason_events`
     - existing BMC launch budget fields remain unchanged.
   - docs aligned:
     - `utils/opentitan_fpv_policy/README.md` now documents per-pack and
       workflow-level BMC+LEC launch reason-event budget controls.
   - regression coverage added/extended:
     - added:
       `test/Tools/run-opentitan-fpv-bmc-policy-profile-packs-launch-budget-parity.test`
     - updated:
       `test/Tools/run-opentitan-fpv-bmc-policy-profiles.test`
       to validate workflow defaults + per-profile overrides for both BMC and
       LEC launch reason allowlists, max rows, budget files, and fail flags.
37. OpenTitan FPV cross-lane objective parity lane bootstrap completed:
   - added generic FPV objective parity checker:
     - `utils/check_opentitan_fpv_objective_parity.py`
     - compares normalized BMC-vs-LEC objective statuses for assertion/cover
       evidence with explicit missing-objective policy:
       `ignore|assertion|all`.
   - `run_formal_all.sh` now supports FPV objective parity controls:
     - `--opentitan-fpv-lec-assertion-results-file`
     - `--opentitan-fpv-lec-cover-results-file`
     - `--opentitan-fpv-objective-parity-file`
     - `--opentitan-fpv-objective-parity-allowlist-file`
     - `--fail-on-opentitan-fpv-objective-parity`
     - `--opentitan-fpv-objective-parity-include-missing`
     - `--opentitan-fpv-objective-parity-missing-policy`
   - added parity lane:
     - `opentitan/FPV_OBJECTIVE_PARITY`
     - emits summary counters:
       - `fpv_objective_parity_rows`
       - `fpv_objective_parity_non_allowlisted_rows`
       - `fpv_objective_parity_allowlisted_rows`
       - `fpv_objective_parity_assertion_rows`
       - `fpv_objective_parity_cover_rows`
       - `fpv_objective_parity_missing_rows`
       - `fpv_objective_parity_assertion_status_rows`
       - `fpv_objective_parity_cover_status_rows`
       - class-specific non-allowlisted counters.
   - strict-gate integration:
     - auto-enables `--fail-on-opentitan-fpv-objective-parity` when FPV BMC is
       enabled and LEC assertion evidence is configured.
     - defaults missing-objective policy to `assertion` under strict-gate
       unless explicitly overridden.
   - regression coverage:
     - `test/Tools/check-opentitan-fpv-objective-parity-fail.test`
     - `test/Tools/run-formal-all-opentitan-fpv-objective-parity-forwarding.test`
     - `test/Tools/run-formal-all-opentitan-fpv-objective-parity-requires-lec-assertions.test`
38. OpenTitan FPV native LEC evidence auto-production integrated:
   - added native runner:
     - `utils/run_opentitan_fpv_circt_lec.py`
     - consumes FPV compile contracts + BMC objective manifests and emits:
       - case-level LEC artifact (`OUT`)
       - assertion objective evidence (`LEC_ASSERTION_RESULTS_OUT`)
       - cover objective evidence (`LEC_COVER_RESULTS_OUT`).
   - `run_formal_all.sh` objective parity lane now auto-produces FPV LEC
     evidence when caller does not pass external
     `--opentitan-fpv-lec-{assertion,cover}-results-file`.
   - strict tool preflight now includes OpenTitan FPV LEC runner/tool checks
     (`circt-verilog`, `circt-opt`, `circt-lec`) for auto-produced path.
   - added regression coverage:
     - `test/Tools/run-opentitan-fpv-circt-lec-basic.test`
     - updated
       `test/Tools/run-formal-all-opentitan-fpv-objective-parity-requires-lec-assertions.test`
       to validate auto-production wiring.
39. OpenTitan FPV LEC assertion semantics hardening completed:
   - `utils/run_opentitan_fpv_circt_lec.py` no longer copies assertion status
     classes from BMC objective rows.
   - assertion rows now use assertion-native case-projected mapping:
     - `PASS -> PROVEN`
     - `FAIL -> FAILING`
     - `UNKNOWN|TIMEOUT|SKIP|ERROR -> same status`
   - cover evidence emission is now explicit opt-in
     (`--emit-cover-evidence` / `LEC_FPV_EMIT_COVER_EVIDENCE=1`) to avoid
     implicit pseudo-reachability signals from case-level LEC runs.
   - added focused regressions:
     - `test/Tools/run-opentitan-fpv-circt-lec-basic.test`
       (assertion-native mapping + cover opt-in behavior)
     - `test/Tools/run-opentitan-fpv-circt-lec-failing-status.test`
       (NEQ -> `FAILING` assertion mapping).

### OpenTitan DVSIM-Equivalent Formal Plan (CIRCT Backend) — February 14, 2026

#### Goal

Enable a CIRCT-native command flow that is operationally equivalent to:

- `util/dvsim/dvsim.py hw/top_earlgrey/formal/top_earlgrey_fpv_*.hjson --select-cfgs ...`
- `util/dvsim/dvsim.py hw/top_earlgrey/formal/chip_conn_cfg*.hjson`

while keeping the backend generic (reusable across OpenTitan, sv-tests,
verilator-verification, and yosys corpora).

#### Architecture Commitments (long-term)

1. Build a generic formal orchestration core with pluggable project adapters.
2. Treat OpenTitan as the first full adapter, not a hard-coded one-off path.
3. Keep artifacts schema-versioned and strict-gate-governed from day one.
4. Preserve deterministic run identity (inputs, options, contracts, toolchain).

#### 1) Native OpenTitan HJSON ingestion with `--select-cfgs` semantics

1. Deliverables:
   - new adapter module for OpenTitan formal cfg ingestion
     (target cfg HJSON + imported cfgs).
   - canonical target-selection semantics matching dvsim behavior:
     - `--select-cfgs` exact target names
     - category cfg expansion (`ip`, `prim`, `sec_cm`)
     - deterministic selection order and dedup.
   - emitted normalized manifest:
     - `formal-target-manifest.tsv/json` (target, dut, flow kind, task, rel_path,
       fusesoc core, cfg source).
2. Implementation steps:
   - parse HJSON graph and `import_cfgs` with cycle checks.
   - normalize cfg entries into a common lane model consumed by formal runners.
   - add CLI entry-point compatible subset:
     - `run_formal_all.sh --opentitan-fpv-cfg <hjson> --select-cfgs <list>`.
3. Tests:
   - lit parser fixtures for malformed/ambiguous cfgs.
   - parity fixtures against known OpenTitan cfg subsets.
   - strict-gate checks for selection drift (`target list` and `target metadata`).
4. Exit criteria:
   - selecting any target from `top_earlgrey_fpv_ip_cfgs.hjson`,
     `top_earlgrey_fpv_prim_cfgs.hjson`, and `top_earlgrey_fpv_sec_cm_cfgs.hjson`
     yields deterministic target manifests with stable IDs.

#### 2) Full FuseSoC/dvsim-style filelist + defines/include resolution

1. Deliverables:
   - formal input resolution layer that produces canonical compile contracts:
     - ordered fileset
     - include dirs
     - defines
     - top module, package roots, generated files.
   - per-target resolved compile artifact:
     - `resolved-compile-contract.tsv/json`.
2. Implementation steps:
   - implement adapter-backed resolution path for OpenTitan FPV targets
     (FuseSoC core + cfg imports + overrides).
   - preserve exact ordering semantics and generated-file handling.
   - include contract fingerprint in strict-gate baseline rows.
3. Tests:
   - fixture-based parity tests comparing resolved contracts against reference
     manifests for representative `ip`, `prim`, and `sec_cm` targets.
   - regression tests for include/define precedence and duplicate suppression.
4. Exit criteria:
   - resolved compile contracts are stable and reproducible per target, and
     strict-gate detects any unintended contract drift.

#### 3) FPV task support parity (`task`, `stopat`, `blackbox`, sec_cm flow)

1. Deliverables:
   - task profile model in formal lane contracts:
     - default FPV
     - `FpvSecCm`
     - connectivity-specific profile hooks.
   - stopat/blackbox transformation support for sec_cm workflows.
2. Implementation steps:
   - implement task-profile interpreter from cfg metadata.
   - add stopat injection pipeline:
     - stopat net selection and symbolic fault-driving wrapper generation.
   - add blackbox policy support:
     - designated modules replaced with abstract stubs during formal elaboration.
   - version and export effective task policy per case in resolved contracts.
3. Tests:
   - synthetic sec_cm fixtures asserting stopat/blackbox policy activation.
   - OpenTitan sec_cm smoke targets with policy provenance assertions.
   - strict-gate checks for task-policy drift.
4. Exit criteria:
   - sec_cm targets run through CIRCT with explicit policy provenance
     (`task`, `stopats`, `blackboxes`) and deterministic outcomes.

#### 4) FPV-compatible per-assertion reporting

1. Deliverables:
   - per-assertion result artifact:
     - `assertion-results.tsv/json` with stable assertion IDs and source locs.
   - summary counters aligned with FPV expectations:
     - proven, failing, vacuous, covered, unreachable, timeout, unknown.
2. Implementation steps:
   - attribute assertion/check IDs through lowering and BMC pipelines.
   - execute per-property classification runs as needed:
     - bounded check
     - induction check (where configured)
     - cover reachability checks for vacuity/coverage classification.
   - add report adapters producing FPV-like summary views.
3. Tests:
   - targeted assertion micro-benches for each status class.
   - OpenTitan module subset with known assertion outcomes.
   - strict-gate rules on assertion-status deltas by assertion ID.
4. Exit criteria:
   - target-level reports are assertion-granular and machine-comparable across
     runs; strict-gate flags assertion-level regressions, not only lane totals.

#### 5) Connectivity formal integration (`chip_conn_cfg*.hjson` + CSV)

1. Deliverables:
   - connectivity adapter that ingests chip conn cfg HJSON and CSV rules.
   - generated connectivity check suite and per-connection result artifacts.
2. Implementation steps:
   - parse connectivity CSV schema into normalized connection contracts.
   - synthesize formal checks (assert/cover assumptions) from connection rules.
   - route through shared orchestration pipeline with lane/task metadata.
3. Tests:
   - synthetic CSV fixture tests for parser + check generation.
   - OpenTitan chip connectivity subset validation with deterministic summaries.
   - strict-gate drift checks for connection-level IDs/results.
4. Exit criteria:
   - `chip_conn_cfg*.hjson` flows execute with connection-level pass/fail reports
     and strict-gate compatibility.

#### 6) Scalability model for large formal target sets

1. Deliverables:
   - shardable execution planner across targets and assertion groups.
   - run-local immutable toolchain pinning and input contract pinning.
   - content-addressed cache layers for frontend/elaboration artifacts.
2. Implementation steps:
   - add scheduler model:
     - shard by target, then by assertion batches for heavy jobs.
   - add robust process isolation:
     - run-local tool copies, per-shard workdirs, bounded retries.
   - add artifact durability:
     - normalized pathing, schema-versioned reports, resumable checkpoints.
3. Tests:
   - stress fixtures for parallel lane execution and shard replay.
   - deterministic artifact hash checks under concurrent execution.
   - throughput regressions and timeout-budget trend checks in strict-gate.
4. Exit criteria:
   - multi-target OpenTitan FPV batches run reproducibly with bounded variance,
     stable artifacts, and actionable strict-gate output.

#### Execution Phases and Milestones

1. Phase A (ingestion + selection):
   - complete item 1, deliver `--select-cfgs` parity and target manifests.
2. Phase B (compile contract parity):
   - complete item 2, deliver resolved compile contracts with fingerprints.
3. Phase C (task parity, sec_cm):
   - complete item 3, deliver stopat/blackbox policy execution + provenance.
4. Phase D (assertion-granular FPV reports):
   - complete item 4, deliver per-assertion status and drift gating.
5. Phase E (connectivity):
   - complete item 5, deliver chip connection cfg+CSV execution path.
6. Phase F (scale hardening):
   - complete item 6, deliver sharding, artifact stability, and runtime SLAs.

#### Execution Status (February 14, 2026)

1. Phase A is complete:
   - `run_formal_all.sh` supports `--opentitan-fpv-cfg` + `--select-cfgs`
     selection semantics with deterministic target manifests.
   - `--opentitan-fpv-cfg` is now repeatable, enabling deterministic
     multi-bundle target selection across FPV category cfgs (`ip`, `prim`,
     `sec_cm`) in a single planning run.
   - duplicate target names across cfg bundles are now fail-closed when payload
     metadata diverges, preventing ambiguous target resolution.
   - unfiltered FPV execution is now available as an explicit operator choice
     (`--opentitan-fpv-allow-unfiltered`) with target-budget governance
     (`--opentitan-fpv-max-targets`).
   - `utils/select_opentitan_formal_cfgs.py` resolves cfg import graphs and
     emits stable selected-target artifacts.
2. Phase B is complete:
   - FuseSoC-backed compile-contract resolution is integrated.
   - OpenTitan FPV execution lane bootstrap is landed:
     - `--with-opentitan-fpv-bmc` (`opentitan/FPV_BMC`)
     - contracts are executable via `utils/run_opentitan_fpv_circt_bmc.py`.
3. Phase C core execution semantics are landed:
   - new generic HW pass `--hw-externalize-modules` converts selected
     `hw.module` symbols to `hw.module.extern`.
   - OpenTitan FPV BMC runner now applies task blackbox policy via
     `BMC_PREPARE_CORE_PASSES` + `hw-externalize-modules`.
   - task stopats (including hierarchical selectors) are now lowered through
     `--hw-stopat-symbolic` and exercised end-to-end in OpenTitan FPV runner
     and `run_formal_all` lanes.
4. Phase D is in progress:
   - OpenTitan FPV runner supports FPV-style assertion summary artifacts.
   - generic pairwise BMC runner now has optional assertion-granular execution
     (`BMC_ASSERTION_GRANULAR=1`) with per-assertion result artifacts.
   - assertion-granular classification now consumes explicit
     `BMC_ASSERTION_STATUS` tags, including first-class `VACUOUS` rows.
   - cover-granular execution (`BMC_COVER_GRANULAR=1`) now provides concrete
     `covered` / `unreachable` evidence for FPV summaries.
   - `run_formal_all.sh` now forwards OpenTitan FPV summary-drift governance
     controls (`baseline/drift/allowlist/fail/update`) into `opentitan/FPV_BMC`,
     with strict-gate auto-enabling drift failure when a baseline is configured.
   - OpenTitan FPV BMC launch-retry/fallback telemetry is now surfaced through
     pairwise artifacts (`BMC_LAUNCH_EVENTS_OUT`) and summarized in
     `run_formal_all.sh` lane metrics (`bmc_launch_*` counters).
   - pairwise BMC frontend now supports unified include-compilation fallback:
     - `BMC_VERILOG_UNIFIED_INCLUDE_UNIT_MODE=auto|on|off` (default: `auto`).
     - in `auto`, macro-visibility/preprocessor failures retry once via an
       auto-generated `circt-verilog.unified-include.sv` compilation unit with
       launch event reason
       `unified_include_unit_macro_visibility`.
     - unified mode now avoids duplicate source forwarding (generated unit is
       the single source argument), preventing duplicate vendor primitive
       definitions when combined with Xilinx shim fallback.
   - non-FPV launch telemetry surfacing is now wired for standard formal lanes:
     - `run_sv_tests_circt_bmc.sh` emits `BMC_LAUNCH_EVENTS_OUT`.
     - `run_verilator_verification_circt_bmc.sh` and
       `run_yosys_sva_circt_bmc.sh` now emit `BMC_LAUNCH_EVENTS_OUT` with
       retry/fallback events for frontend launch retries and copy-fallback.
     - `run_sv_tests_circt_lec.sh`,
       `run_verilator_verification_circt_lec.sh`, and
       `run_yosys_sva_circt_lec.sh` emit `LEC_LAUNCH_EVENTS_OUT`.
     - `run_formal_all.sh` now forwards and summarizes:
       - `bmc_launch_*` in `sv-tests/BMC` and `sv-tests-uvm/BMC_SEMANTICS`
       - `bmc_launch_*` in `verilator-verification/BMC` and
         `yosys/tests/sva/BMC`
       - `lec_launch_*` in `sv-tests/LEC`,
         `verilator-verification/LEC`, and `yosys/tests/sva/LEC`.
   - strict-gate launch-counter policy coupling is now enabled by default:
     - `--strict-gate` auto-enables `--fail-on-new-bmc-counter-prefix bmc_launch_`
       and `--fail-on-new-lec-counter-prefix lec_launch_`.
5. Phase E execution lane is landed:
   - new connectivity adapter utility:
     - `utils/select_opentitan_connectivity_cfg.py`
   - supports OpenTitan connectivity cfg import-graph composition and
     placeholder expansion (`{proj_root}`, `{conn_csvs_dir}`, etc.).
   - parses connectivity CSV rows into normalized rule contracts:
     - `CONNECTION`
     - `CONDITION`
   - emits deterministic artifacts:
     - `opentitan-connectivity-target-manifest.tsv`
     - `opentitan-connectivity-rules-manifest.tsv`
   - `run_formal_all.sh` now supports connectivity planning lane wiring:
     - `--with-opentitan-connectivity-parse` (`opentitan/CONNECTIVITY_PARSE`)
     - `--opentitan-connectivity-cfg`
     - `--opentitan-connectivity-target-manifest`
     - `--opentitan-connectivity-rules-manifest`
   - connectivity BMC execution lane is now wired:
     - `--with-opentitan-connectivity-bmc` (`opentitan/CONNECTIVITY_BMC`)
     - `--opentitan-connectivity-rule-filter`
     - `--opentitan-connectivity-bmc-rule-shard-count`
     - `--opentitan-connectivity-bmc-rule-shard-index`
   - connectivity BMC now emits connectivity-status evidence beyond pass/fail:
     - per-rule guard-activation cover objectives are synthesized with each
       generated connectivity checker.
     - `utils/run_opentitan_connectivity_circt_bmc.py` now supports
       cover-granular delegation into `run_pairwise_circt_bmc.py`
       (`BMC_COVER_GRANULAR` + deterministic cover shard controls).
     - `run_formal_all.sh` now records connectivity BMC cover counters in lane
       summaries (`bmc_cover_total`, `bmc_cover_covered`,
       `bmc_cover_unreachable`, ...).
   - connectivity BMC status drift governance is now wired:
     - per-rule status summary artifact:
       - `opentitan-connectivity-bmc-status-summary.tsv`
     - baseline/drift/allowlist/fail/update controls through
       `run_formal_all.sh`:
       - `--opentitan-connectivity-bmc-status-baseline-file`
       - `--opentitan-connectivity-bmc-status-drift-file`
       - `--opentitan-connectivity-bmc-status-drift-allowlist-file`
       - `--update-opentitan-connectivity-bmc-status-baseline`
       - `--fail-on-opentitan-connectivity-bmc-status-drift`
     - strict-gate auto-enables connectivity BMC status drift failure when a
       baseline is configured.
   - connectivity LEC execution lane is now wired:
     - `--with-opentitan-connectivity-lec` (`opentitan/CONNECTIVITY_LEC`)
     - `--opentitan-connectivity-rule-filter`
     - `--opentitan-connectivity-lec-rule-shard-count`
     - `--opentitan-connectivity-lec-rule-shard-index`
   - connectivity LEC status drift governance is now wired:
     - per-rule status summary artifact:
       - `opentitan-connectivity-lec-status-summary.tsv`
     - baseline/drift/allowlist/fail/update controls through
       `run_formal_all.sh`:
       - `--opentitan-connectivity-lec-status-baseline-file`
       - `--opentitan-connectivity-lec-status-drift-file`
       - `--opentitan-connectivity-lec-status-drift-allowlist-file`
       - `--update-opentitan-connectivity-lec-status-baseline`
     - `--fail-on-opentitan-connectivity-lec-status-drift`
     - strict-gate auto-enables connectivity LEC status drift failure when a
       baseline is configured.
   - connectivity cross-lane status parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_status_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-status-parity-file`
       - `--opentitan-connectivity-status-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-status-parity`
     - strict-gate auto-enables parity failure when both connectivity lanes
       are active.
   - connectivity cross-lane contract-fingerprint parity governance is now
     wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_contract_fingerprint_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_CONTRACT_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-contract-parity-file`
       - `--opentitan-connectivity-contract-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-contract-parity`
     - strict-gate auto-enables contract-parity failure when both
       connectivity lanes are active.
   - connectivity cross-lane cover-counter parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_cover_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_COVER_PARITY`
     - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-cover-parity-file`
       - `--opentitan-connectivity-cover-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-cover-parity`
     - checker compares only shared cover counters across lanes (no synthetic
       placeholders); strict-gate auto-enables cover-parity failure when both
       connectivity lanes are active.
   - connectivity cross-lane objective-level parity governance is now wired:
     - new checker utility:
       - `utils/check_opentitan_connectivity_objective_parity.py`
     - parity lane in `run_formal_all.sh`:
       - `opentitan/CONNECTIVITY_OBJECTIVE_PARITY`
   - parity controls through `run_formal_all.sh`:
       - `--opentitan-connectivity-objective-parity-file`
       - `--opentitan-connectivity-objective-parity-allowlist-file`
       - `--fail-on-opentitan-connectivity-objective-parity`
       - `--opentitan-connectivity-objective-parity-include-missing`
       - `--opentitan-connectivity-objective-parity-missing-policy`
    - checker now emits structured objective metadata rows
      (`objective_id/objective_class/objective_key/rule_id/kind`) and compares
      normalized objective-level statuses across shared objective IDs by
      default, with explicit missing-objective policy controls
      (`ignore|case|all`).
   - new connectivity runner:
     - `utils/run_opentitan_connectivity_circt_bmc.py`
     - `utils/run_opentitan_connectivity_circt_lec.py`
   - `CONNECTION` rules are synthesized into bind-check cases and executed via
     generic `run_pairwise_circt_bmc.py` with deterministic rule sharding.
   - `CONDITION` rows are now consumed as guard semantics on the owning
     connectivity assertion:
     - conditions are associated to the preceding `CONNECTION` in CSV order.
     - `--opentitan-connectivity-rule-filter` now matches both connection rule
       IDs/names and associated condition rule IDs/names.
     - guarded implication lowering uses:
       - conjunction of all expected-true condition checks
       - OR fallback over expected-false condition checks (when provided).
6. Phase F scalability bootstrap is landed for deterministic sharded execution:
   - `run_opentitan_fpv_circt_bmc.py` now supports deterministic target sharding
     (`--target-shard-count`, `--target-shard-index`).
   - `run_opentitan_fpv_circt_bmc.py` now forwards deterministic per-target case
     sharding into pairwise execution:
     - `--case-shard-count`
     - `--case-shard-index`
   - `run_pairwise_circt_bmc.py` now supports deterministic sharding at three
     levels:
     - case-level (`--case-shard-count`, `--case-shard-index`)
     - assertion-level (`--assertion-shard-count`, `--assertion-shard-index`)
     - cover-level (`--cover-shard-count`, `--cover-shard-index`)
   - `run_formal_all.sh` now forwards OpenTitan FPV shard controls via:
     - `--opentitan-fpv-bmc-target-shard-count`
     - `--opentitan-fpv-bmc-target-shard-index`
     - `--opentitan-fpv-bmc-case-shard-count`
     - `--opentitan-fpv-bmc-case-shard-index`
   - empty shard partitions now return deterministic zero-work success, which
     enables stable static sharding across heterogeneous target counts.

#### Success Definition (program-level)

1. CIRCT can run selected OpenTitan FPV targets from official formal cfg HJSON
   with dvsim-equivalent target selection semantics.
2. Reports are assertion-granular and strict-gate-enforced.
3. Connectivity cfg+CSV flows run in the same governance plane.
4. The implementation remains adapter-based and reusable outside OpenTitan.

### Remaining Formal Limitations (BMC/LEC/mutation focus)

1. **FPV status derivation depth gap**: explicit assertion statuses (`PROVEN/FAILING/VACUOUS/UNKNOWN`) and cover-granular `covered/unreachable` evidence are now wired, but fully automatic vacuity/coverage derivation for targets that do not emit explicit status tags is still pending.
2. **BMC/LEC operational robustness**: launcher retry/copy-fallback controls are telemetry-visible in OpenTitan FPV and non-FPV formal lanes (`bmc_launch_*`, `lec_launch_*`), strict-gate enforces launch-counter drift plus nonzero/max reason-event budgets and selector-based per-reason budget files (`exact|prefix|regex|*`) with allowlist-aware semantics, OpenTitan FPV policy-pack wrappers now carry canonical check-only budget-file forwarding/defaults for both BMC and LEC, and retry reason taxonomy includes `permission_denied` / `posix_spawn_failed` / `resource_temporarily_unavailable` / `stale_file_handle` / `too_many_open_files` / `cannot_allocate_memory`; remaining gap is cohort-scale budget tuning and adding further platform-specific transient classes with deterministic governance.
3. **Frontend triage ergonomics**: sv-tests BMC now preserves frontend error logs via `KEEP_LOGS_DIR`, and launch retry is in place for transient launcher failures; host-side tool relink contention can still surface as launcher-level `Permission denied`/ETXTBSY noise until binaries stabilize.
4. **Frontend scalability blocker on semantic closure buckets**: `sv-tests` UVM `16.11` (sequence-subroutine) and `16.13` (multiclock) currently hit frontend OOM in `circt-verilog` during import; this blocks clean semantic closure measurement for those buckets.
5. **Assertion/cover-granular scalability gap**: deterministic objective sharding is now available, but adaptive batch sizing, runtime-budget aware shard planning, and strict-gate policy guardrails for large targets are still pending.
6. **LEC provenance parity**: BMC resolved-contract fingerprinting is stronger than LEC/mutation lanes; strict-gate cross-lane provenance equivalence remains incomplete.
7. **Mutation cross-lane governance**: mutation strict gates are lane-scoped, but deeper policy coupling to BMC/LEC semantic buckets and resolved contracts is still pending.
8. **FPV LEC semantic depth gap**: OpenTitan FPV objective-level parity now has
   native CIRCT LEC evidence auto-production and assertion-native status
   classes (no BMC status reuse), but evidence is still case-projected rather
   than per-assertion LEC proof extraction. Cover semantics remain opt-in and
   non-reachability-native.
9. **OpenTitan macro frontend residual gap**: for representative IP targets
   (e.g. `pinmux_fpv`), retries now progress through
   `single_unit_preprocessor_failure` + Xilinx stub + unified include fallback,
   but `circt-verilog` can still terminate with
   `macro_operators_may_only_be_used_within_a_macro_definition`; remaining work
   is macro parser compatibility for complex OpenTitan macro stacks without
   placeholder shims.

### Next Long-Term Features (best long-term path)

1. Validate and tune selector-based launch reason-event budget files (and allowlists) on larger OpenTitan and non-OpenTitan cohorts, then extend retry classification with additional platform-specific transient launch classes and policy-pack defaults.
2. Extend resolved-contract artifact/fingerprint semantics to LEC and mutation runners, then enforce strict-gate drift checks on shared `(case_id, fingerprint)` tuples.
3. Add dedicated OpenTitan+sv-tests semantic-closure dashboards in strict-gate summaries (multiclock/sequence-subroutine/disable-iff/local-var buckets) to drive maturity from semantic evidence, not pass-rate alone.
4. Deepen native OpenTitan FPV LEC evidence from case-projected objective
   statuses to per-assertion proof/counterexample extraction, then add
   cover-reachability-native objective classes with strict-gate parity
   governance.

### Iteration Update (2026-02-21)

- SVA/LTL closure:
  - Added unbounded `first_match` regression:
    `test/Conversion/LTLToCore/first-match-unbounded.mlir`.
  - `LTLToCore` now lowers unbounded `first_match` with first-hit semantics:
    accepting next states define `match`, and next-state updates are masked by
    `!match` after first satisfaction.
  - reduced duplicate transition masking in first-match lowering (bounded and
    unbounded paths) by caching per-state/per-condition masks.
- Validation snapshots:
  - `ninja -C build-test circt-opt`
  - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-unbounded.mlir`
  - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/first-match-unbounded.mlir`
  - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`
  - profiling sample:
    - `time build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core` (`~0.01s`)
  - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-restrict-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_restrict bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-restrict-e2e.sv --check-prefix=CHECK-BMC`

- Additional SVA importer closure:
  - concurrent `restrict property` is now accepted and lowered as assumption
    semantics, including clocked and procedural-hoisted forms.
  - concurrent `cover sequence` is now accepted and lowered as cover semantics,
    including clocked and procedural-hoisted forms.
  - abort-style property operators (`accept_on`, `reject_on`, `sync_accept_on`,
    `sync_reject_on`) are now lowered instead of rejected.
  - `strong(...)` / `weak(...)` property wrappers are now accepted and lowered.
  - `case` property expressions are now accepted and lowered.
  - `case` property selector matching now preserves multi-bit bitvector
    equality semantics.

- Additional SVA/LTL closure:
  - sequence assertion warmup in `LTLToCore` now uses minimum sequence length
    (including unbounded-repeat forms), not only exact bounded lengths.
  - new regression:
    - `test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir`
  - sequence event-control lowering now caches per-source-state transition
    terms to reduce duplicated combinational terms in large NFA-based waits.
  - validation:
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sequence-event-control.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sequence-event-control.sv`

- Additional SVA/LTL closure:
  - `LTLToCore` now supports both-edge clock normalization for direct lowering
    of clocked sequence/property checks on `i1` clocks.
  - new regression:
    - `test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
  - validation:
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - sync abort operators (`sync_accept_on` / `sync_reject_on`) now sample the
    abort condition on assertion clocking controls, rather than lowering
    identically to async variants.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-abort-on.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-abort-on.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-abort-on.sv build-test/test/Tools/circt-bmc/sva-abort-on-e2e.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - `strong(...)` and `weak(...)` wrappers now lower differently.
  - `strong(expr)` is lowered to `ltl.and(expr, ltl.eventually expr)`.
  - `weak(expr)` preserves direct lowering.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-IMPORT`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-MOORE`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog/sva-strong-weak.sv build-test/test/Tools/circt-bmc/sva-strong-weak-e2e.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - `strong(expr)` lowers with explicit eventual-progress requirement.
  - `weak(expr)` remains direct.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-strong-weak.sv`

- Additional SVA/LTL closure:
  - empty `first_match` sequences now lower as immediate success in
    `LTLToCore`.
  - regression:
    - `test/Conversion/LTLToCore/first-match-empty.mlir`

- Additional ImportVerilog SVA closure:
  - assertion clock event-list lowering now deduplicates repeated equivalent
    clock events, avoiding redundant `ltl.clock` / `ltl.or` generation.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-clock-event-list-dedup.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - mixed sequence+signal event-list lowering now infers sequence clocking for
    unclocked sequence events when signal-event clocks are uniform.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - assertion timing controls now support sequence-valued clocking events
    (`@seq`) via sequence match-based event predicates.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - explicit assertion clocking now correctly suppresses outer default-clocking
    reapplication (prevents nested `ltl.clock` wrappers after explicit `@...`).
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-defaults-property.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-defaults-property.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-defaults.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-defaults.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - mixed sequence+signal event-list inference now also supports non-uniform
    signal clocks by expanding unclocked sequences into per-signal clocked
    variants and lowering via multi-clock event-control machinery.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - multiclock mixed event waits now emit explicit edge-specific wakeup detects
    for signal-event entries (`posedge` / `negedge`) alongside conservative
    generic wakeups.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-clock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - unclocked sequence event controls now fall back to global clocking when
    default clocking is absent.
  - applies to procedural sequence event controls, mixed sequence event lists,
    and sequence-valued assertion clocking events.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-global-clocking.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - mixed sequence event lists now support named event entries (event-typed
    expressions), e.g. `always @(s or e)`.
  - when such entries are present, lowering emits sequence-match wakeups via
    `ltl.matched` alongside direct signal/named-event wakeup detects.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-control-infer-multiclock.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - named events are now supported in assertion clock controls, including mixed
    sequence+named-event assertion clock event lists.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - composed explicit assertion clock controls are now preserved and no longer
    rewrapped by default clocking.
  - explicit LTL timing-control conversion tags roots with
    `sva.explicit_clocking` and assertion defaulting respects this.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - mixed sequence event lists now accept clocking-block entries by resolving
    them to canonical signal events (e.g. `always @(s or cb)` with
    `clocking cb @(posedge clk);`).
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - mixed sequence event lists now also resolve `$global_clock` entries through
    the scope global clocking declaration.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - assertion mixed clock event-lists now support clocking-block entries
    (`assert property (@(s or cb) c);`) with robust sequence-clock inference
    for non-assertion list members.
  - fixed regression in named-event mixed assertion lists introduced while
    extending symbol resolution (`assert property (@(s or e) d);`).
  - new regression:
    - `test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-assert-clock-list-clocking-block.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assert-clock-sequence-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-named-event.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-global-clock.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-event-list-clocking-block.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - sequence method `.matched` now lowers in assertion expressions via
    `ltl.matched` (parity with existing `.triggered` support for sequence
    methods).
  - new regression:
    - `test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-matched-method.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-moore test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sequence-event-control.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sequence-event-control.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - `$assertcontrol` now maps fail-message control types in addition to
    procedural assertion enable controls:
    - `8` => fail-message on
    - `9` => fail-message off
  - this aligns `$assertcontrol(8/9)` with existing `$assertfailon/off`
    behavior for immediate assertion action-block gating.
  - new regression:
    - `test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/system-calls-complete.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/system-calls-complete.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`

- Additional ImportVerilog SVA closure:
  - bounded unary temporal forms on property-valued operands now fail with
    explicit frontend diagnostics instead of generating invalid MLIR.
  - covered operators:
    - bounded `eventually` / bounded `s_eventually`
    - `nexttime` / `s_nexttime`
    - `always` / `s_always`
  - new regression:
    - `test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog --verify-diagnostics test/Conversion/ImportVerilog/sva-bounded-unary-property-error.sv`
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-matched-method.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='.' utils/run_yosys_sva_circt_bmc.sh`
