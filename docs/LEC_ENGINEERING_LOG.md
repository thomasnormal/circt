## 2026-02-28 - Connectivity LEC resource-guard retry + timing frontier

- realization:
  - OpenTitan connectivity LEC (`ast_clkmgr.csv`) was spending most wall time
    in frontend/lowering, not in Z3.
  - `circt-verilog` RSS guard failures were causing expensive split/retry churn
    in the connectivity runner.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-resource-guard-auto-relax.test`
    to reproduce first-run RSS guard failure and require in-place retry with
    raised `--max-rss-mb`.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - `LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD` (`0|1`, default `1`)
    - `LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_MAX_RSS_MB` (default `24576`)
    - `LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_RSS_LADDER_MB`
    - retry-on-RSS-guard path for `circt-verilog` with next ladder budget.
    - per-case mirror log: `circt-verilog.resource-guard-rss.log`.

- validation:
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    24/24 pass.
  - real OpenTitan Z3 replay:
    `connectivity::ast_clkmgr.csv:AST_CLK_ES_IN` -> `PASS ... EQ`.
  - direct `circt-lec --mlir-timing` profile on generated core MLIR:
    - total wall ~99.26s
    - `Run SMT-LIB via z3` ~0.007s
    - dominant passes include `FlattenModules`, `Canonicalizer`, `Mem2RegPass`.

## 2026-02-28 - Connectivity LEC bytecode frontend path + fallback

- realization:
  - For real `connectivity::ast_clkmgr.csv:AST_CLK_ES_IN`, text MLIR parse
    was still a large fixed cost in each `circt-lec` invocation.
  - A/B replay (`circt-lec` direct, same case):
    - text `.mlir`: total ~98.29s, parse ~13.46s
    - bytecode `.mlirbc`: total ~87.16s, parse ~6.31s
  - In end-to-end runner replay with bytecode enabled:
    - total `circt-lec` ~82.81s, parse ~3.12s, `PASS ... EQ` (Z3 path).

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-opt-bytecode-default.test`
    to require default `circt-opt --emit-bytecode` and `.mlirbc` input to
    `circt-lec`.
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-opt-bytecode-auto-fallback.test`
    to require auto fallback when `--emit-bytecode` is unsupported.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new env knob `LEC_OPT_EMIT_BYTECODE_MODE` (`auto|on|off`, default `auto`)
    - default path emits `connectivity.core.mlirbc` from `circt-opt`
    - `auto` fallback: on `--emit-bytecode` unknown-option failure, retry
      `circt-opt` without `--emit-bytecode`
    - per-case mirror log for fallback diagnostics:
      `circt-opt.emit-bytecode.log`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    26/26 pass.
  - real OpenTitan replay:
    `connectivity::ast_clkmgr.csv:AST_CLK_ES_IN` -> `PASS ... EQ` with
    `.mlirbc` shared core input.

## 2026-02-28 - Timeout frontier stabilization (retry budgets + fallback robustness)

- realization:
  - fallback-path timeout logging in
    `run_opentitan_connectivity_circt_lec.py` could crash when
    `subprocess.TimeoutExpired` carried `bytes` payloads (copied test-local
    runner without shared helpers).
  - low canonicalizer retry setting
    `--lec-canonicalizer-max-iterations=1` can segfault on real OpenTitan
    `AST_CLK_ES_IN` cores (direct `circt-lec` repro), so retry defaults needed
    a safer profile.
  - `AST_CLK_SNS_IN` sits on a real timeout frontier: repeated direct runs with
    identical inputs at `CIRCT_TIMEOUT_SECS=120` oscillated between PASS and
    TIMEOUT, indicating scheduler/load sensitivity, not a deterministic logic
    failure.

- TDD:
  - updated
    `test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-retry.test`
    to:
    - require copied first-timeout log preservation,
    - require rewrites-only canonicalizer retry invocation,
    - require extended retry timeout handling.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - fallback logger now coerces timeout payloads (`str|bytes|None`) robustly.
    - canonicalizer-timeout retry knobs:
      - `LEC_CANONICALIZER_TIMEOUT_RETRY_MODE` (`auto|on|off`)
      - `LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_ITERATIONS` (default `0`)
      - `LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_NUM_REWRITES` (default `40000`)
      - `LEC_CANONICALIZER_TIMEOUT_RETRY_TIMEOUT_SECS` (default `180`, applied
        only to the retry attempt when canonicalizer timeout retry is active)
    - safer default retry profile now uses rewrites-only budget unless user
      opts in to iteration override.
    - timeout retry diagnostic now reports timeout transition, e.g.
      `(timeout=60s->180s)`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    27/27 pass.
  - real OpenTitan repro:
    - `AST_CLK_SNS_IN` showed PASS/TIMEOUT jitter at 120s in repeated direct
      runs (baseline and rewrites-only).
    - forced-timeout recovery demonstrated with retry timeout override:
      `CIRCT_TIMEOUT_SECS=60` +
      `LEC_CANONICALIZER_TIMEOUT_RETRY_TIMEOUT_SECS=180`
      produced `PASS ... EQ` after canonicalizer-timeout retry.

## 2026-02-28 - Mixed-top connectivity batch isolation (per-batch source pruning)

- realization:
  - mixed-rule OpenTitan runs can require different bind tops in one invocation
    (for example local `top_earlgrey.*` rules plus `u_ast.*` rules that require
    chip-wrapper fallback).
  - global source pruning was insufficient: a local-top case could still ingest
    chip-wrapper sources and fail in `circt-verilog` with
    `CIRCT_VERILOG_ERROR`, even though the fallback case itself was valid.
  - source pruning is asymmetric:
    - local-top batches should drop chip-wrapper sources where possible.
    - fallback chip-wrapper batches must keep inner top sources.

- TDD:
  - updated
    `test/Tools/run-opentitan-connectivity-circt-lec-per-rule-top-mixed.test`
    to require:
    - mixed bind tops are split into separate frontend batches,
    - local-top batch prunes chip-wrapper sources,
    - chip-wrapper fallback batch still includes chip-wrapper source.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - `ConnectivityLECCase` now carries `bind_top`.
    - mixed-top `batch_cases` are split by `bind_top` before shared frontend
      invocation.
    - source pruning is now computed per batch (not globally):
      - default-top prune for non-fallback override batches,
      - fallback-top prune for non-fallback batches,
      - preserve full source set for fallback-top batches that require nested
        hierarchy modules.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    28/28 pass.
  - real OpenTitan Z3 replay:
    - filter: `ALERT_HANDLER_PWRMGR_ESC_CLK|AST_CLK_ES_IN`
    - result: `2/2 PASS` with isolated shared batches:
      - batch 0 wrappers bind `top_earlgrey`
      - batch 1 wrappers bind `chip_earlgrey_asic`
    - `AST_CLK_SNS_IN` spot checks after the refactor:
      - `CIRCT_TIMEOUT_SECS=120`: `PASS ... EQ`
      - `CIRCT_TIMEOUT_SECS=60`: retry path
        `(timeout=60s->180s)` then `PASS ... EQ`

## 2026-02-28 - Frontend retry propagation across connectivity batches

- realization:
  - OpenTitan connectivity runs with multiple shared frontend batches were
    repeatedly rediscovering the same retry knobs (`--max-rss-mb`,
    `--allow-multi-always-comb-drivers`) per batch.
  - this duplicated failed `circt-verilog` attempts and increased timeout risk
    on large rule sets.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-frontend-retry-propagation.test`.
  - test forces both retry classes (`resource guard triggered` and
    `always_comb procedure`) and asserts the second batch starts with learned
    flags (4 total invocations instead of 6).

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - learned per-run frontend state:
      - `learned_verilog_timescale_override`
      - `learned_verilog_allow_multi_driver`
      - `learned_verilog_max_rss_mb`
    - when a retry succeeds in one batch, later batches inherit those settings
      immediately.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    29/29 pass.
  - real OpenTitan Z3 replay:
    - filter: `ALERT_HANDLER_PWRMGR_ESC_CLK|AST_CLK_ES_IN`
    - result: `2/2 PASS`.
    - observed retries: only one
      `--allow-multi-always-comb-drivers` retry on `batch=0`; no retry on
      `batch=1` (learned state reused).

## 2026-02-28 - Canonicalizer-timeout budget propagation across cases

- realization:
  - canonicalizer-timeout handling was still per-case local: even after a first
    case proved bounded canonicalizer budget was required, later cases started
    without the budget and could re-hit timeout-first retry churn.
  - this directly increases wall time and timeout fragility on OpenTitan rule
    groups with similar LEC complexity.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-budget-propagation.test`.
  - test synthesizes two connectivity cases where unbudgeted `circt-lec`
    always times out.
  - expected behavior:
    - first case times out once, then succeeds with canonicalizer budget.
    - second case starts with learned canonicalizer budget and succeeds without
      a new timeout-first attempt.
    - total `circt-lec` invocations = 3 (not 4).

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added per-run learned state:
      - `learned_canonicalizer_timeout_budget`
    - case loop now initializes
      `lec_enable_canonicalizer_timeout_budget` from learned state.
    - on first timeout-driven canonicalizer retry, learned state is promoted so
      subsequent cases begin with bounded canonicalizer budget.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    30/30 pass.
  - real OpenTitan Z3 replay:
    - filter: `AST_CLK_SNS_IN|AST_CLK_ES_IN|AST_HISPEED_SEL_IN`
      with `CIRCT_TIMEOUT_SECS=60`.
    - result: `3/3 PASS`.
    - observed canonicalizer-timeout retry messages: 1 total for the 3-case
      run (instead of re-triggering on each case).

## 2026-02-28 - Connectivity LEC LLHD abstraction parity (default accept + retry path)

- realization:
  - real OpenTitan connectivity `ALERT_` rules (for example
    `clkmgr_cg_en.csv:CLKMGR_IO_DIV4_PERI_ALERT_3_CG_EN`) were returning
    `LEC_RESULT=UNKNOWN` with `LEC_DIAG=LLHD_ABSTRACTION`, producing hard FAIL
    rows despite being inconclusive abstraction diagnostics.
  - root cause was parity drift between wrappers:
    - `run_opentitan_circt_lec.py` already defaulted
      `--accept-llhd-abstraction`.
    - `run_opentitan_connectivity_circt_lec.py` did not.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added `LEC_ACCEPT_LLHD_ABSTRACTION` (default `1`) and automatic
      `--accept-llhd-abstraction` injection for connectivity LEC.
    - added optional LLHD abstraction retry mode
      `LEC_LLHD_ABSTRACTION_ASSUME_KNOWN_INPUTS_RETRY_MODE` (`auto|on|off`,
      default `auto`):
      - when LLHD abstraction still reports UNKNOWN and accept is disabled,
        retry once with `--assume-known-inputs`,
      - learned per-run propagation across later cases.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-llhd-default-accept.test`
    to require default `--accept-llhd-abstraction` behavior.
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-llhd-auto-retry.test`
    (with `LEC_ACCEPT_LLHD_ABSTRACTION=0`) to require LLHD UNKNOWN retry via
    `--assume-known-inputs`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    32/32 pass.
  - real OpenTitan Z3 replay:
    - filter:
      `CLKMGR_IO_DIV4_PERI_ALERT_3_CG_EN|CLKMGR_USB_PERI_ALERT_9_CG_EN`
      with `CIRCT_TIMEOUT_SECS=60`.
    - before fix: both rules ended `FAIL ... LLHD_ABSTRACTION`.
    - after fix: both rules `PASS ... LLHD_ABSTRACTION` with
      `LEC_RESULT=EQ` in case logs.
  - broader OpenTitan stress replay:
    - filter: `ALERT_`, shard `0/6`, `CIRCT_TIMEOUT_SECS=60`.
    - result: `11/11 PASS` with zero timeout rows.
    - diag split: `EQ=6`, `LLHD_ABSTRACTION=5`.
    - confirms formerly failing `clkmgr_cg_en` alert rows now pass in
      mixed `alert_handler/clkmgr/rstmgr` shard context.

## 2026-02-28 - Proactive canonicalizer budget on low-timeout Z3 runs

- realization:
  - timeout-frontier OpenTitan AST runs were still paying a deterministic
    first-case timeout in `auto` mode before learning canonicalizer budget.
  - direct timing profile for
    `connectivity::ast_clkmgr.csv:AST_CLK_SNS_IN` with bounded rewrites:
    - total wall: ~104.26s
    - `Run SMT-LIB via z3`: ~0.0071s
    - dominant wall passes remained frontend/lowering:
      - `FlattenModules` ~30.11s
      - `'hw.module' Pipeline` ~24.90s
      - `Canonicalizer` (late) ~8.85s
  - confirms timeout frontier is primarily lowering-side, not solver-side.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added
      `LEC_CANONICALIZER_TIMEOUT_RETRY_AUTO_PREENABLE_TIMEOUT_SECS`
      (default `120`).
    - when retry mode is `auto`, `LEC_RUN_SMTLIB=1`, no explicit canonicalizer
      budget is provided, and `CIRCT_TIMEOUT_SECS` is non-zero and at/below
      threshold, the run now pre-enables bounded canonicalizer rewrites from
      case 1 (instead of paying timeout-first retry churn).
    - emits explicit stderr diagnostic when this pre-enable path activates.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-auto-preenable.test`.
  - test asserts first case starts with bounded canonicalizer budget in `auto`
    mode and avoids creation of `circt-lec.canonicalizer-timeout.log`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `33/33` pass.
  - real OpenTitan Z3 replay:
    - filter: `AST_CLK_SNS_IN|AST_CLK_ES_IN|AST_HISPEED_SEL_IN`,
      `CIRCT_TIMEOUT_SECS=60`.
    - before change: first case hit timeout then retried with bounded
      canonicalizer budget.
    - after change: startup logs pre-enable diagnostic and no
      `retrying circt-lec with bounded canonicalizer budget` line appears.
    - result remains `3/3 PASS`.

## 2026-02-28 - Runner launch robustness for transient tool invocation errors

- realization:
  - broader OpenTitan shard replays can hit transient process-launch failures
    (for example `PermissionError`/`EACCES` on `build_test/bin/circt-verilog`
    during active local rebuild races), which previously surfaced as uncaught
    Python exceptions and aborted the full run.
  - these failures should be treated as normal per-case tool failures (with
    logs and `ERROR` rows), not framework crashes.

- implemented:
  - `utils/formal/lib/runner_common.py`:
    - `run_command_logged` now catches `OSError` from `subprocess.run`,
      records diagnostic log text, applies existing retry policy hooks
      (`retryable_exit_codes`/`retryable_output_patterns`), and raises a
      normalized `CalledProcessError(returncode=127)` when retries are
      exhausted.
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - fallback (no-shared-helper) `run_and_log` now mirrors the same
      normalization for `OSError` launch failures.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-tool-invoke-permission-error.test`
    to enforce:
    - no traceback on non-executable `CIRCT_VERILOG`,
    - per-case log contains `PermissionError`,
    - result row is `ERROR ... CIRCT_VERILOG_ERROR`.
  - added
    `test/Tools/formal-runner-common-tool-invoke-permission-error.test`
    to enforce shared helper normalization:
    `CalledProcessError` with return code `127` and logged `PermissionError`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    `utils/formal/lib/runner_common.py` passes.
  - `llvm-lit -sv`
    `test/Tools/run-opentitan-connectivity-circt-lec-*.test`
    `test/Tools/formal-runner-common-tool-invoke-permission-error.test`:
    `35/35` pass.

## 2026-02-28 - Persistent always_comb batch failure fast-path to singleton frontends

- realization:
  - for some OpenTitan connectivity clusters, shared frontend batching can fail
    in `circt-verilog` with
    `variable ... driven by always_comb procedure` even after auto-retry with
    `--allow-multi-always-comb-drivers`.
  - isolated single-rule replays from the same rule family can proceed through
    `circt-verilog`/`circt-opt` and reach `circt-lec`, indicating a
    batch-interaction pathology rather than a uniformly failing rule.
  - previous binary split behavior (`N -> N/2 -> ...`) incurred avoidable
    expensive failing frontend attempts on intermediate batches.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - detect persistent always_comb multi-driver failure in verilog stage after
      multi-driver retry is already active.
    - when that condition occurs for a multi-case batch, split directly into
      single-case frontend batches instead of binary halving.
    - emit explicit stderr diagnostic:
      `splitting batch into single-case frontends after persistent always_comb multi-driver failure`.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-always-comb-persistent-singleton-split.test`.
  - test models a frontend that:
    - fails for multi-rule wrappers with always_comb diagnostic,
    - succeeds for single-rule wrappers.
  - required invocation count drops from prior split churn (`8`) to direct
    singleton path (`6`).

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv`
    `test/Tools/run-opentitan-connectivity-circt-lec-*.test`
    `test/Tools/formal-runner-common-tool-invoke-permission-error.test`:
    `36/36` pass.

## 2026-02-28 - Segfault-class circt-lec retry via single-threaded MLIR fallback

- realization:
  - while probing lower canonicalizer rewrite budgets on a real OpenTitan case
    (`CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN`), `circt-lec` intermittently crashed
    with a segfault-class failure and stack trace rooted in
    `LowerLECLLVM.cpp` (`LowerLECLLVMPass::runOnOperation`).
  - this class of failure should be treated as recoverable infrastructure
    instability for runner robustness, not a hard per-case terminal error.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new retry mode knob:
      `LEC_DISABLE_THREADING_RETRY_MODE` (`auto|on|off`, default `auto`).
    - detects segfault/abort signatures (return codes `-11/-6/134/139` or
      stack-dump-style crash text) on `circt-lec` failure.
    - retries once with `--mlir-disable-threading` (unless explicitly set by
      user args), and propagates learned retry state to later cases.
    - mirrors pre-retry log to
      `circt-lec.disable-threading.log` for case-level diagnostics.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-disable-threading-auto-retry.test`.
  - test forces a first `circt-lec` segfault-like exit (`139`) and requires:
    - retry invocation includes `--mlir-disable-threading`,
    - retry-diagnostic log copy exists and contains segfault marker,
    - final case result is `PASS ... EQ`.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `36/36` pass.

## 2026-02-28 - Root-caused LowerLECLLVM segfault to use-after-free

- realization:
  - intermittent OpenTitan LEC segfaults (seen on
    `CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN` with low canonicalizer rewrite budgets)
    pointed at `LowerLECLLVM.cpp` in
    `LowerLECLLVMPass::runOnOperation`.
  - root cause is a use-after-free in
    `rewriteAllocaBackedLLHDRef`:
    - `castOp` can be erased as part of `refCasts`,
    - code then dereferenced `castOp->getParentOp()` to build
      `DominanceInfo`.

- implemented:
  - `lib/Tools/circt-lec/LowerLECLLVM.cpp`:
    - capture parent op (`domRoot`) before any cast erasure,
    - use captured parent to construct `DominanceInfo` after erasure.
  - added regression coverage:
    `test/Tools/circt-lec/lower-lec-llvm-ref-alloca-dominance-regression.mlir`
    (exercises alloca-backed multi-ref cast rewrite path).

- validation:
  - rebuilt: `utils/ninja-with-lock.sh -C build_test circt-opt circt-lec`
  - lit:
    - `llvm-lit -sv test/Tools/circt-lec/lower-lec-llvm-*.mlir`
      `test/Tools/circt-lec/lec-lower-llvm-*.mlir`: `12/12` pass.
    - `llvm-lit -sv`
      `test/Tools/run-opentitan-connectivity-circt-lec-disable-threading-auto-retry.test`:
      pass.
  - real OpenTitan stress replay (same prior crash-prone command):
    - `--lec-canonicalizer-max-num-rewrites=500`, threading enabled,
      repeated 3x on
      `connectivity.core.mlirbc` + `CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN`.
    - results:
      - run1 `rc=0`, `89.107s`, `LEC_RESULT=EQ`, `LEC_DIAG=LLHD_ABSTRACTION`
      - run2 `rc=0`, `88.632s`, `LEC_RESULT=EQ`, `LEC_DIAG=LLHD_ABSTRACTION`
      - run3 `rc=0`, `88.477s`, `LEC_RESULT=EQ`, `LEC_DIAG=LLHD_ABSTRACTION`
    - no segfault/stack-dump markers observed.

## 2026-02-28 - Canonicalizer rewrite-ladder retry on strict timeout frontiers

- realization:
  - mixed OpenTitan shard (`32-way shard index 2`) reproduces a real strict-timeout
    frontier on
    `connectivity::alert_handler_esc.csv:ALERT_HANDLER_LC_CTRL_ESC0_RST`
    at `CIRCT_TIMEOUT_SECS=90` (Z3 path).
  - direct `circt-lec --mlir-timing` on the shared core for that case shows wall
    time dominated by lowering/canonicalization, not solver:
    - `--run-smtlib`: total `137.38s`, `Run SMT-LIB via z3` `0.0074s`
    - `--emit-mlir`: total `175.84s`
    - top wall contributors: `Canonicalizer`, `FlattenModules`, and
      `'hw.module' Pipeline`.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new env knob:
      `LEC_CANONICALIZER_TIMEOUT_RETRY_REWRITE_LADDER`
      (default: `20000,10000,5000,2000,1000,500`).
    - when a case times out while canonicalizer timeout budget is already
      enabled (including auto-preenable mode), runner now retries with the
      next tighter `--lec-canonicalizer-max-num-rewrites` value from the
      ladder, instead of failing immediately.
    - learned tighter rewrite budget now propagates to later cases, avoiding
      repeated timeout-first attempts.
    - per-case timeout retry evidence is mirrored to
      `circt-lec.canonicalizer-timeout-rewrite.log`.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-rewrite-ladder.test`.
  - test enforces:
    - auto-preenabled first attempt times out at high rewrite budget,
    - retry uses tighter rewrite budget and passes,
    - learned tightened rewrite budget is reused by the second case,
    - total `circt-lec` calls are reduced (`3` calls for `2` cases).

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-*.test`:
    `4/4` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `37/37` pass.

## 2026-02-28 - Connectivity LEC symbol-pruning parity with auto fallback

- realization:
  - connectivity runner did not enable `circt-lec --prune-unreachable-symbols`,
    leaving large shared-core symbol sets active for each case even though LEC
    compares only selected `-c1/-c2` modules.
  - this is a direct gap vs commercial-style frontends that aggressively prune
    unreachable design fragments before heavy lowering/canonicalization.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new env knob:
      `LEC_PRUNE_UNREACHABLE_SYMBOLS_MODE` (`auto|on|off`, default `auto`).
    - in `auto/on`, injects `--prune-unreachable-symbols` into per-case
      `circt-lec` invocation unless user already set
      `--prune-unreachable-symbols`/`--no-prune-unreachable-symbols`.
    - `auto` fallback: if tool reports unknown option for
      `--prune-unreachable-symbols`, retry once without the flag and propagate
      learned disabled state to later cases.
    - mirrors first-failure log to
      `circt-lec.prune-unreachable-symbols.log`.

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-prune-unreachable-default.test`
    to require default flag injection.
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-prune-unreachable-auto-fallback.test`
    to require unknown-option retry and clean final success without the flag.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `39/39` pass.

- profiling note:
  - real OpenTitan direct A/B on
    `ALERT_HANDLER_LC_CTRL_ESC0_RST` under current host load (`~100` load avg,
    many concurrent formal jobs) caused both baseline and pruned direct runs to
    hit a hard `timeout 360` cap.
  - deterministic lit coverage now guards behavior; clean wall-time quantification
    of this optimization should be repeated on a quieter host.

## 2026-02-28 - No-flatten timeout frontier retry with safe fallback

- realization:
  - on strict OpenTitan timeout frontiers, flattening can dominate pre-solver
    runtime (`FlattenModules` + downstream canonicalization in prior timing
    probes), while `z3` itself may be near-zero.
  - `--flatten-hw=false` can cut that frontend cost on some cases, but is not
    universally safe because some designs still fail with
    `solver must not contain any non-SMT operations`.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new env knob:
      `LEC_NO_FLATTEN_TIMEOUT_RETRY_MODE` (`auto|on|off`, default `auto`).
    - in `auto`, for SMTLIB (non-smoke) runs with no explicit user
      `--flatten-hw*` setting:
      - if a case times out, retry once with `--flatten-hw=false`.
      - propagate learned no-flatten state to later cases after success.
    - safety fallback:
      - if a no-flatten retry fails with
        `solver must not contain any non-SMT operations`, retry once with
        flattening re-enabled and clear learned no-flatten mode.
    - new per-case retry artifacts:
      - `circt-lec.no-flatten-timeout.log`
      - `circt-lec.no-flatten-failure.log`

- TDD:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-no-flatten-timeout-retry.test`
    to require timeout-triggered no-flatten retry and learned reuse on the next
    case.
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-no-flatten-auto-fallback.test`
    to require non-SMT failure fallback back to flattened mode.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-no-flatten-*.test`:
    `2/2` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `41/41` pass.
  - direct real-case probe (same `ALERT_HANDLER_LC_CTRL_ESC0_RST` shared core,
    `--run-smtlib` with/without `--flatten-hw=false`) under current host load
    still hit hard external timeout caps before tool-complete output; behavior
    is now regression-guarded in lit while noisy-host wall-time quantification
    remains pending.

## 2026-02-28 - circt-lec no-flatten hierarchy fail-fast for runner fallback

- realization:
  - `--flatten-hw=false` currently cannot solve hierarchical compared modules;
    it eventually fails with
    `solver must not contain any non-SMT operations` at instance-like ops.
  - on large OpenTitan rules, this failure could take a full timeout slice,
    delaying the runner's automatic fallback back to flattened mode.

- implemented:
  - `tools/circt-lec/circt-lec.cpp`:
    - added an early pre-pipeline check for instance-like hierarchy in selected
      `-c1/-c2` modules when `--flatten-hw=false` and output mode requires
      solver lowering (non-`--emit-mlir`).
    - if hierarchy is present, `circt-lec` now emits
      `solver must not contain any non-SMT operations` immediately, instead of
      timing out deep in the lowering pipeline.
  - added regression:
    - `test/Tools/circt-lec/lec-no-flatten-instance-fast-fail.mlir`.
  - created toy repro (for fast local iteration):
    - `toy_models/no_flatten_non_smt_toy/simple_hier.moore.mlir`
      (`flatten` => `LEC_RESULT=EQ`, `no-flatten` => immediate non-SMT error).

- validation:
  - rebuilt `circt-lec` in `build_test`.
  - `llvm-lit -sv test/Tools/circt-lec/lec-no-flatten-instance-fast-fail.mlir`:
    pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-no-flatten-*.test`:
    `2/2` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `41/41` pass.
  - real OpenTitan direct check on
    `.../connectivity.core.mlirbc` +
    `__circt_conn_rule_0_AST_CLK_ES_IN_{ref,impl}` with
    `--flatten-hw=false --run-smtlib` now fails in ~3.5s with the expected
    non-SMT error (previously this could consume full timeout windows).

## 2026-02-28 - Auto-preenable canonicalizer budget starts at ladder head

- realization:
  - low-timeout auto-preenable (`CIRCT_TIMEOUT_SECS<=120`) previously started
    at `LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_NUM_REWRITES` (default `40000`),
    which repeatedly burned full timeout windows before converging to useful
    ladder budgets on OpenTitan timeout-frontier cases.
  - observed in real shard replay (`32-way shard index 2`, timeout `90s`):
    repeated retries from `40000 -> 20000 -> 10000` on
    `CLKMGR_INFRA_CLK_SRAM_CTRL_MAIN_OTP_CLK`, then `10000 -> 5000` on
    `CLKMGR_POWERUP_CLK_PWRMGR_LC_CLK`.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - new helper `select_auto_preenable_rewrite_budget`.
    - in low-timeout auto-preenable mode, initial rewrite budget now uses the
      highest ladder value within the max rewrite cap, instead of always using
      the raw max cap.
    - startup log now prints the chosen preenabled rewrite budget.

- TDD:
  - updated
    `test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-rewrite-ladder.test`
    to enforce ladder-head preenable behavior (`20000`) and subsequent
    tightening (`1000`).

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-canonicalizer-timeout-*.test`:
    `4/4` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `41/41` pass.
  - real OpenTitan check on
    `clkmgr_infra.csv:CLKMGR_INFRA_CLK_SRAM_CTRL_MAIN_OTP_CLK` now logs
    preenable at `max-num-rewrites=20000` and reaches `PASS` under Z3.

## 2026-02-28 - Case batching mode knob (csv vs bind-top)

- realization:
  - connectivity frontend compilation was repeated per CSV batch, even when
    cases shared the same bind top and could potentially reuse a single shared
    frontend artifact.
  - however, real OpenTitan A/B showed that aggressively merging batches by
    bind top can inflate per-case LEC workload and introduce new timeouts.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added `LEC_CASE_BATCH_MODE=csv|bind-top` (default `csv`).
    - `csv` preserves current behavior and reliability baseline.
    - `bind-top` is opt-in for experimentation with fewer frontend compiles.
    - startup summary now prints `batch_mode=...`.
    - refactored internal case-grouping via `group_cases_by_key`.
  - added regressions:
    - `test/Tools/run-opentitan-connectivity-circt-lec-batch-mode-csv.test`
    - `test/Tools/run-opentitan-connectivity-circt-lec-batch-mode-bind-top.test`
    - updated
      `test/Tools/run-opentitan-connectivity-circt-lec-frontend-retry-propagation.test`
      to pin `LEC_CASE_BATCH_MODE=csv` for deterministic multi-batch behavior.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-batch-mode-*.test`:
    `2/2` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `43/43` pass.
  - real OpenTitan smoke A/B on two rules:
    - `csv`: 2 shared frontend batches, both cases pass.
    - `bind-top`: 1 shared frontend batch, but second case hit
      `CIRCT_LEC_TIMEOUT`.
  - conclusion: keep `csv` as default; use `bind-top` only as explicit tuning.

## 2026-03-01 - Canonicalize alias-backed 4-state detection in LLHD stripping

- realization:
  - LLHD interface/signal stripping uses `isFourStateStructType` in multiple
    resolution paths.
  - that helper did not canonicalize `hw.typealias` types, so alias-backed
    `!hw.struct<value,unknown>` signals were misclassified as non-4-state.
  - consequence: the pass could spuriously inject `*_unknown` abstraction
    inputs (`multi_driver_unknown_resolution`) even when 4-state resolution was
    available, inflating `circt.lec_abstracted_llhd_interface_inputs` and
    sometimes flipping expected status from error to pass/fail drift.

- implemented:
  - `include/circt/Support/FourStateUtils.h`:
    - canonicalize type via `hw::getCanonicalType(type)` in:
      - `isFourStateStructType`
      - `getFourStateValueWidth`
  - new regression:
    - `test/Tools/circt-lec/lec-strip-llhd-signal-strength-resolve-typealias.mlir`
      validates alias-backed 4-state signal strength resolution without
      abstraction input insertion.
  - updated expectation:
    - `test/Tools/circt-lec/sv-tests-lec-smoke.mlir`
      now expects `pass=2 error=0` (previously `pass=1 error=1`), matching the
      resolved alias-backed 4-state handling.

- validation:
  - rebuilt `circt-opt` and `circt-lec` in `build_test`.
  - `llvm-lit -sv test/Tools/circt-lec/lec-strip-llhd*.mlir`: `36/36` pass.
  - `llvm-lit -sv test/Tools/circt-lec/*.mlir`: `153/153` pass.
  - real OpenTitan direct repro (`CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN`) remains
    `LEC_RESULT=UNKNOWN` with `LEC_DIAG=LLHD_ABSTRACTION`; this fix removes a
    concrete abstraction inflation class but does not yet eliminate the main
    timeout-frontier root cause on that case.

## 2026-03-01 - Scope LLHD abstraction classification to selected LEC circuits

- realization:
  - `circt-lec` classified SAT mismatches as `LLHD_ABSTRACTION` by reading the
    module-level total attr `circt.lec_abstracted_llhd_interface_inputs`.
  - that total is global for the whole design and can include unrelated modules
    not in the selected `-c1/-c2` comparison.
  - consequence: real mismatches could be downgraded to `UNKNOWN` and accepted
    by `--accept-llhd-abstraction` even when the compared circuits themselves
    had no LLHD abstraction inputs.

- implemented:
  - `lib/Tools/circt-lec/ConstructLEC.cpp`:
    - record selected-circuit abstraction count on the top module as
      `circt.lec_selected_abstracted_llhd_interface_inputs`
      (sum of `circt.bmc_abstracted_llhd_interface_inputs` on selected
      `c1`/`c2` modules).
  - `tools/circt-lec/circt-lec.cpp`:
    - prefer selected-circuit attr for LLHD abstraction classification; only
      fall back to the legacy global attr when selected metadata is absent.
    - emit `LEC_DIAG_LLHD_ABSTRACTED_INPUTS=<N>` when LLHD abstraction diag is
      reported (strict or accepted path).
  - regressions added:
    - `test/Tools/circt-lec/lec-run-smtlib-llhd-abstraction-scope-unrelated.mlir`
      (global attr unrelated to selected modules must remain `NEQ`).
    - `test/Tools/circt-lec/lec-run-smtlib-llhd-abstraction-selected-count.mlir`
      (selected module abstraction drives `UNKNOWN/LLHD_ABSTRACTION` and count).
  - updated:
    - `test/Tools/circt-lec/lec-run-smtlib-llhd-abstraction-sat-unknown.mlir`
      to encode selected-module abstraction via module attr instead of global
      module total.

- validation:
  - `llvm-lit -sv test/Tools/circt-lec/lec-run-smtlib-llhd*.mlir`: `3/3` pass.
  - `llvm-lit -sv test/Tools/circt-lec/*.mlir`: `155/155` pass.
  - targeted repro:
    - synthetic scope repro with only global attr now returns `LEC_RESULT=NEQ`
      (previously misclassified as `UNKNOWN/LLHD_ABSTRACTION`).
  - real OpenTitan repro (`CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN`) now returns
    `LEC_RESULT=NEQ` (previously `UNKNOWN/LLHD_ABSTRACTION`), showing the prior
    LLHD abstraction acceptance on this case was classification scope drift.

## 2026-03-01 - Preserve indexed-element width without illegal `$bits(<hierarchical>)`

- realization:
  - the first width-preserving fix for `rewrite_const_indexed_bit` used
    `$bits(base_expr)` and `$size(base_expr)` directly on hierarchical paths.
  - real OpenTitan run on
    `clkmgr_cg_en.csv:CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN` failed at frontend:
    `error: hierarchical references are not allowed in calls to '$bits'`.
  - this is parser-context sensitive: `$bits(dut.sig)` is rejected in these
    wrapper expressions, while `$bits(dut.sig[0])` is accepted.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - rewrote constant index lowering to avoid `$bits(<hierarchical>)` in
      constant-expression contexts.
    - new form:
      - `elem_width = $bits(base[0])`
      - shift right by `index * elem_width`
      - clear upper bits via left/right trim using
        `($size(base) - 1) * elem_width`
    - this preserves selected element width (e.g. packed `mubi4_t` arrays)
      without illegal hierarchical `$bits(base)`.
  - test updates:
    - `test/Tools/run-opentitan-connectivity-circt-lec-bit-select-rewrite.test`
    - `test/Tools/run-opentitan-connectivity-circt-lec-indexed-array-width-rewrite.test`
    - both now assert:
      - no raw `[index]` remains,
      - no lossy `& 1'b1` pattern,
      - width handling uses `$bits(base[0])` + `$size(base)`,
      - illegal `$bits(base)` form is absent.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
    passes.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-bit-select-rewrite.test`
    and
    `test/Tools/run-opentitan-connectivity-circt-lec-indexed-array-width-rewrite.test`:
    `2/2` pass.
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`:
    `44/44` pass.
  - real OpenTitan Z3 repro (no smoke):
    - command uses `LEC_RUN_SMTLIB=1`, `LEC_SMOKE_ONLY=0`,
      `LEC_ACCEPT_LLHD_ABSTRACTION=0`, `CIRCT_TIMEOUT_SECS=120`.
    - result:
      `FAIL connectivity::clkmgr_cg_en.csv:CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN ... NEQ`
      with `circt-lec` log `LEC_RESULT=NEQ`.
    - importantly, this now reaches solver result (no `CIRCT_VERILOG_ERROR`).

## 2026-03-01 - Add OpenTitan connectivity LEC batch precheck fast path

- realization:
  - real OpenTitan `ast_clkmgr` Z3 runs were spending most wall time before
    solver invocation, repeatedly per rule.
  - single-case profiling on
    `__circt_conn_rule_0_AST_CLK_SYS_OUT_{ref,impl}` with `--emit-mlir`
    showed `~111s` wall with dominant passes:
    - `Canonicalizer`: `~67s`
    - `FlattenModules`: `~24.8s`
    - `Mem2RegPass`: `~22.5s`
  - this means N per-rule LEC invocations multiply the same heavy lowering
    cost, even when all rules eventually prove `EQ`.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added `LEC_BATCH_PRECHECK_MODE` (`auto|on|off`, default `auto`)
    - added `LEC_BATCH_PRECHECK_MIN_CASES` (default `4`)
    - each shared batch can now synthesize aggregate wrappers:
      - ref: vector `result='1`
      - impl: one `result[i] = <rule_assertion_i>` per rule
    - runner executes one aggregate `circt-lec` precheck before per-rule runs.
      - if precheck proves `EQ` (or succeeds in smoke mode), all rules in that
        batch are marked `PASS` immediately.
      - otherwise (NEQ/UNKNOWN/error/timeout), runner falls back to existing
        per-rule flow unchanged.
    - refactored wrapper synthesis to keep/store per-rule assertion
      expressions for aggregate wrapper generation.

- regressions:
  - added `test/Tools/run-opentitan-connectivity-circt-lec-batch-precheck-pass.test`
    - proves aggregate PASS short-circuits per-rule invocations.
  - added `test/Tools/run-opentitan-connectivity-circt-lec-batch-precheck-fallback.test`
    - proves NEQ aggregate result falls back to per-rule checks.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
  - `llvm-lit -sv` on:
    - `run-opentitan-connectivity-circt-lec-batch-precheck-pass.test`
    - `run-opentitan-connectivity-circt-lec-batch-precheck-fallback.test`
    - `run-opentitan-connectivity-circt-lec-verify-each-auto.test`
    - `run-opentitan-connectivity-circt-lec-always-comb-auto-preenable.test`
    - `run-opentitan-connectivity-circt-lec-resource-guard-auto-preenable.test`
  - implementation surprise + fix:
    - first real run failed batch precheck with
      `error: circuit '__circt_connectivity_batch_precheck_..._ref' ... not found`.
    - root cause: aggregate modules were emitted but not listed in
      `circt-verilog --top`, so frontend top-pruning dropped them.
    - fixed by adding aggregate precheck `--top=` entries in
      `build_shared_verilog_cmd`.
  - real OpenTitan repro after fix:
    - command:
      - `LEC_RUN_SMTLIB=1 LEC_SMOKE_ONLY=0 CIRCT_TIMEOUT_SECS=180`
      - `CIRCT_LEC_ARGS='--verify-each=false'`
      - filter: `ast_clkmgr.csv:` (`17` connectivity rules)
    - result:
      - aggregate precheck: `batch precheck PASS (batch=0, cases=17)`
      - summary: `total=17 pass=17 fail=0 xfail=0 xpass=0 error=0 skip=0`
      - only one case directory emitted (`connectivity_batch_precheck_0`),
        confirming no per-rule `circt-lec` fallback was needed.

## 2026-03-01 - Cap frontend batch size to avoid large-top import stalls

- realization:
  - very large per-CSV frontend batches can still trigger heavyweight
    `circt-verilog` behavior (high-RSS and intermittent disk-sleep pressure)
    before batch precheck can run.
  - concrete repro:
    - `clkmgr_cg_en.csv` (`22` rules) in one frontend batch showed
      pathological import pressure.
  - singleton reruns for historical frontier rules now pass (`EQ`) under real
    Z3:
    - `clkmgr_cg_en.csv:CLKMGR_IO_DIV4_PERI_ALERT_1_CG_EN`
    - `alert_handler_esc.csv:ALERT_HANDLER_LC_CTRL_ESC0_RST`
    - `clkmgr_infra.csv:CLKMGR_INFRA_CLK_SRAM_CTRL_MAIN_OTP_CLK`
    - this shifted focus to frontend scaling, not per-rule solver correctness.

- implemented:
  - `utils/run_opentitan_connectivity_circt_lec.py`:
    - added `LEC_FRONTEND_MAX_CASES_PER_BATCH` (default `18`, `0` disables).
    - proactive split of oversized same-top batches before frontend commands.
    - emits diagnostic:
      `splitting large frontend batch (size=... max=...)`.
  - this composes with existing batch-precheck so each split shard can still
    fast-pass via aggregate LEC.

- regressions:
  - added
    `test/Tools/run-opentitan-connectivity-circt-lec-frontend-max-cases-per-batch.test`
    to prove proactive split occurs and keeps per-frontend top count bounded.

- validation:
  - `python3 -m py_compile utils/run_opentitan_connectivity_circt_lec.py`
  - `llvm-lit -sv test/Tools/run-opentitan-connectivity-circt-lec-*.test`
    (`53/53` pass).
  - real OpenTitan Z3 repro:
    - `clkmgr_cg_en.csv` full group (`22` rules) with default cap.
    - observed:
      - `splitting large frontend batch (size=22 max=18)`
      - `batch precheck PASS (batch=0, cases=18)`
      - `batch precheck PASS (batch=1, cases=4)`
    - result:
      `total=22 pass=22 fail=0 xfail=0 xpass=0 error=0 skip=0`.
  - additional historical singleton check:
    - `rstmgr_rst_en.csv:RSTMGR_LC_IO_DIV4_D0_ALERT_0_RST_EN`
    - result: `PASS ... EQ` under real Z3 (previously observed as
      `CIRCT_LEC_ERROR`/`CIRCT_VERILOG_ERROR` in older artifacts).
