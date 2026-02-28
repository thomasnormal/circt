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
