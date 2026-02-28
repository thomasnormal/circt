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
