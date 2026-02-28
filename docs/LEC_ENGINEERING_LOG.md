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
