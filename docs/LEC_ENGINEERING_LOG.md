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
