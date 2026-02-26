# SVA Parity Checklist

Last updated: 2026-02-25
Owner: CIRCT SVA/BMC/Sim

This is the canonical SVA parity checklist for SVA work only.

## Canonical Sources

- Language / importer baseline: Yosys+Verific decomp findings (recorded in `docs/SVA_ENGINEERING_LOG.md`).
- CIRCT capability evidence: `test/Conversion/ImportVerilog`, `test/Tools/circt-sim`, `test/Tools/circt-bmc`.
- Large-suite parity: `utils/run_yosys_sva_circt_bmc.sh`, sv-tests harnesses.

## Status Legend

- `DONE`: implemented with regression coverage.
- `PARTIAL`: implemented for common paths; edge semantics or diagnostics still missing.
- `GAP`: missing or unproven.

## Snapshot Matrix

| Area | Status | Evidence | Remaining Work |
|---|---|---|---|
| Core directives (`assert/assume/cover/restrict/expect`) | DONE | `test/Conversion/ImportVerilog/sva-restrict-property.sv`, `test/Tools/circt-bmc/sva-assume-e2e.sv`, `test/Tools/circt-bmc/sva-cover-sat-e2e.sv`, `test/Tools/circt-bmc/sva-expect-e2e.sv` | Keep expanding sv-tests parity, no architectural gap identified. |
| Delay/repeat families (`##`, `[*]`, `[->]`, `[=]`, unbounded forms) | DONE | `sva-repeat-*`, `sva-goto-*`, `sva-nonconsecutive-*` suites under `test/Tools/circt-bmc` and runtime tests under `test/Tools/circt-sim` | Continue stress tests for deep NFA/state explosion scenarios. |
| `first_match`, `intersect`, `throughout`, `within` | DONE | `test/Tools/circt-sim/sva-firstmatch-*.sv`, `sva-intersect-*.sv`, `sva-throughout-runtime.sv`, `sva-within-implication-*.sv` | Add more mixed nesting combinations to lock semantics. |
| `nexttime/s_nexttime`, `always/s_always`, `eventually/s_eventually`, `until/*` | DONE | `test/Conversion/ImportVerilog/sva-open-range-*.sv`, `test/Tools/circt-sim/sva-*-open-range-*.sv`, `test/Tools/circt-bmc/sva-strong-until.mlir` | Add very-large bound performance checks and solver scaling guardrails. |
| Abort controls (`disable iff`, `accept_on/reject_on`, sync variants) | DONE | `test/Conversion/ImportVerilog/sva-abort-on.sv`, `test/Tools/circt-bmc/sva-sync-abort-on-e2e.sv`, `test/Tools/circt-sim/sva-sync-*.sv` | Keep proving async pulse edge-cases and VCD observability paths. |
| Sampled-value funcs (`$past/$rose/$fell/$stable/$changed`) | DONE | `test/Conversion/ImportVerilog/sva-sampled-*.sv`, `test/Tools/circt-bmc/sva-rose-fell-vector-lsb-unsat-e2e.sv`, `sva-stable-*.sv` | Add more aggregate/packed/unpacked mixed-shape sampled regressions. |
| Sequence methods (`.matched`, `.triggered`, `.ended`) | DONE | `test/Conversion/ImportVerilog/sva-sequence-matched-method.sv`, `test/Conversion/ImportVerilog/sva-sequence-ended-method.sv`, `test/Tools/circt-sim/sva-matched-runtime.sv`, `test/Tools/circt-sim/sva-triggered-runtime.sv`, `test/Tools/circt-sim/sva-ended-runtime.sv`, `test/Tools/circt-bmc/sva-ended-e2e.sv` | Keep adding mixed clock/local-var edge-case coverage. |
| Unsupported-construct policy parity (`unsupported_sva` + continue-on-error mode) | PARTIAL | Importer supports `--sva-continue-on-unsupported` with tagged placeholder assert-like ops (`circt.unsupported_sva`), `circt-bmc` supports `--drop-unsupported-sva`, Yosys harness policy wiring supports `UNSUPPORTED_SVA_POLICY={strict,lenient}`, immediate/procedural unsupported `$past(... sampled-controls ...)` now continues in lenient mode, and non-concurrent sampled-value calls on unsupported operand types now fall back in lenient mode; see `test/Conversion/ImportVerilog/sva-continue-on-unsupported.sv`, `test/Conversion/ImportVerilog/sva-immediate-past-event-continue-on-unsupported.sv`, `test/Conversion/ImportVerilog/sva-immediate-sampled-continue-on-unsupported.sv`, `test/Tools/circt-bmc/drop-unsupported-sva.mlir`, and `test/Tools/run-yosys-sva-bmc-unsupported-sva-policy.test`. | Extend tolerant handling across remaining unsupported assertion-expression / assertion-port-binding timing-control actual paths. |
| Constraint-solver trio (`constraint-inside-basic`, `constraint-signed-basic`, `constraint-unique-narrow`) | DONE (for current lit tests) | Focused lit run on 2026-02-25: all 3 pass | Still need full solver-completeness roadmap for broader SV constraints. |
| Cross coverage (`cross-var-inline`, `cross-var-linear-sum`) | DONE (for current lit tests) | Focused lit run on 2026-02-25: both pass | Expand coverage model support beyond currently tested inline/linear forms. |
| Fork-disable/I3C runtime regressions | DONE (for current lit tests) | Focused lit run on 2026-02-25: `fork-disable-ready-wakeup`, both `i3c-samplewrite-*` pass | Keep stress-testing race-ordering and monitor patterns from AVIP/UVM. |

## Priority Work Queue

1. Unsupported-SVA tolerant mode (P0)
- Extend tolerant handling coverage to remaining importer SVA unsupported paths (primarily assertion-port-binding timing-control actual paths).
- Keep `circt.unsupported_sva` tagging stable and documented for downstream filters.

2. Infinite-time / liveness stress closure (P1)
- Add targeted tests for infinite-time sequence corner cases and liveness obligations.
- Add bounded proof obligations and diagnostics for ambiguous infinite semantics.

3. Constraint/coverage completeness (P1)
- Treat current passing lit tests as baseline, not endpoint.
- Add broader solver and cross-coverage feature matrix (range+set+cross combinations, weighted bins, scaling tests).

## Exit Criteria for "Commercial Parity"

- No open `GAP` rows in this checklist.
- All rows have deterministic regressions in ImportVerilog + sim and/or bmc (as applicable).
- Yosys/Verific decomp-only features are either implemented or explicitly superseded with documented CIRCT behavior.
- sv-tests/yosys/OVL/AVIP parity dashboards remain green with strict XPASS policy.
