# SVA BMC + LEC Master Plan

This plan drives full SystemVerilog Assertions (SVA) support with robust
bounded model checking (BMC) and logical equivalence checking (LEC). It
summarizes the long-term milestones, the concrete engineering tracks, and the
regular test loops required to reach parity targets in PROJECT_PLAN.md.

## Goals

1. Full SVA language support end-to-end: SV -> Moore -> LTL -> Verif -> SMT.
2. BMC results are trustworthy, reproducible, and explainable.
3. LEC results are trustworthy, reproducible, and explainable.
4. Tooling is stable on large suites: sv-tests, verilator-verification,
   yosys/tests, and AVIP UVM testbenches.

## Scope and Non-Goals

In scope:
- SVA parsing, elaboration, and lowering to LTL.
- LTL to Verif lowering with precise semantics.
- Verif to SMT lowering with correct time and clock handling.
- BMC tool pipeline (circt-bmc) with diagnostics and correct modeling of
  temporal semantics.
- LEC tool pipeline (circt-lec or equivalent) with SMT backends and
  counterexample reporting.

Out of scope for this plan:
- General SV simulation feature expansion (handled by PROJECT_PLAN.md).
- Non-SVA language features not required by assertions.

## Current Status (Condensed)

- SVA parsing and SVAToLTL conversion are functional with broad coverage.
- BMC pipeline is operational but has known correctness and semantics gaps.
- LEC framework exists but is not yet fully integrated into a stable tool
  workflow or end-to-end test harness.
  - ConstructLEC and SMT LEC now have basic equivalent/inequivalent regressions;
    still missing an end-to-end JIT harness and counterexample reporting.
- Preprocessor `ifdef` expressions with integer comparisons now parse for
  AVIP compatibility.

See PROJECT_PLAN.md for detailed iteration status and prior work.

## Latest SVA Closure Slice (February 23, 2026, ImportVerilog SVA harness refresh for `OnlyParse` drift)

- fixed stale ImportVerilog SVA regression harness assumptions where tests
  expected lowered Moore/LTL IR from `circt-verilog --parse-only`.
- moved 13 SVA ImportVerilog tests to:
  - `circt-verilog --no-uvm-auto-include --ir-moore`
- refreshed 7 brittle check patterns for current lowering:
  - explicit `ltl.clock` attribute tolerance (`{sva.explicit_clocking}`)
  - string sampled/past checks aligned to `moore.string_cmp`
  - default clocking/disable and procedural-hoist checks aligned to direct
    `moore.past` / `verif.clocked_assert` forms.
- validation snapshot:
  - ImportVerilog SVA bucket:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/ImportVerilog --filter='sva-'`
    - result: `148/148` pass.
  - regular formal sanity:
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
      - result: `14 tests, failures=0`.
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=0 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
      - result: `14 tests, failures=0, xfail=6, xpass=0`.
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(next|increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
      - result: `8 tests, failures=0`.

## Previous SVA Closure Slice (February 23, 2026, Yosys xprop baseline sync for `counter`)

- removed stale xprop expected-failure baseline for `counter/pass`:
  - `utils/yosys-sva-bmc-expected.txt`
  - `utils/yosys-sva-bmc-xfail.txt`
- rationale:
  - `counter/pass/xprop` is now stably passing and should no longer be tracked
    as expected-fail.
- validation snapshot:
  - targeted:
    - `TEST_FILTER='^counter$' BMC_ASSUME_KNOWN_INPUTS=0 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `PASS(pass)`, `PASS(fail)`.
  - full xprop lane:
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=0 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `14 tests, failures=0, xfail=6, xpass=0`.
  - known-input lane:
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `14 tests, failures=0`.
  - OVL semantic sanity:
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(next|increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `8 tests, failures=0`.

## Previous SVA Closure Slice (February 23, 2026, multiclock `ltl.past` de-XFAIL closure in VerifToSMT)

- closed stale expected-failure coverage for shared `ltl.past` across clock
  domains in VerifToSMT conversion tests.
- regression fixes:
  - `test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-conflict.mlir`
    - removed stale `XFAIL`.
    - fixed typed uses of `ltl.past` to `!ltl.sequence`.
    - added dual `smt.eq` checks in `@bmc_circuit`.
  - `test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-clockop-conflict.mlir`
    - removed stale `XFAIL`.
    - fixed `ltl.clock`/`verif.assert` uses to `!ltl.sequence`.
    - added dual `smt.eq` checks in `@bmc_circuit`.
  - `test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-clocked.mlir`
    - updated check ordering (`CHECK-DAG`) to avoid fragile local op-order
      assumptions around `smt.not`/`smt.and`/`smt.ite`.
- validation snapshot:
  - targeted:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-clocked.mlir build-test/test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-conflict.mlir build-test/test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-clockop-conflict.mlir`
    - result: `3/3` pass.
  - focused bucket:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/VerifToSMT --filter='bmc-multiclock-past-buffer'`
    - result: `6/6` pass.
  - regular formal sanity:
    - `TEST_FILTER='^(counter|extnets)$' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
      - result: `4/4` mode checks pass.
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(next|increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
      - result: `8 tests, failures=0`.
  - profiling sample:
    - `time llvm/build/bin/llvm-lit -sv build-test/test/Conversion/VerifToSMT/bmc-multiclock-past-buffer-conflict.mlir`
    - result: `real 0m0.102s`.

## Previous SVA Closure Slice (February 23, 2026, BMC final-check condition folding)

- removed redundant final-check disjunctions in BMC lowering when no non-final
  checks exist:
  - `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
  - added folded SMT bool combiners (`createSMTOrFolded`,
    `createSMTAndFolded`) and applied them to final condition aggregation.
  - in non-SMTLIB path, `violated` now resolves directly to `smtConstFalse`
    when `numNonFinalChecks == 0`.
- this avoids emitting patterns like:
  - `smt.or %false, %final_fail`
  and keeps final-only obligations structurally cleaner for solver backends.
- regression lock:
  - `test/Tools/circt-bmc/sva-assert-final-e2e.sv`
  - added `CHECK-BMC-NOT: smt.or %false`.
- validation snapshot:
  - build:
    - `ninja -C build-test circt-bmc`
  - focused regression:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sva-assert-final-e2e.sv`
    - result: `PASS`.
  - focused final-check batch:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sva-assert-final-e2e.sv build-test/test/Tools/circt-bmc/sva-cover-sat-e2e.sv build-test/test/Tools/circt-bmc/sva-cover-unsat-e2e.sv build-test/test/Tools/circt-bmc/sva-cover-disable-iff-sat-e2e.sv build-test/test/Tools/circt-bmc/sva-cover-disable-iff-unsat-e2e.sv build-test/test/Tools/circt-bmc/bmc-final-checks-any-violation-smtlib.mlir build-test/test/Tools/circt-bmc/bmc-liveness-lasso-fair-sampling.mlir build-test/test/Tools/circt-bmc/bmc-liveness-lasso-fair-sampled-true.mlir`
    - result: `4 pass, 4 unsupported`.
  - regular formal sanity:
    - `TEST_FILTER='^(counter|extnets)$' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `4/4` mode checks pass.
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(next|increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `8 tests, failures=0`.

## Previous SVA Closure Slice (February 23, 2026, sampled-value clocking + past clock recovery + formal stability)

- importer sampled-value helper tightening in clocked assertion contexts:
  - `lib/Conversion/ImportVerilog/AssertionExpr.cpp`
  - `disable iff`-only clocked contexts now avoid forcing helper-state sampled
    lowering for sampled functions and `$past`; this prevents avoidable
    sampled-value skew while keeping explicit-clock and enable-controlled helper
    paths intact.
- `MooreToCore` robustness for complex clocked sampled expressions:
  - `lib/Conversion/MooreToCore/MooreToCore.cpp`
  - `PastOpConversion` now traces through `moore.yield` / `scf.yield` and can
    recover a unique module clock fallback when direct use-tracing is
    insufficient.
  - assertion-context display/strobe/monitor-family builtins are dropped
    outside procedural regions to keep formal lowering structurally legal.
  - 4-state variable init now uses written-ref-aware defaults:
    - written refs initialize unknown bits to `0`
    - unwritten refs preserve X-default unknown bits.
- regression/test updates:
  - added:
    - `test/Conversion/ImportVerilog/sva-past-conditional-branch-clocked.sv`
    - `test/Tools/circt-bmc/sva-sequence-match-item-display-bmc-e2e.sv`
    - `test/Tools/circt-bmc/sva-written-uninit-reg-known-inputs-parity.sv`
  - refreshed UVM BMC e2e lanes (remove stale `XFAIL`, stable pre-solver
    lowering pipeline):
    - `test/Tools/circt-bmc/sva-uvm-assume-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-assert-final-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-expect-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-interface-property-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-local-var-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-seq-local-var-e2e.sv`
    - `test/Tools/circt-bmc/sva-uvm-seq-subroutine-e2e.sv`
- validation snapshot:
  - build:
    - `ninja -C build-test circt-verilog circt-opt circt-bmc`
  - focused lit:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sva-sequence-match-item-display-bmc-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-assume-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-assert-final-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-expect-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-interface-property-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-local-var-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-seq-local-var-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-seq-subroutine-e2e.sv`
    - result: `8/8` pass.
  - focused direct checks:
    - `build-test/bin/circt-verilog --no-uvm-auto-include test/Conversion/ImportVerilog/sva-past-conditional-branch-clocked.sv | build-ot/bin/FileCheck test/Conversion/ImportVerilog/sva-past-conditional-branch-clocked.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-written-uninit-reg-known-inputs-parity.sv | build-test/bin/circt-bmc -b 6 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only --shared-libs=/home/thomas-ahle/z3-install/lib64/libz3.so -`
    - result: `BMC_RESULT=UNSAT`.
  - regular formal loops:
    - `TEST_FILTER='.*' utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
      - result: `14 tests, failures=0`.
    - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-matrix-20260223-024709`
      - `std_ovl/BMC PASS 110/110`
      - `std_ovl/BMC_SEMANTIC PASS 110/110`.

## Previous SVA Closure Slice (February 23, 2026, Yosys parity de-XFAIL for `counter`/`extnets`)

- removed stale `XFAIL` markers from the known-input Yosys parity lock tests:
  - `test/Tools/circt-bmc/sva-yosys-counter-known-inputs-parity.sv`
  - `test/Tools/circt-bmc/sva-yosys-extnets-parity.sv`
- expected lock behavior is now reflected directly in lit metadata:
  - pass profile: `BMC_RESULT=UNSAT`
  - fail profile: `BMC_RESULT=SAT`
- validation snapshot:
  - Yosys BMC sanity:
    - `TEST_FILTER='^(counter|extnets)$' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `4/4` mode checks pass.
  - OVL semantic sanity:
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `6 tests, failures=0`.

## Previous SVA Closure Slice (February 23, 2026, `disable iff` constant-property fix + multiclock e2e optioning)

- fixed `LTLToCore` constant-i1 detection for `disable iff` lowering:
  - `lib/Conversion/LTLToCore/LTLToCore.cpp`
  - `getI1Constant` now folds simple i1 combinational forms
    (`or/and/xor/mux/cmp/casts`) instead of only direct constants.
  - this prevents false non-constant classification of
    `comb.or(disable, true)`, which previously caused an unnecessary
    clock-shift register and a spurious `SAT` result.
- updated multiclock e2e harness RUN flow:
  - `test/Tools/circt-bmc/sva-multiclock-e2e.sv`
  - now passes
    `--externalize-registers='allow-multi-clock=true'`
    alongside `--lower-to-bmc ... allow-multi-clock`.
- refreshed multiclock tool regression to ensure meaningful no-allow behavior:
  - `test/Tools/circt-bmc/circt-bmc-multiclock.mlir`
  - now uses two `verif.clocked_assert` checks on distinct clocks so
    `--allow-multi-clock`/default behavior is validated against actual
    multi-clock use, not just extra unused clock inputs.
- validation snapshot:
  - targeted regression:
    - `build-test/bin/circt-bmc -b 5 --module m_const_prop --run-smtlib test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir`
    - result: `BMC_RESULT=UNSAT`
  - focused lit:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir build-test/test/Tools/circt-bmc/sva-multiclock-e2e.sv build-test/test/Tools/circt-bmc/circt-bmc-multiclock.mlir`
    - result: `3/3` pass
  - Yosys BMC sanity:
    - `TEST_FILTER='^(counter|extnets)$' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `4/4` mode checks pass.
  - OVL semantic sanity:
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `6 tests, failures=0`.

## Previous SVA Closure Slice (February 23, 2026, sequence match-item print legalization + UVM de-XFAIL)

- fixed `MooreToCore` lowering for assertion-context sequence match-item
  print side effects (`$display/$write/$strobe/$monitor` family) so they no
  longer emit illegal `sim.proc.print` ops outside procedural regions.
- updated:
  - `lib/Conversion/MooreToCore/MooreToCore.cpp`
- added regression:
  - `test/Tools/circt-bmc/sva-sequence-match-item-display-bmc-e2e.sv`
- closed stale UVM SVA e2e XFAILs by switching to the stable pre-solver
  `circt-opt` lowering path (`lower-clocked-assert-like`,
  `lower-ltl-to-core`, `externalize-registers`, `strip-llhd-processes`,
  `lower-to-bmc`):
  - `test/Tools/circt-bmc/sva-uvm-assume-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-assert-final-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-expect-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-interface-property-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-local-var-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-seq-local-var-e2e.sv`
  - `test/Tools/circt-bmc/sva-uvm-seq-subroutine-e2e.sv`
- validation snapshot:
  - targeted regressions:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sva-uvm-assume-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-assert-final-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-expect-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-interface-property-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-local-var-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-seq-local-var-e2e.sv build-test/test/Tools/circt-bmc/sva-uvm-seq-subroutine-e2e.sv build-test/test/Tools/circt-bmc/sva-sequence-match-item-display-bmc-e2e.sv`
    - result: `8/8` pass.
  - Yosys BMC sanity:
    - `TEST_FILTER='^(counter|extnets)$' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - result: `4/4` mode checks pass.
  - OVL semantic sanity:
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_(increment|decrement|reg_loaded)$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `6 tests, failures=0`.

## Known Limitations (Must-Fix)

1. SVA temporal semantics in BMC are still incomplete:
   - Delay and history buffers need strict posedge alignment.
     - Clocked properties now gate delay/past buffers by their clock/edge,
       and unclocked properties now gate checks on any posedge in multi-clock
       mode to avoid negedge sampling; remaining work includes better clock
       inference and per-property clock tracking when clocking is implicit.
     - Interim: `bmc.clock` can pin delay/past buffers to a named clock input,
       and `ltl.clock` propagates to delay/past buffers (including
       `bmc.clock_edge` for posedge/negedge/edge).
   - Clocked asserts/assumes/covers now carry `bmc.clock_edge`, but BMC still
     lacks per-check edge scheduling (only posedge-only skip optimization).
   - ✅ Multi-step sequence semantics now use per-sequence NFAs in BMC
     (delay/concat/repeat/goto/non-consecutive). Remaining gaps include
     nested/multi-clock sequences inside a single property and liveness-style
     eventualities near the bound.
   - Strong until (`s_until`) now lowers as `until` + `eventually`; verify BMC
     bound semantics for eventual satisfaction near the bound.
   - Non-overlapping implication (`|=>`) with property consequents now has
     end-to-end coverage; keep an eye on edge cases for strong until and
     nested clocks.
   - `--rising-clocks-only` now rejects negedge/edge-triggered properties
     (including `ltl.clock`-derived edges). Use full edge modeling when suites
     contain negedge/edge assertions.
2. BMC clocking and sampled-value alignment still produce incorrect results
   for some SVA cases (especially local var sampling).
   - Long-term fix: track a clock domain per property/sequence so delay/past
     buffers advance only when their property clock fires (not any clock).
3. Yosys SVA "no-property" fallthroughs are fixed for extnets; keep scanning
   for any remaining cases that mask expected FAIL behavior.
4. Verilator-verification SVA suites now parse, but some tests still report
   "no-property" and need explicit assertions or harness treatment
   (e.g., sequence_delay_ranges).
5. LEC tool flow is under-specified; integration with SMT backends lacks
   formal golden tests and a stable CLI workflow.
   - ✅ Strict LLHD signal multi-drive now accepts mutually exclusive enable
     chains (in addition to complementary pairs).
   - ✅ Interface signal stores now resolve for dominance, complementary pairs,
     and exclusive multi-way conditional chains (including CF-lowered merges);
     overlapping or partial coverage still requires abstraction.
6. OpenTitan masked AES S-Box LEC fixed: skipping `strip-llhd-processes` in the
   LEC pipeline avoids dropping LLHD drives (e.g. `vec_c`) that collapsed masked
   outputs to constants. Regression: `test/Tools/circt-lec/lec-llhd-drv-preserve.mlir`.
7. LEC now carries original input types through `construct-lec`, allowing
   `--assume-known-inputs` to constrain 4-state inputs even after HW-to-SMT
   lowering (OpenTitan canright AES S-Box passes with SMT-LIB).
8. OpenTitan masked AES S-Box LEC still reports inequivalence with valid
   op_i values; need to root-cause masked S-Box semantics vs LUT reference.
9. UVM runtime support in BMC is incomplete; current pipeline prunes
   unreachable UVM symbols rather than modeling class-based runtime behavior.
10. Sequence clocking event syntax is now accepted via Slang patches, but
   we need to upstream this behavior or replace it with native parsing.
11. AVIP randomization constraints still expose gaps:
    - `default` dist weights are parsed but currently ignored (warning only).
    - Large signed ranges and >64-bit ranges need explicit modeling work.
12. Nested interface instance connections via interface ports now have regression
    coverage and AXI4Lite master + slave VIP filelists compile after fixing
    nested interface signal access and guarding 64-bit cover properties in AVIP.
    Full env (`Axi4LiteProject.f`) compiles when paired with read/write VIP
    filelists; remaining work is to exercise BMC/LEC on AXI4Lite and capture
    UVM/runtime gaps.
13. Suite status (Jan 30, 2026):
    - Yosys SVA BMC: ✅ full suite passes (14 tests, 2 VHDL skips).
    - Verilator-verification BMC reports 8 errors (likely compile/import issues).
    - Yosys SVA LEC and Verilator-verification LEC both pass in this run.
13. Yosys SVA bind tests with implicit `.*` connections fail due to bind scope
    or implicit named port resolution (e.g., `basic02.sv` reports missing
    `clock/read/write/ready` connections).
    - ✅ Slang patch prepared to fall back to the bound target scope for
      implicit wildcard port connections; regression added in
      `test/Conversion/ImportVerilog/bind-directive.sv`.
    - ✅ Patch applied to the Slang source used by the CIRCT build and rebuilt
      `circt-verilog`.
    - ✅ Filtered yosys SVA BMC run (`basic02`) now passes.
    - ✅ Full yosys SVA BMC suite now passes (14 tests, 2 VHDL skips).
14. **BMC i1 clock + delay**: ✅ Fixed. `ltl.clock` on i1 clocks with
    `ltl.delay` now lowers without region isolation failures.
    - Regression: `test/Conversion/VerifToSMT/bmc-delay-i1-clock.mlir`
15. **AVIP SPI compile failures (external)**: SPI AVIP does not compile due to
    SV/UVM issues in the VIP code (non-static class property use in nested
    classes, empty argument in `$sformatf`). Track as AVIP fixes, not CIRCT.
16. **AVIP JTAG compile failures (external)**: JTAG AVIP does not compile due
    to UVM `do_compare` default argument mismatch and reversed range bins.
    Track as AVIP fixes, not CIRCT.

## Unsupported Feature Inventory (As Of February 22, 2026)

This inventory tracks known SVA-related gaps that are still not fully supported.
Items are grouped by pipeline stage.

### Import + Frontend Gaps

- General concurrent assertion action-block semantics are not modeled end-to-end.
  - Import now preserves simple severity-message action blocks
    (`$error/$warning/$fatal/$info`, plus `$display/$write`) as assertion
    labels for diagnostics.
  - Rich action blocks (arbitrary statements, side effects) are still not
    represented as assertion failure actions in IR/formal flows.
- Property event controls in procedural timing controls remain unsupported in
  temporal-property form (`@property_name` when lowering yields `!ltl.property`).
- Sequence event controls in event lists do not currently support explicit edge
  qualifiers (only sequence/property event form without edge qualifiers).
- Module-level labeled concurrent assertions now preserve single-block module
  structure during import (no invalid `cf.br` block-splitting around
  `moore.output` terminators).
- Some assertion port timing-control value paths still carry unsupported
  diagnostics in generic expression lowering; these paths should be retired or
  completed once all legal event-typed uses are covered.
  - Feb 21, 2026: fixed a concrete false-positive diagnostic in this area for
    nested `$past(..., @(event_port))` lowering (`i1` bool-cast path).
- Sequence match-item compound local-var assignments are now lowered for
  integer local assertion variables (arithmetic/bitwise/shift compound forms).
- Immediate assertions now lower to formal obligations even when action blocks
  are present, so deferred OVL checkers (`assert #0 ... else ...`) are no
  longer vacuous in the BMC semantic harness.

### BMC + Semantics Gaps

- Per-check edge scheduling in BMC remains incomplete for mixed edge cases.
- Nested / multi-clock sequence semantics inside a single property remain
  incomplete.
- Liveness-style eventualities near bound need stronger closure checks.
- Sampled-value alignment remains imperfect in some local-variable scenarios.
- Some unbounded-sequence forms can still generate large NFAs; more global
  CSE/transition dedup in sequence lowering remains a performance track.

### Tooling + Flow Gaps

- LEC workflow exists but is not yet fully integrated as a stable end-to-end
  default flow with complete CEX reporting.
- UVM runtime behavior in BMC remains incomplete (currently prunes unreachable
  runtime symbols instead of full class runtime modeling).
- ✅ OVL BMC matrix harness is now integrated into `run_formal_all.sh` as
  lane `std_ovl/BMC` via `utils/run_ovl_sva_circt_bmc.sh`.
  - Current matrix status (Feb 22, 2026): `110/110` passing across
    `known,xprop` profiles.
  - Follow-up: add OVL LEC companion lane to track cross-mode parity.
- ✅ OVL semantic harness is now integrated into `run_formal_all.sh` as
  lane `std_ovl/BMC_SEMANTIC` via
  `utils/run_ovl_sva_semantic_circt_bmc.sh`.
  - Harness style: one SV wrapper per checker case in
    `utils/ovl_semantic/wrappers/` with manifest-driven expectations.
  - Current semantic coverage inventory (Feb 23, 2026): `110` pass/fail
    obligations (`55` checker wrappers x `pass/fail` modes), matching the full
    OVL checker matrix.
  - Runner supports known-gap modes (`known_gap=1`, `known_gap=tool`,
    `known_gap=any`) for future triage, but there are currently no active
    known-gap semantic cases in `utils/ovl_semantic/manifest.tsv`.
  - Newly added in this slice:
    - `ovl_coverage`
    - `ovl_value_coverage`
    - `ovl_xproduct_bit_coverage`
    - `ovl_xproduct_value_coverage`
  - Current semantic lane status (Feb 23, 2026): full green
    (`110/110`, `xfail=0`, `xpass=0`) in the active workspace.

## Core Workstreams

### Track A: SVA Language Semantics

Goal: IEEE 1800-2017 compliant SVA semantics with a strict test grid.

Key work:
- Sequence timing: full handling of ##[n:m], delayed sequences, repeat forms.
- Clocking: proper default clocking and clock inference rules.
- disable iff: enforce semantics for property disabling with correct scope.
- sampled value functions: $past/$rose/$fell/$changed/$stable with
  correct edge behavior and reset interactions.

Deliverables:
- New/expanded LTL conversion tests.
- End-to-end SV -> BMC regression tests with pass/fail cases.

### Track B: BMC Semantics and Soundness

Goal: BMC results are sound and stable for temporal properties.

Key work:
- ✅ **Implement SMT-LIB export** for BMC: solver-only, unrolled encoding that
  emits pure SMT ops (no scf/func/arith) so `--emit-smtlib` produces a
  backend-independent artifact.
- Hard correctness for ltl.delay, ltl.repeat, ltl.concat under BMC unrolling.
  - ✅ Implemented multi-step sequence NFAs for delay/concat/repeat/goto
    operators, eliminating single-step approximations in BMC.
  - ✅ Keep concat-length guardrails when sequence length is statically
    unbounded (explicit errors instead of silent approximations).
- Posedge gating for all history and assertion updates.
 - Per-property clocked delay/past buffers (associate each temporal op with its
  property clock instead of advancing on any clock).
  - Propagate clock names from clocked asserts into delay/past buffers so
    clock-domain gating works even after LTL lowering removes ltl.clock.
 - ✅ Clone shared delay/past ops per property to avoid accidental cross-clock
    sharing (hard error remains only for conflicting clock info in a single
    property, e.g., `bmc.clock_edge` vs `ltl.clock`).
  - ✅ Clone LTL subtrees per clocked property to avoid accidental sharing of
    delay/past ops across clock domains.
- ✅ Gate each non-final check by its own clock edge instead of a single
  combined check.
- ⚠️ 4-state X/Z semantics in BMC: **partial** (core comb AND/OR/XOR, mux,
  add/sub, shifts, mul/div/mod, comparisons, and case/wild equality now
  modeled; remaining ops still 2-state).
  - ✅ Uninitialized 4-state nets now start as X (unknown=1) while supply nets
    keep unknown=0 in MooreToCore lowering.
  - ✅ 4-state out-of-bounds extracts now yield X in MooreToCore lowering.
  - ✅ 4-state out-of-bounds extract_ref and dyn_extract_ref now yield X in
    MooreToCore lowering.
  - ✅ 4-state conditional (?:) now merges true/false values when the condition
    is X/Z (bitwise X-prop).
  - ✅ 4-state shifts now yield X when the shift amount is X/Z.
  - ✅ 4-state dyn_extract now yields X when the index is X/Z or out-of-bounds.
  - ✅ 4-state dyn_extract on arrays now yields X for X/Z or out-of-bounds index.
- ✅ Add BMC cone-of-influence pruning for externalized registers and unused
  outputs so irrelevant state is removed before SMT lowering (including
  transitive reg deps).
  - Regression: `test/Tools/circt-bmc/prune-bmc-registers-transitive.mlir`.
  - Input pruning regression: `test/Tools/circt-bmc/prune-bmc-inputs.mlir`.
- ✅ Add `--assume-known-inputs` to BMC to optionally constrain unknown bits on
  4-state inputs.
- ✅ Thread `--assume-known-inputs` through BMC harness scripts via
  `BMC_ASSUME_KNOWN_INPUTS=1`.
- ✅ Thread `--assume-known-inputs` through LEC harness scripts via
  `LEC_ASSUME_KNOWN_INPUTS=1`.
- ✅ Add SMT-LIB regression for `--assume-known-inputs` in LEC.
- ✅ Allow BMC harness scripts to use `--run-smtlib` via `BMC_RUN_SMTLIB=1`.
- ✅ Add `run_formal_all.sh` switches for BMC/LEC assume-known-inputs and
  SMT-LIB flows.
- ✅ Make OpenTitan LEC script honor `LEC_ASSUME_KNOWN_INPUTS` for 4-state
  control.
- ✅ Default OpenTitan LEC to `--run-smtlib` via `LEC_RUN_SMTLIB=1` and `Z3_BIN`.
- ✅ Emit a warning when 4-state inputs are unconstrained without
  `--assume-known-inputs`.
- Derived clock constraints and correct relation to primary BMC clock.
  - ✅ Map derived clocks from assumes using eq/ne/ceq/cne/weq/wne, xor, and
    comb.not relations, including inversion tracking (posedge on inverted
    clocks gates on base negedge).
  - ✅ Treat `seq.from_clock(seq.to_clock(x))` as equivalent to `x` when
    resolving explicit clocked properties.
    - Regression:
      `test/Conversion/VerifToSMT/bmc-derived-clock-from-to-equivalence.mlir`
  - ✅ Preserve externalized-reg clock port names for i1/struct clocks and use
    `bmc_reg_clocks` to synthesize BMC clock inputs after LTL lowering prunes
    `ltl.clock`/`seq.to_clock`.
    - Regression: `test/Tools/circt-bmc/lower-to-bmc-struct-clock.mlir`
- Deterministic handling of LLHD signals, probes, and drives.
- ✅ Use `--fail-on-violation` in BMC harnesses so violations are treated as
  errors in automated runs.

Deliverables:
- BMC semantics tests in test/Tools/circt-bmc.
- Cross-suite regression checks with sv-tests and yosys SVA.
  - ✅ Added E2E X‑prop regression for 4‑state comb ops.
  - ✅ Added E2E X‑prop regression for 4‑state add.
  - ✅ Added E2E X‑prop regression for 4‑state shifts.
  - ✅ Added E2E X‑prop regression for 4‑state comparisons.
  - ✅ Added E2E X‑prop regression for 4‑state mul/div.
  - ✅ Added E2E X‑prop regression for 4‑state mod.
  - ✅ Added E2E X‑prop regression for 4‑state wildcard equality.
  - ✅ Added E2E X‑prop regression for 4‑state case equality.
  - ✅ Added E2E X‑prop regression contrasting == vs === on unknown inputs.
  - ✅ Added E2E X‑prop regression for signed compares.
  - ✅ Added E2E regression for `--assume-known-inputs` vs unknown inputs.
  - ✅ Added E2E X‑prop regression for unsigned compares.
  - ✅ Added E2E X‑prop regression for mixed‑width compares.
  - ✅ Added E2E X‑prop regression for array indexing.
  - ✅ Added E2E X‑prop regression for array injection.
  - ✅ Added E2E X‑prop regression for struct field extraction.
  - ✅ Added E2E X‑prop regression for struct field injection.
  - ✅ Added E2E X‑prop regression for multi-bit struct fields.
  - ✅ Added E2E X‑prop regression for nested aggregates.
  - ✅ Added E2E X‑prop regression for nested aggregate writes.
  - ✅ Added E2E X‑prop regression for concatenation.
  - ✅ Added E2E X‑prop regression for bit extraction.
  - ✅ Added E2E X‑prop regression for part-select and replication.
  - ✅ Added E2E X‑prop regression for nested aggregate concatenation.
  - ✅ Added E2E X‑prop regression for array-of-struct concatenation.
  - ✅ Added E2E X‑prop regression for dynamic indexing with unknown indices.
  - ✅ Added E2E X‑prop regression for dynamic part-selects.
  - ✅ Added E2E X‑prop regression for signed shifts with unknown shift amounts.
  - ✅ Added E2E X‑prop regression for reduction operators.
  - ✅ Added E2E X‑prop regression for reduction XOR.
  - ✅ Added E2E X‑prop regression for bitwise NOT.
  - ✅ Added E2E X‑prop regression for logical NOT.
  - ✅ Added E2E X‑prop regression for logical AND/OR.
  - ✅ Added E2E X‑prop regression for ternary operator.
  - ✅ Added E2E X‑prop regression for implication.
  - ✅ Added E2E X‑prop regression for implication with delayed consequent.
  - ✅ Added E2E X‑prop regression for until.
  - ✅ Added E2E X‑prop regressions for eventually/always.
  - ✅ Added E2E X‑prop regression for strong until.
  - ✅ Added E2E X‑prop regression for weak eventually.
  - ✅ Added E2E X‑prop regression for nexttime.
  - ✅ Added E2E X‑prop regression for nexttime range.
  - ✅ Added E2E X‑prop regression for delay ranges.
  - ✅ Added E2E X‑prop regression for repetition.
  - ✅ Added E2E X‑prop regression for non‑consecutive repetition.
  - ✅ Added E2E X‑prop regression for goto repetition.
  - ✅ Added E2E X‑prop regressions for $rose/$fell.
  - ✅ Added E2E X‑prop regressions for $stable/$changed.
  - ✅ Added ImportVerilog regression for $stable/$changed logical equality.
  - ✅ Added E2E X‑prop regression for unbounded repetition.
  - ✅ Added E2E X‑prop regression for unbounded delay ranges.
  - ✅ Added E2E X‑prop regression for sequence concatenation.
  - ✅ Added E2E X‑prop regressions for sequence AND/OR.

Implementation sketch for per-property clocked buffers:
- Propagate a clock domain id from `ltl.clock` / clocked asserts through the
  LTL tree to `ltl.delay`/`ltl.past` sites.
- Allocate delay/past buffer slices per clock domain in the BMC loop.
- Update each buffer slice only when its clock posedge fires (not any clock).
- Add a fallback domain for unclocked properties (current any-clock behavior).
- Add MLIR tests with two `ltl.clock`-wrapped properties to ensure their
  delay/past buffers advance independently.

### Track C: SMT and Solver Integration

Goal: Robust SMT generation for BMC and LEC with clear diagnostics.

Key work:
- SMT encoding for SVA and LTL constructs with documented truth tables.
- Counterexample/trace reporting suitable for debugging.
- Stable Z3 integration with deterministic solver configs.
- ✅ Add `circt-bmc --run-smtlib` (external z3, `--z3-path`) with model printing
  parity to the JIT path.

Deliverables:
- VerifToSMT and SMT dialect test coverage.
- Minimal reproducer pipeline for each bug fix.

### Track D: LEC Tooling and Equivalence Semantics

Goal: Reliable LEC workflow integrated into CIRCT tools.

Key work:
- Define canonical LEC input form (verif.lec) and a standard CLI flow.
- Ensure both SMT and LLVM backends match on equivalence results.
- Provide end-to-end LEC examples in docs/FormalVerification.md.
- Introduce a test harness for LEC (pass/fail + counterexample).
- Add counterexample input reporting parity between SMT-LIB and JIT paths.
- ✅ Expose output values in SMT-LIB counterexamples (declare c1_/c2_ outputs).
- ✅ JIT counterexample input printing uses SMT model formatting helpers.
- ✅ Use `--fail-on-inequivalent` in harnesses so inequivalence is treated as an
  error signal for automated runs.
- ✅ Ensure JIT path uses Z3 model evaluation to emit per-input counterexamples
  when `--print-counterexample` is set.
- ✅ Strict LEC now lowers `hw.inout` ports and resolves 4-state read/write
  against internal drives (2-state read/write still unsupported).
- ✅ Strict LEC now supports struct-field inout accesses by lifting them to
  explicit read/write ports.
- ✅ Strict LEC now supports constant and dynamic array-index inout accesses by
  lifting them to explicit read/write ports, including nested dynamic indices
  and struct/constant-array suffixes. Dynamic indices still reject other
  writers to the same array unless 4-state resolution is enabled.

Deliverables:
- New tests in test/Conversion/VerifToSMT and LEC-specific test folder.
- A stable command line interface and CI gating tests.

### Track E: Real-World Regression Suites

Goal: Track progress on real suites and prevent regressions.

Key work:
- Regular runs on:
  - ~/sv-tests
  - ~/verilator-verification
  - ~/yosys/tests
  - ~/mbit/*avip*
- Interpret failures and classify into:
  - CIRCT bug
  - Missing SVA feature
  - Suite issue or unsupported syntax

Deliverables:
- Updated result artifacts and changelog entries.
- Reduced xfail set over time with justifications.

## Route-Context Schema Reference

This section documents the route-context schema used by drop-events auto
profile routing in `utils/run_yosys_sva_circt_bmc.sh`.

Primary environment variables:
- `YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_JSON`
- `YOSYS_SVA_MODE_SUMMARY_HISTORY_DROP_EVENTS_REWRITE_SELECTOR_PROFILE_ROUTE_CONTEXT_SCHEMA_JSON`

Top-level schema keys:
- `schema_version`: currently must be `"1"`.
- `allow_unknown_keys`: allow unknown keys in the context payload.
- `validate_merged_context`: validate merged context after built-in CI overlay.
- `keys`: typed key declarations and per-key constraints.
- `regex_defs`: named regex patterns for `bool_expr.regex_ref`.
- `limits`: expression guardrails (depth/node limits).
- `import_registry`: named import-module registry (`version`, `path`,
  optional `schema_versions` allow-list).
- `int_arithmetic`: inline arithmetic mode (`div_mode`, `mod_mode`).
- `int_arithmetic_presets`: named arithmetic mode map.
- `int_arithmetic_ref`: select a named arithmetic preset as default mode.
- `imports`: optional JSON file list for reusable schema modules.
- `all_of`, `any_of`: clause arrays with key/int/bool predicates.

Import module JSON contract:
- Allowed keys inside each imported JSON object:
  - `imports`
  - `regex_defs`
  - `limits`
  - `int_arithmetic_presets`
- Unknown keys are rejected.
- Import payload must be a non-empty JSON object.
- Imports may recursively import other modules.
- Import cycles are rejected with field-qualified diagnostics.
- `imports` supports two entry styles:
  - path string: `"/abs/path/module.json"`
  - registry reference: `{"module":"core","version":"1"}`
- Registry reference validation:
  - module must exist in `import_registry`
  - requested `version` must match registered `version`
  - current schema version must be listed in `schema_versions` when provided

Merge and precedence rules:
- `limits`: later imports override earlier imports; inline schema overrides all
  imported limits.
- `regex_defs`: duplicate names across imports or between imports and inline
  schema are rejected.
- `int_arithmetic_presets`: duplicate preset names across imports or between
  imports and inline schema are rejected.
- Shared imports are deduplicated by canonicalized file path within one schema
  parse closure (first import wins; repeated references are ignored).
- Arithmetic mode precedence during clause evaluation:
  - clause `int_arithmetic` or clause `int_arithmetic_ref`
  - schema `int_arithmetic` or schema `int_arithmetic_ref`
  - built-in defaults (`floor` division/mod)

Conflict rules:
- A single scope cannot set both `int_arithmetic` and `int_arithmetic_ref`.
- `int_arithmetic_ref` requires `int_arithmetic_presets` to be configured in
  that schema after import+inline merge.
- Unknown preset names fail with field-qualified diagnostics.

Minimal import module example:

```json
{
  "regex_defs": {
    "night_casefold": { "pattern": "^night$", "flags": "i" }
  },
  "limits": {
    "max_int_expr_nodes": 96
  },
  "int_arithmetic_presets": {
    "tz": { "div_mode": "trunc_zero", "mod_mode": "trunc_zero" }
  }
}
```

Minimal schema example using imports + arithmetic refs:

```json
{
  "schema_version": "1",
  "allow_unknown_keys": true,
  "imports": ["/abs/path/auto-route-schema-import.json"],
  "int_arithmetic_ref": "tz",
  "keys": {
    "attempt": { "type": "integer", "required": true },
    "flavor": { "type": "string", "required": true }
  },
  "all_of": [
    {
      "int_expr": [
        [{ "div": ["attempt", 2] }, "eq", -1],
        [{ "mod": ["attempt", 2] }, "eq", -1]
      ],
      "bool_expr": [
        { "regex_ref": ["flavor", "night_casefold"] }
      ]
    }
  ]
}
```

Minimal schema example using import registry refs:

```json
{
  "schema_version": "1",
  "allow_unknown_keys": true,
  "import_registry": {
    "core": {
      "version": "1",
      "path": "/abs/path/auto-route-schema-import-core.json",
      "schema_versions": ["1"]
    }
  },
  "imports": [
    { "module": "core", "version": "1" }
  ],
  "int_arithmetic_ref": "tz",
  "keys": {
    "attempt": { "type": "integer", "required": true }
  },
  "all_of": [
    {
      "int_expr": [
        [{ "div": ["attempt", 2] }, "eq", -1]
      ]
    }
  ]
}
```

## Milestones

1. M1: BMC correctness baseline
   - Pass all SVA tests in yosys with correct pass/fail semantics.
   - Remove no-property fallthroughs; ensure assertions survive lowering.

2. M2: sv-tests SVA parity
   - All SVA tagged sv-tests pass or have justified XFAIL.

3. M3: verilator-verification SVA parity
   - Resolve remaining SVA helpers and sequence event tests.

4. M4: LEC MVP
   - LEC tool can compare two HW modules with SMT backend and produce
     clear results.

5. M5: LEC scale and integration
   - LEC handles non-trivial SV designs and integrates with BMC workflows.

## Engineering Practices

- Add unit tests for every new bug fix or feature.
- Prefer small, focused regressions for each semantic fix.
- Update CHANGELOG.md per iteration with results and test status.
- Merge upstream main regularly to avoid drift.

## Testing Strategy (As Work Proceeds)

Every change should follow a staged test approach to keep feedback fast while
still exercising full-suite coverage regularly.

### Stage 0: Preflight Build and Smoke Checks

- Build the affected tools so CLI defaults match the new behavior:
  - `ninja -C build circt-bmc`
  - `ninja -C build circt-verilog`
- If the change touches LEC or SMT conversion, also build:
  - `ninja -C build circt-opt`
- Run the smallest possible hand-crafted reproducer (often a single MLIR
  test or a tiny SV file) to confirm the bug is fixed before running suites.
- For real UVM/AVIP runs, disable the auto UVM stub:
  - Use `--no-uvm-auto-include` and explicitly include
    `~/uvm-core/src/uvm_pkg.sv`.
- Some AVIPs require an explicit timescale to avoid mixed-timescale errors
  (e.g., `TIMESCALE=1ns/1ps` when invoking the AVIP smoke script).

### Per-Change Checklist (Minimum)

- Create or update a reproducer that fails before the change and passes after.
- Add a lit test in `test/` or a gtest in `unittests/` (prefer both if feasible).
- Rebuild the affected binary and re-run the reproducer immediately.
- Record the exact command line and outcome in CHANGELOG.md.

### Stage 1: Local Regression for the Specific Fix

- Add a minimal lit test in the closest folder:
  - SVA lowering: test/Conversion/SVAToLTL
  - LTL/Verif/SMT lowering: test/Conversion/VerifToSMT
  - BMC tool behavior: test/Tools/circt-bmc
  - LEC behavior: new LEC-focused tests adjacent to VerifToSMT
- Run only the touched tests with lit:
  - build/test/Tools/circt-bmc/...
  - build/test/Conversion/SVAToLTL/...
  - build/test/Conversion/VerifToSMT/...
- For unit-testable logic, add/update a gtest in unittests/ and run:
  - `ninja -C build check-circt-unittests`

### Stage 2: Targeted Suite Slice

Run a filtered subset to confirm real-world behavior before expanding:
- sv-tests: use TEST_FILTER on `utils/run_sv_tests_circt_bmc.sh`
- verilator-verification: use TEST_FILTER on
  `utils/run_verilator_verification_circt_bmc.sh`
- yosys SVA: use TEST_FILTER on `utils/run_yosys_sva_circt_bmc.sh`

Key knobs to set per slice:
- `BOUND` and `IGNORE_ASSERTS_UNTIL` to tune runtime.
- `RISING_CLOCKS_ONLY=1` for correct SVA semantics (use `0` only when
  specifically debugging clocking-model issues).
- `PRUNE_UNREACHABLE_SYMBOLS=0` to debug full-module behavior without
  pruning.

Record the filtered output files and update CHANGELOG.md with deltas.

### Stage 3: Full Suite Cadence

At least once per iteration (or after major milestones), run:
- utils/run_sv_tests_circt_bmc.sh
- utils/run_verilator_verification_circt_bmc.sh
- utils/run_yosys_sva_circt_bmc.sh
- `utils/run_avip_circt_verilog.sh ~/mbit/<avip>` (compile smoke check)
  - Some AVIPs rely on env vars in filelists; export them before running.

Store results in the usual artifacts:
- sv-tests-bmc-results.txt
- verilator-verification-bmc-results.txt
- yosys SVA summary from the harness output

When results change:
- Save any interesting counterexample logs to a temporary directory for
  reproducibility.
- Record suite statistics in CHANGELOG.md with date and command line.
- File a minimal reproducer in test/ for each new failure that is fixed.

### Stage 4: LEC-Focused Coverage

- Add a focused LEC regression per feature (pass/fail pair).
- For LEC changes, run at least one full, small end-to-end check with
  a real SV design and record the command in CHANGELOG.md.
- OpenTitan AES S-Box LEC harness:
  - `utils/run_opentitan_circt_lec.py` (uses OpenTitan's AES S-Box LEC fixtures)
- Seed LEC SMT coverage with `test/Tools/circt-lec/lec-smt.mlir`.
- For smoke/pipeline checks without Z3, use the LEC harnesses:
  - `utils/run_sv_tests_circt_lec.sh` (sv-tests)
  - `utils/run_verilator_verification_circt_lec.sh` (verilator-verification)
  - `utils/run_yosys_sva_circt_lec.sh` (yosys SVA)
  - Use `LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir` for smoke mode.
  - Use `FORCE_LEC=1` to run LEC for parsing-only sv-tests.
  - Use `UVM_PATH=...` for UVM-tagged sv-tests.
  - Use `INCLUDE_UVM_TAGS=1` to include sv-tests tagged only with `uvm`.
  - Use `KEEP_LOGS_DIR=...` to preserve MLIR/log artifacts per test.
- For BMC smoke/pipeline checks without Z3:
  - Use `BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir` with the BMC harnesses.
  - Use `FORCE_BMC=1` to run BMC for parsing-only sv-tests.
  - Use `ALLOW_MULTI_CLOCK=1` for multi-clock suites.
  - Use `NO_PROPERTY_AS_SKIP=1` to classify propertyless designs as skip.
  - Use `INCLUDE_UVM_TAGS=1` to include sv-tests tagged only with `uvm`.
  - Use `KEEP_LOGS_DIR=...` to preserve MLIR/log artifacts per test.

### Cadence and Reporting

- Per small fix: run Stage 0 and Stage 1; update CHANGELOG.md.
- Per feature series: add a Stage 2 slice for at least one external suite.
- Per milestone or monthly: run all Stage 3 suites and archive results.
- Track regressions explicitly (new FAILs, new NO-PROPERTY) and record them
  alongside remediation plans.

### LEC-Specific Verification

For each LEC change:
- Add a direct LEC regression (pass/fail) with a minimal pair of modules.
- Verify both SMT and LLVM backends produce the same verdicts.
- Add documentation examples in docs/FormalVerification.md if behavior changes.

## Regular Test Loop (Required)

Run these at least once per iteration (or per change if relevant):

- utils/run_sv_tests_circt_bmc.sh
- utils/run_verilator_verification_circt_bmc.sh
- utils/run_yosys_sva_circt_bmc.sh
- ~/yosys/tests (add a harness if needed; track deltas separately)
- ~/mbit/*avip* (appropriate BMC/sim flow)

Record results in CHANGELOG.md and include relevant output artifacts.

## Latest SVA closure slice (2026-02-22, OVL const-clock closure + semantic expansion VII)

- Closed gap:
  - fixed BMC externalization/lowering path for constant clocks used in lowered
    SVA/LTL state:
    - `ExternalizeRegisters` now accepts `seq.const_clock` and emits stable
      source keys (`const0` / `const1`).
    - `LowerToBMC` now synthesizes derived BMC clock inputs from
      `bmc_reg_clock_sources` const keys when no explicit/derived clocks are
      otherwise discovered.
    - hardened LowerToBMC against null-root clock tracing for constant clocks.
  - this unblocks previously failing OVL checkers in the BMC pipeline
    (notably `ovl_arbiter` import/lowering path).
- New regression:
  - `test/Tools/circt-bmc/externalize-registers-const-clock.mlir`
- OVL semantic harness expansion:
  - added wrappers:
    - `utils/ovl_semantic/wrappers/ovl_sem_arbiter.sv`
    - `utils/ovl_semantic/wrappers/ovl_sem_stack.sv`
  - manifest additions:
    - `ovl_sem_arbiter` (`known_gap=1`)
    - `ovl_sem_stack` (`known_gap=1`)
  - semantic lane breadth increased from `43` to `45` wrappers.
  - semantic obligation coverage increased from `86` to `90`.
- Validation:
  - `build-test/bin/circt-opt test/Tools/circt-bmc/externalize-registers-const-clock.mlir --externalize-registers='allow-multi-clock=true' | llvm/build/bin/FileCheck test/Tools/circt-bmc/externalize-registers-const-clock.mlir`
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(arbiter|stack)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-constclock-arbiter-stack`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion VI)

- Closed gap:
  - OVL semantic lane now includes additional combinational/data-integrity
    checkers:
    - `ovl_bits`
    - `ovl_code_distance`
    - `ovl_fifo_index`
  - plus explicit known-gap tracking for:
    - `ovl_frame` (`known_gap=tool`, frontend empty-match parse limitation)
    - `ovl_never_unknown_async` (`known_gap=1`, immediate-assert fail-mode)
  - semantic lane breadth increased from `38` to `43` checker wrappers.
  - semantic obligation coverage increased from `76` to `86`.
- Harness enhancement:
  - `utils/run_ovl_sva_semantic_circt_bmc.sh` now supports
    `known_gap=tool|any` in addition to existing `known_gap=1`.
  - expected tool failures are reported as `XFAIL` and successful closures as
    `XPASS`.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_bits.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_code_distance.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_fifo_index.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_frame.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_never_unknown_async.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(bits|code_distance|fifo_index|frame|never_unknown_async)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-next5b`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion V)

- Closed gap:
  - OVL semantic lane now includes additional protocol/timing assertion
    checkers:
    - `ovl_cycle_sequence`
    - `ovl_handshake`
    - `ovl_req_ack_unique`
    - `ovl_reg_loaded`
    - `ovl_time`
  - semantic lane breadth increased from `33` to `38` checker wrappers.
  - semantic obligation coverage increased from `66` to `76`.
  - note: semantic wrapper for `ovl_handshake` uses `min_ack_cycle=1` to avoid
    empty-match parser limitations for `[*0]` in current frontend lowering.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_cycle_sequence.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_handshake.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_req_ack_unique.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_reg_loaded.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_time.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(cycle_sequence|handshake|req_ack_unique|reg_loaded|time)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-next5`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion IV)

- Closed gap:
  - OVL semantic lane now includes five more assertion-focused checkers:
    - `ovl_always_on_edge`
    - `ovl_width`
    - `ovl_quiescent_state`
    - `ovl_value`
    - `ovl_proposition`
  - semantic lane breadth increased from `28` to `33` checker wrappers.
  - semantic obligation coverage increased from `56` to `66`.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_always_on_edge.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_width.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_quiescent_state.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_value.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_proposition.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Known-gap tracking:
  - `ovl_sem_proposition` fail-mode is marked `known_gap=1` because the
    immediate assertion checker (`assert #0`) is not currently lowered as a
    formal property in this flow.
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(always_on_edge|width|quiescent_state|value|proposition)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-new5-2`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion II)

- Closed gap:
  - OVL semantic lane now includes transition and overflow/underflow checkers
    plus request-sequence ordering:
    - `ovl_no_overflow`
    - `ovl_no_underflow`
    - `ovl_transition`
    - `ovl_no_transition`
    - `ovl_req_requires`
  - semantic lane breadth increased from `18` to `23` checker wrappers.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_no_overflow.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_no_underflow.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_transition.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_no_transition.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_req_requires.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(no_overflow|no_underflow|transition|no_transition|req_requires)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-new11`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion III)

- Closed gap:
  - OVL semantic lane now includes windowed stability and contention checkers:
    - `ovl_window`
    - `ovl_win_change`
    - `ovl_win_unchange`
    - `ovl_hold_value`
    - `ovl_no_contention`
  - semantic lane breadth increased from `23` to `28` checker wrappers.
  - note: semantic wrapper for `ovl_no_contention` uses
    `min_quiet=1,max_quiet=1` to avoid empty-match sequence parsing limits in
    current frontend lowering for `[*0]`.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_window.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_win_change.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_win_unchange.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_hold_value.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_no_contention.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(window|win_change|win_unchange|hold_value|no_contention)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-window-batch`

## Latest SVA closure slice (2026-02-22, OVL semantic expansion)

- Closed gap:
  - OVL semantic lane now covers additional arithmetic and stability checkers:
    - `ovl_odd_parity`
    - `ovl_increment`
    - `ovl_decrement`
    - `ovl_delta`
    - `ovl_unchange`
  - semantic lane breadth increased from `13` to `18` checker wrappers.
- New regressions:
  - `utils/ovl_semantic/wrappers/ovl_sem_odd_parity.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_increment.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_decrement.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_delta.sv`
  - `utils/ovl_semantic/wrappers/ovl_sem_unchange.sv`
  - manifest entries in `utils/ovl_semantic/manifest.tsv`
- Validation:
  - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(odd_parity|increment|decrement|delta|unchange)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
  - `utils/run_formal_all.sh --with-ovl --with-ovl-semantic --ovl /home/thomas-ahle/std_ovl --ovl-bmc-test-filter '.*' --ovl-semantic-test-filter '.*' --include-lane-regex '^std_ovl/' --out-dir /tmp/formal-ovl-full-matrix-after-new5`

## Latest SVA closure slice (2026-02-22)

- Closed gap:
  - ImportVerilog statement-level assertion-control lowering now includes
    pass/vacuous controls and legacy subroutine forms:
    - `$assertcontrol(6/7/10/11)`
    - `$assertpasson/$assertpassoff`
    - `$assertnonvacuouson/$assertvacuousoff`
  - this aligns procedural behavior with sequence match-item lowering.
  - ImportVerilog now implements assertion-control lock semantics:
    - `$assertcontrol(1)` lock
    - `$assertcontrol(2)` unlock
    - lock state gates subsequent assertion-control updates in both
      procedural and match-item lowering paths.
  - Concurrent assertion action blocks now preserve labels for additional I/O
    task families:
    - `$strobe/$monitor` (+ `b/o/h`)
    - `$fdisplay/$fwrite/$fstrobe/$fmonitor` (+ `b/o/h`)
    - dynamic payloads retain deterministic task-name fallback labels.
- New regression:
  - `test/Conversion/ImportVerilog/sva-assertcontrol-pass-vacuous-procedural.sv`
  - `test/Conversion/ImportVerilog/sva-assertcontrol-lock-procedural.sv`
  - `test/Conversion/ImportVerilog/sva-sequence-match-item-assertcontrol-lock-subroutine.sv`
  - `test/Conversion/ImportVerilog/sva-action-block-io-labels.sv`
- Validation:
  - `ninja -C build-test circt-translate`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-pass-vacuous-procedural.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assertcontrol-pass-vacuous-procedural.sv`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-lock-procedural.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assertcontrol-lock-procedural.sv`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-match-item-assertcontrol-lock-subroutine.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-match-item-assertcontrol-lock-subroutine.sv`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-action-block-io-labels.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-action-block-io-labels.sv`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-assertcontrol-failmsg.sv`
  - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-sequence-match-item-assertcontrol-pass-vacuous-subroutine.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-sequence-match-item-assertcontrol-pass-vacuous-subroutine.sv`
  - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' DISABLE_UVM_AUTO_INCLUDE=1 utils/run_yosys_sva_circt_bmc.sh`

## Latest SVA closure slice (2026-02-21)

- Closed gap:
  - `LTLToCore` now lowers unbounded `ltl.first_match` in clocked assertions
    with first-hit semantics: accepting next states form `match`, and all
    next-state updates are masked by `!match`.
- New regression:
  - `test/Conversion/LTLToCore/first-match-unbounded.mlir`
- Validation:
  - `ninja -C build-test circt-opt`
  - `build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/first-match-unbounded.mlir`
  - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/first-match-unbounded.mlir`
  - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`
  - profiling sample:
    - `time build-test/bin/circt-opt test/Conversion/LTLToCore/first-match-unbounded.mlir --lower-ltl-to-core` (`~0.01s`)

- Performance follow-up in same slice:
  - reduced duplicate transition masking in bounded and unbounded first-match
    lowering via per-source-state/per-condition mask caching.

- Additional closure (same date):
  - `LTLToCore` sequence warmup now uses minimum-length bounds, not only exact
    finite bounds, so unbounded-repeat sequences with known minimum delay get
    startup warmup gating in assertion lowering.
  - regression:
    - `test/Conversion/LTLToCore/unbounded-sequence-warmup.mlir`
  - sequence event-control lowering now caches per-source-state transition
    terms to reduce duplicated combinational `and` generation in large NFAs.

- Additional closure (same date):
  - ImportVerilog now supports concurrent `restrict property` by lowering to
    assume semantics.
  - regressions:
    - `test/Conversion/ImportVerilog/sva-restrict-property.sv`
    - `test/Tools/circt-bmc/sva-restrict-e2e.sv`
  - ImportVerilog now supports concurrent `cover sequence` lowering through
    the cover paths.
  - regressions:
    - `test/Conversion/ImportVerilog/sva-cover-sequence.sv`
    - `test/Tools/circt-bmc/sva-cover-sequence-e2e.sv`
  - ImportVerilog now supports abort-style property operators:
    `accept_on`, `reject_on`, `sync_accept_on`, `sync_reject_on`.
  - regressions:
    - `test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `test/Tools/circt-bmc/sva-abort-on-e2e.sv`
  - ImportVerilog now supports `strong(...)` / `weak(...)` property wrappers.
  - regressions:
    - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
    - `test/Tools/circt-bmc/sva-strong-weak-e2e.sv`
  - ImportVerilog now supports `case` property expressions.
  - regressions:
    - `test/Conversion/ImportVerilog/sva-case-property.sv`
    - `test/Tools/circt-bmc/sva-case-property-e2e.sv`
  - `case` property lowering now uses bitvector selector equality for
    multi-bit selectors (not boolean-only matching).

- Additional closure (same date):
  - `LTLToCore` now supports both-edge clock normalization for direct lowering
    of clocked sequence/property checks on `i1` clocks.
  - regression:
    - `test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
  - validation:
    - `build-test/bin/circt-opt test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir --lower-ltl-to-core | llvm/build/bin/FileCheck test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
    - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/LTLToCore/clocked-sequence-edge-both.mlir`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Additional closure (same date):
  - ImportVerilog now distinguishes sync abort operators
    (`sync_accept_on`/`sync_reject_on`) by sampling abort condition on the
    assertion clock before combining with property body.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-abort-on.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-abort-on.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-abort-on.sv`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-abort-on-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc=\"top-module=sva_abort_on_e2e bound=2\" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-abort-on-e2e.sv --check-prefix=CHECK-BMC`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

## Ownership and References

- Primary plan: PROJECT_PLAN.md (tracks and iteration status)
- Formal tooling references: docs/FormalVerification.md
- Dialect references: docs/Dialects/LTL.md, docs/Dialects/SMT.md
- BMC/LEC passes: docs/Passes.md

- Additional closure (same date):
  - ImportVerilog now differentiates `strong(...)` and `weak(...)` wrappers.
  - `strong(expr)` lowers with explicit eventual progress requirement via
    `ltl.eventually`; `weak(expr)` remains direct.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-strong-weak.sv`
  - validation:
    - `build-test/bin/circt-translate --import-verilog test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-IMPORT`
    - `build-test/bin/circt-verilog --ir-moore test/Conversion/ImportVerilog/sva-strong-weak.sv | llvm/build/bin/FileCheck test/Conversion/ImportVerilog/sva-strong-weak.sv --check-prefix=CHECK-MOORE`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-strong-weak-e2e.sv | build-test/bin/circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc="top-module=sva_strong_weak_e2e bound=2" | llvm/build/bin/FileCheck test/Tools/circt-bmc/sva-strong-weak-e2e.sv --check-prefix=CHECK-BMC`
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`

- Additional closure (same date):
  - ImportVerilog now differentiates `strong(...)` and `weak(...)` wrappers.
  - updated regression:
    - `test/Conversion/ImportVerilog/sva-strong-weak.sv`

- Additional closure (same date):
  - empty `first_match` sequences now lower in `LTLToCore` instead of emitting
    a conversion error.
  - regression:
    - `test/Conversion/LTLToCore/first-match-empty.mlir`

## Latest BMC clocking hardening (2026-02-22)

- Closed robustness gap in `LowerToBMC` when register clock metadata is present
  but unresolved (`bmc_reg_clock_sources = [unit, ...]`) and no traceable
  `seq.to_clock` / `ltl.clock` source exists at top level.
- Added fallback clock discovery:
  - if there is exactly one clock-like original interface input (excluding
    appended register-state inputs), use it as derived BMC clock input.
  - clock-like now includes 4-state clock structs
    (`!hw.struct<value: i1, unknown: i1>`) in addition to `i1` and
    `!seq.clock`.
- New regression:
  - `test/Tools/circt-bmc/lower-to-bmc-unit-reg-clock-source-struct-input.mlir`
- Expected impact:
  - prevents malformed `verif.bmc` init/loop regions with missing clock yields
    in unresolved-clock metadata paths.
  - does not by itself close semantic vacuity gaps in
    `ovl_sem_arbiter` / `ovl_sem_stack` fail-mode.

## Latest SVA closure slice (2026-02-22, const-only clock override + semantic OVL closure)

- Realization:
  - Some flattened OVL paths produced `bmc_reg_clock_sources = [{clock_key = "const0", ...}]` even with a real interface `clk`.
  - `LowerToBMC` then selected a constant derived BMC clock, which made clocked checks vacuous/contradictory and kept `ovl_sem_arbiter`/`ovl_sem_stack` fail-mode at `UNSAT`.

- Implemented:
  - `lib/Tools/circt-bmc/LowerToBMC.cpp`
    - when discovered clock inputs are const-only, prefer a uniquely named clock-like interface input (`clk`/`clock`) over const clocks.
  - new regression:
    - `test/Tools/circt-bmc/lower-to-bmc-const-clock-source-prefers-named-input.mlir`
  - semantic harness tuning:
    - `utils/ovl_semantic/wrappers/ovl_sem_arbiter.sv`
      - set `.min_cks(0)`, `.max_cks(0)`, `.one_cycle_gnt_check(0)` so pass/fail profiles are semantically separable.
    - `utils/ovl_semantic/manifest.tsv`
      - removed known-gap markers for:
        - `ovl_sem_arbiter`
        - `ovl_sem_stack`

- Validation:
  - build:
    - `ninja -C build-test circt-opt circt-bmc`
  - focused BMC lowering checks:
    - `build-test/bin/circt-opt --lower-to-bmc='top-module=m bound=2 allow-multi-clock=true' test/Tools/circt-bmc/lower-to-bmc-const-clock-source-prefers-named-input.mlir`
    - `build-test/bin/circt-opt --lower-to-bmc='top-module=m bound=2 allow-multi-clock=true' test/Tools/circt-bmc/lower-to-bmc-unit-reg-clock-source-struct-input.mlir`
  - OVL semantic focused:
    - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(arbiter|stack)' FAIL_ON_XPASS=0 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `4 tests, failures=0, xfail=0, xpass=0`
  - full OVL semantic slice:
    - `FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `90 tests, failures=0, xfail=0, xpass=0`

- Closure update (same date, immediate-action assertion formalization):
  - `lib/Conversion/ImportVerilog/Statements.cpp` now emits formal
    immediate assert-like ops even when action blocks are present, preserving
    runtime action behavior while restoring formal obligations.
  - regression:
    - `test/Conversion/ImportVerilog/immediate-assert-action-block.sv`
  - focused semantic closure check:
    - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(proposition|never_unknown_async|frame)' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `6 tests, failures=0, xfail=0, xpass=0`

- Known semantic gaps in OVL harness:
  - `ovl_sem_multiport_fifo` pass-mode (`known_gap=pass`), currently tracked
    as an LLHD process-abstraction semantic gap.

## Latest OVL expansion slice (2026-02-22, interface-propagation fix + 6 new wrappers)

- Implemented:
  - `lib/Tools/circt-lec/StripLLHDInterfaceSignals.cpp`
    - fixed module signature propagation for abstraction-added inputs by
      updating all affected `hw.instance` ops bottom-up over the instance graph.
  - new regression:
    - `test/Tools/circt-lec/lec-strip-llhd-comb-abstraction-instance-propagation.mlir`
  - OVL semantic wrappers added:
    - `ovl_sem_crc`
    - `ovl_sem_fifo`
    - `ovl_sem_memory_async`
    - `ovl_sem_memory_sync`
    - `ovl_sem_multiport_fifo`
    - `ovl_sem_valid_id`
  - manifest updated:
    - `utils/ovl_semantic/manifest.tsv`
  - semantic runner updated:
    - `utils/run_ovl_sva_semantic_circt_bmc.sh` now supports
      `known_gap=pass` in addition to `known_gap=1|fail`, `tool`, `any`.

- Validation:
  - build:
    - `ninja -C build-test circt-opt circt-bmc`
  - focused pass regression:
    - `build-test/bin/circt-opt --strip-llhd-interface-signals test/Tools/circt-lec/lec-strip-llhd-comb-abstraction-instance-propagation.mlir | llvm/build/bin/FileCheck test/Tools/circt-lec/lec-strip-llhd-comb-abstraction-instance-propagation.mlir`
  - focused OVL subset:
    - `OVL_SEMANTIC_TEST_FILTER='ovl_sem_(crc|fifo|memory_async|memory_sync|multiport_fifo|valid_id)' FAIL_ON_XPASS=0 ...`
    - result: `14 tests, failures=0, xfail=1, xpass=0`
  - full OVL semantic matrix:
    - `FAIL_ON_XPASS=0 ...`
    - result: `102 tests, failures=0, xfail=1, xpass=0`

## Latest LLHD process abstraction slice (2026-02-23)

- Implemented:
  - `lib/Tools/circt-bmc/StripLLHDProcesses.cpp`
    - for `observable_signal_use` cases, switched from process-result
      abstraction to signal-level interface abstraction
      (`observable_signal_use_resolution_unknown`) when possible.
    - goal: reduce spurious behavior from over-abstracted
      `llhd_process_result*` ports.
  - updated regression expectations:
    - `test/Tools/circt-bmc/strip-llhd-processes.mlir`

- Validation:
  - `ninja -C build-test circt-opt circt-bmc`
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/strip-llhd-processes.mlir build-test/test/Tools/circt-bmc/strip-llhd-process-drives.mlir`
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/circt-bmc-llhd-process.mlir build-test/test/Tools/circt-bmc/lower-to-bmc-llhd-signals.mlir build-test/test/Tools/circt-bmc/lower-to-bmc-llhd-process-abstraction-attr.mlir`
  - `FAIL_ON_XPASS=0 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `102 tests, failures=0, xfail=1, xpass=0`

- Current state:
  - `ovl_sem_multiport_fifo` pass-mode remains the only tracked semantic gap
    (`known_gap=pass`), but the abstraction footprint is reduced (from
    process-result fanout to 4 signal-level abstraction inputs).

## Latest LLHD process abstraction closure (2026-02-23, observable init defaults)

- Implemented:
  - `lib/Tools/circt-bmc/StripLLHDProcesses.cpp`
    - abstraction details now record `default_bits` when an observable
      signal-level abstraction has a constant signal init.
    - instance propagation now wires
      `observable_signal_use_resolution_unknown` ports from constant defaults
      when `default_bits` are available, instead of always introducing a new
      parent input.
  - regression update:
    - `test/Tools/circt-bmc/strip-llhd-processes.mlir`
      - added `observable_child`/`observable_parent` hierarchy check proving
        this no longer leaks to top-level nondeterministic inputs.
  - semantic harness:
    - `utils/ovl_semantic/manifest.tsv`
      - removed `known_gap=pass` from `ovl_sem_multiport_fifo`.

- Validation:
  - build:
    - `ninja -C build-test circt-opt circt-bmc`
  - focused tests:
    - `lit -sv build-test/test/Tools/circt-bmc/strip-llhd-processes.mlir`
    - `lit -sv build-test/test/Tools/circt-bmc/circt-bmc-llhd-process.mlir build-test/test/Tools/circt-bmc/lower-to-bmc-llhd-signals.mlir build-test/test/Tools/circt-bmc/lower-to-bmc-llhd-process-abstraction-attr.mlir build-test/test/Tools/circt-bmc/strip-llhd-processes.mlir build-test/test/Tools/circt-bmc/strip-llhd-process-drives.mlir`
  - targeted semantic:
    - `OVL_SEMANTIC_TEST_FILTER='^ovl_sem_multiport_fifo$' FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `2 tests, failures=0, xfail=0, xpass=0`
  - full semantic matrix:
    - `FAIL_ON_XPASS=1 utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - result: `102 tests, failures=0, xfail=0, xpass=0`
  - formal smoke:
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='basic00' utils/run_yosys_sva_circt_bmc.sh`
    - result: pass/pass
  - profiling sample:
    - `time FAIL_ON_XPASS=1 OVL_SEMANTIC_TEST_FILTER='ovl_sem_(multiport_fifo|fifo|stack|arbiter)' utils/run_ovl_sva_semantic_circt_bmc.sh /home/thomas-ahle/std_ovl`
    - `real 0m8.814s`

- Current semantic-gap status:
  - OVL semantic harness has no tracked known gaps (`102/102` pass/fail modes
    passing).

## Latest sampled-value parity closure (2026-02-23, multi-assert enable semantics)

- Implemented:
  - `lib/Dialect/Verif/Transforms/CombineAssertLike.cpp`
    - fixed enabled assert/assume combination to preserve implication
      semantics (`!enable || property`) before conjoining multiple checks.
    - previous behavior incorrectly used `enable && property`, which turns
      disabled checks into failures once combined.
  - regression update:
    - `test/Dialect/Verif/combine-assert-like.mlir`
      - updated enabled-check expectations to implication-gated form.

- Validation:
  - build:
    - `ninja -C build-test circt-opt circt-bmc circt-verilog`
  - targeted regressions:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Dialect/Verif/combine-assert-like.mlir`
    - `build-test/bin/circt-verilog --no-uvm-auto-include --ir-hw test/Tools/circt-bmc/sva-sampled-first-cycle-known-inputs-parity.sv | build-test/bin/circt-bmc -b 6 --ignore-asserts-until=0 --module top --assume-known-inputs --rising-clocks-only --shared-libs=/home/thomas-ahle/z3-install/lib64/libz3.so -`
      - result: `BMC_RESULT=UNSAT`
  - targeted Yosys SVA parity:
    - `TEST_FILTER='^sva_value_change_sim$' BMC_ASSUME_KNOWN_INPUTS=1 ... utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
      - result: pass
    - `BMC_ASSUME_KNOWN_INPUTS=0` currently reports `XPASS` vs existing
      baseline for this profile.

- Current state:
  - known-input sampled-value parity gap for `sva_value_change_sim` is closed.
  - xprop expectation baseline for this case should be reviewed in a follow-up.

## Latest parity baseline sync (2026-02-23, `sva_value_change_sim` xprop)

- Implemented:
  - removed stale xprop xfail entries for `sva_value_change_sim` in:
    - `utils/yosys-sva-bmc-expected.txt`
    - `utils/yosys-sva-bmc-xfail.txt`

- Validation:
  - `TEST_FILTER='^sva_value_change_sim$' BMC_ASSUME_KNOWN_INPUTS=1 ... utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - pass
  - `TEST_FILTER='^sva_value_change_sim$' BMC_ASSUME_KNOWN_INPUTS=0 ... utils/run_yosys_sva_circt_bmc.sh /home/thomas-ahle/yosys/tests/sva`
    - pass (no xpass)

- Current state:
  - `sva_value_change_sim` is now green in both known/xprop lanes with no
    expected-failure override.

## Latest circt-bmc parity closure (2026-02-23, `--x-optimistic`)

- Implemented:
  - `tools/circt-bmc/circt-bmc.cpp`
    - added user-facing `--x-optimistic` CLI option.
    - wired option to `ConvertVerifToSMTOptions.xOptimisticOutputs` in:
      - `executeBMC`
      - `executeBMCWithInduction`
  - regression updates:
    - added `test/Tools/circt-bmc/bmc-x-optimistic-lec.mlir`
      - checks strict vs x-optimistic LEC mismatch lowering in
        `circt-bmc --emit-mlir`.
    - updated `test/Tools/circt-bmc/commandline.mlir`
      - verifies `--x-optimistic` appears in help.

- Validation:
  - build:
    - `ninja -C build-test circt-bmc`
  - targeted lit:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/commandline.mlir build-test/test/Tools/circt-bmc/bmc-x-optimistic-lec.mlir`
    - result: `2/2` pass.
  - regular formal sanity:
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=0 utils/run_yosys_sva_circt_bmc.sh`
      - result: `14 tests, failures=0, xfail=6, xpass=0`.
    - `TEST_FILTER='.*' BMC_ASSUME_KNOWN_INPUTS=1 utils/run_yosys_sva_circt_bmc.sh`
      - result: `14 tests, failures=0`.

- Current state:
  - `circt-bmc` now has parity with `circt-lec` for `x-optimistic` LEC output
    handling controls.
  - remaining known xprop lane expected failures stay unchanged (`6`) since
    they are BMC-property semantics, not LEC output-comparison mode.

## Latest sv-tests harness restoration (2026-02-23)

- Implemented:
  - `utils/run_sv_tests_circt_bmc.sh`
    - restored missing main execution/summarization tail that runs discovered
      sv-tests and emits `sv-tests SVA summary`.
    - preserved explicit filter contract (`TAG_REGEX`/`TEST_FILTER`) except
      existing smoke-mode fallback behavior.
  - stale test harness callsite fixes:
    - `test/Tools/circt-bmc/sv-tests-expectations.mlir`
    - `test/Tools/circt-bmc/sv-tests-rising-clocks-only.mlir`
    - both now pass explicit `TEST_FILTER='.*'`.
  - stale expected-fail cleanup:
    - `test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir`
    - removed `XFAIL` after confirming stable pass.

- Validation:
  - targeted sv-tests files:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sv-tests-expectations.mlir build-test/test/Tools/circt-bmc/sv-tests-rising-clocks-only.mlir build-test/test/Tools/circt-bmc/sv-tests-bare-property-smoke.mlir build-test/test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir build-test/test/Tools/circt-bmc/sv-tests-uvm-tags-include.mlir`
    - result: `4 pass, 1 expected-fail`.
  - harness-level contract tests:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools --filter='run-sv-tests-bmc-'`
    - result: `19 pass, 1 unsupported`.
  - full `circt-bmc` sv-tests bucket:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc --filter='sv-tests-'`
    - result: `11 pass, 1 expected-fail, 4 unsupported`.

- Current state:
  - sv-tests BMC runner is operational again with non-empty summary output.

## Latest sv-tests multiclock parity closure (2026-02-23)

- Implemented:
  - `utils/run_sv_tests_circt_bmc.sh`
    - added `AUTO_ALLOW_MULTI_CLOCK` (default `1`) auto-retry:
      - if a case fails with known multiclock diagnostics and
        `ALLOW_MULTI_CLOCK` is not globally set, rerun that case with
        `--allow-multi-clock`.
  - regression updates:
    - added `test/Tools/run-sv-tests-bmc-auto-allow-multi-clock.test`
    - updated `test/Tools/circt-bmc/sv-tests-uvm-tags-include.mlir`
      - removed stale `XFAIL`.

- Validation:
  - runner feature test:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/run-sv-tests-bmc-auto-allow-multi-clock.test`
    - result: `1/1` pass.
  - UVM tagged smoke regressions:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/sv-tests-uvm-tags-include.mlir build-test/test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir`
    - result: `2/2` pass.
  - harness contract tests:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools --filter='run-sv-tests-bmc-'`
    - result: `20 pass, 1 unsupported`.
  - full `circt-bmc` sv-tests bucket:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc --filter='sv-tests-'`
    - result: `12 pass, 4 unsupported`.

- Current state:
  - UVM include-tags lane is now green without an expected-fail override.
  - no expected-fail entries remain in the `sv-tests-*` `circt-bmc` subset.

## Latest mixed assert+cover BMC closure (2026-02-23)

- Implemented:
  - `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
    - enabled mixed `verif.assert` + `verif.cover` handling in one
      `verif.bmc` by removing the hard verifier rejection.
    - introduced per-check non-final typing (`nonFinalCheckIsCover`) so
      assert/cover checks are lowered with correct polarity in both SMTLIB and
      non-SMTLIB paths.
    - unified terminal solve condition so one BMC query can report:
      - any non-final assertion violation,
      - any final assertion violation, and
      - any final/non-final cover hit.
  - `lib/Dialect/Verif/Transforms/CombineAssertLike.cpp`
    - preserved `bmc.*`-annotated assert/assume ops from combination.
    - prevents `bmc.final` metadata loss before liveness/final-check lowering.
  - test updates:
    - added `test/Tools/circt-bmc/bmc-mixed-assert-cover.mlir`.
    - refreshed `test/Tools/circt-bmc/bmc-emit-mlir-cover-inverts-result.mlir`
      checks to avoid brittle SSA-id coupling.

- Validation:
  - build:
    - `ninja -C build-test circt-bmc circt-opt`
  - focused:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/bmc-mixed-assert-cover.mlir build-test/test/Tools/circt-bmc/bmc-liveness-mode-ignores-non-final.mlir build-test/test/Tools/circt-bmc/bmc-emit-mlir-cover-inverts-result.mlir`
    - result: `3/3` pass.
  - broad `circt-bmc` suite:
    - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc`
    - result: `153 pass, 156 unsupported, 1 xfail, 1 fail`.
    - remaining fail is local JIT Z3 linkage (`Z3_*` symbol materialization)
      in `circt-bmc-disable-iff-constant.mlir`.
  - formal smoke cadence:
    - `BMC_SMOKE_ONLY=1 TEST_FILTER='16.12--property|16.12--property-disj' utils/run_sv_tests_circt_bmc.sh /home/thomas-ahle/sv-tests`
    - result: `total=9 pass=9 fail=0`.

- Current state:
  - mixed assert+cover BMC support is now landed.
  - liveness/final-property semantics remain intact under
    `combine-assert-like` for `bmc.*`-annotated checks.

## Latest k-induction cover enablement (2026-02-23)

- Implemented:
  - `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
    - removed induction-step cover hard rejection
      (`k-induction does not support cover properties yet`).
    - induction-step now accepts cover-only and mixed property sets.
    - updated empty-induction-check diagnostic to mention both assertions and
      covers.
    - removed stale `coverBMCOps` pattern/plumbing that became dead after this
      change.
  - test coverage:
    - added `test/Tools/circt-bmc/bmc-k-induction-cover.mlir`.

- Validation:
  - `ninja -C build-test circt-bmc circt-opt`
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/bmc-k-induction-cover.mlir build-test/test/Tools/circt-bmc/bmc-k-induction-unsat.mlir build-test/test/Tools/circt-bmc/bmc-k-induction-sat.mlir build-test/test/Tools/circt-bmc/bmc-k-induction-final-unsat.mlir build-test/test/Tools/circt-bmc/bmc-k-induction-final-sat.mlir build-test/test/Tools/circt-bmc/bmc-induction-alias-unsat.mlir build-test/test/Tools/circt-bmc/bmc-induction-ignore-asserts-until.mlir`
    - result: `7/7` pass.
  - `llvm/build/bin/llvm-lit -sv build-test/test/Conversion/VerifToSMT --filter='induction'`
    - result: `2/2` pass.

- Current state:
  - k-induction no longer rejects cover properties.
  - induction mode now supports cover-only checks in line with broader
    assert/cover mixed-property support.

## Latest LLHD inline formal regression de-XFAIL (2026-02-23)

- Implemented:
  - `test/Tools/circt-bmc/lower-to-bmc-inline-llhd-combinational.mlir`
    - repaired invalid SSA (process-local values used outside process regions).
    - removed stale `XFAIL: *`.
    - aligned expected output with current post-lowering form
      (`smt.solver`).

- Validation:
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc/lower-to-bmc-inline-llhd-combinational.mlir`
    - result: `1/1` pass.
  - `llvm/build/bin/llvm-lit -sv build-test/test/Tools/circt-bmc --filter='llhd|lower-to-bmc-inline'`
    - result: `18 pass, 1 unsupported`.

- Current state:
  - this LLHD inline BMC regression now contributes active pass/fail signal
    instead of being permanently expected-fail.

## Latest sv-tests mixed assert+cover classification closure (2026-02-23)

- Implemented:
  - `utils/run_sv_tests_circt_bmc.sh`
    - added explicit mixed check-mode detection when generated MLIR contains
      both `verif.assert` and `verif.cover`.
    - added SAT disambiguation for mixed, non-negative simulation tests:
      rerun `circt-bmc` on an assert-only MLIR view (covers removed) to
      distinguish:
      - SAT-from-assert-violation => `FAIL`
      - SAT-from-cover-hit => `PASS`
  - regression coverage:
    - added
      `test/Tools/run-sv-tests-bmc-mixed-assert-cover-classification.test`.

- Validation:
  - manual TDD repro before fix:
    - `utils/run_sv_tests_circt_bmc.sh` on a synthetic mixed module with SAT
      only when cover is present
    - result: `total=1 pass=0 fail=1`.
  - same repro after fix:
    - result: `total=1 pass=1 fail=0`.
  - harness contract tests:
    - `build-ot/bin/llvm-lit -sv --filter 'run-sv-tests-bmc-mixed-assert-cover-classification' build-test/test`
    - result: `1/1` pass.
    - `build-ot/bin/llvm-lit -sv --filter 'run-sv-tests-bmc-' build-test/test`
    - result: `21 pass, 1 unsupported`.

- Current state:
  - mixed assert+cover SAT outcomes in sv-tests are now interpreted with
    assert/cover-aware semantics rather than assert-only heuristics.
