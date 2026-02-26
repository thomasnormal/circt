// RUN: circt-bmc -b 1 --allow-multi-clock --module top --emit-mlir %s | FileCheck %s

// Regression: when a check's clock metadata cannot be resolved to a named
// top-level clock (bmc.clock_key = "expr:*"), keep the original clocked LTL
// form so LowerToBMC can materialize a derived BMC clock instead of failing
// in ConvertVerifToSMT.
// CHECK-LABEL: func.func @top()
// CHECK: smt.solver

module {
  hw.module @top(in %clk_i : !hw.struct<value: i1, unknown: i1>,
                 in %other_clk : !seq.clock,
                 in %in : i1) {
    %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %delay = llhd.constant_time <0ns, 0d, 1e>
    %clk_sig = llhd.sig name "clk_sig" %zero : !hw.struct<value: i1, unknown: i1>
    llhd.drv %clk_sig, %clk_i after %delay : !hw.struct<value: i1, unknown: i1>

    %clk_prb = llhd.prb %clk_sig : !hw.struct<value: i1, unknown: i1>
    %clk_v = hw.struct_extract %clk_prb["value"] : !hw.struct<value: i1, unknown: i1>
    %clk_u = hw.struct_extract %clk_prb["unknown"] : !hw.struct<value: i1, unknown: i1>
    %true = hw.constant true
    %clk_nu = comb.xor %clk_u, %true : i1
    %clk_i1 = comb.and bin %clk_v, %clk_nu : i1

    %clk = seq.to_clock %clk_i1
    %r0 = seq.compreg %in, %clk : i1
    %r1 = seq.compreg %in, %other_clk : i1

    verif.clocked_assert %r0, posedge %clk_i1 : i1
    // This unresolved check clock used to force bmc.clock_key="expr:*" and
    // fail with "clocked property uses a clock that is not a BMC clock input".
    verif.clocked_assert %r1, posedge %r1 : i1
    hw.output
  }
}
