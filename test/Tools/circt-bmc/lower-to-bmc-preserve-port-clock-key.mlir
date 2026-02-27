// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=1 allow-multi-clock=true" | FileCheck %s

// Regression: preserve pre-existing `port:*` check clock keys when lowering to
// BMC, even when the structural key synthesis for the clock expression would
// produce an `expr:*` hash.
//
// CHECK: verif.bmc
// CHECK: ltl.clock {{.*}}bmc.clock_key = "port:clk_i"
// CHECK: verif.assert {{.*}}bmc.clock_key = "port:clk_i"

module {
  hw.module @top(in %clk_i : !hw.struct<value: i1, unknown: i1>, in %sig : i1)
      attributes {num_regs = 0 : i32, initial_values = []} {
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

    %seq = ltl.delay %sig, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %clk_i1 {bmc.clock_key = "port:clk_i"} : !ltl.sequence
    verif.assert %clocked {
      bmc.clock = "clk_i",
      bmc.clock_key = "port:clk_i",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : !ltl.sequence
    hw.output
  }
}
