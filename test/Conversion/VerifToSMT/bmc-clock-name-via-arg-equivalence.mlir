// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Regression: check metadata can reference a 4-state source clock port name
// (clk_i/port:clk_i) while BMC clock inputs are keyed through a different i1
// source. If an assume-equivalence proves the source clock i1 view equals a
// BMC clock input, resolve bmc.clock/bmc.clock_key through that arg root.
// CHECK: smt.solver
func.func @bmc_clock_name_via_arg_equivalence() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bmc_clk", "clk_i", "sig"],
    bmc_clock_keys = ["port:bmc_clk"]
  } init {
    %f = hw.constant false
    %c0 = seq.to_clock %f
    verif.yield %c0 : !seq.clock
  } loop {
  ^bb0(%c0: !seq.clock):
    verif.yield %c0 : !seq.clock
  } circuit {
  ^bb0(%bmc_clk: i1, %clk_i: !hw.struct<value: i1, unknown: i1>, %sig: i1):
    %v = hw.struct_extract %clk_i["value"] : !hw.struct<value: i1, unknown: i1>
    %u = hw.struct_extract %clk_i["unknown"] : !hw.struct<value: i1, unknown: i1>
    %t = hw.constant true
    %not_u = comb.xor %u, %t : i1
    %src_clk = comb.and %v, %not_u : i1
    %eq = comb.icmp eq %src_clk, %bmc_clk : i1
    verif.assume %eq : i1
    verif.assert %sig {
      bmc.clock = "clk_i",
      bmc.clock_key = "port:clk_i",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : i1
    verif.yield %sig : i1
  }
  func.return %res : i1
}
