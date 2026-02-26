// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Ensure bmc.clock_key metadata on check ops is accepted as the primary
// clock mapping source even when bmc.clock names are stale.
// CHECK: smt.solver
func.func @bmc_check_clock_key_attr() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_clock_keys = ["port:clk0", "port:clk1"]
  } init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  } loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  } circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %sig: i1):
    verif.assert %sig {bmc.clock = "stale_clk", bmc.clock_key = "port:clk1", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
