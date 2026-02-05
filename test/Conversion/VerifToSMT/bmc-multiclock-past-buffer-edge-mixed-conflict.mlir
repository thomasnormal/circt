// RUN: not circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect 2>&1 | FileCheck %s

// Conflicting edge between bmc.clock_edge and ltl.clock is rejected.
// CHECK: ltl.delay/ltl.past used with conflicting clock information
func.func @past_edge_mixed_conflict() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk", "sig"]
  }
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from = seq.from_clock %clk
    %true = hw.constant true
    %nclk = comb.xor %from, %true : i1
    %new = seq.to_clock %nclk
    verif.yield %new : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %past = ltl.past %sig, 1 {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    %clk_i1 = seq.from_clock %clk
    %clocked = ltl.clock %past, negedge %clk_i1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
