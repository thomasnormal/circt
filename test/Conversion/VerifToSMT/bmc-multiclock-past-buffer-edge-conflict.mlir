// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// XFAIL: *
// Known conversion bug: ltl.past sequence lowering is unsupported in this case.

// Shared ltl.past with different clock edges is cloned per property.
// CHECK-LABEL: func.func @past_edge_conflict
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-LABEL: func.func @bmc_circuit
func.func @past_edge_conflict() -> i1 {
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
    %past = ltl.past %sig, 1 : i1
    verif.assert %past {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : !ltl.sequence
    verif.assert %past {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge negedge>} : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
