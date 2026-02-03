// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that edge=both clock gating is handled inside bmc_circuit.
// The circuit returns !smt.bool which incorporates edge detection.

// CHECK-LABEL: func.func @bmc_nonfinal_check_edge_both() -> i1
// CHECK: scf.for
// Loop is called first
// CHECK:   func.call @bmc_loop
// Circuit returns outputs + !smt.bool for the clocked property
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool)
// CHECK:   smt.not
// CHECK:   smt.and
func.func @bmc_nonfinal_check_edge_both() -> i1 {
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
    %prop = ltl.delay %sig, 0, 0 : i1
    verif.assert %prop {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge edge>} : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
