// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that negedge-clocked assertions are handled via the bmc.clock_edge
// attribute on verif.assert, with edge detection done inside the circuit.

// CHECK-LABEL: func.func @bmc_negedge_check() -> i1
// CHECK: scf.for
// Circuit returns outputs + !smt.bool for the clocked property
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool)
// CHECK:   func.call @bmc_loop
// CHECK:   smt.not
// CHECK:   smt.and
func.func @bmc_negedge_check() -> i1 {
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
    %prop = ltl.boolean_constant true
    verif.assert %prop {bmc.clock_edge = #ltl<clock_edge negedge>} : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
