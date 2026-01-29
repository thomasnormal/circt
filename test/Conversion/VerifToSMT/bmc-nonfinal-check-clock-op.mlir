// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that clock edge detection via ltl.clock is handled inside bmc_circuit.
// The circuit returns !smt.bool which already incorporates the clock gating.

// CHECK-LABEL: func.func @bmc_clock_op_check() -> i1
// CHECK: scf.for
// Circuit returns outputs + !smt.bool for the clocked property
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool)
// CHECK:   func.call @bmc_loop
// CHECK:   smt.not
// CHECK:   smt.and
// CHECK:   smt.push 1
// CHECK:   smt.assert
// CHECK:   smt.check
func.func @bmc_clock_op_check() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1"]
  }
  init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    %from0 = seq.from_clock %clk0
    %from1 = seq.from_clock %clk1
    %true = hw.constant true
    %nclk0 = comb.xor %from0, %true : i1
    %nclk1 = comb.xor %from1, %true : i1
    %new0 = seq.to_clock %nclk0
    %new1 = seq.to_clock %nclk1
    verif.yield %new0, %new1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    %true = ltl.boolean_constant true
    %clk1_i1 = seq.from_clock %clk1
    %clocked = ltl.clock %true, negedge %clk1_i1 : !ltl.property
    verif.assert %clocked : !ltl.property
    %out = seq.from_clock %clk0
    verif.yield %out : i1
  }
  func.return %bmc : i1
}
