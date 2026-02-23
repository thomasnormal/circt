// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multiclock past buffer with clocked property checking.

// CHECK-LABEL: func.func @past_multiclock_clocked
// CHECK: scf.for
// Loop is called first
// CHECK: func.call @bmc_loop
// Clock edge detection
// CHECK: smt.bv.not
// CHECK: smt.bv.and
// CHECK: smt.eq
// Circuit returns outputs + past buffer + !smt.bool property
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
// Property checking
// CHECK-DAG: smt.not
// CHECK-DAG: smt.and
// Past buffer update
// CHECK-DAG: smt.ite
// CHECK: smt.push
// CHECK: smt.assert
// CHECK: smt.check
// CHECK: smt.pop
func.func @past_multiclock_clocked() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1", "sig"]
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
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %sig: i1):
    %true = ltl.boolean_constant true
    %past = ltl.past %sig, 1 {bmc.clock = "clk0"} : i1
    %prop = ltl.and %true, %past : !ltl.property, !ltl.sequence
    verif.assert %prop {bmc.clock_edge = #ltl<clock_edge negedge>} : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
