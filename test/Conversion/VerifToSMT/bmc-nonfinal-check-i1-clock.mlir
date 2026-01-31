// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// CHECK-LABEL: func.func @bmc_i1_clock_ltlclock
// CHECK: scf.for
// CHECK:   func.call @bmc_loop
// CHECK:   %[[NOT_NEW:.*]] = smt.bv.not
// CHECK:   %[[EDGE_AND:.*]] = smt.bv.and
// CHECK:   %[[NEGEDGE:.*]] = smt.eq %[[EDGE_AND]]
// CHECK:   func.call @bmc_circuit
// CHECK:   %[[NOT_PROP:.*]] = smt.not
// CHECK:   %[[GATED:.*]] = smt.and %[[NEGEDGE]], %[[NOT_PROP]]
// CHECK:   smt.assert %[[GATED]]
func.func @bmc_i1_clock_ltlclock() -> i1 {
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
  ^bb0(%clk0: i1, %clk1: i1, %sig: i1):
    %seq = ltl.delay %sig, 0, 0 : i1
    %clocked = ltl.clock %seq, negedge %clk1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
