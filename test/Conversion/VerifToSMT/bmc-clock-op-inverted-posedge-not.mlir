// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that a posedge ltl.clock on a comb.not-inverted clock value maps to
// negedge gating of the underlying BMC clock.

// CHECK-LABEL: func.func @bmc_clock_op_inverted_posedge_not() -> i1
// CHECK: scf.for
// Posedge on inverted clock = negedge on original clock = old & ~new
// CHECK: [[NEWLOW:%.+]] = smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK: [[NEGBV:%.+]] = smt.bv.and {{%.+}}, [[NEWLOW]] : !smt.bv<1>
// CHECK: [[NEG:%.+]] = smt.eq [[NEGBV]], {{%.+}} : !smt.bv<1>
// CHECK: smt.ite [[NEG]], {{%.+}}, {{%.+}} : !smt.bv<1>
func.func @bmc_clock_op_inverted_posedge_not() -> i1 {
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
    %seq = ltl.delay %sig, 1, 0 : i1
    %clk_i1 = seq.from_clock %clk
    %true = hw.constant true
    %inv = comb.xor %clk_i1, %true : i1
    %clocked = ltl.clock %seq, posedge %inv : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
