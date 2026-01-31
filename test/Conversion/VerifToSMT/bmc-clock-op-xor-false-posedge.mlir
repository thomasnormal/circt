// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test XOR with false does not invert the clock edge.

// CHECK-LABEL: func.func @bmc_clock_op_xor_false_posedge() -> i1
// CHECK: scf.for
// CHECK: [[OLDLOW:%.+]] = smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK: [[POSEBV:%.+]] = smt.bv.and [[OLDLOW]], {{%.+}} : !smt.bv<1>
// CHECK: [[POSE:%.+]] = smt.eq [[POSEBV]], {{%.+}} : !smt.bv<1>
// CHECK: smt.ite [[POSE]], {{%.+}}, {{%.+}} : !smt.bv<1>
func.func @bmc_clock_op_xor_false_posedge() -> i1 {
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
    %false = hw.constant false
    %same = comb.xor %from, %false : i1
    %new = seq.to_clock %same
    verif.yield %new : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %seq = ltl.delay %sig, 1, 0 : i1
    %clk_i1 = seq.from_clock %clk
    %false = hw.constant false
    %same = comb.xor %clk_i1, %false : i1
    %clocked = ltl.clock %seq, posedge %same : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
