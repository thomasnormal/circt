// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @delay_multiclock_clocked_edge
// CHECK: scf.for
// CHECK: func.call @bmc_loop
// CHECK: func.call @bmc_circuit
// The bmc_circuit function checks delay buffer values and applies clock edge
// CHECK-LABEL: func.func @bmc_circuit
// CHECK: smt.eq {{%.+}}, {{%.+}}
// CHECK: smt.eq {{%.+}}, {{%.+}}
// CHECK: smt.or
// CHECK: smt.and
// CHECK: ltl.clock {{%.+}}, negedge
func.func @delay_multiclock_clocked_edge() -> i1 {
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
    %del = ltl.delay %sig, 1, 1 {bmc.clock = "clk0"} : i1
    %prop = ltl.and %true, %del : !ltl.property, !ltl.sequence
    %clk0_i1 = seq.from_clock %clk0
    %clocked = ltl.clock %prop, negedge %clk0_i1 : !ltl.property
    verif.assert %clocked : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
