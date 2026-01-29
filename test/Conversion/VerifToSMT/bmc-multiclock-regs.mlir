// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_multiclock_reg_updates
// CHECK: scf.for
// CHECK-DAG: [[POSEDGE0BV:%.+]] = smt.bv.and {{%.+}}, {{%.+}}
// CHECK-DAG: [[POSEDGE1BV:%.+]] = smt.bv.and {{%.+}}, {{%.+}}
// CHECK-DAG: [[POSEDGE0:%.+]] = smt.eq [[POSEDGE0BV]], {{%.+}}
// CHECK-DAG: [[POSEDGE1:%.+]] = smt.eq [[POSEDGE1BV]], {{%.+}}
// CHECK: smt.ite [[POSEDGE0]]
// CHECK: smt.ite [[POSEDGE1]]
func.func @test_multiclock_reg_updates() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 2 initial_values [unit, unit] attributes {
    bmc_input_names = ["clk0", "clk1", "reg0_state", "reg1_state"],
    bmc_reg_clocks = ["clk0", "clk1"]
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
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %reg0: i1, %reg1: i1):
    verif.assert %reg0 : i1
    verif.assert %reg1 : i1
    verif.yield %reg0, %reg1 : i1, i1
  }
  func.return %bmc : i1
}
