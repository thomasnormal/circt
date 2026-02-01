// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.bv.not %arg2
// CHECK: [[NEW_LOW:%[^ ]+]] = smt.bv.not
// CHECK: [[NEGEDGE_BV:%[^ ]+]] = smt.bv.and %arg2, [[NEW_LOW]]
// CHECK: [[NEGEDGE:%[^ ]+]] = smt.eq [[NEGEDGE_BV]], %{{.+}}
// CHECK: [[REGNEXT:%[^ ]+]] = smt.ite [[NEGEDGE]], %{{.+}}, %arg3
func.func @bmc_reg_clock_sources_invert() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 1 initial_values [unit] attributes {
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_reg_clock_sources = [
      {arg_index = 1 : i32, invert = true}
    ]
  }
  init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %state: i1):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %state : i1
  }
  func.return %bmc : i1
}
