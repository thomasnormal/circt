// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Regression: keep clocked checks legal outside verif.bmc so helper/library
// modules do not cause legalization failures during whole-module BMC lowering.
// CHECK: smt.solver
// CHECK: hw.module private @helper
// CHECK: verif.clocked_assert
func.func @bmc_with_unrelated_hw_clocked_assert() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values []
  init {
    %f = hw.constant false
    %clk = seq.to_clock %f
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock):
    %clk_i1 = seq.from_clock %clk
    %t = hw.constant true
    verif.clocked_assert %t, posedge %clk_i1 : i1
    verif.yield %clk : !seq.clock
  }
  func.return %res : i1
}

hw.module private @helper(in %clk: i1, in %en: i1, in %cond: i1) {
  verif.clocked_assert %cond if %en, posedge %clk : i1
  hw.output
}
