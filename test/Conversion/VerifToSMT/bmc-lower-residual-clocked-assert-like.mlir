// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect 2>&1 | FileCheck %s

// Regression: clocked-only BMC properties must be recognized as real
// properties (not propertyless), so lowering does not short-circuit to a
// trivial result.
// CHECK: smt.solver
// CHECK-NOT: no property provided to check in module
func.func @bmc_lower_residual_clocked_assert_like() -> i1 {
  %res = verif.bmc bound 2 num_regs 0 initial_values []
  init {
    %f = hw.constant false
    %clk = seq.to_clock %f
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %clk_i1 = seq.from_clock %clk
    %t = hw.constant true
    %next = comb.xor %clk_i1, %t : i1
    %next_clk = seq.to_clock %next
    verif.yield %next_clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %clk_i1 = seq.from_clock %clk
    %del = ltl.delay %sig, 1, 0 : i1
    %prop = ltl.implication %sig, %del : i1, !ltl.sequence
    verif.clocked_assert %prop, posedge %clk_i1 : !ltl.property
    verif.yield %sig : i1
  }
  func.return %res : i1
}
