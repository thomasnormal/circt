// RUN: circt-opt %s --convert-verif-to-smt="rising-clocks-only=true" \
// RUN:   -allow-unregistered-dialect 2>&1 | FileCheck %s

// Test that BMC with only assume (no assert) trivially returns true.
// When there's no property to check, the module returns constant true.

// CHECK: warning: no property provided to check in module - will trivially find no violations
// CHECK-LABEL: func.func @bmc_rising_clocks_only_assume_negedge() -> i1
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   return %[[TRUE]]
func.func @bmc_rising_clocks_only_assume_negedge() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk"]
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
  ^bb0(%clk: !seq.clock):
    %true = hw.constant true
    %seq = ltl.delay %true, 0, 0 : i1
    verif.assume %seq {bmc.clock_edge = #ltl<clock_edge negedge>} : !ltl.sequence
    verif.yield %true : i1
  }
  func.return %bmc : i1
}
