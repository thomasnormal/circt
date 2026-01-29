// RUN: circt-opt %s --convert-verif-to-smt="rising-clocks-only=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Test that rising-clocks-only mode handles clock-op negedge properties.
// The clock edge information is incorporated into bmc_circuit's return.

// CHECK-LABEL: func.func @bmc_rising_clocks_only_clock_op_negedge() -> i1
// CHECK:   scf.for
// CHECK:     func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool)
// CHECK:     func.call @bmc_loop
func.func @bmc_rising_clocks_only_clock_op_negedge() -> i1 {
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
    %neg = hw.constant -1 : i1
    %nclk = comb.xor %from, %neg : i1
    %new = seq.to_clock %nclk
    verif.yield %new : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock):
    %t = ltl.boolean_constant true
    %clk_i1 = seq.from_clock %clk
    %prop = ltl.clock %t, negedge %clk_i1 : !ltl.property
    verif.assert %prop : !ltl.property
    %out = hw.constant true
    verif.yield %out : i1
  }
  func.return %bmc : i1
}
