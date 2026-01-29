// RUN: circt-opt %s --convert-verif-to-smt="rising-clocks-only=true" \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Test that rising-clocks-only mode handles edge=both properties.
// The clock edge information is incorporated into bmc_circuit's return.

// CHECK-LABEL: func.func @bmc_rising_clocks_only_edge_both() -> i1
// CHECK:   scf.for
// CHECK:     func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool)
// CHECK:     func.call @bmc_loop
func.func @bmc_rising_clocks_only_edge_both() -> i1 {
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
    verif.assert %seq {bmc.clock_edge = #ltl<clock_edge edge>} : !ltl.sequence
    verif.yield %true : i1
  }
  func.return %bmc : i1
}
