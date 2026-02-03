// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that past buffer updates are gated by posedge OR negedge when edge=both.

// CHECK-LABEL: func.func @bmc_past_buffer_edge_both() -> i1
// CHECK: scf.for
// Loop is called first
// CHECK:   func.call @bmc_loop
// Edge detection: posedge OR negedge (both bv.nots first, then both bv.ands, then both eqs)
// CHECK:   smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.and {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.and {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// Circuit returns outputs + past buffer + !smt.bool for the property
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
// Property check happens between edge detection and buffer update
// CHECK:   smt.or {{%.+}}, {{%.+}}
// Past buffer update conditioned on edge
// CHECK:   smt.ite {{%.+}}, {{%.+}}, {{%.+}} : !smt.bv<1>
func.func @bmc_past_buffer_edge_both() -> i1 {
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
    %seq = ltl.past %sig, 1 {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge edge>} : i1
    verif.assert %seq : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
