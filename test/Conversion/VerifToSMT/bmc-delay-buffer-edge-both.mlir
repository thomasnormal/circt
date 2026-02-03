// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that edge=both delay buffer updates are gated by posedge OR negedge.
// posedge = NOT old_clock AND new_clock
// negedge = old_clock AND NOT new_clock

// CHECK-LABEL: func.func @bmc_delay_buffer_edge_both() -> i1
// CHECK: scf.for
// Loop is called first
// CHECK:   func.call @bmc_loop
// Edge detection: posedge OR negedge (before circuit call)
// CHECK:   smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.and {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.and {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// Circuit returns outputs + delay buffer + !smt.bool for the property
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
// Delay buffer update conditioned on edge
// CHECK:   smt.or {{%.+}}, {{%.+}}
// CHECK:   smt.ite {{%.+}}, {{%.+}}, {{%.+}} : !smt.bv<1>
func.func @bmc_delay_buffer_edge_both() -> i1 {
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
    %seq = ltl.delay %sig, 1, 0 {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge edge>} : i1
    verif.assert %seq : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
