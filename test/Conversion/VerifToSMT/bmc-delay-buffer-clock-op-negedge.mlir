// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// Test that delay buffer updates are gated by negedge derived from ltl.clock.
// The negedge is computed as: old_clock AND NOT new_clock.

// CHECK-LABEL: func.func @bmc_delay_buffer_clock_op_negedge() -> i1
// CHECK: scf.for
// CHECK:   func.call @bmc_loop
// Circuit returns outputs + delay buffer + !smt.bool for the property
// Negedge detection: old_clock AND NOT new_clock
// CHECK:   smt.bv.not {{%.+}} : !smt.bv<1>
// CHECK:   smt.bv.and {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:   func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
// Delay buffer update conditioned on negedge
// CHECK:   smt.ite {{%.+}}, {{%.+}}, {{%.+}} : !smt.bv<1>
func.func @bmc_delay_buffer_clock_op_negedge() -> i1 {
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
    %seq = ltl.delay %sig, 1, 0 : i1
    %clk_i1 = seq.from_clock %clk
    %clocked = ltl.clock %seq, negedge %clk_i1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
