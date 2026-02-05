// RUN: circt-bmc --emit-smtlib --allow-multi-clock -b 1 --module top %s | FileCheck %s

// This exercises multi-clock BMC when register clocks are derived from complex
// i1 expressions (so they cannot be traced to a single input root). In this
// case, ExternalizeRegisters uses an expression key, and VerifToSMT maps it via
// bmc_clock_keys.

// CHECK: (declare-const d (_ BitVec 1))
// CHECK: (declare-const r1_state (_ BitVec 1))
// CHECK: (declare-const r2_state (_ BitVec 1))
// CHECK: (check-sat)

hw.module @top(in %a : i1, in %b : i1, in %c : i1, in %d : i1) {
  %clk1_i1 = comb.and %a, %b : i1
  %clk2_i1 = comb.or %b, %c : i1
  %clk1 = seq.to_clock %clk1_i1
  %clk2 = seq.to_clock %clk2_i1
  %r1 = seq.compreg %d, %clk1 : i1
  %r2 = seq.compreg %r1, %clk2 : i1
  verif.assert %r2 : i1
  hw.output
}
