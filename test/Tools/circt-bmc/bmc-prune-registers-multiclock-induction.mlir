// RUN: circt-bmc --run-smtlib -b 1 --module top --k-induction --prune-bmc-registers=true --allow-multi-clock %s | FileCheck %s

// CHECK: BMC_BASE=UNSAT
// CHECK: BMC_STEP=UNSAT
// CHECK: BMC_RESULT=UNSAT
// CHECK: Induction holds.

// This design has no properties, so both induction stages are trivially UNSAT.
// The key regression here is that prune + allow-multi-clock is accepted in the
// induction entry path (it used to fail early on option validation).
hw.module @top(in %clk_a : !seq.clock, in %clk_b : !seq.clock, in %in : i1) {
  %r0 = seq.compreg %in, %clk_a : i1
  %r1 = seq.compreg %in, %clk_b : i1
  hw.output
}
