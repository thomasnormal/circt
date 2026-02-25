// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=modA -c2=modB %s | FileCheck %s

module attributes {circt.lec_abstracted_llhd_interface_inputs = 1 : i32} {
  hw.module @modA(in %in: i1, out out: i1) {
    hw.output %in : i1
  }

  // This models a side that reached LEC with unresolved LLHD interface
  // abstraction; mismatches should be reported as inconclusive.
  hw.module @modB(in %in: i1, in %llhd_comb: i1, out out: i1) {
    hw.output %llhd_comb : i1
  }
}

// CHECK: c1 != c2
// CHECK: LEC_RESULT=UNKNOWN
// CHECK: LEC_DIAG=LLHD_ABSTRACTION
