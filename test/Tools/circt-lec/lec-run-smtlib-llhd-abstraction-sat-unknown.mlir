// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=modA -c2=modB %s 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: circt-lec --run-smtlib --accept-llhd-abstraction -c1=modA -c2=modB %s 2>&1 | FileCheck %s --check-prefix=ACCEPT

module {
  hw.module @modA(in %in: i1, out out: i1) {
    hw.output %in : i1
  }

  // This models a side that reached LEC with unresolved LLHD interface
  // abstraction; mismatches should be reported as inconclusive.
  hw.module @modB(in %in: i1, in %llhd_comb: i1, out out: i1)
      attributes {circt.bmc_abstracted_llhd_interface_inputs = 1 : i32} {
    hw.output %llhd_comb : i1
  }
}

// STRICT: c1 != c2
// STRICT: LEC_RESULT=UNKNOWN
// STRICT: LEC_DIAG=LLHD_ABSTRACTION
// ACCEPT: note: accepting LLHD abstraction mismatch (--accept-llhd-abstraction)
// ACCEPT: c1 == c2
// ACCEPT: LEC_RESULT=EQ
// ACCEPT: LEC_DIAG=LLHD_ABSTRACTION
