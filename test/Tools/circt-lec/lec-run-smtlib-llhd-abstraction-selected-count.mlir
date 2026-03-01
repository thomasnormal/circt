// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=a -c2=b %s 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: circt-lec --run-smtlib --accept-llhd-abstraction -c1=a -c2=b %s 2>&1 | FileCheck %s --check-prefix=ACCEPT

// Selected-module LLHD abstraction metadata should drive SAT classification and
// report the selected abstraction input count.
module {
  hw.module @a(out out: i1) {
    %true = hw.constant true
    hw.output %true : i1
  }

  hw.module @b(out out: i1) attributes {circt.bmc_abstracted_llhd_interface_inputs = 1 : i32} {
    %false = hw.constant false
    hw.output %false : i1
  }
}

// STRICT: c1 != c2
// STRICT: LEC_RESULT=UNKNOWN
// STRICT: LEC_DIAG=LLHD_ABSTRACTION
// STRICT: LEC_DIAG_LLHD_ABSTRACTED_INPUTS=1

// ACCEPT: c1 == c2
// ACCEPT: LEC_RESULT=EQ
// ACCEPT: LEC_DIAG=LLHD_ABSTRACTION
// ACCEPT: LEC_DIAG_LLHD_ABSTRACTED_INPUTS=1
