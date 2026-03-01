// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=a -c2=b %s 2>&1 | FileCheck %s

// Top-level LLHD abstraction totals may include unrelated modules. LEC should
// classify LLHD abstraction from the selected compared circuits only.
module attributes {circt.lec_abstracted_llhd_interface_inputs = 99 : i32} {
  hw.module @a(out out: i1) {
    %true = hw.constant true
    hw.output %true : i1
  }

  hw.module @b(out out: i1) {
    %false = hw.constant false
    hw.output %false : i1
  }
}

// CHECK: c1 != c2
// CHECK: LEC_RESULT=NEQ
// CHECK-NOT: LEC_DIAG=LLHD_ABSTRACTION
