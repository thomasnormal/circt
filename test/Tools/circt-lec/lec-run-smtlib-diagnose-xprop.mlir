// RUN: circt-lec --run-smtlib --diagnose-xprop --print-solver-output \
// RUN:   --z3-path=%S/Inputs/fake-z3-xdiag.sh -c1=modA -c2=modB %s 2>&1 \
// RUN:   | FileCheck %s

hw.module @modA(in %fs: !hw.struct<value: i8, unknown: i8>, out out: i1) {
  %false = hw.constant false
  hw.output %false : i1
}

hw.module @modB(in %fs: !hw.struct<value: i8, unknown: i8>, out out: i1) {
  %unk = hw.struct_extract %fs["unknown"] : !hw.struct<value: i8, unknown: i8>
  %c0 = hw.constant 0 : i8
  %neq = comb.icmp ne %unk, %c0 : i8
  hw.output %neq : i1
}

// CHECK: note: LEC mismatch only exists when 4-state inputs are unconstrained
// CHECK: counterexample inputs:
// CHECK: fs = value=8'h00 unknown=8'h01
// CHECK: c1 != c2
// CHECK: LEC_DIAG_ASSUME_KNOWN_RESULT=UNSAT
// CHECK: LEC_DIAG=XPROP_ONLY
