// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @top(in %a : i1, in %b : i1) {
    %0 = llhd.combinational -> i1 {
      %1 = comb.xor %a, %b : i1
      llhd.yield %1 : i1
    }
    verif.assert %0 : i1
    hw.output
  }
}

// CHECK: smt.solver
// CHECK-NOT: llhd.
