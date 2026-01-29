// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @top(in %cond : i1, out out : i1) {
    %init = hw.constant 0 : i1
    %sig_a = llhd.sig %init : i1
    %sig_b = llhd.sig %init : i1
    %0 = llhd.combinational -> i1 {
      %1 = comb.mux %cond, %sig_a, %sig_b : !llhd.ref<i1>
      %2 = llhd.prb %1 : i1
      llhd.yield %2 : i1
    }
    hw.output %0 : i1
  }
}

// CHECK: smt.solver
// CHECK-NOT: llhd.
