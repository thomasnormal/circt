// RUN: circt-lec --emit-mlir -c1=with_loop -c2=plain %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK-NOT: llhd.

module {
  hw.module @with_loop(in %a : i1, out o : i1) {
    %0 = llhd.combinational -> i1 {
      cf.br ^bb1(%a : i1)
    ^bb1(%v: i1):  // 2 preds: ^bb0, ^bb1
      cf.cond_br %a, ^bb1(%v : i1), ^bb2
    ^bb2:  // pred: ^bb1
      llhd.yield %v : i1
    }
    hw.output %0 : i1
  }

  hw.module @plain(in %a : i1, out o : i1) {
    %0 = llhd.combinational -> i1 {
      llhd.yield %a : i1
    }
    hw.output %0 : i1
  }
}
