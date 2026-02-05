// RUN: circt-lec --emit-mlir --strict-llhd -c1=m -c2=m %s %s 2>&1 | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @m(in %a : i1, in %b : i1, in %c : i1, in %sel0 : i1, in %sel1 : i1, out out_o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : i1
    %comb = llhd.combinational -> i1 {
      cf.cond_br %sel0, ^bb1, ^bb2
    ^bb1:
      llhd.drv %sig, %a after %t0 : i1
      cf.br ^bb4
    ^bb2:
      cf.cond_br %sel1, ^bb3, ^bb5
    ^bb3:
      llhd.drv %sig, %b after %t0 : i1
      cf.br ^bb4
    ^bb5:
      llhd.drv %sig, %c after %t0 : i1
      cf.br ^bb4
    ^bb4:
      %p = llhd.prb %sig : i1
      llhd.yield %p : i1
    }
    hw.output %comb : i1
  }
}
