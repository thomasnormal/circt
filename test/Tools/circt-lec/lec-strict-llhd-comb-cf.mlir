// RUN: not circt-lec --emit-mlir --strict-llhd -c1=m -c2=m %s %s 2>&1 | FileCheck %s

// CHECK: LLHD combinational control flow requires abstraction; rerun without --strict-llhd

module {
  hw.module @m(in %a : i1, in %b : i1, in %sel : i1, out out_o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : i1
    %comb = llhd.combinational -> i1 {
      cf.cond_br %sel, ^bb1, ^bb2
    ^bb1:
      llhd.drv %sig, %a after %t0 : i1
      cf.br ^bb3
    ^bb2:
      llhd.drv %sig, %b after %t0 : i1
      cf.br ^bb3
    ^bb3:
      %p = llhd.prb %sig : i1
      llhd.yield %p : i1
    }
    hw.output %comb : i1
  }
}
