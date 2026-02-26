// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

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

// The pass flattens control-flow and resolves mutually-exclusive drives to a
// concrete mux chain.
// CHECK: comb.mux
// CHECK-NOT: sig_unknown
// CHECK-NOT: llhd.
