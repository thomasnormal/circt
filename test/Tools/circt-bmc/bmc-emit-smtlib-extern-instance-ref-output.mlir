// RUN: circt-bmc --emit-smtlib -b 1 --module top %s | FileCheck %s

// Extern module instances can carry auxiliary ref-typed outputs from imported
// assertion scaffolding. SMT-LIB export should still lower the live i1 result
// path and not leave hw.instance ops in the solver.

module {
  hw.module.extern @prim_ext(in %clk : !seq.clock, in %a : i1,
                             out y : i1, out dbg : !llhd.ref<i1>)

  hw.module @top(in %clk : !seq.clock, in %a : i1) {
    %y, %dbg = hw.instance "u" @prim_ext(clk: %clk: !seq.clock, a: %a: i1) -> (y: i1, dbg: !llhd.ref<i1>)
    verif.assert %y : i1
    hw.output
  }
}

// CHECK: (check-sat)
