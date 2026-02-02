// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s | FileCheck %s

// Verify that multi-drive signals without enable are now handled in strict mode
// (producing an abstracted unknown input).

// CHECK: smt.solver
// CHECK: smt.declare_fun "a"
// CHECK: smt.declare_fun "b"
// CHECK: smt.declare_fun "sig_unknown"
// CHECK-NOT: llhd.drv
// CHECK-NOT: llhd.prb

module {
  hw.module @top(in %a : i1, in %b : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : i1
    llhd.drv %sig, %a after %t0 : i1
    llhd.drv %sig, %b after %t0 : i1
    %p = llhd.prb %sig : i1
    hw.output %p : i1
  }
}
