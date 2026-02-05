// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: smt.solver

module {
  hw.module @top(in %a : i1, in %b : i1, in %en : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %true = hw.constant true
    %not_en = comb.xor %en, %true : i1
    %sig = llhd.sig %a : i1
    llhd.drv %sig, %a after %t0 if %en : i1
    llhd.drv %sig, %b after %t0 if %not_en : i1
    %p = llhd.prb %sig : i1
    hw.output %p : i1
  }
}
