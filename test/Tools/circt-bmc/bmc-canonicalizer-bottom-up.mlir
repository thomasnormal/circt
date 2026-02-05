// RUN: circt-bmc --emit-mlir --verbose-pass-executions -b 1 --module m %s -o /dev/null 2>&1 | FileCheck %s

module {
  hw.module @m(in %in_i : i1) {
    %false = hw.constant false
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %in_i : i1
    %p:2 = llhd.process -> i1, i1 {
      cf.br ^bb1(%false, %false : i1, i1)
    ^bb1(%v: i1, %en: i1):
      llhd.wait yield (%v, %en : i1, i1), delay %t0, ^bb1
    }
    llhd.drv %sig, %p#0 after %t0 if %p#1 : i1
    %prb = llhd.prb %sig : i1
    %clk = seq.to_clock %prb
    %reg = seq.firreg %false clock %clk {name = "r"} : i1
    verif.assert %reg : i1
    hw.output
  }
}

// CHECK: canonicalize
// CHECK-SAME: top-down=false
// CHECK-SAME: region-simplify=disabled
// CHECK-SAME: max-num-rewrites=200000
