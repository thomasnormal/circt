// RUN: circt-bmc --emit-mlir -b 1 --module m %s | FileCheck %s

module {
  hw.module @m() {
    %false = hw.constant false
    %true = hw.constant true
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %sig_init : !hw.struct<value: i1, unknown: i1>
    %p:2 = llhd.process -> !hw.struct<value: i1, unknown: i1>, i1 {
      cf.br ^bb1(%sig_init, %true : !hw.struct<value: i1, unknown: i1>, i1)
    ^bb1(%v: !hw.struct<value: i1, unknown: i1>, %en: i1):
      llhd.wait yield (%v, %en : !hw.struct<value: i1, unknown: i1>, i1), delay %t0, ^bb1
    }
    llhd.drv %sig, %p#0 after %t0 if %p#1 : !hw.struct<value: i1, unknown: i1>
    %prb = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    %val = hw.struct_extract %prb["value"] : !hw.struct<value: i1, unknown: i1>
    %clk = seq.to_clock %val
    %reg = seq.firreg %false clock %clk {name = "r"} : i1
    verif.assert %reg : i1
    hw.output
  }
}

// CHECK: smt.solver
// CHECK-NOT: llhd.process
