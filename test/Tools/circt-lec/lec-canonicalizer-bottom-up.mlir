// RUN: circt-lec --emit-mlir --assume-known-inputs --verbose-pass-executions -c1=ref -c2=dut %s -o /dev/null 2>&1 | FileCheck %s

module {
  hw.module @ref(in %in_i : i1, out out_o : i1) {
    hw.output %in_i : i1
  }

  hw.module @dut(in %in_i : i1, out out_o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %in_i : i1
    %comb = llhd.combinational -> i1 {
      llhd.yield %in_i : i1
    }
    llhd.drv %sig, %comb after %t0 : i1
    %prb = llhd.prb %sig : i1
    hw.output %prb : i1
  }
}

// CHECK: canonicalize
// CHECK-SAME: max-num-rewrites=200000
// CHECK-SAME: region-simplify=disabled
// CHECK-SAME: top-down=false
