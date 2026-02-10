// RUN: not circt-opt --strip-llhd-interface-signals %s 2>&1 | FileCheck %s --check-prefix=FAIL
// RUN: circt-opt --strip-llhd-interface-signals='require-no-llhd=false' %s | FileCheck %s --check-prefix=PASS

hw.module @top() {
  %c0_i1 = hw.constant false
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %sig = llhd.sig %c0_i1 : i1

  llhd.process {
    %v = llhd.prb %sig : i1
    llhd.drv %sig, %v after %delta : i1
    llhd.halt
  }

  hw.output
}

// FAIL: error: LLHD operations are not supported by circt-lec
// PASS: hw.module @top
// PASS: llhd.process
