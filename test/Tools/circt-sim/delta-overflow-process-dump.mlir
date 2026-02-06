// RUN: not circt-sim %s --max-deltas=5 2>&1 | FileCheck %s

// CHECK: ERROR(DELTA_OVERFLOW)
// CHECK: [circt-sim] Process states:
// CHECK: [circt-sim] Signals changed in last delta:
// CHECK: [circt-sim] Processes executed in last delta:

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %true = hw.constant true
  %sig = llhd.sig %false : i1

  llhd.process {
    llhd.drv %sig, %true after %eps : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    %val = llhd.prb %sig : i1
    %next = comb.xor %val, %true : i1
    llhd.drv %sig, %next after %eps : i1
    llhd.wait delay %eps, ^bb1
  }

  hw.output
}
