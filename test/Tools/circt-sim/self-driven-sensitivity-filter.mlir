// RUN: circt-sim %s --max-deltas=5 2>&1 | FileCheck %s

// CHECK: [circt-sim] Simulation completed
// CHECK-NOT: ERROR(DELTA_OVERFLOW)

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %true = hw.constant true
  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1

  llhd.process -> i1 {
    %pa = llhd.prb %a : i1
    %pb = llhd.prb %b : i1
    %next = comb.xor %pa, %true : i1
    llhd.drv %a, %next after %eps : i1
    llhd.wait yield (%pa : i1), (%pa, %pb : i1, i1), ^bb1
  ^bb1:
    %pa1 = llhd.prb %a : i1
    %pb1 = llhd.prb %b : i1
    %next1 = comb.xor %pa1, %true : i1
    llhd.drv %a, %next1 after %eps : i1
    llhd.wait yield (%pa1 : i1), (%pa1, %pb1 : i1, i1), ^bb1
  }

  hw.output
}
