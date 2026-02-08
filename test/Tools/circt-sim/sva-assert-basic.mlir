// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that SVA verification ops (assert/assume/cover) halt their process
// cleanly instead of spinning forever or crashing.

// CHECK: [circt-sim] Simulation completed

module {
  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %true = hw.constant true

    // Simulate a process with an assertion evaluation loop.
    // Without the verif.assert handler, this would spin forever.
    llhd.process {
      cf.br ^check
    ^check:
      verif.assert %true : i1
      llhd.wait delay %t1, ^check
    }
    hw.output
  }
}
