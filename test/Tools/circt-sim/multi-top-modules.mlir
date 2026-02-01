// RUN: circt-sim %s --top top1 --top top2 --max-time 1000000000 2>&1 | FileCheck %s

// This test verifies that circt-sim correctly supports simulating multiple top
// modules simultaneously, which is essential for UVM testbenches that use
// separate hdl_top (DUT + interfaces) and hvl_top (UVM test) modules.

// CHECK: [circt-sim] Simulating 2 top modules: top1, top2
// CHECK: [circt-sim] Found {{[0-9]+}} LLHD processes
// CHECK: [circt-sim] Registered {{[0-9]+}} LLHD signals and {{[0-9]+}} LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed

// Test module 1 with a clock-like process
hw.module @top1() {
  %c1 = hw.constant 1 : i1
  %c0 = hw.constant 0 : i1
  %t1 = llhd.constant_time <10ns, 0d, 0e>
  %clk = llhd.sig %c0 : i1

  llhd.process {
    cf.br ^check
  ^check:
    %current = llhd.prb %clk : i1
    %next = comb.xor %current, %c1 : i1
    llhd.drv %clk, %next after %t1 : i1
    // Use delay-based wait to avoid infinite delta loop
    llhd.wait delay %t1, ^check
  }
  hw.output
}

// Test module 2 with a simple process - simulates hvl_top pattern
hw.module @top2() {
  %c0 = hw.constant 0 : i1
  %t100ns = llhd.constant_time <100ns, 0d, 0e>
  %data = llhd.sig %c0 : i1

  // This process waits for a delay - simulating hvl_top's run_test() pattern
  // In real testbenches, this would be driven by UVM phases
  llhd.process {
    cf.br ^wait
  ^wait:
    // Wait for a delay to allow top1's clock to run
    llhd.wait delay %t100ns, ^done
  ^done:
    llhd.halt
  }
  hw.output
}
