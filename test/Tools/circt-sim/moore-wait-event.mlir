// RUN: circt-sim %s --max-time=1000000000 --sim-stats | FileCheck %s

// Test that moore.wait_event and arith.constant operations are handled
// correctly by circt-sim. This test verifies:
// 1. arith.constant values can be retrieved from within llhd.process bodies
// 2. moore.wait_event operations properly suspend process execution
// 3. moore.detect_event operations inside wait_event bodies are processed

// CHECK: [circt-sim] Found {{[0-9]+}} LLHD process
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1000000000 fs
// CHECK: Processes executed:
// CHECK-SAME: {{[0-9]+}}
// CHECK: Signal updates:
// CHECK-SAME: {{[0-9]+}}
// CHECK: [circt-sim] Simulation finished successfully

hw.module @MooreWaitEventTest() {
  %true = hw.constant true
  %false = hw.constant false
  // Use arith.constant to verify it's handled correctly (unlike hw.constant)
  // This was previously causing simulation to hang at time 0
  %c500000_i64 = arith.constant 500000 : i64   // 500ps clock period (half)
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Clock signal
  %clk = llhd.sig %false : i1

  // Clock generation process using arith.constant for delay
  // This tests that arith.constant values are accessible during process execution
  %0 = llhd.process -> i1 {
    cf.br ^bb1(%false : i1)
  ^bb1(%clk_val: i1):
    %delay = llhd.int_to_time %c500000_i64
    llhd.wait yield (%clk_val : i1), delay %delay, ^bb2
  ^bb2:
    %clk_inv = comb.xor %clk_val, %true : i1
    cf.br ^bb1(%clk_inv : i1)
  }
  llhd.drv %clk, %0 after %eps : i1

  // Process with moore.wait_event to test that it properly suspends
  // moore.wait_event should suspend the process until a signal changes
  llhd.process {
    // First moore.wait_event - should wait for clk change
    moore.wait_event {
      %clk_prb = llhd.prb %clk : i1
      %clk_moore = builtin.unrealized_conversion_cast %clk_prb : i1 to !moore.l1
      moore.detect_event posedge %clk_moore : l1
    }
    // After one clock edge, halt
    llhd.halt
  }

  hw.output
}
