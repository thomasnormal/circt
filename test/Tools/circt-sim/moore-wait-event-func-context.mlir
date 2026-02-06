// RUN: circt-sim %s --max-time=10000000000 --sim-stats 2>&1 | FileCheck %s

// Test that moore.wait_event works correctly when appearing in func.func context
// (e.g., from forked UVM code). The conversion should use __moore_wait_event
// runtime call instead of llhd.wait (which requires llhd.process parent).

// CHECK: [circt-sim] Found {{[0-9]+}} LLHD process
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time
// CHECK: [circt-sim] Simulation completed

// A function that contains wait_event - this simulates forked code.
// In real UVM, this would be inside a class task called from a fork.
func.func @wait_for_event(%eventPtr: !llhd.ref<i8>) {
  // Wait for any change on the event memory location.
  // This tests the runtime fallback path since we're in func.func, not llhd.process.
  moore.wait_event {
    %event_val = llhd.prb %eventPtr : i8
    %event_moore = builtin.unrealized_conversion_cast %event_val : i8 to !moore.i8
    moore.detect_event any %event_moore : i8
  }
  return
}

hw.module @TestWaitEventFuncContext() {
  %true = hw.constant true
  %false = hw.constant false
  %zero_i8 = hw.constant 0 : i8
  %one_i8 = hw.constant 1 : i8
  %c500000_i64 = arith.constant 500000 : i64   // 500ps clock period
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Clock signal
  %clk = llhd.sig %false : i1

  // Event signal (simulates a UVM event stored in memory)
  %event_sig = llhd.sig %zero_i8 : i8

  // Clock generation process
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

  // Process that triggers the event after a few clock cycles.
  llhd.process {
    // Wait for 3 posedges of clk
    moore.wait_event {
      %clk_prb = llhd.prb %clk : i1
      %clk_moore = builtin.unrealized_conversion_cast %clk_prb : i1 to !moore.l1
      moore.detect_event posedge %clk_moore : l1
    }
    moore.wait_event {
      %clk_prb = llhd.prb %clk : i1
      %clk_moore = builtin.unrealized_conversion_cast %clk_prb : i1 to !moore.l1
      moore.detect_event posedge %clk_moore : l1
    }
    moore.wait_event {
      %clk_prb = llhd.prb %clk : i1
      %clk_moore = builtin.unrealized_conversion_cast %clk_prb : i1 to !moore.l1
      moore.detect_event posedge %clk_moore : l1
    }

    // Trigger the event
    llhd.drv %event_sig, %one_i8 after %eps : i8
    llhd.halt
  }

  // Process that waits for the event using a function call.
  // This tests the func.func context for wait_event.
  llhd.process {
    // Call the function that contains wait_event.
    // The wait_event inside should use __moore_wait_event runtime.
    func.call @wait_for_event(%event_sig) : (!llhd.ref<i8>) -> ()

    // After receiving the event, halt
    llhd.halt
  }

  hw.output
}
