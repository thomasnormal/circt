// RUN: true
// UNSUPPORTED: true
// This test requires UVM runtime simulation, not circt-verilog compilation.

//===----------------------------------------------------------------------===//
// UVM Phase wait_for_state Test
//===----------------------------------------------------------------------===//
//
// This test verifies that uvm_phase::wait_for_state() method and the
// uvm_phase_state enum are available, as required by real-world UVM code.
//
// Regression test for: Missing uvm_phase::wait_for_state() method
// The AXI4-Lite AVIP uses this pattern in assertion modules to synchronize
// with the start_of_simulation phase before enabling assertions.
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"
import uvm_pkg::*;

module uvm_phase_wait_for_state_test;

  // Test that uvm_phase_state enum values are accessible
  initial begin
    uvm_phase_state state;

    $display("Testing uvm_phase_state enum values...");

    // Test all enum values
    state = UVM_PHASE_DORMANT;
    $display("  UVM_PHASE_DORMANT = %0d", state);

    state = UVM_PHASE_SCHEDULED;
    $display("  UVM_PHASE_SCHEDULED = %0d", state);

    state = UVM_PHASE_SYNCING;
    $display("  UVM_PHASE_SYNCING = %0d", state);

    state = UVM_PHASE_STARTED;
    $display("  UVM_PHASE_STARTED = %0d", state);

    state = UVM_PHASE_EXECUTING;
    $display("  UVM_PHASE_EXECUTING = %0d", state);

    state = UVM_PHASE_READY_TO_END;
    $display("  UVM_PHASE_READY_TO_END = %0d", state);

    state = UVM_PHASE_ENDED;
    $display("  UVM_PHASE_ENDED = %0d", state);

    state = UVM_PHASE_CLEANUP;
    $display("  UVM_PHASE_CLEANUP = %0d", state);

    state = UVM_PHASE_DONE;
    $display("  UVM_PHASE_DONE = %0d", state);

    state = UVM_PHASE_JUMPING;
    $display("  UVM_PHASE_JUMPING = %0d", state);

    $display("All uvm_phase_state enum values accessible - PASS");
  end

  // Test that wait_for_state() method exists on phase handles
  initial begin
    $display("Testing uvm_phase::wait_for_state() method...");

    // This is the pattern used in AXI4-Lite AVIP assertion modules
    start_of_simulation_ph.wait_for_state(UVM_PHASE_STARTED);

    $display("wait_for_state() method accessible - PASS");
  end

  // Test get_state() method
  initial begin
    uvm_phase_state current_state;

    $display("Testing uvm_phase::get_state() method...");

    current_state = build_ph.get_state();
    $display("  build_ph state = %0d", current_state);

    $display("get_state() method accessible - PASS");
  end

  // Final message
  initial begin
    #1;
    $display("All uvm_phase wait_for_state tests PASSED");
    $finish;
  end

endmodule
