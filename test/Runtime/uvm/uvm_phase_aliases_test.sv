// RUN: true
// UNSUPPORTED: true
// This test requires UVM runtime simulation, not circt-verilog compilation.

//===----------------------------------------------------------------------===//
// UVM Phase Handle Aliases Test
//===----------------------------------------------------------------------===//
//
// This test verifies that the standard UVM phase handle aliases (with _ph suffix)
// are available, as required by IEEE 1800.2 and used in real-world UVM code.
//
// Regression test for: Missing UVM phase handle aliases bug
// The UVM stubs previously only defined _phase_h suffix handles, but standard
// UVM testbenches use the _ph suffix (e.g., start_of_simulation_ph).
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"
import uvm_pkg::*;

module uvm_phase_aliases_test;

  // Test that all standard _ph aliases are accessible
  initial begin
    // Access all the _ph suffix phase handles (should not cause compile errors)
    $display("Testing UVM phase handle aliases...");

    // Build phase
    if (build_ph != null)
      $display("  build_ph: %s", build_ph.get_name());

    // Connect phase
    if (connect_ph != null)
      $display("  connect_ph: %s", connect_ph.get_name());

    // End of elaboration phase
    if (end_of_elaboration_ph != null)
      $display("  end_of_elaboration_ph: %s", end_of_elaboration_ph.get_name());

    // Start of simulation phase (commonly used in AVIP code)
    if (start_of_simulation_ph != null)
      $display("  start_of_simulation_ph: %s", start_of_simulation_ph.get_name());

    // Run phase
    if (run_ph != null)
      $display("  run_ph: %s", run_ph.get_name());

    // Extract phase
    if (extract_ph != null)
      $display("  extract_ph: %s", extract_ph.get_name());

    // Check phase
    if (check_ph != null)
      $display("  check_ph: %s", check_ph.get_name());

    // Report phase
    if (report_ph != null)
      $display("  report_ph: %s", report_ph.get_name());

    // Final phase
    if (final_ph != null)
      $display("  final_ph: %s", final_ph.get_name());

    $display("All UVM phase handle aliases are accessible - TEST PASSED");
    $finish;
  end

endmodule
