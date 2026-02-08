// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top event_clearing_tb 2>&1 | FileCheck %s

// Test: Per IEEE 1800-2017 §15.5.3, the .triggered property of an event
// should return true only within the time slot where the event was triggered.
// After time advances (e.g., via #1), .triggered should return false.

// CHECK: triggered at time 0 = 1
// CHECK: triggered at time 1 = 0
// CHECK: triggered at time 2 = 1
// CHECK: [circt-sim] Simulation completed
module event_clearing_tb();
  event ev;
  initial begin
    // Trigger event at time 0
    ->ev;
    $display("triggered at time 0 = %0d", ev.triggered);

    // Advance time - event should auto-clear
    #1;
    $display("triggered at time 1 = %0d", ev.triggered);

    // Trigger again at time 2 and verify
    #1;
    ->ev;
    $display("triggered at time 2 = %0d", ev.triggered);

    $finish;
  end
endmodule
