// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top event_triggered_tb 2>&1 | FileCheck %s

// Test: event .triggered property correctly reports whether an event
// has been triggered in the current time step.

// CHECK: event triggered ok
// CHECK: [circt-sim] Simulation completed
module event_triggered_tb();
  event ev;
  initial begin
    ->ev;
    if (ev.triggered)
      $display("event triggered ok");
    else
      $display("FAIL: event not triggered");
  end
endmodule
