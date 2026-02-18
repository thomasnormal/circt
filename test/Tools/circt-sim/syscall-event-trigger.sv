// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test event trigger and wait
module top;
  event e;

  initial begin
    #5;
    -> e;  // trigger event
  end

  initial begin
    @(e);  // wait for event
    // CHECK: event_triggered
    $display("event_triggered");
    $finish;
  end
endmodule
