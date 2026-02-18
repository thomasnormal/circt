// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $fatal — should terminate simulation and print message
module top;
  initial begin
    // CHECK: before_fatal
    $display("before_fatal");
    $fatal(0, "fatal_reason: %0d", 42);
    // This line should NOT be reached
    // CHECK-NOT: after_fatal
    $display("after_fatal");
  end
endmodule
