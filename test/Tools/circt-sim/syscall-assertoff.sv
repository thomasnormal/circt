// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: $assertoff not yet disabling immediate assertions at runtime.
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("ASSERT_FAILED: a != 1");
  end

  initial begin
    // First, trigger assertion failure normally
    #1 clk = 1; #1 clk = 0;
    // CHECK: ASSERT_FAILED: a != 1

    // Now disable assertions
    $assertoff;
    // CHECK: assertoff_called
    $display("assertoff_called");
    a = 0;
    #1 clk = 1; #1 clk = 0;
    // After $assertoff, no assertion failure should appear
    // CHECK-NOT: ASSERT_FAILED
    // CHECK: after_assertoff
    $display("after_assertoff");
    $finish;
  end
endmodule
