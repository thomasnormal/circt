// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("ASSERT_KILLED_FAIL");
  end

  initial begin
    // Kill all active assertions
    $assertkill;
    // CHECK: assertkill_called
    $display("assertkill_called");
    #1 clk = 1; #1 clk = 0;
    // CHECK-NOT: ASSERT_KILLED_FAIL
    // CHECK: after_assertkill
    $display("after_assertkill");
    $finish;
  end
endmodule
