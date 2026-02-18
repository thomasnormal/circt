// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("ASSERT_FAILED");
  end

  initial begin
    // Disable assertions
    $assertoff;
    #1 clk = 1; #1 clk = 0;
    // No failure expected here

    // Re-enable assertions
    $asserton;
    #1 clk = 1; #1 clk = 0;
    // CHECK: ASSERT_FAILED

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
