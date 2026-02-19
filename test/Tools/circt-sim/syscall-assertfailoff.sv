// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $assertfailoff suppresses assertion failure messages.
// Bug: $assertfailoff is a no-op — assertion failures still appear.
// IEEE 1800-2017 Section 20.12: $assertfailoff should suppress the
// execution of assertion fail statements (the else clause).
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("FAIL_MSG_VISIBLE");
  end

  initial begin
    // First verify assertion fires normally
    #1 clk = 1; #1 clk = 0;
    // CHECK: FAIL_MSG_VISIBLE

    // Now suppress failure messages
    $assertfailoff;
    #1 clk = 1; #1 clk = 0;
    // The failure message should NOT appear after assertfailoff
    // CHECK-NOT: FAIL_MSG_VISIBLE

    // CHECK: test_done
    $display("test_done");
    $finish;
  end
endmodule
