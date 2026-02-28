// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: simplifying procedure globals into one-time local shadows must
// not stale across event waits. For blocking compound assignments in clocked
// processes, the LHS read must observe the latest global value each trigger.

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  logic [7:0] data = 0;
  logic [7:0] shadow;

  always @(posedge clk)
    shadow |= data;

  initial begin
    shadow = 0;
    repeat (8) begin
      @(posedge clk);
      data <= data + 1;
    end
    #1;
    $display("SUMMARY shadow=%0d", shadow);
    // CHECK: SUMMARY shadow=7
    #1 $finish;
  end
endmodule
