// RUN: not circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s
// Test $past, $rose, $fell, $stable, $changed — these sampling value functions
// require SVA concurrent assertion context with clock inference. In procedural
// context without a clocked assertion, MooreToCore correctly rejects them because
// the moore.past op cannot be lowered without a clock.
module top;
  reg clk = 0;
  reg [7:0] val = 0;

  always #5 clk = ~clk;

  initial begin
    val = 8'h00;
    @(posedge clk);
    val = 8'hFF;
    @(posedge clk);

    // CHECK: error: non-boolean moore.past requires a clocked assertion
    $display("past=%0d", $past(val));
    $finish;
  end
endmodule
