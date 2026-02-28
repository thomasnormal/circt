// RUN: circt-verilog %s --ir-llhd --single-unit -o %t.mlir
// RUN: circt-sim --resource-guard=false --mode interpret --top tb %t.mlir 2>&1 | FileCheck %s
// REQUIRES: circt-sim
//
// Regression: instance-boundary NBA visibility ordering.
// Expected behavior matches xrun: a nonblocking update to `we` in TB at a
// given posedge must not be observed by DUT's always_ff until the next posedge.

module dut(input logic clk, input logic rst_n, input logic we, output logic [7:0] y);
  always_ff @(posedge clk) begin
    $display("DUT we=%0d y_pre=%0d", we, y);
    if (!rst_n)
      y <= 8'd0;
    else
      y <= we ? 8'd3 : 8'd0;
  end
endmodule

module tb;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n = 0;
  logic we = 0;
  logic [7:0] y;
  int i;

  dut d(.clk, .rst_n, .we, .y);

  initial begin
    repeat (2) @(posedge clk);
    rst_n <= 1;
    for (i = 0; i < 3; i++) begin
      @(posedge clk);
      $display("TB pre i=%0d we=%0d y=%0d", i, we, y);
      we <= (i == 0);
    end
    #1 $finish;
  end
endmodule

// CHECK: DUT we=0 y_pre=x
// CHECK: DUT we=0 y_pre=0
// CHECK: TB pre i=0 we=0 y=0
// CHECK: DUT we=0 y_pre=0
// CHECK: TB pre i=1 we=1 y=0
// CHECK: DUT we=1 y_pre=0
// CHECK: TB pre i=2 we=0 y=3
// CHECK: DUT we=0 y_pre=3
