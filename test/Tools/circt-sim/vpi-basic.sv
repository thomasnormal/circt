// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top vpi_test_top 2>&1 | FileCheck %s
// Simple SystemVerilog module for VPI testing.
// Has a clock, reset, and a counter signal.
// With no external clock or reset driving, the 4-state counter remains X; the
// initial block fires
// after 100 time units and displays the value.
// CHECK: counter=x

module vpi_test_top(input logic clk, input logic rst);
  logic [7:0] counter;

  always_ff @(posedge clk or posedge rst) begin
    if (rst)
      counter <= 8'h00;
    else
      counter <= counter + 8'h01;
  end

  initial begin
    #100;
    $display("counter=%0d", counter);
    $finish;
  end
endmodule
