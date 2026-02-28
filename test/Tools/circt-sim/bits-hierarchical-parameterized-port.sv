// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.default.mlir 2>/dev/null
// RUN: circt-sim --top tb %t.default.mlir 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit --allow-hierarchical-const -o %t.hierconst.mlir 2>/dev/null
// RUN: circt-sim --top tb %t.hierconst.mlir 2>&1 | FileCheck %s --check-prefix=CHECK
// REQUIRES: slang

// Regression: $bits(hierarchical-parameterized-port) should return the
// elaborated bit width in default mode, not 0.
// CHECK: bits_hier=3 bits_local=3
// CHECK: PASS

module sram #(parameter int DEPTH = 16, parameter int WIDTH = 8) (
  input  logic                      clk,
  input  logic                      we,
  input  logic [$clog2(DEPTH)-1:0] addr,
  input  logic [WIDTH-1:0]         wdata,
  output logic [WIDTH-1:0]         rdata
);
  logic [WIDTH-1:0] mem [0:DEPTH-1];
  always_ff @(posedge clk) begin
    if (we) mem[addr] <= wdata;
    rdata <= mem[addr];
  end
endmodule

module tb;
  logic clk = 0;
  always #5 clk = ~clk;

  logic [2:0] addr4;
  logic [3:0] wdata4, rdata4;
  logic we4;
  sram #(.DEPTH(8), .WIDTH(4)) u_small(
      .clk, .we(we4), .addr(addr4), .wdata(wdata4), .rdata(rdata4));

  initial begin
    $display("bits_hier=%0d bits_local=%0d", $bits(u_small.addr), $bits(addr4));
    if ($bits(u_small.addr) !== 3) $display("FAIL");
    else                           $display("PASS");
    $finish;
  end
endmodule
