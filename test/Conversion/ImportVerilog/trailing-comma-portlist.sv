// RUN: circt-verilog --parse-only %s
// REQUIRES: slang

module trailing_comma(
  input logic clk,
  input logic rst,
);
endmodule
