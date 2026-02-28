// RUN: not circt-verilog --parse-only %s 2>&1 | FileCheck %s --check-prefix=ERR
// REQUIRES: slang

module trailing_comma(
  input logic clk,
  input logic rst,
);
endmodule

// ERR: misplaced trailing ','
