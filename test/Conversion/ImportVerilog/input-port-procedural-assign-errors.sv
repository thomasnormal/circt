// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Test: module input ports are read-only in procedural assignments.
module InputPortProcAssign(
  input logic clk,
  input logic [7:0] wdata,
  output logic [7:0] q
);
  // expected-error @below {{cannot assign to input port 'wdata'}}
  always @(posedge clk) wdata <= q;
endmodule

// -----

// Test: input var ports are also read-only in procedural assignments.
module InputVarPortProcAssign(
  input logic clk,
  input var logic [7:0] wdata,
  output logic [7:0] q
);
  // expected-error @below {{cannot assign to input port 'wdata'}}
  always @(posedge clk) wdata <= q;
endmodule
