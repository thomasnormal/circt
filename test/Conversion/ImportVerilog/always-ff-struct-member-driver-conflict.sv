// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR

typedef struct packed {
  logic a;
  logic b;
} packed_s_t;

module StructMemberMultiAlwaysFFDriver(
  input logic clk,
  input logic x,
  output packed_s_t y
);
  always_ff @(posedge clk) y.a <= x;
  always_ff @(posedge clk) y.a <= ~x;
endmodule

// ERR: error: variable 'y' driven by always_ff procedure
