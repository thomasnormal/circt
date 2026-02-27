// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR

module WholeStructAndFieldMultiAlwaysFFDriver(
  input logic clk,
  input logic a,
  input logic b
);
  typedef struct packed {
    logic x;
    logic y;
  } s_t;
  s_t s;
  always_ff @(posedge clk) s <= '{x: a, y: b};
  always_ff @(posedge clk) s.x <= b;
endmodule

// ERR: error: variable 's' driven by always_ff procedure
