// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaCaseEqUnpackedArray(input logic clk);
  typedef logic [3:0] arr_t [0:1];
  arr_t x, y;

  // CHECK: moore.uarray_cmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk) (x === y));

  // CHECK: moore.uarray_cmp ne
  // CHECK: verif.assert
  assert property (@(posedge clk) (x !== y));
endmodule
