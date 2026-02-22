// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedArray(input logic clk);
  logic [1:0] s [2];

  // CHECK: moore.past
  // CHECK: moore.uarray_cmp eq
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk) $changed(s));

  // CHECK: moore.past
  // CHECK: moore.uarray_cmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk) $stable(s));
endmodule
