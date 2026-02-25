// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedArray(input logic clk);
  logic [1:0] s [2];

  // CHECK: moore.past
  // CHECK: moore.uarray_cmp eq
  // CHECK: moore.not
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $changed(s));

  // CHECK: moore.past
  // CHECK: moore.uarray_cmp eq
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $stable(s));
endmodule
