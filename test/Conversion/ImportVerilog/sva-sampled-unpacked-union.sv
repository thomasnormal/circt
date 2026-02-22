// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedUnion(input logic clk_a, input logic clk_b);
  typedef union {
    logic [1:0] a;
    logic [1:0] b;
  } u_t;
  u_t u;

  // Regular assertion-clock sampled union support.
  // CHECK: moore.past
  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: moore.and
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $changed(u));

  // Explicit sampled clock should also work (helper path).
  // CHECK: moore.procedure always
  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $stable(u, @(posedge clk_b)));
endmodule
