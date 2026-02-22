// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedRoseFellExplicitClock(input logic clk_a,
                                               input logic clk_b);
  typedef logic [1:0] arr_t [0:1];
  typedef struct {
    logic [1:0] a;
    logic b;
  } s_t;
  typedef union {
    logic [1:0] a;
    logic [1:0] b;
  } u_t;
  arr_t arr;
  s_t st;
  u_t un;

  // CHECK: moore.procedure always
  // CHECK: moore.dyn_extract
  // CHECK: moore.or
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $rose(arr, @(posedge clk_b)));

  // CHECK: moore.procedure always
  // CHECK: moore.struct_extract
  // CHECK: moore.or
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $fell(st, @(posedge clk_b)));

  // CHECK: moore.procedure always
  // CHECK: moore.union_extract
  // CHECK: moore.or
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $rose(un, @(posedge clk_b)));
endmodule
