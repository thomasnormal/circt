// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedRoseFell(input logic clk);
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

  // CHECK: moore.dyn_extract
  // CHECK: moore.or
  // CHECK: moore.past
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(arr));

  // CHECK: moore.struct_extract
  // CHECK: moore.or
  // CHECK: moore.past
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk) $fell(st));

  // CHECK: moore.union_extract
  // CHECK: moore.or
  // CHECK: moore.past
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(un));
endmodule
