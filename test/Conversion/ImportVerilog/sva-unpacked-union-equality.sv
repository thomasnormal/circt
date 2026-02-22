// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaUnpackedUnionEquality(input logic clk);
  typedef union {
    logic [1:0] a;
    logic [1:0] b;
  } u_t;
  u_t x, y;

  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk) (x == y));

  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk) (x != y));

  // CHECK: moore.union_extract
  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk) (x === y));

  // CHECK: moore.union_extract
  // CHECK: moore.case_eq
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk) (x !== y));
endmodule
