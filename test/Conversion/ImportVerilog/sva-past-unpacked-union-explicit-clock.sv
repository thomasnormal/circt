// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastUnpackedUnionExplicitClock(input logic clk_a, input logic clk_b);
  typedef union {
    logic [1:0] a;
    logic [1:0] b;
  } u_t;
  u_t u;

  // CHECK: moore.variable : <uunion
  // CHECK: moore.procedure always
  // CHECK: moore.blocking_assign
  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: verif.assert
  assert property (@(posedge clk_a)
                   ($past(u, 1, @(posedge clk_b)).a == u.a));
endmodule
