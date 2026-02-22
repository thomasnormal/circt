// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastUnpackedStructExplicitClock(input logic clk_a, input logic clk_b);
  typedef struct {
    logic [1:0] a;
    logic b;
  } s_t;
  s_t s;

  // CHECK: moore.variable : <ustruct
  // CHECK: moore.procedure always
  // CHECK: moore.blocking_assign
  // CHECK: moore.struct_extract
  // CHECK: moore.eq
  // CHECK: moore.and
  // CHECK: verif.assert
  assert property (@(posedge clk_a) ($past(s, 1, @(posedge clk_b)) == s));
endmodule
