// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaCaseEqUnpackedStruct(input logic clk);
  typedef struct {
    logic [1:0] a;
    logic b;
  } s_t;
  s_t x, y;

  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) (x === y));

  // CHECK: moore.case_eq
  // CHECK: moore.not
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) (x !== y));
endmodule
