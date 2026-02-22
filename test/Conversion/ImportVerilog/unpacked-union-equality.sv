// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module UnpackedUnionEquality(input logic [1:0] a0, input logic [1:0] b0,
                             input logic [1:0] a1, input logic [1:0] b1,
                             output logic eq_o, output logic ne_o,
                             output logic ceq_o, output logic cne_o);
  typedef union {
    logic [1:0] a;
    logic [1:0] b;
  } u_t;
  u_t x, y;

  always_comb begin
    x.a = a0;
    x.b = b0;
    y.a = a1;
    y.b = b1;
    eq_o = (x == y);
    ne_o = (x != y);
    ceq_o = (x === y);
    cne_o = (x !== y);
  end

  // CHECK: moore.union_extract
  // CHECK: moore.eq
  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: moore.not
  // CHECK: moore.blocking_assign
endmodule
