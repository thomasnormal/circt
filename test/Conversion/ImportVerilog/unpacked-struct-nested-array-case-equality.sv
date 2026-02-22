// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module UnpackedStructNestedArrayCaseEquality(input logic [1:0] a00,
                                             input logic [1:0] a01,
                                             input logic b0,
                                             input logic [1:0] a10,
                                             input logic [1:0] a11,
                                             input logic b1,
                                             output logic ceq_o,
                                             output logic cne_o);
  typedef logic [1:0] a2_t [0:1];
  typedef struct {
    a2_t a;
    logic b;
  } s_t;
  s_t x, y;

  always_comb begin
    x.a[0] = a00;
    x.a[1] = a01;
    x.b = b0;
    y.a[0] = a10;
    y.a[1] = a11;
    y.b = b1;
    ceq_o = (x === y);
    cne_o = (x !== y);
  end

  // CHECK: moore.struct_extract
  // CHECK: moore.uarray_cmp eq
  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: moore.not
  // CHECK: moore.blocking_assign
endmodule
