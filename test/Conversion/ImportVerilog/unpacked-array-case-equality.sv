// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module UnpackedArrayCaseEquality(input logic [1:0] a0, input logic [1:0] a1,
                                 input logic [1:0] b0, input logic [1:0] b1,
                                 output logic ceq_o, output logic cne_o);
  typedef logic [1:0] arr_t [0:1];
  arr_t x, y;

  always_comb begin
    x[0] = a0;
    x[1] = a1;
    y[0] = b0;
    y[1] = b1;
    ceq_o = (x === y);
    cne_o = (x !== y);
  end

  // CHECK: moore.uarray_cmp eq
  // CHECK: moore.uarray_cmp ne
  // CHECK: moore.blocking_assign
endmodule
