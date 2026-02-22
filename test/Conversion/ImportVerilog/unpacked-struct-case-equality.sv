// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module UnpackedStructCaseEquality(input logic [1:0] a0, input logic b0,
                                  input logic [1:0] a1, input logic b1,
                                  output logic ceq_o, output logic cne_o);
  typedef struct {
    logic [1:0] a;
    logic b;
  } s_t;
  s_t x, y;

  always_comb begin
    x.a = a0;
    x.b = b0;
    y.a = a1;
    y.b = b1;
    ceq_o = (x === y);
    cne_o = (x !== y);
  end

  // CHECK: moore.struct_extract
  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: moore.not
  // CHECK: moore.blocking_assign
endmodule
