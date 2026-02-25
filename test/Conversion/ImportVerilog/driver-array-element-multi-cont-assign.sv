// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module leaf(output logic o);
  assign o = 1'b0;
endmodule

module top;
  logic [1:0] a;
  leaf u0(.o(a[0]));
  leaf u1(.o(a[0]));

  // CHECK: moore.module @top
  // CHECK: %u0.o = moore.instance "u0" @leaf()
  // CHECK: moore.assign
  // CHECK: %u1.o = moore.instance "u1" @leaf()
  // CHECK: moore.assign
endmodule
