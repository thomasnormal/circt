// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module child(input logic in);
endmodule

module top;
  // CHECK-LABEL: moore.module @top
  // CHECK: %[[VAR:.*]] = moore.variable : <l1>
  // CHECK: %[[READ:.*]] = moore.read %[[VAR]] : <l1>
  // CHECK: moore.instance {{.*}}(in: %[[READ]]:
  logic valid;
  child u(.in(valid));
  initial valid = 1'b1;
endmodule
