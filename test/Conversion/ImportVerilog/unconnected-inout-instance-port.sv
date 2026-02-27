// RUN: circt-verilog %s --no-uvm-auto-include --ir-moore | FileCheck %s

module child(inout wire io);
endmodule

module top;
  // Regression: unconnected inout instance ports should synthesize a local
  // placeholder lvalue instead of hard-failing as unsupported.
  child u0(.io());
endmodule

// CHECK-LABEL: moore.module @top
// CHECK: moore.net
// CHECK: moore.instance "u0"
