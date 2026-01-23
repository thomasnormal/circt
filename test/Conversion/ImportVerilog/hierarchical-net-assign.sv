// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | FileCheck %s
// REQUIRES: slang

module top(input i);
  A A();
  assign A.i = i;
endmodule

module A;
  wire i;
endmodule

// CHECK: hw.module @top
// CHECK: hw.module private @A
