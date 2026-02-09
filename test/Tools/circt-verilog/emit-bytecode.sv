// Test that circt-verilog can emit MLIR bytecode format.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include --emit-bytecode -o %t.mlirbc
// RUN: circt-opt %t.mlirbc | FileCheck %s

// CHECK: hw.module @top
module top(input logic clk, output logic q);
  always_ff @(posedge clk)
    q <= ~q;
endmodule
