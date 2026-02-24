// RUN: circt-verilog %s --ignore-unknown-modules --ir-hw | FileCheck %s

module tb;
  unknown_mod #(.N(1)) dut();
endmodule

// CHECK: hw.module @tb
