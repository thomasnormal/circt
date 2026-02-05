// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

module ConcatCompound;
  logic [9:0] hi;
  logic [9:0] lo;
  logic [19:0] mask;

  always_comb begin
    {hi, lo} &= mask;
  end
endmodule

// CHECK: hw.module @ConcatCompound
