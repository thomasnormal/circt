// RUN: circt-verilog --no-uvm-auto-include --ir-llhd %s | FileCheck %s
// REQUIRES: slang

module ImmediateAssertProcGlobalStaticInit(input logic i);
  always @* assert (i);
endmodule

// CHECK: llvm.mlir.global internal @__circt_proc_assertions_enabled(true)
// CHECK-NOT: llvm.mlir.global_ctors ctors = [@__moore_global_init___circt_proc_assertions_enabled]
