// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
`timescale 1ns/1ps
module top;
  initial begin
    // $printtimescale should display the timescale of the current module
    // CHECK: 1ns
    // CHECK: 1ps
    $printtimescale;
    $finish;
  end
endmodule
