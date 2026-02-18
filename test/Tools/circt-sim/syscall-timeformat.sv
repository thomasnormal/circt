// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $timeformat
`timescale 1ns/1ps
module top;
  initial begin
    $timeformat(-9, 3, " ns", 15);
    #42;
    // CHECK: time= 42.000 ns
    $display("time=%t", $time);
    $finish;
  end
endmodule
