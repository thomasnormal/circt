// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
`timescale 1ns/1ps
module top;
  initial begin
    // $timeunit returns the time unit exponent (1ns = 10^-9 → -9)
    // CHECK: timeunit=-9
    $display("timeunit=%0d", $timeunit);
    $finish;
  end
endmodule
