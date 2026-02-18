// RUN: circt-verilog %s --no-uvm-auto-include --language-version 1800-2023 -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
`timescale 1ns/1ps
module top;
  initial begin
    // $timeprecision returns the precision exponent (1ps = 10^-12 → -12)
    // CHECK: timeprecision=-12
    $display("timeprecision=%0d", $timeprecision);
    $finish;
  end
endmodule
