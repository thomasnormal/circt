// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
`timescale 10ns/1ns
module top;
  initial begin
    // $timeformat(units, precision, suffix, min_width)
    $timeformat(-9, 2, " ns", 10);
    #15;
    // After #15 with timescale 10ns, that's 150ns
    // CHECK: time=150.00 ns
    $display("time=%t", $time);
    $finish;
  end
endmodule
