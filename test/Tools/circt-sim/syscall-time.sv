// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $time, $stime, $simtime
`timescale 1ns/1ns
module top;
  initial begin
    // CHECK: time_0=0
    $display("time_0=%0d", $time);

    #10;
    // CHECK: time_10=10
    $display("time_10=%0d", $time);

    // $stime returns 32-bit time
    // CHECK: stime_10=10
    $display("stime_10=%0d", $stime);

    #5;
    // CHECK: time_15=15
    $display("time_15=%0d", $time);

    $finish;
  end
endmodule
