// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $stime — 32-bit simulation time (IEEE 1800-2017 20.7.1)
`timescale 1ns/1ns
module top;
  integer t32;
  longint t64;

  initial begin
    // At time 0, both should be 0
    t32 = $stime;
    t64 = $time;
    // CHECK: stime_0=0
    $display("stime_0=%0d", t32);
    // CHECK: time_0=0
    $display("time_0=%0d", t64);

    #25;
    t32 = $stime;
    t64 = $time;
    // CHECK: stime_25=25
    $display("stime_25=%0d", t32);
    // CHECK: time_25=25
    $display("time_25=%0d", t64);

    // Verify $stime and $time agree for small values
    // CHECK: agree=1
    $display("agree=%0d", t32 == t64);

    $finish;
  end
endmodule
