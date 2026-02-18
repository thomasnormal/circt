// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
`timescale 1ns/1ps
module top;
  realtime t;

  initial begin
    #10;
    t = $realtime;
    // CHECK: realtime=10
    $display("realtime=%0d", $rtoi(t));
    $finish;
  end
endmodule
