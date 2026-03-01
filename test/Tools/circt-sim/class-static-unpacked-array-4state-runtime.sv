// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=100000000 2>&1 | FileCheck %s

// Regression for issue #25: static unpacked arrays of 4-state class members
// must preserve assigned values.

module tb;
  class C;
    logic [7:0] arr[4];
    bit   [7:0] good[4];
    logic [7:0] dyn[];
  endclass

  C c;

  initial begin
    c = new;
    c.arr[0] = 8'hAB;
    c.good[0] = 8'hAB;
    c.dyn = new[2];
    c.dyn[0] = 8'hAB;

    $display("arr[0]=%h", c.arr[0]);
    $display("good[0]=%h", c.good[0]);
    $display("dyn[0]=%h", c.dyn[0]);
    $finish;
  end

  // CHECK: arr[0]=ab
  // CHECK: good[0]=ab
  // CHECK: dyn[0]=ab
endmodule
