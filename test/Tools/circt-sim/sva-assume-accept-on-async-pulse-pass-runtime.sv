// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assumption failed

// Runtime semantics: async accept_on(c) on an assume must vacuously pass when
// c pulses high between sampled assertion clock edges.

module top;
  reg clk;
  reg a, b, c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent sampled at cycle 2
    b = 1'b0;

    #1 c = 1'b1;    // async pulse between sampled edges
    #1 c = 1'b0;

    @(posedge clk); // cycle 2: would fail without async accept_on handling
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 3
    $finish;
  end

  assume property (@(posedge clk) accept_on(c) (a |-> ##1 b));
endmodule
