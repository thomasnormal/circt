// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: sync_accept_on sampled-high abort
// CHECK-NOT: SVA assertion failed

// Runtime semantics: sync_accept_on(c) should abort only when c is sampled
// high on the assertion clock edge.

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

    @(negedge clk);
    c = 1'b1;       // visible at next sampled edge

    @(posedge clk); // cycle 2: sync abort triggers
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 3
    $display("SVA_PASS: sync_accept_on sampled-high abort");
    $finish;
  end

  assert property (@(posedge clk) sync_accept_on(c) (a |-> ##1 b));
endmodule
