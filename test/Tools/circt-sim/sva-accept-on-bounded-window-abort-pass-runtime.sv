// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: accept_on cleared bounded pending obligation
// CHECK-NOT: SVA assertion failed

// Runtime semantics: accept_on(c) must clear pending implication obligations,
// including bounded windows, once c becomes true while they are active.

module top;
  reg clk;
  reg a, b, c;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent sampled at cycle 2

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b0;       // keep consequent false

    @(negedge clk);
    c = 1'b1;       // abort while ##[1:3] obligation is pending

    @(posedge clk); // cycle 3
    c = 1'b0;

    @(posedge clk); // cycle 4
    @(posedge clk); // cycle 5
    $display("SVA_PASS: accept_on cleared bounded pending obligation");
    $finish;
  end

  assert property (@(posedge clk) accept_on(c) (a |-> ##[1:3] b));
endmodule
