// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=300000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_CONCAT_ORDER
// CHECK-NOT: SVA assertion failed at time

// Positive control for ordered concat semantics.
//
// b is sampled high at +1 and c at +2 relative to the antecedent trigger,
// which satisfies ((##[1:2] b) ##[1:2] c).

module top;
  reg clk;
  reg a;
  reg b;
  reg c;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    a = 1'b0;
    b = 1'b1;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b1;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    repeat (3) @(posedge clk);
    $display("SVA_PASS_CONCAT_ORDER");
    $finish;
  end

  assert property (@(posedge clk) a |-> ((##[1:2] b) ##[1:2] c));
endmodule
